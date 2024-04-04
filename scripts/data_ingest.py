# %%
#
from __future__ import annotations

import gzip
import json
import math
import pickle
import random
import shutil
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Tuple
from xml.etree import ElementTree

import git
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import pykanto.plot as pyplot
import ray
import seaborn as sns
import soundfile as sf
import umap
from grpc import dynamic_ssl_server_credentials
from idna import uts46_remap
from imageio import save
from imblearn.ensemble import BalancedRandomForestClassifier
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from numba import njit
from pykanto.dataset import SongDataset
from pykanto.parameters import Parameters
from pykanto.signal.analysis import (approximate_minmax_frequency,
                                     spec_centroid_bandwidth)
from pykanto.signal.filter import (dereverberate, hz_to_mel_lib, mels_to_hzs,
                                   norm)
from pykanto.signal.segment import (ReadWav, drop_zero_len_units,
                                    get_segment_info, segment_files,
                                    segment_files_parallel)
from pykanto.signal.spectrogram import (get_indv_units, get_unit_spectrograms,
                                        get_vocalisation_units,
                                        pad_spectrogram, retrieve_spectrogram)
from pykanto.utils.compute import (calc_chunks, dictlist_to_dict, flatten_list,
                                   get_chunks, print_parallel_info,
                                   to_iterator, tqdmm)
from pykanto.utils.custom import parse_sonic_visualiser_xml
from pykanto.utils.paths import (ProjDirs, get_file_paths, get_wav_filepaths,
                                 get_wavs_w_annotation, link_project_data)
from pykanto.utils.read import read_json
from pykanto.utils.write import makedir
from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.analysis import (extract_flatness, extract_mfccs,
                          extract_minmax_frequencies,
                          extract_spectral_centroids_bw)
from src.io import chipper_move_remaining, chipper_units_to_json

# %load_ext autoreload
# %autoreload 2


# %%──── METHODS ───────────────────────────────────────────────────────────────


# Parse Sonic Visualiser xml files
# NOTE: XML file name has to be strictly filename.xml - to go with filename.wav


# external drive or


# %%
# PETREL

# Find the project's root directory.
PROJECT_ROOT = Path(
    git.Repo('.', search_parent_directories=True).working_tree_dir)

# Name of the folder containing the data that you want to segment
DATASET = 'STORM_PETREL_2021'
RAW_DATA = PROJECT_ROOT / 'data' / 'raw' / DATASET

DATA_LOCATION = Path('/media/nilomr/My Passport/SONGDATA')
link_project_data(DATA_LOCATION, PROJECT_ROOT / 'data')

DIRS = ProjDirs(PROJECT_ROOT, RAW_DATA)


# %%
# Load an existing dataset
DATASET_ID = "STORM_PETREL_SONGS"
out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
dataset = pickle.load(open(out_dir, "rb"))
# %%


# %% Create a new class inheriting the ReadWav and SegmentMetadata classes to
# add any extra fields that your project might require. You can then
# monkey-patch the original classes, saving yourself from having to define the
# full classes and their methods again from scratch.
#
# Note: For consistency, you might also want to extend the dictionary types
# returned by your custom methods to include any new keys and their data types.


# class CustomReadWav(ReadWav):
#     def get_metadata(self) -> Dict[str, Any]:
#         add_to_dict = {
#             'tags': str(self.all_metadata['tags'])
#         }
#         return {**self.metadata, **add_to_dict}


# ReadWav = CustomReadWav


# class CustomSegmentMetadata(SegmentMetadata):
#     def get(self) -> Dict[str, Any]:
#         new_dict = {
#             'tags': self.all_metadata['tags']
#         }
#         return {**self.metadata, **new_dict}


# SegmentMetadata = CustomSegmentMetadata


# Build a list of tuples containing paths to the wav file and its annotations
# (here, in .xml format). Wav files and their annotations should have the same
# file names. This is not automated to give the user more flexibility to use
# diferent file and directory systems.

wav_filepaths = get_file_paths(DIRS.RAW_DATA, ['.wav'])
xml_filepaths = get_file_paths(DIRS.RAW_DATA, ['.xml'])
datapaths = get_wavs_w_annotation(wav_filepaths, xml_filepaths)


# The `min_duration` and `min_freqrange` parameters can help you filter out
# clearly erroneous segmentation annotations (too short or too narrow).

# Settings
min_duration: float = .1
min_freqrange: int = 200
resample: int = 22050
labels_to_ignore: List[str] = ["FIRST", "NOISE"]

# Make sure output folders exists
wav_outdir = makedir(DIRS.SEGMENTED / "WAV")
json_outdir = makedir(DIRS.SEGMENTED / "JSON")


# %%

segment_files(datapaths,
              wav_outdir,
              json_outdir,
              resample=resample,
              parser_func=parse_sonic_visualiser_xml,
              min_duration=min_duration, min_freqrange=min_freqrange,
              labels_to_ignore=labels_to_ignore)

segment_files_parallel(
    datapaths,
    wav_outdir,
    json_outdir,
    resample=resample,
    parser_func=parse_sonic_visualiser_xml,
    min_duration=min_duration,
    min_freqrange=min_freqrange,
    labels_to_ignore=labels_to_ignore)


# %%──── SETTINGS ──────────────────────────────────────────────────────────────


# Get info on available song selections
lengths_dict = get_segment_info(DIRS.RAW_DATA, min_duration, min_freqrange)

# Get list of wav files to segment
wav_filelist = get_wav_filepaths(DIRS.RAW_DATA)

# Plot segment lengths
ax = sns.histplot(lengths_dict, bins=50, log_scale=False)


# ────── MAIN ───────────────────────────────────────────────────────────────────


# %%
# Segment songs into units

# Recording quality (SNR, etc.) is too variable to do this automatically,
# so I used Chipper (REF) to find optimal parameters for each recording.


# Methods to help segmentation:

# 1. Chipper does not skip songs that already have segmentation information
# when the app is restarted, so this function moves any song that remains to be
# segmented to a different folder ('to_annotate').
# Run this after each annotation session.

to_annotate = DIRS.SEGMENTED / 'WAV' / 'to_annotate'
annotated = DIRS.SEGMENTED / 'WAV'
chipper_move_remaining(DIRS.SEGMENTED, annotated, to_annotate)

# %%

# 2. Chipper outputs segmentation information to a gzip file. Let's add this to
# the JSON metadata created above.

directory = DIRS.SEGMENTED
chipper_units_to_json(directory)


# %%──── PREPARE DATASET ───────────────────────────────────────────────────────

# Define parameters
params = Parameters(
    # Spectrogramming
    window_length=512,
    hop_length=32,
    n_fft=2048,
    num_mel_bins=240,
    sr=22050,
    top_dB=65,                  # top dB to keep
    lowcut=300,
    highcut=10000,
    dereverb=True,
    # general settings
    song_level=True,
    subset=(0, -1),
    verbose=False,
)

# np.random.seed(123)
# random.seed(123)
DATASET_ID = "STORM_PETREL_SONGS"
dataset = SongDataset(
    DATASET_ID, DIRS, parameters=params, overwrite_dataset=True,
    overwrite_data=False)

# Segmente into individual units using information from chipper,
# then check a few.
dataset.segment_into_units()

# %%
for key in dataset.vocalisations.index[:3]:
    dataset.plot_vocalisation_segmentation(
        key, cmap='bone')


# %%

dataset.show_extreme_samples(
    n_songs=2, query='duration',
    order='descending'
)

k = dataset.vocalisations.index[0]

get_vocalisation_units(dataset, k, song_level=False)
indvus = get_indv_units(
    dataset, dataset.vocalisations.index, individual='faroes')

sample_sizes = dataset.vocalisations['ID'].value_counts()

dataset.vocalisations[dataset.vocalisations.isna().any(axis=1)]

dataset.segment_into_units()

# %%

dataset.parameters.update(song_level=False, num_cpus=1)
dataset.get_units()
dataset.cluster_individuals()
# TODO:

# %%
dataset.vocalisations.dropna(inplace=True)

dataset.save_to_disk()
dataset = dataset.reload()

dataset.prepare_interactive_data(spec_length=8)
dataset.open_label_app()


dataset.vocalisations = dataset.vocalisations.sample(frac=0.5)


# Get mean and SD of MFCC for average purr note or breathe note

# What notes to get


kk = extract_spectral_centroids_bw(dataset, note_type='breathe')
kk = extract_spectral_centroids_bw(dataset, note_type='purr')


plt.imshow(note)


plt.figure(figsize=(25, 10))

note = np.hstack(notes[0])

key = dataset.vocalisations.index[50]
wavpath = dataset.vocalisations.at[key, 'wav_file']

spec = retrieve_spectrogram(dataset.vocalisations.at
                            [key, 'spectrogram_loc'])

# centroid, bandwidth = spec_centroid_bandwidth(dataset, key=key, plot=True)

for key in dataset.vocalisations.index[50:58]:
    minfreqs, maxfreqs = approximate_minmax_frequency(
        dataset, key=key, plot=True)

# %%
#
# Extract song features for analysis
#

atlantic_pops = ['wales', 'faroes', 'scotland',
                 'norway', 'ireland', 'iceland',
                 'molene', 'montana_clara']

features_dict = {
    'ID': dataset.vocalisations.ID.values,
    'group': ['pelagicus' if x in atlantic_pops
              else 'melitensis'
              for x in dataset.vocalisations.ID.values],
    'mean_purr_duration': [np.mean(
        r[:-1]) for r in dataset.vocalisations.unit_durations],
    'std_purr_duration': [np.std(
        r[:-1]) for r in dataset.vocalisations.unit_durations],
    'breathe_duration': [r[-1]
                         for r in dataset.vocalisations.unit_durations],
    'purr_duration': [np.sum(r[:-1])
                      for r in dataset.vocalisations.unit_durations],
    'mean_silence_duration': [
        np.mean(r[:-1]) for r in dataset.vocalisations.silence_durations],
    'std_silence_duration': [
        np.std(r[:-1]) for r in dataset.vocalisations.silence_durations],
    'n_purr_notes': [len(r[:-1]) for r in dataset.vocalisations.unit_durations],
    'song_length': dataset.vocalisations.length_s.values,
    'purr_ioi': [np.mean((
        np.append(r[1:], 0) - r)[:-1]) for r in dataset.vocalisations.onsets],
    'purr_ioi_std': [np.std((
        np.append(r[1:], 0) - r)[:-1]) for r in dataset.vocalisations.onsets],
}


for nt in ['breathe', 'purr']:
    features_dict[f"{nt}_mfcc"] = extract_mfccs(
        dataset, note_type=nt, n_mfcc=12)
    features_dict[f'{nt}_centroid_bw'] = extract_spectral_centroids_bw(
        dataset, note_type=nt)
    features_dict[f'{nt}_minmax_freq'] = extract_minmax_frequencies(
        dataset, note_type=nt)
    features_dict[f'{nt}_flatness'] = extract_flatness(
        dataset, note_type=nt)


features = pd.DataFrame(features_dict, index=dataset.vocalisations.index)

arraycols = [
    col for col in features.columns
    if isinstance(features[col][0],
                  np.ndarray)]

dfs = []
for col in arraycols:
    tmpdf = features[col].apply(pd.Series)
    dfs.append(tmpdf.rename(columns=lambda x: f'{col}_{str(x)}'))

features = pd.concat(dfs + [features], axis=1).drop(columns=arraycols)

# %%
# Random Forest

features.dropna(inplace=True)
#features = features[~features['ID'].isin(['norway', 'montana_clara'])]

['purr_centroid_bw']

features.groupby(['group'])['purr_minmax_freq_2'].agg(['mean', 'std'])

melitensis_maxfreq = features.query('group == "melitensis"')[
    'purr_minmax_freq_2'].values
pelagicus_maxfreq = features.query('group == "pelagicus"')[
    'purr_minmax_freq_2'].values

features[['group', 'purr_minmax_freq_2']]

ax = sns.boxplot(
    x='group', y='breathe_minmax_freq_0',
    data=features[['group', 'breathe_minmax_freq_0']], color=".9", whis=np.inf)

ax = sns.stripplot(
    x='group', y='breathe_minmax_freq_0',
    data=features[['group', 'breathe_minmax_freq_0']])
# Remove the marettimo population: only 4 observations

subspecies = True
conf_mat, feature_importances, report, samples = [], [], [], []
min_sample = 20


for i in tqdmm(range(100)):

    features_equalclass = features.groupby(
        'ID').filter(lambda x: len(x) > min_sample)
    features_equalclass = features_equalclass.groupby('ID').apply(
        lambda x: x.sample(
            min(features_equalclass.ID.value_counts()),
            replace=True)).droplevel('ID')

    if subspecies:
        features_equalclass = (
            features_equalclass.groupby('group').apply(
                lambda x: x.sample(
                    min(features_equalclass.group.value_counts())))
            .droplevel('group'))
        y = features_equalclass.group.values

    else:
        y = features_equalclass.ID.values

    samples.append(features_equalclass.index.values)

    X = features_equalclass.drop(['ID', 'group'], axis=1)
    strata = features_equalclass.ID.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=i, stratify=strata)

    # Randomise y
    #y_train = np.random.choice(y_train, size=len(y_train))

    features.ID.value_counts()
    features_equalclass.ID.value_counts()
    features_equalclass.group.value_counts()

    #  Scale features
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(sc.transform(X_test), columns=X.columns)

    # Train and fir Random Forest
    randforest = BalancedRandomForestClassifier(
        n_estimators=100, random_state=i, class_weight='balanced_subsample',
        max_features='sqrt')
    randforest.fit(X_train, y_train)
    y_pred = randforest.predict(X_test)

    accuracy_score(y_test, y_pred)

    conf_mat.append(confusion_matrix(y_test, y_pred, normalize='all'))
    feature_importances.append(randforest.feature_importances_)
    report.append(
        classification_report(
            y_test, y_pred, target_names=np.unique(y_test),
            output_dict=True))


report_df = pd.concat([pd.DataFrame(x) for x in [report]], axis=0)
report_df.reset_index(level=0, inplace=True)
report_df[report_df['index'] == 'precision']


# %%
# Plot confusion matrix

figsize = (7, 7)

conf_mat = np.array(conf_mat)
mean_conf_mat = np.sum(conf_mat, axis=0) / conf_mat.shape[0]
mean_conf_mat = mean_conf_mat.astype(
    'float') / mean_conf_mat.sum(axis=1)[:, np.newaxis]

if subspecies:
    labels = ['$\it{H. p. }$'f'$\it{i}$' for i in np.unique(y_test)]
else:
    labels = [x.title().replace("_", " ") for x in np.unique(y_test)]

fig, ax = plt.subplots(figsize=figsize)
sns.set(font_scale=1.4)  # for label size
sns.heatmap(
    data=mean_conf_mat,
    annot=True,
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"size": 16},
    square=True, fmt='.2f',
    cbar=False,
    cmap="BuPu")

ax.set_xlabel("\nPredicted", fontsize=16)
ax.set_ylabel("True label\n", fontsize=16)
plt.xticks(rotation=45, ha='right')

plt.show()


key = dataset.vocalisations.query('ID=="greece"').index[50]
key = dataset.vocalisations.query('ID=="faroes"').index[50]


# %%

# Get and save feature importance from RF runs
feature_importances = np.array(feature_importances)
feats = [(X_train.columns[i], feature_importances[:, i])
         for i in range(len(X_train.columns))]
feats_df = pd.DataFrame(
    feats, columns=['feature', 'value']).explode(
    column='value')
feats_df.to_csv(
    DIRS.RESOURCES /
    f"{'sbsp_' if subspecies else ''}rf_feature_importance.csv",
    index=False)


# Quick plot of mean feature importance
# Sort the feature importance in descending order
mean_feature_importances = np.sum(
    feature_importances, axis=0) / feature_importances.shape[0]
sorted_indices = np.argsort(mean_feature_importances)[::-1]

fig, ax = plt.subplots(figsize=(19, 5))
plt.title('Feature Importance')
plt.bar(
    range(X_train.shape[1]),
    mean_feature_importances[sorted_indices],
    align='center')
plt.xticks(
    range(X_train.shape[1]),
    X_train.columns[sorted_indices],
    rotation=90)
plt.tight_layout()
plt.show()

# %%
# Prepare features

include = X_train.columns[sorted_indices][:4]
exclude = [col for col in features.columns if col not in include]


labels = features.group
reduce_features = features.drop(exclude, axis=1)
scaled_features = StandardScaler().fit_transform(
    reduce_features.values)

# %%
# PCA

pca = PCA(n_components=3)
embedding = pca.fit_transform(scaled_features)

# pca = PCA(n_components=2)
# embedding = pca.fit_transform(scaled_features)

# explained_variance = pca.explained_variance_ratio_


# plot
labels = features.group
coldict = {str(x): i for i, x in enumerate(np.unique(labels))}
colours = [sns.color_palette(palette='Set3')[x]
           for x in labels.map(coldict)]

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colours,
)
plt.gca().set_aspect('equal', 'datalim')

# %%

# %%
# UMAP

reducer = umap.UMAP(n_neighbors=15, min_dist=.2)
umap_embedding = reducer.fit_transform(scaled_features)

# %%

back_colour = 'white'
text_colour = '#3d3d3d'
grid_colour = '#f2f2f2'


# Plot
labels = features.ID
if 'pelagicus' in np.unique(labels):
    coldict = {'pelagicus': '#196182', 'melitensis': '#e67717'}
    colours = np.array([x for x in labels.map(coldict)])
else:
    labels = pd.Series([l.title().replace("_", " ") for l in features.ID])
    coldict = {str(x): i for i, x in enumerate(np.unique(labels))}
    colours = np.array([sns.color_palette(palette='Paired')[x]
                        for x in labels.map(coldict)])


fig, ax = plt.subplots(figsize=(10, 10), facecolor=back_colour)

for group in np.unique(labels):
    ix = np.where(labels == group)
    ax.scatter(
        embedding[:, 0][ix],
        embedding[:, 1][ix],
        c=colours[ix],
        alpha=0.7,
        s=80,
        label='$\it{H. p. }$'
        f'$\it{group}$' if 'pelagicus' in np.unique(labels) else group)

ax.legend(loc='lower left', frameon=False,
          labelcolor=text_colour,
          handletextpad=0.1,
          fontsize=12)

ax.set_facecolor(grid_colour)
plt.grid(visible=None)
plt.axis('equal')


plt.xticks([])
plt.yticks([])
ax.set_xlabel('PC1', fontsize=20, labelpad=20, color=text_colour)
ax.set_ylabel('PC2', fontsize=20, labelpad=20, color=text_colour)

plt.show()

# %%
features.explode()
#  Note and silence durations
dataset.vocalisations['mean_purr_duration'] = [
    np.mean(r[:-1]) for r in dataset.vocalisations.unit_durations]
dataset.vocalisations['breathe_note_duration'] = [
    r[-1]
    for r in dataset.vocalisations.unit_durations
]
dataset.vocalisations['mean_silence_duration'] = [
    np.mean(r[:-1]) for r in dataset.vocalisations.silence_durations]

#  MFCCs
dataset.vocalisations['breathe_note_mfcc'] = extract_mfccs(
    dataset, note_type='breathe', n_mfcc=12)
dataset.vocalisations['purr_mfcc'] = extract_mfccs(
    dataset, note_type='purr', n_mfcc=12)

# (mean) Spectral centroids and bandwidths
dataset.vocalisations['breathe_spec_centroid_bw'] = extract_spectral_centroids_bw(
    dataset, note_type='breathe')
dataset.vocalisations['purr_spec_centroid_bw'] = extract_spectral_centroids_bw(
    dataset, note_type='purr')


# Random Forest


# %%

len(mean_sd_mfccs) == len(dataset.vocalisations)

librosa.feature.spectral_centroid(S=)

print(dataset.DIRS.VOCALISATION_LABELS['already_checked'])

dataset.parameters.song_level = False
units = [get_vocalisation_units(
    dataset, key, song_level=False) for key in dataset.vocalisations.index]


units = dictlist_to_dict(units)

key = list(units)[0]

len(units[key])
pyplot.melspectrogram(units[key][-1])
pykplt

plt.imshow(units[key][-1])

max_frames = max([unit.shape[1]for ls in units.values() for unit in ls])

# %%
units = {key: [pad_spectrogram(spec, max_frames)
               for spec in ls] for key, ls in units.items()}

for key, ls in units.items():
    for spec in ls:
        try:
            pad_spectrogram(spec, max_frames)
        except:
            print('Nope', f'{spec.shape=}')

# %%

# Load an existing dataset
DATASET_ID = "STORM_PETREL_SONGS"
out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
dataset = pickle.load(open(out_dir, "rb"))


# %%
dataset.sample_info()
dataset.summary_plot(variable='all')

# %%
for key in dataset.vocalisations.index[:3]:
    dataset.plot_vocalisation_segmentation(
        key, cmap='bone')
# %%


# %%


# %%
# Tell the dataset that analyses will be at the unit level
dataset.parameters.update(song_level=False)
dataset.segment_into_units(overwrite=True)
dataset.get_units()
dataset.cluster_individuals()
dataset.prepare_interactive_data()
dataset.open_label_app()


sr = dataset.parameters.sr
for key in dataset.vocalisations.index[:3]:

    spectrogram = retrieve_spectrogram(dataset.vocalisations.at
                                       [key, 'spectrogram_loc'])
    pyplot.melspectrogram(spectrogram, parameters=dataset.parameters)

    uspecs = get_unit_spectrograms(
        spectrogram, dataset.vocalisations.at[key, 'onsets'],
        dataset.vocalisations.at[key, 'offsets'],
        sr=dataset.parameters.sr, hop_length=dataset.parameters.hop_length)

    onsets = dataset.vocalisations.at[key, 'onsets'][-1]
    offsets = dataset.vocalisations.at[key, 'offsets'][-1]

    onsets = math.floor(onsets * sr / dataset.parameters.hop_length)
    offsets = math.floor(offsets * sr / dataset.parameters.hop_length)

    c = spectrogram[:, onsets:offsets]
    # units.append(unit)

    pyplot.melspectrogram(uspecs[-1], parameters=dataset.parameters)
    plt.imshow(uspecs[-1])

it = 0
for spec in uspecs:
    try:
        pyplot.melspectrogram(spec, parameters=dataset.parameters)
        it += 1

    except:
        pass

pyplot.melspectrogram(uspecs[-1], parameters=dataset.parameters)

pyplot.melspectrogram(
    spectrogram
    [:,
     int(2 * dataset.parameters.hop_length_ms):
     int(3 * dataset.parameters.hop_length_ms)],
    parameters=dataset.parameters)

spectrogram[:, int(0.99174603*dataset.parameters.hop_length_ms):int(1.536009*dataset.parameters.hop_length_ms)]

((N-OVERLAP)/SAMPLING)*1000  # from chipper defaults
s_per_frame = ((1024-1010)/22050)
t = (2570)*s_per_frame

math.floor(t * 22050 / 64)

math.floor((2570 * ((1024-1010)/22050)) * 1010/22050)


pyplot.segmentation(dataset, key)
key = spkey
# plot??


def segmentation(
        dataset: SongDataset, key: str = None,
        spectrogram: bool | np.ndarray = False,
        onsets_offsets: bool | Tuple[np.ndarray, np.ndarray] = False, **kwargs) -> None:
    """
    Plots a vocalisation and overlays the results
    of the segmentation process.

    Args:
        dataset (SongDataset): A SongDataset object.
        key (str, optional): Vocalisation key. Defaults to None.
        spectrogram (bool | np.ndarray, optional): [description].
            Defaults to False.
        onsets_offsets (bool | Tuple[np.ndarray, np.ndarray], optional):
            Tuple containing arrays with unit onsets and offsets. Defaults to False.
        kwargs: Keyword arguments to be passed to
                :func:`~pykanto.plot.melspectrogram`
    """

    params = dataset.parameters
    if not isinstance(spectrogram, np.ndarray):
        spectrogram = retrieve_spectrogram(dataset.vocalisations.at
                                           [key, 'spectrogram_loc'])
        onsets_offsets = [dataset.vocalisations.at[key, i]
                          for i in ['onsets', 'offsets']]

    ax = melspectrogram(spectrogram, parameters=params,
                        title=key if key else '')

    # Add unit onsets and offsets
    ylmin, ylmax = ax.get_ylim()
    ysize = (ylmax - ylmin) * 0.05
    ymin = ylmax - ysize
    patches = []
    for onset, offset in zip(onsets_offsets[0], onsets_offsets[1]):
        # ax.axvline(onset, color="#FFFFFF", ls="-", lw=0.5, alpha=0.3)
        # ax.axvline(offset, color="#FFFFFF", ls="-", lw=0.5, alpha=0.3)
        patches.append(Rectangle(xy=(onset, ylmin),
                                 width=offset - onset, height=(ylmax - ylmin)))
    collection = PatchCollection(patches, color="red", alpha=0.7)
    ax.add_collection(collection)
    plt.show()
