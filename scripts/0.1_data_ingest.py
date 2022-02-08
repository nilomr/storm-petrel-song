# %%
#
from __future__ import annotations
from pykanto.signal.spectrogram import retrieve_spectrogram
from numba.core.decorators import njit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import gridspec
import librosa.display
from typing import TYPE_CHECKING, Dict, List, Tuple
import warnings

import datetime as dt
import gzip
import json
import math
import os
import pickle
import random
import re
import shutil
import signal
import wave
from pathlib import Path
from typing import Any, Dict, List, Tuple
from xml.etree import ElementTree

import audio_metadata
import git
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pykanto.plot as pyplot
import ray
import seaborn as sns
import soundfile as sf
from audio_metadata.formats import id3v2frames
from librosa.core.spectrum import perceptual_weighting
from pykanto import parameters
from pykanto.dataset import SongDataset
from pykanto.parameters import Parameters
from pykanto.signal.filter import dereverberate
from pykanto.signal.segment import get_segment_info, segment_into_songs
from pykanto.signal.spectrogram import (get_indv_units, get_unit_spectrograms,
                                        get_vocalisation_units,
                                        pad_spectrogram, retrieve_spectrogram)
from pykanto.utils.compute import (dictlist_to_dict, flatten_list, to_iterator,
                                   tqdmm)
from pykanto.utils.paths import ProjDirs, get_wav_filepaths, link_project_data
from pykanto.utils.read import read_json
from pykanto.utils.write import safe_makedir

# %load_ext autoreload
# %autoreload 2

# %%──── METHODS ───────────────────────────────────────────────────────────────


def segment_into_songs(
    DATA_DIR: Path,
    WAV_FILEDIR: Path,
    RAW_DATA_ID: str,
    min_duration: float = .5,
    min_freqrange: int = 200,
    species: str = "Big Bird",
    ignore_labels: List[str] = ["FIRST", "first"],
    resample: int | None = 22050
) -> None:
    """
    Segments long .wav files recorderd with AudioMoth units into shorter
    segments, using segmentation metadata from .xml files output by
    Sonic Visualiser. Works well with large files (only reads one chunk at a
    time).

    Note:
        .xml files should be in the same folder as the .wav files.

    Args:
        DATA_DIR (Path): Directory where to output files (will create subfolders).
        WAV_FILEDIR (Path): Path to wav file.
        RAW_DATA_ID (str): Name of dataset.
        min_duration (float, optional): Minimum duration for a segment to be
            considered (in seconds). Defaults to .5.
        min_freqrange (int, optional): Minimum frequency range for a segment to be
            considered (in hertz). Defaults to 200.
        species (str, optional): Species name. Defaults to "Big Bird".
        ignore_labels (List[str], optional): Ignore segments with these labels.
            Defaults to ["FIRST", "first"].

    Raises:
        Exception: Will raise an exception if sample rates in the file and the
        metadata do not coincide. You should check why.
    """
    wavfile_str = str(WAV_FILEDIR)
    XML_FILEDIR = WAV_FILEDIR.parents[0] / str(WAV_FILEDIR.stem + ".xml")

    if Path(XML_FILEDIR).is_file():
        with sf.SoundFile(wavfile_str) as wavfile:

            if not wavfile.seekable():
                raise ValueError(
                    f"Cannot seek through this file ({wavfile_str}).")

            sr = wavfile.samplerate

            # Get xml metadata for this file.
            # This xml file structure is particular to Sonic Visualiser
            root = ElementTree.parse(XML_FILEDIR).getroot()

            # Get sample rate of file
            xml_sr = int(root.findall('data/model')[0].get('sampleRate'))
            species = species  # Here you would obtain the species label.

            # Get minimum n of frames for a segment to be kept
            min_frames = min_duration * sr

            # Prepare audio metadata
            mtdt = audio_metadata.load(WAV_FILEDIR)
            metadata = {
                'bit_depth': mtdt["streaminfo"].bit_depth,
                'bitrate': mtdt["streaminfo"].bitrate
            }

            # Where to save output?
            OUT_DIR = DATA_DIR / "segmented" / RAW_DATA_ID

            # Iterate over segments and save them (+ metadata)
            for cnt, segment in enumerate(root.findall('data/dataset/point')):
                # Ignore very short segments, segments that have very narrow
                # bandwidth, and anything passed to `ignore_labels`.
                if (int(
                        float(segment.get('duration'))) < min_frames or int(
                        float(segment.get('extent'))) < min_freqrange or segment.get(
                        'label') in ignore_labels):
                    continue
                else:
                    save_segment(segment, WAV_FILEDIR, OUT_DIR, species,
                                 sr, wavfile, metadata, cnt, resample=resample)


def save_segment(
    segment: Dict[str, object],
    WAV_FILEDIR: Path,
    OUT_DIR: Path,
    species: str, sr: int,
    wavfile: sf.SoundFile,
    metadata: Dict[str, Any],
    cnt: int,
    resample: int | None = 22050
) -> None:
    """
    Save wav and json files for a single song segment present in WAV_FILEDIR

    Args:
        segment (Dict[str, object]): [description]
        WAV_FILEDIR (Path): [description]
        OUT_DIR (Path): [description]
        species (str): [description]
        sr (int): [description]
        wavfile (sf.SoundFile): [description]
        metadata (Dict[str, Any]): [description]
        cnt (int): [description]
        resample (int, optional): [description]. Defaults to 22050.
    """
    # Extract relevant information from xml file
    start, duration, lowfreq, freq_extent = [
        int(float(segment.get(value)))
        for value in ['frame', 'duration', 'value', 'extent']]
    label = segment.get('label')
    ID = f'{WAV_FILEDIR.parents[0].name}_{WAV_FILEDIR.stem}'

    # Get segment frames
    wavfile.seek(start)
    audio_section = wavfile.read(duration)

    if len(audio_section.shape) == 2:
        audio_section = librosa.to_mono(np.swapaxes(audio_section, 0, 1))
    # Save .wav
    wav_out = (OUT_DIR / "WAV" / f'{ID}_{str(cnt)}.wav')
    safe_makedir(wav_out)

    if resample:
        audio_section = librosa.resample(audio_section, sr, resample)
        sf.write(wav_out.as_posix(), audio_section, resample)
    else:
        sf.write(wav_out.as_posix(), audio_section, sr)

    # Make a JSON dictionary to go with the .wav file
    json_dict = {"species": species,
                 "ID": ID,
                 "label": label,
                 "samplerate_hz": resample if resample else sr,
                 "length_s": len(audio_section) / resample if resample else sr,
                 "lower_freq": lowfreq,
                 "upper_freq": lowfreq + freq_extent,
                 "max_amplitude": float(max(audio_section)),
                 "min_amplitude": float(min(audio_section)),
                 "bit_depth": metadata['bit_depth'],
                 "source_loc": WAV_FILEDIR.as_posix(),
                 "wav_loc": wav_out.as_posix()}

    # Dump json
    json_out = (OUT_DIR / "JSON" / (wav_out.name + ".JSON"))
    safe_makedir(json_out)
    f = open(json_out.as_posix(), "w")
    print(json.dumps(json_dict, indent=2), file=f)
    f.close()


def get_file_paths(root_dir: Path, extensions: List[str]) -> List[Path]:
    """
    Returns paths to files with given extension. Recursive.

    Args:
        root_dir (Path): Root directory to search recursively.
        extensions (List[str]): File extensions to look for (e.g., .wav)

    Raises:
        FileNotFoundError: No files found.

    Returns:
        List[Path]: List with path to files.
    """
    file_list: List[Path] = []
    ext = "".join([f"{x} and/or "
                   if i != len(extensions)-1 and len(extensions) > 1 else f"{x}"
                   for i, x in enumerate(extensions)])

    for root, _, files in os.walk(str(root_dir)):
        for file in files:
            if file.endswith(tuple(extensions)):
                file_list.append(Path(root) / file)
    if len(file_list) == 0:
        raise FileNotFoundError(
            f"There are no {ext} files in this directory")
    else:
        print(f"Found {len(file_list)} {ext} files in {root_dir}")
    return file_list


def open_gzip(file: Path) -> Tuple[
    Dict[str, Any],
    Dict[str, List[int]],
    float
]:
    """
    Reads syllable segmentation generated
    with `Chipper <https://github.com/CreanzaLab/chipper>`_.


    Args:
        file (Path): Path to the .gzip file

    Returns:
        Tuple[Dict[str, Any], Dict[str, List[int]], float]: Tuple containing
        two dictionaries (the first contains chipper parameters, the second
        has two keys ['Onsets', 'Offsets']) and a parameter 'timeAxisConversion'.
    """

    with gzip.open(file, "rb") as f:
        data = f.read()

    song_data = pickle.loads(data, encoding="utf-8")

    return song_data[0], song_data[1], np.array(
        song_data[2]['Sonogram']), song_data[3]['timeAxisConversion']


# %%──── SETTINGS ──────────────────────────────────────────────────────────────

PROJECT_DIR = Path(
    git.Repo('.', search_parent_directories=True).working_tree_dir)
DIRS = ProjDirs(PROJECT_DIR)
print(DIRS)

RAW_DATA = 'STORM_PETREL_2021'
DIRS.append('ORIGIN', DIRS.DATA / "raw" / RAW_DATA)
DIRS.append('WAVFILES', DIRS.DATA / "segmented" / RAW_DATA)


# Link your data with the project's data folder:
origin = '/media/nilomr/My Passport/SONGDATA'
try:
    link_project_data(origin, DIRS.DATA)
except ValueError as e:
    print(e)

file_list = get_file_paths(DIRS.ORIGIN, ['.wav', '.WAV'])

# Settings
species = "european storm petrel"
min_duration = .5
min_freqrange = 200
ignore_labels = ["NOISE"]

# Get info on available song selections
lengths_dict = get_segment_info(DIRS.ORIGIN, min_duration, min_freqrange)

# Get list of wav files to segment
wav_filelist = get_wav_filepaths(DIRS.ORIGIN)

# Plot segment lengths
ax = sns.histplot(lengths_dict, bins=50, log_scale=False)


# ────── MAIN ───────────────────────────────────────────────────────────────────

# %% Run serially

failed_list = []
for WAV_FILEDIR in tqdmm(wav_filelist, desc='Initiate: Segmenting raw files'):
    try:
        segment_into_songs(
            DIRS.DATA, WAV_FILEDIR, RAW_DATA, min_duration=min_duration,
            min_freqrange=min_freqrange, species=species,
            ignore_labels=ignore_labels, resample=22050)
    except:
        print(f'{WAV_FILEDIR=} failed')
        failed_list.append(WAV_FILEDIR)
print(f'Segmentation failed for {len(failed_list)} files:',
      *failed_list, sep='\n')

# %% Run in parallel (w/ ray)

segment_into_songs_r = ray.remote(segment_into_songs)

obj_ids = [
    segment_into_songs_r.remote(
        DIRS.DATA, WAV_FILEDIR, RAW_DATA, min_duration=min_duration,
        min_freqrange=min_freqrange, species=species,
        ignore_labels=ignore_labels, resample=22050)
    for WAV_FILEDIR in tqdmm(
        wav_filelist, desc='Initiate: Segmenting raw files')]

for obj in tqdmm(
        to_iterator(obj_ids),
        desc='Segmenting raw files', total=len(wav_filelist)):
    pass


# %%
# Segment songs into units

# Recording quality (SNR, etc.) is too variable to do this automatically,
# so I used Chipper (REF).

# Methods to help segmentation:

# 1. Chipper does not skip songs that already have segmentation information
# when the app is restarted, so this function moves any song that remains to be
# segmented to a different folder ('to_annotate').
# Run this after each annotation session.

annotate = DIRS.WAVFILES / 'WAV' / 'to_annotate'
annotated = DIRS.WAVFILES / 'WAV'

gzip_files = get_file_paths(DIRS.WAVFILES, [".gzip", ".GZIP"])
wav_files = get_file_paths(DIRS.WAVFILES, ['.wav', '.WAV'])
gzip_filenames = [file.stem.replace('SegSyllsOutput_', '')
                  for file in gzip_files]

for loc in [annotate, annotated]:
    if 'annotate' in str(loc):
        to_move = [file for file in wav_files
                   if file.stem not in gzip_filenames]
        safe_makedir(loc)
    else:
        to_move = [file for file in wav_files if file.stem in gzip_filenames]

    moved = 0
    for i, file in tqdmm(enumerate(to_move), desc='Moving files'):
        try:
            if file != loc / file.name:
                shutil.move(file, loc / file.name)
                moved += 1
        except:
            pass  # TODO: handle exceptions properly
    print(f'Moved {moved}/{i} files to {loc}')
# %%

# 2. Chipper outputs segmentation information to a '' file. Let's add this to
# the JSON metadata created above.

# TODO: refactor, parallelise
jsons = get_file_paths(DIRS.WAVFILES, ['.json', '.JSON'])
gzips = get_file_paths(DIRS.WAVFILES, [".gzip", ".GZIP"])
wavs = get_file_paths(DIRS.WAVFILES, [".WAV", ".wav"])

json_file_dict = {path.stem[:-4]: path for path in jsons}
gzip_file_dict = {path.stem.replace(
    'SegSyllsOutput_', ''): path for path in gzips}
wav_file_dict = {path.stem: path for path in wavs}

it = 0
overwrite_json: bool = True
for gz_name, gz_path in tqdmm(
        gzip_file_dict.items(),
        desc='Adding unit onset/offset information to json files'):

    if gz_name in json_file_dict:
        chipper_params, gf, spec_img, ms_frame = open_gzip(gz_path)
        onsets = np.array(gf['Onsets'])
        offsets = np.array(gf['Offsets'])
        s_frame = ms_frame/1000

        with wave.open(str(wav_file_dict[gz_name]), 'r') as obj:
            nframes = obj.getnframes()
            nframes_chipper = chipper_params['BoutRange'][1]

        coef_correction = nframes / nframes_chipper
        dic = read_json(json_file_dict[gz_name])
        sr = dic['samplerate_hz']

        if 'onsets' in dic and not overwrite_json:
            raise FileExistsError(
                'Json files already contain unit onset/offset times.'
                'Set `overwrite_json = True` if you want to overwrite them.')
        dic['onsets'], dic['offsets'] = (
            (onsets)*coef_correction/sr,
            (offsets)*coef_correction/sr
        )

        print(f"{dic['offsets'][-1]=}")
        print(f"{dic['length_s']=}")
        it += 1
        if it == 1:
            break

        # with open(json_file_dict[gz_name].as_posix(), "w") as f:
        #     json.dump(dic, f, indent=2)


spec = spec_img
unit = spec[:, gf['Onsets'][-1]:gf['Offsets'][-1]]
plt.imshow(spec)
plt.imshow(unit)


dat, sr = librosa.load(str(wav_file_dict[gz_name]))

mel_spectrogram = librosa.feature.melspectrogram(
    dat, sr=dataset.parameters.sr, n_fft=2048,
    win_length=512,
    hop_length=32,
    n_mels=240,
    fmin=500,
    fmax=6000)

mel_spectrogram = librosa.amplitude_to_db(
    mel_spectrogram, top_db=dataset.parameters.top_dB, ref=np.max)

onsets = np.array(dic['onsets'])
offsets = np.array(dic['offsets'])


units = []
# Seconds to frames
# onsets = (onsets * sr / dataset.parameters.hop_length).astype(np.int32)
# offsets = (offsets * sr / dataset.parameters.hop_length).astype(np.int32)
# for on, off in zip(onsets, offsets):
#     unit = mel_spectrogram[:, on:off]
#     units.append(unit)

# for unit in units:
#     pyplot.melspectrogram(unit, parameters=dataset.parameters)
#     # pyplot.melspectrogram(mel_spectrogram, parameters=dataset.parameters)

# pyplot.segmentation(dataset, spectrogram=mel_spectrogram,
#                     onsets_offsets=(onsets, offsets))


spectrogram = mel_spectrogram


if TYPE_CHECKING:
    from pykanto.dataset import SongDataset

params = dataset.parameters
key = 'mierda'
onsets_offsets = (onsets, offsets)


pyplot.segmentation(dataset, spectrogram=spectrogram,
                    onsets_offsets=onsets_offsets)


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
    lowcut=0,
    highcut=10000,
    dereverb=False,
    # Segmentation,
    max_dB=-30,                 # Max threshold for segmentation
    dB_delta=5,                 # n thresholding steps, in dB
    silence_threshold=0.1,      # Between 0.1 and 0.3 tends to work
    max_unit_length=0.3,        # Maximum unit length allowed
    min_unit_length=0.005,       # Minimum unit length allowed
    min_silence_length=0.001,   # Minimum silence length allowed
    gauss_sigma=1,              # Sigma for gaussian kernel
    # general settings
    song_level=True,
    subset=(10, 20),
    n_jobs=-1,
    verbose=False,
)

np.random.seed(123)
random.seed(123)
DATASET_ID = "STORM_PETREL_SONGS"
dataset = SongDataset(
    DATASET_ID, DIRS, parameters=params, overwrite_dataset=True,
    overwrite_data=True)

# Set population as ID
dataset.vocalisations['ID'] = dataset.vocalisations['ID'].map(
    lambda x: x.split('_')[0])

# Remove files with missing data
dataset.vocalisations.dropna(inplace=True)


for key in dataset.vocalisations.index[:3]:
    dataset.plot_vocalisation_segmentation(
        key, cmap='bone')

# %%

dataset.show_extreme_samples(
    n_songs=10, query='duration',
    order='ascending'
)
k = dataset.vocalisations.index[0]

get_vocalisation_units(dataset, k, song_level=False)
indvus = get_indv_units(
    dataset, dataset.vocalisations.index, individual='faroes')


units = [get_vocalisation_units(
    dataset, key, song_level=False) for key in dataset.vocalisations.index]
units = dictlist_to_dict(units)

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

spectrogram[:, int(0.99174603*dataset.parameters.hop_length_ms)            :int(1.536009*dataset.parameters.hop_length_ms)]

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
