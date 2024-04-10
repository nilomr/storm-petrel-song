import pickle

import numpy as np
import pandas as pd
from pykanto.utils.io import load_dataset
from pykanto.utils.paths import ProjDirs
from settings import ROOT_DIR

from tempesta.analysis import (
    extract_flatness,
    extract_mfccs,
    extract_minmax_frequencies,
    extract_spectral_centroids_bw,
)

# ──── PROJECT SETUP ──────────────────────────────────────────────────────────

dataset_name = "storm-petrel"
raw_data = ROOT_DIR / "data" / "raw" / dataset_name
DIRS = ProjDirs(ROOT_DIR, raw_data, dataset_name, mkdir=False)

# load dataset
dataset = load_dataset(DIRS.DATASET, DIRS)

# Allocate missing population names for 2022 data
dataset.data.loc[dataset.data.index.str.contains("^AM"), "ID"] = "sardinia"
dataset.data.loc[dataset.data.index.str.contains("^Dia|^dia"), "ID"] = (
    "benidorm"
)

# Remove Nolsoy_colony_song_and_call_25_July_2021 - something went wrong with
# the recording
dataset.data = dataset.data[
    ~dataset.data.index.str.contains("Nolsoy_colony_song_and_call_25_July_2021")
]
dataset.files = dataset.files[
    ~dataset.files.index.str.contains(
        "Nolsoy_colony_song_and_call_25_July_2021"
    )
]

# remove the corresponding units from the pickled files
fdict = dataset.files.units[dataset.files.index.str.contains("faroes")][0]
with open(fdict, "rb") as f:
    kk = pickle.load(f)
ofkey = [
    k for k in kk.keys() if "Nolsoy_colony_song_and_call_25_July_2021" in k
]
for k in ofkey:
    kk.pop(k)
with open(fdict, "wb") as f:
    pickle.dump(kk, f)

# Remove rows where ID is 'marettimo' as the sample size is too small
dataset.data = dataset.data[dataset.data.ID != "marettimo"]
dataset.files = dataset.files[dataset.files.ID != "marettimo"]

# Define populations
atlantic_pops = [
    "wales",
    "faroes",
    "scotland",
    "norway",
    "ireland",
    "iceland",
    "molene",
    "montana_clara",
]

# Temporal features
features_dict = {
    "ID": dataset.data.ID.values,
    "group": [
        "pelagicus" if x in atlantic_pops else "melitensis"
        for x in dataset.data.ID.values
    ],
    "mean_purr_duration": [
        np.mean(r[:-1]) for r in dataset.data.unit_durations
    ],
    "std_purr_duration": [np.std(r[:-1]) for r in dataset.data.unit_durations],
    "breathe_duration": [r[-1] for r in dataset.data.unit_durations],
    "purr_duration": [np.sum(r[:-1]) for r in dataset.data.unit_durations],
    "mean_silence_duration": [
        np.mean(r[:-1]) for r in dataset.data.silence_durations
    ],
    "std_silence_duration": [
        np.std(r[:-1]) for r in dataset.data.silence_durations
    ],
    "n_purr_notes": [len(r[:-1]) for r in dataset.data.unit_durations],
    "song_length": dataset.data.length_s.values,
    "purr_ioi": [
        np.mean((np.append(r[1:], 0) - r)[:-1]) for r in dataset.data.onsets
    ],
    "purr_ioi_std": [
        np.std((np.append(r[1:], 0) - r)[:-1]) for r in dataset.data.onsets
    ],
}

# Spectral features
for nt in ["breathe", "purr"]:
    features_dict[f"{nt}_mfcc"] = extract_mfccs(
        dataset, note_type=nt, n_mfcc=24
    )
    features_dict[f"{nt}_centroid_bw"] = extract_spectral_centroids_bw(
        dataset, note_type=nt
    )
    features_dict[f"{nt}_minmax_freq"] = extract_minmax_frequencies(
        dataset, note_type=nt
    )
    features_dict[f"{nt}_flatness"] = extract_flatness(dataset, note_type=nt)


# Prepare the dataframe
features = pd.DataFrame(features_dict, index=dataset.data.index)

arraycols = [
    col for col in features.columns if isinstance(features[col][0], np.ndarray)
]

dfs = []
for col in arraycols:
    tmpdf = features[col].apply(pd.Series)
    dfs.append(tmpdf.rename(columns=lambda x: f"{col}_{str(x)}"))

features = pd.concat([features] + dfs, axis=1).drop(columns=arraycols)

# Save the features
features.to_csv(DIRS.RESOURCES / "features.csv")
