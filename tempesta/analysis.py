# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
Audio feature calculations (spectral centroids, peak frequencies, etc.)
"""

# ──── IMPORTS ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, List

import librosa
import numpy as np
from pykanto.signal.analysis import (
    approximate_minmax_frequency,
    get_mean_sd_mfcc,
    spec_centroid_bandwidth,
)
from pykanto.utils.compute import with_pbar

if TYPE_CHECKING:
    from pykanto.dataset import KantoData

# ──── FUNCTIONS ───────────────────────────────────────────────────────────────


def extract_mfccs(
    dataset: KantoData, note_type: str, n_mfcc: int = 20
) -> List[np.ndarray]:
    """
    Calculates the mean and SD for n MFCCs extracted from each note type in each
    song in the dataset. In the case of purr notes it return the mean mean and
    sd for all purr notes in a song.

    Args:
        dataset (KantoData): Song dataset object.
        note_type (str): One of 'purr', 'breathe'.
        n_mfcc (int, optional): Number of MFCCs to return. Defaults to 20.

    Returns:
        List[np.ndarray]: List containing one array per song in the database
    """

    if "units" not in dataset.files.columns:
        raise KeyError(
            "This function requires the output of "
            "pykanto.KantoData.get_units()"
        )

    idx = "-1" if note_type == "breathe" else "0:-1"
    # Iterates over IDs because data is stored gruped by ID to reduce I/O
    # bottleneck.
    unitdata = set(dataset.files.units.to_list())
    mean_sd_mfccs = []

    for song in with_pbar(unitdata, desc="Calculating MFCCs"):
        with open(song, "rb") as f:
            kk = pickle.load(f)

        notes = []
        for k, v in kk.items():
            notes.append(eval("v[" + idx + "]"))

        for note in notes:
            if note_type == "breathe":
                mean_sd = get_mean_sd_mfcc(note, n_mfcc)
            else:
                mean_sd_purrs = []
                for purr in note:
                    mean_sd_purr = get_mean_sd_mfcc(purr, n_mfcc)
                    mean_sd_purrs.append(mean_sd_purr)
                mean_sd = np.mean(mean_sd_purrs, axis=0)
            mean_sd_mfccs.append(mean_sd)

    return mean_sd_mfccs


def extract_spectral_centroids_bw(
    dataset: KantoData, note_type: str
) -> List[np.ndarray]:
    """
    Calculates the mean + SD of the spectral centroids and spectral bandwidth of
    each song in the dataset. In the case of purr notes it return the mean mean
    and sd for all purr notes in a song.

    Args:
        dataset (KantoData): Dataset with vocalisations.
        note_type (str): One of 'purr', 'breathe'.

    Returns:
        List[np.ndarray]: A list containing arrays with
        [mean centroid, sd, mean bandwidth, sd]
    """
    if "units" not in dataset.files.columns:
        raise KeyError(
            "This function requires the output of "
            "pykanto.KantoData.get_units() with the parameter "
            "`song_level = False` ."
        )

    idx = "-1" if note_type == "breathe" else "0:-1"
    unitdata = set(dataset.files.units.to_list())
    c_bw_sd = []

    for song in with_pbar(unitdata, desc="Calculating spectral centroid + SD"):
        with open(song, "rb") as f:
            kk = pickle.load(f)

        notes = []
        for k, v in kk.items():
            notes.append(eval("v[" + idx + "]"))

        for note in notes:
            if note_type == "purr":
                note = np.hstack(note)
            centroid, bandwidth = spec_centroid_bandwidth(dataset, spec=note)

            mean_sd = np.array(
                [[np.nanmean(i), np.nanstd(i)] for i in [centroid, bandwidth]]
            ).flatten()
            c_bw_sd.append(mean_sd)

    return c_bw_sd


def extract_minmax_frequencies(
    dataset: KantoData, note_type: str
) -> List[np.ndarray]:
    """
    Calculates the approximate minimum and maximum frequencies of each note type
    in each song in the dataset.

    Args:
        dataset (KantoData): Dataset with vocalisations.
        note_type (str): One of 'purr', 'breathe'.

    Returns:
        List[np.ndarray]: A list containing arrays with
        [mean minimum frequency, sd, mean maximum frequency, sd]
    """
    if "units" not in dataset.files.columns:
        raise KeyError(
            "This function requires the output of "
            "pykanto.KantoData.get_units() with the parameter "
            "`song_level = False` ."
        )

    idx = "-1" if note_type == "breathe" else "0:-1"
    unitdata = set(dataset.files.units.to_list())
    c_bw_sd = []

    for song in with_pbar(
        unitdata, desc="Calculating mean and maximum frequencies + SD"
    ):
        with open(song, "rb") as f:
            kk = pickle.load(f)

        notes = []
        for k, v in kk.items():
            notes.append(eval("v[" + idx + "]"))

        for note in notes:
            if note_type == "purr":
                note = np.hstack(note)
            minfreqs, maxfreqs = approximate_minmax_frequency(
                dataset, spec=note
            )

            mean_sd = np.array(
                [[np.nanmean(i), np.nanstd(i)] for i in [minfreqs, maxfreqs]]
            ).flatten()
            c_bw_sd.append(mean_sd)

    return c_bw_sd


def extract_flatness(dataset: KantoData, note_type: str) -> List[np.ndarray]:
    """
    Calculates the mean + SD spectral flatness of each note type
    in each song in the dataset.

    Args:
        dataset (KantoData): Dataset with vocalisations.
        note_type (str): One of 'purr', 'breathe'.

    Returns:
        List[np.ndarray]: A list containing arrays with
        [mean spectral flatness, SD]
    """
    if "units" not in dataset.files.columns:
        raise KeyError(
            "This function requires the output of "
            "pykanto.KantoData.get_units() with the parameter "
            "`song_level = False` ."
        )

    idx = "-1" if note_type == "breathe" else "0:-1"
    unitdata = set(dataset.files.units.to_list())
    flat_sd = []

    for song in with_pbar(unitdata, desc="Calculating spectral flatness + SD"):
        with open(song, "rb") as f:
            kk = pickle.load(f)

        notes = []
        for k, v in kk.items():
            notes.append(eval("v[" + idx + "]"))

        for note in notes:
            if note_type == "purr":
                note = np.hstack(note)

            offset = 0
            if np.min(note) < 0:
                offset = abs(np.min(note))

            flatness = librosa.feature.spectral_flatness(
                S=note + offset,
                hop_length=dataset.parameters.hop_length,
                n_fft=dataset.parameters.fft_size,
                win_length=dataset.parameters.window_length,
            )

            mean_sd = np.array([np.mean(flatness), np.std(flatness)])

            flat_sd.append(mean_sd)

    return flat_sd
