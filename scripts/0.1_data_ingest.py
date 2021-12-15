from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List
from xml.etree import ElementTree

import audio_metadata
import git
import librosa
import ray
import seaborn as sns
import soundfile as sf
from pykanto.signal.segment import get_segment_info, segment_into_songs
from pykanto.utils.compute import flatten_list, to_iterator, tqdmm
from pykanto.utils.paths import ProjDirs, get_wav_filepaths, link_project_data
from pykanto.utils.write import safe_makedir

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


file_list: List[Path] = []
for root, _, files in os.walk(str(DIRS.ORIGIN)):
    for file in files:
        if file.endswith(".wav") or file.endswith(".WAV"):
            file_list.append(Path(root) / file)
if len(file_list) == 0:
    raise FileNotFoundError(
        "There are no .wav files in this directory")
else:
    print(f"Found {len(file_list)} .wav files in {DIRS.ORIGIN}")


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


WAV_FILEDIR = wav_filelist[0]


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
        wavfile = sf.SoundFile(wavfile_str)
        if not wavfile.seekable():
            raise ValueError(f"Cannot seek through this file ({wavfile_str}).")
        sr = wavfile.samplerate

        # Get xml metadata for this file.
        # This xml file structure is particular to Sonic Visualiser
        root = ElementTree.parse(XML_FILEDIR).getroot()

        # Get sample rate of file
        xml_sr = int(root.findall('data/model')[0].get('sampleRate'))
        species = species  # Here you would obtain the species label.

        # Check that sample rates coincide
        if not sr == xml_sr:
            print(
                f"Sample rates do not coincide for {WAV_FILEDIR.name}. "
                f"(XML: {xml_sr}, WAV: {sr}.)"
            )

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

    # Save .wav
    wav_out = (OUT_DIR / "WAV" / f'{ID}_{str(cnt)}.wav')
    safe_makedir(wav_out)

    if resample:
        audio_section = librosa.resample(audio_section, sr, resample)
        sf.write(wav_out, audio_section, resample)
    else:
        sf.write(wav_out, audio_section, sr)

    # Make a JSON dictionary to go with the .wav file
    json_dict = {"species": species,
                 "ID": ID,
                 "label": label,
                 "samplerate_hz": sr,
                 "length_s": len(audio_section) / sr,
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


segment_into_songs_r = ray.remote(segment_into_songs)

obj_ids = [
    segment_into_songs_r.remote(
        DIRS.DATA, WAV_FILEDIR, RAW_DATA, min_duration=min_duration,
        min_freqrange=min_freqrange, species=species,
        ignore_labels=ignore_labels, resample=48000)
    for WAV_FILEDIR in tqdmm(
        wav_filelist, desc='Initiate: Segmenting raw files')]

for obj in tqdmm(
        to_iterator(obj_ids),
        desc='Segmenting raw files', total=len(wav_filelist)):
    pass
