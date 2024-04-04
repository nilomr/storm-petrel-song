# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
Functions and methods to read, parse, save and move files.
"""

# ─── LIBRARIES ────────────────────────────────────────────────────────────────

import gzip
import json
from pathlib import Path
import pickle
import shutil
from typing import Any, Dict, List, Tuple
import numpy as np
from pykanto.utils.compute import timing, tqdmm
from pykanto.utils.paths import get_file_paths, makedir
from pykanto.utils.read import read_json

# ──── CLASSES AND FUNCTIONS ───────────────────────────────────────────────────


def open_gzip(file: Path) -> Tuple[
    Dict[str, Any],
    Dict[str, List[int]],
    float
]:
    """
    Reads syllable segmentation generated
    with `Chipper <https://github.com/CreanzaLab/chipper>`_.

    Args:
        file (Path): Path to the .gzip file.

    Returns:
        Tuple[Dict[str, Any], Dict[str, List[int]], float]: Tuple containing
        two dictionaries (the first contains chipper parameters, the second
        has two keys ['Onsets', 'Offsets']) and a parameter 'timeAxisConversion'.
    """
    with gzip.open(file, "rb") as f:
        data = f.read()
    song_data = pickle.loads(data, encoding="utf-8")

    return song_data[0], song_data[1], song_data[3]['timeAxisConversion']


@timing
def chipper_units_to_json(
        directory: Path, n_fft: int = 1024, overlap: int = 1010, pad: int = 150,
        window_offset: bool = True, overwrite_json: bool = False, pbar: bool = True):

    woffset: int = n_fft // 2 if window_offset else 0

    jsons, gzips = [
        get_file_paths(directory, ext)
        for ext in (['.json', '.JSON'],
                    [".gzip", ".GZIP"])]

    jsons = {path.stem: path for path in jsons}
    gzips = {path.stem.replace(
        'SegSyllsOutput_', ''): path for path in gzips}

    for gz_name, gz_path in tqdmm(
            gzips.items(),
            desc='Adding unit onset/offset information '
            'from .gzip to .json files',
            disable=False if pbar else True
    ):

        if gz_name in jsons:

            jsondict = read_json(jsons[gz_name])
            if 'onsets' in jsondict and not overwrite_json:
                raise FileExistsError(
                    'Json files already contain unit onset/offset times.'
                    'Set `overwrite_json = True` if you want '
                    'to overwrite them.')

            sr = jsondict['sample_rate']
            gzip_onoff = open_gzip(gz_path)[1]
            on, off = np.array(
                gzip_onoff['Onsets']), np.array(
                gzip_onoff['Offsets'])

            jsondict['onsets'], jsondict['offsets'] = [(
                ((arr - pad) * (n_fft - overlap) + woffset) / sr).tolist()
                for arr in (on, off)]

            with open(jsons[gz_name].as_posix(), "w") as f:
                json.dump(jsondict, f, indent=2)


def chipper_move_remaining(
        directory: Path, annotated: Path, to_annotate: Path) -> None:
    """
    Moves wav files for which there is not a .gzip file produced by chipper
    to a new subdirectory 'to_annotate'. 

    Args:
        annotated (Path): Directory where already annotated files should be.
        to_annotate (Path): Directory containing remaining files.
    """
    gzip_files = get_file_paths(directory, [".gzip", ".GZIP"])
    wav_files = get_file_paths(directory, ['.wav', '.WAV'])
    gzip_filenames = [file.stem.replace('SegSyllsOutput_', '')
                      for file in gzip_files]

    for loc in [to_annotate, annotated]:
        if 'to_annotate' in str(loc):
            to_move = [file for file in wav_files
                       if file.stem not in gzip_filenames]
            makedir(loc)
        else:
            to_move = [file for file in wav_files
                       if file.stem in gzip_filenames]

        moved = 0
        for i, file in tqdmm(enumerate(to_move), desc='Moving files'):
            try:
                if file != loc / file.name:
                    shutil.move(file, loc / file.name)
                    moved += 1
            except:
                pass  # TODO: handle exceptions properly
        print(f'Moved {moved}/{len(to_move)} files to {loc}')
