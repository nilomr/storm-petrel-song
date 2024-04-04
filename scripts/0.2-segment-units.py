import json
from itertools import groupby
from pathlib import Path
from shutil import copy2

from pykanto.utils.paths import ProjDirs, get_file_paths
from settings import ROOT_DIR

# Project setup
dataset_name = "storm-petrel"
raw_data = ROOT_DIR / "data" / "raw" / dataset_name
DIRS = ProjDirs(ROOT_DIR, raw_data, dataset_name, mkdir=False)

# ──── FUNCTION DEFINITIONS ───────────────────────────────────────────────────


def rename_files(dirs: ProjDirs, preview: bool = False) -> None:
    # For backwards compatibility, change file names to use consecutive integers
    # instead of onset frames.
    wav_files = []
    json_files = []
    wav_files = get_file_paths(dirs.SEGMENTED / "WAV", [".wav"])
    json_files = get_file_paths(dirs.SEGMENTED / "JSON", [".JSON", ".json"])

    for files in [wav_files, json_files]:
        file_groups = {
            k: sorted(g, key=lambda f: int(f.stem.rsplit("_", 1)[1]))
            for k, g in groupby(
                sorted(files, key=lambda f: f.stem.rsplit("_", 1)[0]),
                key=lambda f: f.stem.rsplit("_", 1)[0],
            )
        }
        for group in file_groups.values():
            for i, file in enumerate(group):
                new_name = f"{file.stem.rsplit('_', 1)[0]}_{i}.{file.suffix.lstrip('.')}"
                if preview:
                    print(f"Old name: {file.name}, New name: {new_name}")
                else:
                    file.rename(file.with_name(new_name))
    # Now open the JSON files and change the 'wav_file' key to match the new
    # name
    json_files = get_file_paths(dirs.SEGMENTED / "JSON", [".JSON", ".json"])
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            old_filepath = Path(data["wav_file"])
            file_n = int(json_file.stem.rsplit("_", 1)[1])
            data["wav_file"] = str(
                old_filepath.with_name(
                    f"{old_filepath.stem.rsplit('_', 1)[0]}_{file_n}.wav"
                )
            )
            if preview:
                # show the old and new wav_file paths
                print(
                    f"Old wav_file: {old_filepath}, New wav_file: {data['wav_file']}"
                )
            else:
                with open(json_file, "w") as f:
                    json.dump(data, f, indent=4)
        except FileNotFoundError as exc:
            raise Warning(f"File {json_file} not found.") from exc


# ──── MAIN ───────────────────────────────────────────────────────────────────

# For backwards compatibility, change file names to use consecutive integers
# instead of onset frames (only do this once).
rename_files(DIRS, preview=False)

# Get the directories starting with 'SegSylls' inside DIRS.SEGMENTED / "WAV"
seg_sylls_dirs = [
    d
    for d in (DIRS.SEGMENTED / "WAV").iterdir()
    if d.is_dir() and d.name.startswith("SegSylls")
]

# Get all files in these directories
file_names = []
for directory in seg_sylls_dirs:
    for file in directory.iterdir():
        if file.is_file():
            file_names.append(file)

# move the files whith name including 'tossed' to a new list
tossed_files = [f for f in file_names if "tossed" in f.name]

# Read the contents of the tossed files (txt files) and store them in a list
# (one line per element)
tossed_filenames = []
for file in tossed_files:
    with open(file, "r") as f:
        next(f)  # Skip the first line
        tossed_filenames.extend(line.strip() for line in f.readlines())

# Create the full path of the tossed files
tossed_files = [DIRS.SEGMENTED / "WAV" / file for file in tossed_filenames]

all_wavs = get_file_paths(DIRS.SEGMENTED / "WAV", [".wav"])
all_wavs = [f for f in all_wavs if f.parent.name != "to_segment"]
all_gzips = get_file_paths(DIRS.SEGMENTED / "WAV", [".gzip"])
# Get the wav files in all_wavs that have a corresponding gzip file
wavs_with_gzips = [
    wav
    for wav in all_wavs
    for gzip_file in all_gzips
    if wav.stem == gzip_file.stem.split("_", 1)[1]
]

# return the names of files that are in all_gzips but not in wavs_with_gzips
missing_wavs = [
    gzip_file
    for gzip_file in all_gzips
    if gzip_file.stem.split("_", 1)[1]
    not in [wav.stem for wav in wavs_with_gzips]
]
# Already processed files
processed_files = tossed_files + wavs_with_gzips

# Get the files that have not been processed yet
unprocessed_files = [
    f for f in all_wavs if f.name not in [pf.name for pf in processed_files]
]

# Copy these unprocessed_files to a 'to_segment' subfolder
if unprocessed_files:
    to_segment_dir = DIRS.SEGMENTED / "WAV" / "to_segment"
    to_segment_dir.mkdir(exist_ok=True)
    for file in unprocessed_files:
        copy2(file, to_segment_dir / file.name)
else:
    print("No unprocessed files to revise.")
