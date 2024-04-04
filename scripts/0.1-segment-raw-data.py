from pykanto.signal.segment import segment_files_parallel
from pykanto.utils.paths import ProjDirs, get_file_paths
from settings import ROOT_DIR

# Project setup
dataset_name = "storm-petrel"
raw_data = ROOT_DIR / "data" / "raw" / dataset_name
DIRS = ProjDirs(ROOT_DIR, raw_data, dataset_name, mkdir=False)


# Prepare a list of filepaths to segment
wav_filepaths, xml_filepaths = [
    get_file_paths(DIRS.RAW_DATA, [ext]) for ext in [".wav", ".xml"]
]
upwav_filepaths = get_file_paths(DIRS.RAW_DATA, [".WAV"])
wav_filepaths.extend(upwav_filepaths)
wav_filepaths = [f for f in wav_filepaths if " " not in f.name]
xml_filepaths = [f for f in xml_filepaths if " " not in f.name]
to_segment = [
    (wav, xml)
    for xml in xml_filepaths
    for wav in wav_filepaths
    if xml.stem == wav.stem and xml.parent == wav.parent
]

# Segment the raw data
segment_files_parallel(
    to_segment,
    DIRS,
    resample=22050,
    min_duration=0.1,
    min_freqrange=200,
)
