from pykanto.utils.paths import ProjDirs, change_data_loc, get_wav_filepaths, get_xml_filepaths
import git
from pathlib import Path


PROJECT_DIR = Path(
    git.Repo('.', search_parent_directories=True).working_tree_dir)

ProjDirs
