import matplotlib.cm as cm
from pykanto.dataset import KantoData
from pykanto.parameters import Parameters
from pykanto.utils.custom import chipper_units_to_json
from pykanto.utils.paths import ProjDirs
from settings import ROOT_DIR

# ──── PROJECT SETUP ──────────────────────────────────────────────────────────

dataset_name = "storm-petrel"
raw_data = ROOT_DIR / "data" / "raw" / dataset_name
DIRS = ProjDirs(ROOT_DIR, raw_data, dataset_name, mkdir=False)

# ──── MAIN ───────────────────────────────────────────────────────────────────

# Add the unit onset/offset information from .gzip to .json files
try:
    chipper_units_to_json(DIRS.SEGMENTED)
except FileExistsError as e:
    print(e)

# Define parameters
params = Parameters(
    # Spectrogramming
    window_length=512,
    hop_length=32,
    n_fft=2048,
    num_mel_bins=240,
    sr=22050,
    top_dB=65,
    lowcut=300,
    highcut=10000,
    dereverb=True,
    # general settings
    song_level=False,
    verbose=False,
)

dataset = KantoData(
    DIRS,
    parameters=params,
    overwrite_dataset=True,
    overwrite_data=True,
)

# Plot a spectrogram of a song to make sure everything is working
dataset.plot(dataset.data.index[500], cmap=cm.magma, segmented=True)

dataset.segment_into_units()
dataset.get_units()

# Save the dataset
dataset.save_to_disk()
