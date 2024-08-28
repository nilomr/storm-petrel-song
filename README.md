

# Tempesta
![Title
Badge](https://img.shields.io/badge/european_storm_petrel_|_purr_call_analysis-k?style=for-the-badge&labelColor=d99c2b&color=d99c2b)
![Python
version](https://img.shields.io/badge/v3.10-4295B3?style=for-the-badge&logo=python&logoColor=white)





## Data log

### 2021
- Greece data converted to wav estimating time from bitrate - might be imprecise.
- Data from Espartar unusable, drop
- Only one recording from Benidorm usable, not representative. Drop.
- Marettimo population only one (pretty bad) recording
  
### 2022
  
## Definitions

A phrase defined as uninterrupted trill or purr + subsequent breath note, the latter included; ends where next purr begins or there is a longer silence.

#### Table of contents
  - [Installation](#installation)
  - [Project Organisation](#project-organisation)
  - [Acknowledgements](#acknowledgements)

## Installation

1. Clone the repository:
`git clone https://github.com/nilomr/storm-petrel-song.git`.
2. Install python package (Install in an isolated environment, avoid a [dependency hell](https://en.wikipedia.org/wiki/Dependency_hell)!):
`pip install .` (install) or `pip install -e .` (developer install).
3. To install the R dependencies, use the following command (requires `renv` and R v4.2.1): `renv::restore()`


> Note to self: As of 2024-03-25, had to pass full path to garden. Also had to `conda install -c conda-forge libstdcxx-ng`


# Project Structure
    .
    ├── .gitignore                  # Specifies files and directories to be ignored by Git
    ├── .Rprofile                   # R configuration file
    ├── config.yml                  # Configuration file for the R scripts
    ├── data/                       # Main data folder (not version-tracked)
    │   ├── external/               # External data
    │   ├── interim/                # Intermediate data
    │   ├── processed/              # Processed data
    │   └── raw/                    # Raw data
    ├── dependencies.R              # R script to manage vscode dependencies
    ├── LICENSE                     # License file for the project
    ├── output/                     # Output directory for generated files
    │   ├── figures/                # Generated figures
    │   └── reports/                # Generated reports
    ├── pyproject.toml              # Python project configuration file
    ├── R/                          # R scripts directory
    │   ├── rplot.R                 # R script for plotting
    │   └── utils.R                 # Utility functions for R
    ├── README.md                   # The top-level README file
    ├── renv/                       # R environment management directory
    ├── renv.lock                   # Lock file for renv
    ├── resources/                  # Directory for resource files
    │   ├── features.csv            # Acoustic features under analysis
    │   ├── rf_feature_importance.csv       # Random forest feature importance
    │   └── sbsp_rf_feature_importance.csv  # Random forest feature importance (binary task)
    ├── scripts/                    # Directory for Python scripts
    │   ├── 0.1-segment-raw-data.py         # Script for segmenting raw data
    │   ├── 0.2-segment-units.py            # Script for segmenting units
    │   ├── 0.3-build-dataset.py            # Script for building dataset
    │   ├── 0.4-extract-measurements.py     # Script for extracting measurements
    │   ├── 0.5-train-model.py              # Script for training model
    │   └── ...                             # Additional scripts
    ├── tempesta/                   # Source code directory for the Tempesta (python) project
    └── tempesta.Rproj              # RStudio project file for Tempesta
