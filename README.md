

![Title
Badge](https://img.shields.io/badge/european_storm_petrel_|_purr_call_analysis-k?style=for-the-badge&labelColor=d99c2b&color=d99c2b)
![Python
version](https://img.shields.io/badge/v3.10-4295B3?style=for-the-badge&logo=python&logoColor=white)


<img src="reports/readme.png" alt="rfmat" width="1000"/>


![version](https://img.shields.io/badge/package_version-0.1.0-orange)
![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)
![license](https://img.shields.io/github/license/mashape/apistatus.svg)
![Open Source Love](https://img.shields.io/badge/open%20source%3F-yes!-lightgrey)
![Python 3.8](https://img.shields.io/badge/python-3.8-brightgreen.svg)

***


# Data log

### 2021
- Greece data converted to wav estimating time from bitrate - might be imprecise.
- Data from Espartar unusable, drop
- Only one recording from Benidorm usable, not representative. Drop.
- Marettimo population only one (pretty bad) recording
  
### 2022
  
# Definitions

A phrase defined as uninterrupted trill or purr + subsequent breath note, the latter included, ends where next purr begins or silence.

#### Table of contents
  - [Installation](#installation)
  - [Project Organisation](#project-organisation)
  - [Acknowledgements](#acknowledgements)

## Installation

Avoid a [dependency hell](https://en.wikipedia.org/wiki/Dependency_hell)!
 > While it is possible to use pip without a virtual environment, it is not advised: virtual environments create a clean Python environment that does not interfere with any existing system installation, can be easily removed, and contain only the package versions your application needs. They help avoid a common challenge known as dependency hell.


- Do not upgrade/downgrade pandas within a project/environment: see https://stackoverflow.com/questions/68625748/spark-attributeerror-cant-get-attribute-new-block-on-module-pandas-core-in


1. Clone the repository:
`git clone https://github.com/nilomr/storm-petrel-song.git`.
2. Install source code:
`pip install .` (install) or `pip install -e .` (developer install).
3. Follow the instructions in the [docs](/docs) in the correct order.


As of 2024-03-25,

had to pass full path to garden
also had to conda install libstdcxx-ng
`conda install -c conda-forge libstdcxx-ng`



# File structure

xml along these lines - one option
```
<?xml version="1.0" encoding="UTF-8"?>
<sv>
  <data>
    <model id="1" name="" sampleRate="48000" start="5767168" end="163020800" 
    type="sparse" dimensions="2" resolution="1" notifyOnAdd="true" dataset="0" 
    subtype="box" minimum="1811.16" maximum="5665.32" units="Hz" />
    <dataset id="0" dimensions="2">
      <point frame="5767168" value="3672.92" duration="1499136" 
        extent="1698.44" label="NOISE" />
      <point frame="90030976" value="2219.44" duration="103040" 
        extent="3070.26" label="" />
      <point frame="90445824" value="2284.77" duration="77056" 
        extent="3086.59" label="" />
    </dataset>
  </data>
  <display>
    <layer id="2" type="boxes" name="Boxes" model="1"  verticalScale="0"  
    colourName="White" colour="#ffffff" darkBackground="true" />
  </display>
</sv>
```
## Project Organisation


    ├── LICENSE
    │
    ├── README.md          <- The top-level README.
    │
    ├── data               <- Main data folder. It is not version-tracked-the relevant program(s)  
    │   ├── external          create it automatically.
    │   ├── interim        
    │   ├── processed      
    │   └── raw            
    │
    ├── dependencies       <- Files required to reproduce the analysis environment(s).
    │
    ├── docs               <- Project documentation (installation instructions, etc.).
    │
    ├── notebooks          <- Jupyter and Rmd notebooks. Naming convention is number (for ordering),
    |   |                     the creator's initials, and a short `-` delimited description, e.g.
    │   └── ...               `1.0_nmr-label-songs`.  
    │                         
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc. Not currently tracked;
    |   |                     created automatically when needed.
    │   └── figures        <- Generated figures.
    │
    ├── setup.py           <- Makes project pip installable (so pykanto can be imported).
    |
    ├── ...                <- R project, .gitignore, etc.
    │
    └── pykanto                <- Source code. Install with `pip install .` (install) or 
                              `pip install -e .` (developer install).

## Acknowledgements


- `Many functions in pykanto/avgn` 
Sainburg T, Thielk M, Gentner TQ (2020) Finding, visualizing, and quantifying latent structure across diverse animal vocal repertoires. PLOS Computational Biology 16(10): e1008228. [DOI](https://doi.org/10.1371/journal.pcbi.1008228)

- `pykanto/vocalseg` Code modified from Tim Sainburg. [RAW_DATAal repository](https://github.com/timsainb/vocalization-segmentation).

- The `dereverberate()` function, [here](https://github.com/nilomr/great-tit-song/blob/24d9527d0512e6d735e9849bc816511c9eb24f99/pykanto/greti/audio/filter.py#L66), is based on code by Robert Lachlan ([Luscinia](https://rflachlan.github.io/Luscinia/)).

--------

<p><small>A project by Nilo M. Recalde | based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
