# 2D velocity-based linear decoding

## Setup

### Coding environment

Assumes Python 3 on Ubuntu.

1. Create a Python venv with `python -m venv .venv`.
2. Activate the venv with `source .venv/bin/activate`.
3. Install the Python dependencies with `pip install -r requirements.txt`.

### Downloading data

1. In the top level of this repo, create a directory called `data/` (it is gitignored).
2. Download the `.nwb` files in https://dandiarchive.org/dandiset/000121/0.220124.2156 and place them inside `data/`. (Directly inside `data/` should be `sub-JenkinsC_ses-20151001T150307_behavior+ecephys.nwb`, etc.)
    - More datasets are available at https://npsl.sites.stanford.edu/data-code.

## Running the code

1. Activate the venv with `source .venv/bin/activate`.
2. Run `python main.py`.

It prints progress in the console. It may take a while the first time you run it, because it is doing some calculations over the whole data set. However, it caches intermediate results using `numpy.save`/`numpy.load`, so subsequent runs should be much faster. To recalculate those intermediate results from scratch, delete the `cache/` directory in this directory.
