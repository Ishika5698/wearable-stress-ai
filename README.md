# Wearable Stress AI

A local project for exploring whether physiological signals differ across WESAD's recorded laboratory conditions. The first model compares baseline, stress-task, and amusement-task labels. Those labels describe what participants were asked to do; they are not direct measurements of someone's internal state.

## First analysis

The pipeline starts with chest ECG, EDA, and temperature. It makes 30-second windows every 15 seconds and keeps only windows with one label from start to finish. For each channel, it calculates mean, spread, range, median, RMS, and the fraction of non-finite samples.

The comparison is a majority-class baseline against a class-weighted random forest. Five-fold stratified group validation keeps participants separate between training and test folds. Imputation is fitted on each training fold. This design avoids placing overlapping windows from one person on both sides of an evaluation fold.

This is a learning exercise, not a stress diagnosis, fatigue detector, or tool for clinical, driving, or motorsport decisions. The stress condition includes a speech and mental arithmetic task; movement, posture, and task context may explain some differences.

## Data access

WESAD is not included. The [University of Siegen dataset page](https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/) links the owners' archive and states that use is for scientific, non-commercial purposes, with credit when publishing. Follow the current terms at the official page and download host. Do not redistribute participant records or commit them to GitHub. See [data/README.md](data/README.md) before downloading.

After extraction, place the participant folders under `data/raw/WESAD/`, for example `data/raw/WESAD/S2/S2.pkl`. The notebook inspects the files before analysis and reports pending status when data is not present. Pickle files can execute code when loaded; use only the archive obtained from the official source.

## Run locally

Use Python 3.10 or newer. From this folder:

```zsh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyter notebook
```

Run `notebooks/01_data_exploration.ipynb` before `notebooks/02_modeling.ipynb`. Select `.venv/bin/python` as the kernel if you use VS Code. When the data is absent, model results stay pending; no example scores are substituted.

## Files

- `src/wesad.py` finds and inspects participant files.
- `src/features.py` builds label-pure windows and time-domain summaries.
- `src/evaluation.py` compares the baseline and random forest with grouped folds.
- `notebooks/` contains the data inspection and modeling walkthroughs.
- `reports/research_notes.md` records the protocol, limitations, and results status.
