# ICU Mortality Prediction Fairness Audit

DS-US 202: Responsible Data Science, Fall 2026  
Team: Guyu Sun, Rachel Ruan, Laura Zhu

## Goal
Audit demographic and intersectional fairness in an existing LightGBM ICU mortality prediction system using WiDS Datathon 2020 data (Option B).

## Current status
The first real-data audit baseline ran successfully on 2026-10-30: 91,713 labeled encounters, patient-disjoint 58,696/14,674/18,343 fit/validation/test split. Held-out AUC 0.9056, recall 0.3607 and precision 0.7358 at threshold 0.5. This is a single-model adaptation, not an exact rerun of the original four-fold ensemble or previous proposal experiment. See [reproduction notes](docs/reproduction-notes.md) and [baseline summary](reports/baseline-summary.md).

## Run locally

Use Python 3.9 or newer. On macOS, LightGBM also needs OpenMP (`brew install libomp`).

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.baseline --smoke-test
```

After obtaining the official competition data and completing its access process, place `training_v2.csv` under `data/raw/`, then run:

```sh
.venv/bin/python -m src.baseline
```

The default run keeps patients separate across fit/validation/test, fits preprocessing on the fit partition only, trains up to 7,000 boosting rounds with early stopping, and evaluates at threshold 0.5. It writes aggregate metrics, subgroup support/error rates, missingness, feature importance, a confusion matrix and a provenance manifest to `results/baseline/`. It does not export individual records/predictions. A missing metric means undefined, not zero. This is a single-model audit adaptation, not the original four-fold competition ensemble.

`notebooks/01_baseline.ipynb` provides the equivalent notebook entry point. A private repository must be made accessible to your own authorized Colab session; no credentials are embedded. The synthetic check always writes to ignored `results/smoke/` and is not an ICU experiment.

## Layout
- `docs/`: proposal, course requirements, and planning notes
- `data/raw/`: locally acquired source data (excluded from Git)
- `data/processed/`: derived encounter-level data (excluded from Git)
- `notebooks/`: notebooks for analysis and Colab submission
- `src/`: reusable preprocessing, modeling, and audit code
- `results/figures/`: aggregate plots for reports
- `results/tables/`: aggregate evaluation tables
- `reports/`: draft and final reports
- `slides/`: presentation materials

## Sources
- Source solution: https://github.com/saminens/Women-in-Data-Science-2020 — `WomenInDS.ipynb`, proposal cites commit `a19d3ba`.
- Competition: https://www.kaggle.com/competitions/widsdatathon2020
- Dataset release: https://doi.org/10.13026/vc0e-th79

Download data through its official source and follow its access and reuse conditions. Store local data under `data/raw/`. Do not commit encounter-level data, credentials, or trained model files. Only reviewed aggregate results should be shared. Credit reused code and analysis at the point of use.

## Next steps
1. Obtain the complete project report outline and confirm current deadlines.
2. Acquire the data and pin the source implementation.
3. Implement patient-disjoint evaluation and reproduce the baseline.
4. Document input/output, subgroup support, and one preliminary finding.
5. Extend calibration, threshold, intersectional, uncertainty, and hospital-disjoint analyses.

## GitHub
This folder is an independent local Git repository. Remote repository: https://github.com/GoodrainCN/icu-fairness-audit (private). Dataset contents and local environments are excluded by `.gitignore`.
