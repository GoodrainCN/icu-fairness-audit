# ICU Mortality Prediction Fairness Audit

DS-US 202: Responsible Data Science, Fall 2026  
Team: Guyu Sun, Rachel Ruan, Laura Zhu

## Goal
Audit demographic and intersectional fairness in an existing LightGBM ICU mortality prediction system using WiDS Datathon 2020 data (Option B).

## Current status
Project workspace initialized. The submitted proposal and supplied course documents are in `docs/`. No project dataset, executable audit pipeline, or reproduced results are present yet. Preliminary numbers in the proposal are historical claims and have not been independently reproduced in this workspace.

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
