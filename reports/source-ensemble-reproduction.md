# Original four-fold LightGBM reproduction

We ran the final four-fold LightGBM training and validation workflow in
[saminens' source notebook at revision a19d3ba](https://github.com/saminens/Women-in-Data-Science-2020/blob/a19d3ba/WomenInDS.ipynb)
on all 91,713 labeled WiDS encounters. The rerun preserves the source's
stratified four-fold split (seed 420), main LightGBM hyperparameters, 7,000-round
cap, 100-round early stopping, APACHE treatment as supplied, category
consolidation, and final feature exclusions.

| Fold | Validation encounters | Archived source AUC | Rerun AUC | Rerun best round |
|---|---:|---:|---:|---:|
| 1 | 22,929 | 0.91123 | 0.91102 | 3,314 |
| 2 | 22,928 | 0.90069 | 0.90008 | 2,871 |
| 3 | 22,928 | 0.90559 | 0.90768 | 2,559 |
| 4 | 22,928 | 0.90561 | 0.90383 | 2,558 |
| Mean fold AUC | | 0.90578 | 0.90565 | |

The source stores each fold's AUC and averages the four models' predictions
for its unlabeled competition test file. Its printed final aggregate reads
the earlier model's accumulator. The archived mean in the table is computed
from the four displayed final-model scores.

## Scope and compatibility

- Uses scikit-learn's IterativeImputer, explicit numeric-only pandas
  correlation, and LightGBM 4.6.0 callbacks and Dataset arguments.
  Best rounds and per-fold AUCs differ from the archived execution.
- The official `unlabeled.csv` was unavailable locally. The recorded run
  learns the category vocabulary from the labeled file and evaluates the
  four validation folds. Competition-test averaging can be run with the
  optional file. No competition-test results are reported here.
- As in the source, preprocessing is learned on the labeled file before
  splitting into folds, and the validation fold is used for early stopping.
  The reported mean is a cross-validation result under that protocol.
  Independent audit performance requires a labeled holdout set apart from
  model fitting, preprocessing, and early stopping.
- The final source configuration uses the three exclusions already selected
  in the notebook (`icu_id`, `urineoutput_apache`, `icu_type`). This
  run applies those selections directly.

Run with `.venv/bin/python -m src.source_ensemble`. See the
[aggregate metrics and run details](../results/source_ensemble/cross_validation.json).
