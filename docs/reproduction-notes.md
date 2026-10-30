# Local reproduction attempt

## Status (2026-10-30)

The official Kaggle training data and dictionary are now available locally after
user login and browser download. The training ZIP passed its integrity check.
The first real-data single-model audit baseline completed successfully.

This is an **audit adaptation**, not exact reproduction of the original
four-fold ensemble. See `reports/baseline-summary.md` and the manifest under
`results/baseline/`. Original data remain under ignored `data/raw/`.

## Verification completed

- Installed a dedicated `.venv` with pinned direct dependencies and recorded
  transitive versions in `requirements-lock.txt` (Python 3.9.6 on this Mac).
- Resolved LightGBM's missing macOS OpenMP library using Homebrew `libomp`.
- Three automated checks passed: repeated-patient split isolation, frozen
  preprocessing under unseen categories/extreme validation values, and correct
  error-rate denominators/undefined-metric handling.
- End-to-end artificial-data run passed: 1,200 synthetic encounters from 600
  synthetic patients, with 768/192/240 encounters in fit/validation/test. It
  trained LightGBM and generated the expected aggregate outputs under ignored
  `results/smoke/`, visibly marked `SYNTHETIC_ONLY.txt`.
- Colab notebook structure validated; no execution results are embedded.
- Verified that raw data, the reference notebook, local environment, and
  synthetic outputs are excluded from Git.

The synthetic checks validate software operation only. A separate real-data run
subsequently completed, with the results described in the baseline summary.
No training-runtime benchmark was recorded.

## Provenance

- Original author: saminens.
- Repository: https://github.com/saminens/Women-in-Data-Science-2020
- Notebook: `WomenInDS.ipynb`, requested revision `a19d3ba`.
- Download: https://raw.githubusercontent.com/saminens/Women-in-Data-Science-2020/a19d3ba/WomenInDS.ipynb
- Retrieved file SHA-256: `4366815304e8e0a98b3c784e36075750cde8142ff90ce30c6d9d3ac9b44926cc`.
- Unmodified notebook is stored under ignored `reference/` for local inspection.
- Data: https://www.kaggle.com/competitions/widsdatathon2020/data
- Official description: https://physionet.org/content/widsdatathon2020/1.0.0/

## What the source actually does

Cells 8-9 consolidate categories and exclude patient/hospital IDs and gender.
Cell 13 uses iterative imputation for five numeric fields and fits separate
imputers on competition training and test data. Cell 19 learns categorical
encodings from the combined training/test vocabulary. Cells 23-25 remove exactly
perfectly correlated features before cross-validation. Cell 32 sets LightGBM
parameters and four-fold stratified CV. Cells 51 and 54 remove `icu_id`,
`urineoutput_apache`, and `icu_type`, and rerun the final model.

The downloaded notebook contains historical stored outputs; those are the
original author's outputs, not executions on this computer. In particular,
cell 54's displayed aggregate score appears inconsistent with its displayed
fold scores because it sums `roc_auc` rather than `roc_auc_new`. Compute fresh
metrics; do not copy this displayed aggregate into our results.

## Explicit adaptations

1. Only labeled data is used. Patient-level stratification uses each patient's
   maximum encounter outcome. Split patients into 80% development / 20% test,
   then development into 80% fit / 20% early-stopping validation (approximately
   64/16/20 overall). Repeated patients stay together; exact encounter proportions
   can differ. Abort if a partition lacks either outcome.
2. All imputation, category vocabularies, and perfect-correlation filtering are
   fit on the fit partition only. Missing/unseen categories use native missing
   values. Remaining numeric missing values are handled by LightGBM.
3. Use scikit-learn's IterativeImputer instead of the old fancyimpute import.
4. Use LightGBM's current callbacks for early stopping and logging. Identify
   categorical features from actual pandas category dtypes rather than stale
   source column positions. Retain original core tree parameters, 7,000-round
   cap, and 100-round early stopping; add fixed seeds and deterministic CPU mode.
5. Fit a single model. The original four-fold ensemble and its competition score
   are not reproduced by this first attempt. Original feature exclusion is kept.
6. Preserve original demographics separately for auditing. No patient-level data
   or predictions are exported. Aggregate subgroup results are exploratory;
   confidence intervals and hospital-disjoint evaluation remain future work.

## Remaining scientific checks

- Verify the data dictionary, feature availability and provenance, particularly
  existing APACHE death-probability fields retained by the source.
- Inspect patient/hospital identifiers, missingness and sample support on actual data.
- Compare the single-model result with proposal numbers as a different protocol;
  the exact previous local split, parameters and environment are unavailable.
- Avoid using test performance to choose thresholds or change preprocessing.
- A missing test-group AUC/recall/precision means undefined, not zero.
- Agree on disclosure and presentation of small-support aggregate groups before sharing results.

## Real-data validation notes

- 91,713 rows, 186 columns; 91,713 unique patient IDs, no duplicate encounter IDs;
  147 distinct hospitals and 7,915 hospital-death labels. Patient IDs are unique
  in this release, so patient stratification reduces to outcome stratification
  after sorting by patient ID. Original row-order splitting may differ.
- Recorded age is 16–89; age is missing for 4,228 encounters, recorded gender for
  25 and ethnicity for 1,395. Grouping uses raw demographics before imputation.
- The dictionary describes hospital_death as death during hospitalization, not
  specifically death in the ICU. The model uses first-day features.
- APACHE fields are existing APACHE IVa risk predictions. There are 2,371
  negative values in the hospital-death probability field and 2,230 in its ICU
  counterpart, plus 7,947 missing values in each. They were retained as supplied
  for source alignment in this first run. Negative values cannot be literal
  probabilities; their sentinel meaning and sensitivity to recoding need review.
- Original category consolidation and final feature exclusions were retained.
  Hyperparameters were not selected using the test results.
- Single-split subgroup metrics do not establish stable disparities. Bootstrap
  intervals, repeated splits, hospital-disjoint evaluation and APACHE sensitivity
  checks remain open tasks.
