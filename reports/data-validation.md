# APACHE and missing-data validation

Date: 2026-10-30. First step in the agreed sequence: data validation, subgroup
metrics, uncertainty, intersectional analysis, then robustness.

## Observed data and source evidence

In the official labeled data (91,713 encounters), both APACHE probability
fields have 7,947 NaNs (8.67%). Hospital-death probability additionally has
2,371 negative entries (2.59%); ICU-death probability has 2,230 (2.43%). All
negative entries are exactly **-1**, and no observed nonmissing values exceed 1.
Combined unavailable counts are 10,318 (11.25%) and 10,177 (11.10%).

The official data dictionary describes these fields as existing APACHE IVa
mortality predictions but does not define -1 in the supplied descriptions.
A participating team's firsthand write-up states that -1 means an unavailable
APACHE score and describes recoding it as null:
https://www.sourceallies.com/2020/04/sa-takes-on-wids/ . This supports the sentinel
interpretation; it is not an organizer's authoritative explanation of every
reason the prediction is unavailable. We do not equate NaNs and -1 with the
same clinical missingness mechanism.

## Current preprocessing and its limits

The preserved source-aligned baseline retains -1 as a numeric predictor and
iteratively imputes NaNs in age, height, weight, and the two risk fields, using
only fit-partition data. Other numeric missing values remain for native LightGBM
handling. Original demographic values, including missingness, define audit groups.

Recoding -1 to NaN while leaving iterative regression unconstrained is not enough:
that diagnostic produced out-of-range imputed values. For example, hospital-risk
predictions after imputation ranged from -0.0111 to 0.99 in fit data, and a
held-out transformed value exceeded 1. These are transformed features, not the
LightGBM model's final probabilities. The diagnostic policy is retained for
inspection but was not used as a fitted sensitivity model.

The bounded alternative constrains imputed APACHE values to [0,1]. It leaves
other feature processing and the train-only fitting boundary unchanged. A second
alternative removes just the two APACHE probability features, retaining other
APACHE covariates. Neither changes original data files or the preserved baseline.

## Missingness differs across groups

For hospital-risk probability, NaN-or-negative rates in the **fit partition** are
11.45% for recorded F and 11.18% for M. Rates by recorded ethnicity include
9.62% for African American, 11.38% for Caucasian, 17.21% for Native American,
and 14.87% for Other/Unknown. The age-missing group has 24.47% unavailable
hospital-risk predictions, compared with about 10–12% in observed age bins.

These are descriptive differences with different group sizes; no missing-at-random
assumption or causal explanation is established. Complete counts by partition
and group are in `results/data_validation/apache_missingness_by_group.csv`.

## Prespecified processing sensitivity comparison

All three models use the same seed 420, patient-disjoint 58,696/14,674/18,343
fit/validation/test split, original core LightGBM parameters, early stopping,
and fixed threshold 0.5. Each fits its own preprocessing on the fit partition.
We report all policies and do not select a winner from the test scores.

| Policy | Test AUC | Recall | Precision | FNR difference F minus M |
|---|---:|---:|---:|---:|
| Retain original -1 encoding (preserved baseline) | 0.9056 | 0.3607 | 0.7358 | -3.36 pp |
| -1 to NaN; bounded iterative imputation | 0.9056 | 0.3575 | 0.7313 | -2.18 pp |
| Remove both APACHE probability fields | 0.9068 | 0.3487 | 0.7321 | -1.82 pp |

Overall ranking performance is similar, but the observed gender FNR gap changes
with preprocessing. This is sensitivity evidence on one split, not proof of a
significant gap, equivalence between policies, or improvement in fairness. The
original model must remain the object of the audit; alternatives are interventions
or sensitivity analyses. Early-stopping iteration differs between policies.

## Reproduce

```sh
MPLCONFIGDIR=/private/tmp/icu-audit-mpl .venv/bin/python -m src.validate_data
MPLCONFIGDIR=/private/tmp/icu-audit-mpl .venv/bin/python -m src.baseline --apache-policy negative-to-missing-bounded --output results/sensitivity/apache-bounded
MPLCONFIGDIR=/private/tmp/icu-audit-mpl .venv/bin/python -m src.baseline --apache-policy drop-probabilities --output results/sensitivity/apache-dropped
```

After both sensitivity runs, regenerate just the comparison with:

```sh
.venv/bin/python -c 'from src.validate_data import compare_runs; compare_runs("results/data_validation")'
```

The validation script also recreates the comparison when these run outputs exist.
Aggregate diagnostics are in `results/data_validation/`; policy-specific aggregate
metrics/manifests are in `results/sensitivity/`. No encounter-level files are exported.

## Next steps

1. Complete explicit reference-group definitions and consolidated gender/age/ethnicity
   tables, treating missing categories separately and flagging low support.
2. Add bootstrap intervals for FNR/FPR differences for the preserved baseline and
   sensitivity policies. State whether resampling is conditional on fitted models;
   bootstrap alone does not include training or hospital-shift uncertainty.
3. Extend to age×gender and gender×ethnicity with support/uncertainty rules.
4. Evaluate a prespecified threshold grid, repeated seeds and hospital-disjoint
   splits, without selecting favorable outcomes from the test results.

AI-use record: Codex implemented diagnostics, policy options and automated checks,
ran the experiments, and drafted this note for team review.
