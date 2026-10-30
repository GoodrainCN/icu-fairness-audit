# First local real-data baseline

Run date: October 30, 2026. This is an audit adaptation of saminens'
`WomenInDS.ipynb` (revision `a19d3ba`), not an exact rerun of the original
four-fold ensemble. See `docs/reproduction-notes.md` for all adaptations.

## Data and protocol

Official Kaggle `training_v2.csv`: 91,713 labeled encounters, 186 columns,
91,713 unique patient IDs and 147 hospitals. The label is **death during the
hospitalization**, not specifically ICU death. There are 7,915 deaths (8.63%).

Patient-level outcome-stratified split, seed 420:

| Partition | Encounters | Deaths |
|---|---:|---:|
| Fit | 58,696 | 5,066 |
| Early-stopping validation | 14,674 | 1,266 |
| Held-out test | 18,343 | 1,583 |

No patient crosses partitions. Imputation, category vocabularies and perfectly
correlated-feature selection are fit on the fit partition only. Original final
LightGBM core parameters are retained; early stopping selected 3,103 rounds.
The decision threshold was fixed at 0.5 before test evaluation.

## Held-out test performance

| Metric | Proposal's historical result | This run |
|---|---:|---:|
| ROC-AUC | 0.904 | 0.9056 |
| Recall / TPR at 0.5 | 0.359 | 0.3607 |
| Precision at 0.5 | 0.717 | 0.7358 |
| Accuracy | Not stated | 0.9337 |
| F1 | Not stated | 0.4841 |
| False-negative rate | Not stated | 0.6393 |
| False-positive rate | Not stated | 0.0122 |

Confusion counts: TN 16,555; FP 205; FN 1,012; TP 571.

**Early takeaway:** AUC around 0.906 coexists with low recall at threshold 0.5:
1,012 of the 1,583 observed deaths were classified below the threshold. AUC
describes ranking across thresholds and does not establish adequacy at a chosen
operating threshold. Clinical implications depend on the actual decision use,
which this retrospective audit does not validate.

Similarity to the proposal supports feasibility; it is not exact numerical
reproduction, because the previous split, environment and implementation are
unavailable and this run uses an explicitly documented evaluation protocol.

## Preliminary demographic comparison

| Recorded gender | Test n | Deaths | Recall | FNR |
|---|---:|---:|---:|---:|
| F | 8,481 | 744 | 0.3777 | 0.6223 |
| M | 9,857 | 837 | 0.3441 | 0.6559 |

Female-minus-male FNR difference: **-0.0336**, similar in direction and magnitude
to the proposal's -0.035. **No confidence interval was computed in this run.**
Do not reuse the proposal's interval for these new predictions or infer stable
discrimination from this single split. Five test encounters have missing gender;
their estimates are flagged as low support. Age/ethnicity tables are saved but
small groups require uncertainty analysis before interpretation.

## Limitations and next steps

- This is a single fitted model and one random split, not the original ensemble.
- Complete confidence intervals and repeated-split checks before disparity claims.
- Examine calibration, thresholds, intersections and hospital-disjoint performance.
- Source APACHE probability features include negative values; investigate sentinel
  semantics and sensitivity while preserving this recorded first baseline.
- Observational labels, missingness, population composition and intended use
  constrain fairness interpretations; group differences are not causal findings.

## Files and attribution

- Aggregate outputs: `results/baseline/`.
- Data validation: `results/data_validation.json`.
- Reproducibility: `results/baseline/run_manifest.json`, `requirements-lock.txt`.
- Entry points: `src/baseline.py`, `notebooks/01_baseline.ipynb`.
- Original code: https://github.com/saminens/Women-in-Data-Science-2020/tree/a19d3ba
- Official data: https://www.kaggle.com/competitions/widsdatathon2020/data
- Proposal comparison: team proposal in `docs/proposal.pdf`.

AI-use record: ChatGPT/Codex helped inspect the source, implement the audit
adaptation and checks, execute the local run, and draft these notes. Team members
must review the analysis and use an accurate disclosure consistent with the
course's adopted policy before submitting any deliverable.
