# Section 4 — Outcomes

**How does the predictive performance of the WiDS mortality model vary across demographic groups, and how sensitive are observed fairness disparities to decision thresholds, patient risk distributions, and hospital populations?** The WiDS task predicts in-hospital mortality from first-day ICU data. We audited continuous scores from an ensemble retrained with the source hyperparameters and fold-local preprocessing. Its 18,343 held-out encounters were excluded from fitting and early stopping but had been inspected during earlier exploratory work. The review rules below were introduced by the audit. [Competition protocol](https://physionet.org/content/widsdatathon2020/1.0.0/) · [Model reproduction](source-ensemble-reproduction.md)

## 4(a) Performance by subgroup

ROC-AUC measures how often a death ranks above a survivor within a group. Calibration asks whether predicted probabilities agree with observed mortality. Both can be evaluated directly from scores. At 8.63% mortality, always predicting survival would achieve 91.37% accuracy while detecting no deaths. We therefore also report recall (fraction of deaths selected) and precision (fraction of selected encounters ending in death) under three hypothetical rules. We compare age because patient characteristics and mortality vary across the life course, and recorded gender and ethnicity because overall averages can conceal differences in prediction and allocation.

**Primary holdout.** The risk column compares mean predicted risk with observed mortality. Each triple gives percentages when selecting the highest-scoring 5%, 10% and 20% of the entire cohort, in that order.

| Group | Encounters | Deaths | AUC | Predicted / observed (%) | Recall (%) | Precision (%) |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 18,343 | 1,583 | 0.9064 | 8.5 / 8.6 | 40.1 / 57.9 / 77.2 | 69.1 / 50.0 / 33.3 |
| Age · Under 45 | 2,671 | 95 | 0.9481 | 4.0 / 3.6 | 45.3 / 63.2 / 80.0 | 69.4 / 53.1 / 36.4 |
| Age · 45–64 | 6,012 | 412 | 0.9197 | 6.9 / 6.9 | 40.3 / 57.8 / 75.7 | 65.9 / 47.4 / 33.3 |
| Age · 65–79 | 5,977 | 583 | 0.8991 | 9.7 / 9.8 | 40.0 / 57.6 / 77.5 | 70.8 / 51.5 / 33.5 |
| Age · 80+ | 2,817 | 376 | 0.8521 | 12.7 / 13.3 | 37.2 / 56.1 / 77.9 | 68.3 / 47.4 / 32.6 |
| Age · Missing | 866 | 117 | 0.8557 | 12.8 / 13.5 | 44.4 / 61.5 / 76.1 | 75.4 / 59.0 / 32.6 |
| Gender · F | 8,481 | 744 | 0.9047 | 8.8 / 8.8 | 40.9 / 57.4 / 76.2 | 70.9 / 52.0 / 32.5 |
| Gender · M | 9,857 | 837 | 0.9076 | 8.3 / 8.5 | 39.2 / 58.3 / 78.0 | 67.5 / 48.4 / 34.0 |
| Gender · Missing † | 5 | 2 | 1.0000 | 46.6 / 40.0 | 100.0 / 100.0 / 100.0 | 100.0 / 50.0 / 50.0 |
| Ethnicity · African American | 1,914 | 151 | 0.9062 | 8.3 / 7.9 | 42.4 / 58.9 / 76.2 | 64.6 / 49.2 / 32.8 |
| Ethnicity · Caucasian | 14,170 | 1,244 | 0.9033 | 8.6 / 8.8 | 38.7 / 57.3 / 76.8 | 69.9 / 50.0 / 33.3 |
| Ethnicity · Hispanic | 730 | 70 | 0.9291 | 9.3 / 9.6 | 48.6 / 62.9 / 85.7 | 75.6 / 57.1 / 39.7 |
| Ethnicity · Asian † | 215 | 17 | 0.9213 | 8.3 / 7.9 | 47.1 / 58.8 / 70.6 | 72.7 / 58.8 / 27.9 |
| Ethnicity · Native American † | 175 | 15 | 0.9446 | 9.0 / 8.6 | 53.3 / 73.3 / 93.3 | 80.0 / 52.4 / 40.0 |
| Ethnicity · Other/Unknown | 858 | 64 | 0.9136 | 8.4 / 7.5 | 43.8 / 57.8 / 73.4 | 54.9 / 42.5 / 29.0 |
| Ethnicity · Missing | 281 | 22 | 0.9347 | 8.0 / 7.8 | 50.0 / 59.1 / 81.8 | 84.6 / 50.0 / 34.0 |

† Fewer than 20 deaths or survivors. Estimates are unstable. The same encounters appear once per demographic dimension. [Subgroup counts](../results/review_budget/groups.csv) · [Calibration by score band](../results/review_budget/calibration_bins.csv)

AUC decreased across the four observed age bands. The 80+ minus under-45 difference was −0.0960 (hospital-bootstrap 95% CI −0.128 to −0.059). Recorded F and M had similar AUC estimates, as did the two largest named ethnicity groups. The Hispanic group had higher observed AUC than either, with 70 deaths and no interval for that contrast. Close group means can hide miscalibration within a risk band. Among encounters aged 80+, the 5–10% risk band averaged 7.1% predicted and 8.7% observed mortality, while the 30–50% band averaged 38.4% predicted and 35.8% observed. Calibration slopes were 1.05 and 1.03 for recorded F and M, and 0.96 and 1.05 for African American and Caucasian encounters (ideal slope 1). These diagnostics are descriptive. [Calibration diagnostics and figure](outcomes-appendix.md) · [Subgroup slopes](../results/review_budget/calibration_summary.csv)

## 4(b) Fairness across hypothetical operating rules

For each rule we select the top fraction across all encounters, round slots down and break ties by encounter ID. Earlier analyses used 10% as an illustrative reference for a shared queue. Neither the competition protocol nor the audit provides staffing or clinical evidence favoring it over 5% or 20%. Equalized odds is informative if a binary flag prompts review: FNR measures the fraction of deaths outside the queue and FPR the fraction of survivors inside. These rates describe who would receive attention. Calibration matters when the score is interpreted as a probability, and judging a review policy also requires evidence about its benefits and burdens.

| Difference in percentage points (first group minus second) | 5% FNR | 5% FPR | 10% FNR | 10% FPR | 20% FNR | 20% FPR |
|---|---:|---:|---:|---:|---:|---:|
| 80+ minus under 45 | +8.03 | +1.93 | +7.04 | +7.53 | +2.07 | +19.66 |
| Recorded F minus M | -1.67 | -0.14 | +0.91 | -0.68 | +1.81 | +1.19 |
| African American minus Caucasian | -3.72 | +0.38 | -1.63 | -0.29 | +0.69 | -1.44 |

Older survivors were selected more often under all three rules, but the FPR gap widened as capacity increased. The age FNR gap narrowed. The gender and ethnicity FNR gaps each changed sign. Pointwise 95% intervals at 10% were +6.27 to +9.02 points for age FPR and −4.77 to +16.19 for age FNR. Only the age AUC and age FPR gaps excluded zero after adjustment for nine comparisons in the primary holdout. Intervals for FNR and FPR were calculated at 10% only; the cross-capacity patterns are descriptive. [Bootstrap methods and intervals](outcomes-appendix.md) · [All contrasts](../results/review_budget/gaps.csv)

## 4(c) Sensitivity to risk distributions, inputs and hospitals

Among survivors, median recorded APACHE IVa hospital-death risk was 10% at ages 80+ and 2% under 45. At each capacity we kept individual flags fixed, calculated the age-specific FPR within six APACHE strata (five risk bands and one unavailable category), and averaged each group's rates using the same pooled stratum weights from 2,441 older and 2,576 younger survivors.

| Fraction selected | Raw age FPR gap (pp) | Standardized FPR, 80+ | Standardized FPR, under 45 | Standardized gap (pp) |
|---|---:|---:|---:|---:|
| 5% | +1.93 | 1.69% | 2.09% | -0.40 |
| 10% | +7.53 | 6.32% | 5.13% | +1.19 |
| 20% | +19.66 | 17.21% | 11.71% | +5.49 |

Standardization reduced the absolute gap at every capacity. The residual changed sign at 5% and remained positive at 20%. APACHE is a model-derived risk proxy and an ensemble input. The association depends on the chosen bands and does not identify a causal mechanism. No interval was estimated for the standardized gaps. [All three standardizations](../results/review_budget/age_standardization_by_capacity.csv)

Across three APACHE input treatments, the ensemble's overall AUC ranged from 0.9064 to 0.9086 and the oldest group had lower AUC than the youngest in each refit. On the same 16,279 encounters with valid APACHE scores, the ensemble exceeded APACHE alone by 0.0469 AUC (paired 95% CI 0.0406 to 0.0537). Both comparisons evaluate ranking independently of a review rule. [Baseline and input sensitivity](outcomes-appendix.md)

Holding out 30 hospitals gave overall AUC 0.9047. Across three hospital splits the older group's AUC was lower each time. At 10% capacity its FNR gap varied from −8.58 to +5.87 points and its FPR gap from +5.99 to +7.97 points. These are overlapping internal splits from the same dataset. [Split-specific contrasts](../results/review_budget/hospital_split_sensitivity.csv)

## Findings and interpretation

The clearest model-level disparity is weaker mortality-risk ranking within the oldest group. Under the three simulated policies, older survivors were flagged more often, with a gap strongly associated with capacity and the distribution of predicted APACHE risk. The evidence does not identify an age group that consistently misses review across hospitals. Survival alone cannot establish whether a review was warranted. A judgment about unfair treatment would require evidence about clinical need, decisions and consequences.
