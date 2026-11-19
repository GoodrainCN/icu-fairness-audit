# Section 4 — Outcomes

If a service reviewed the highest-scoring 10% of ICU encounters, which groups would be more likely to enter the queue or miss review? We evaluated a four-fold LightGBM ensemble retrained with the source hyperparameters and fold-local preprocessing. The 18,343 held-out encounters were excluded from training and early stopping but had been inspected in earlier exploratory analyses. The local file contained one encounter per recorded patient ID. WiDS supplied first-day measurements and in-hospital mortality labels. Review capacity and downstream action are hypothetical. [Competition protocol](https://physionet.org/content/widsdatathon2020/1.0.0/) · [Model reproduction](source-ensemble-reproduction.md)

## 4(a) Performance by subgroup

Across 1,583 deaths and 16,760 survivors, ROC-AUC was **0.9064**. The 1,834 review slots covered 917 deaths, giving **57.9% recall** and **50.0% precision**. AUC measures how often a death ranks above a survivor; recall and precision tell us how many deaths a fixed workload captures and how many reviewed encounters end in death. Age was the clearest difference in ranking: AUC was **0.8521 at ages 80+** and **0.9481 below 45**. We examined age because baseline risk varies across the life course, and recorded gender and ethnicity because an aggregate result can conceal differences in access to review. [Subgroup counts and metrics](../results/review_budget/groups.csv)

## 4(b) Fairness under a common review budget

All groups share one queue. We rank scores from highest to lowest, round the number of slots down and break ties by encounter ID. **FNR** measures the fraction of eventual deaths outside the queue. **FPR** measures the fraction of eventual survivors inside it. Equalized odds asks whether these two conditional rates are equal across groups. It makes differences in review allocation visible. The clinical value of any individual review depends on the information available at the time and what action review enables.

**Who is missed at 10% capacity?** The queue contained 1,834 encounters, including 917 eventual deaths and 917 survivors. The other 666 deaths fell outside it. The age comparison spans groups with different observed mortality. The gender comparison asks whether review captures deaths similarly across the two largest recorded categories. For ethnicity we emphasize the two largest named categories to give the rates reasonably supported denominators. Other groups, including missing categories, remain in the full tables. The demographic comparisons overlap and their counts should not be summed.

| Group | Deaths missed / deaths (FNR) | Survivors flagged / survivors (FPR) |
|---|---:|---:|
| 80 years and above | 165 / 376 (43.9%) | 234 / 2,441 (9.6%) |
| Under 45 | 35 / 95 (36.8%) | 53 / 2,576 (2.1%) |
| Recorded F | 317 / 744 (42.6%) | 394 / 7,737 (5.1%) |
| Recorded M | 349 / 837 (41.7%) | 521 / 9,020 (5.8%) |
| African American | 62 / 151 (41.1%) | 92 / 1,763 (5.2%) |
| Caucasian | 531 / 1,244 (42.7%) | 712 / 12,926 (5.5%) |

| First group minus second | AUC gap (95% CI) | FNR gap in pp (95% CI) | FPR gap in pp (95% CI) |
|---|---:|---:|---:|
| 80+ minus under 45 | −0.096 (−0.128, −0.059) | +7.04 (−4.77, +16.19) | +7.53 (+6.27, +9.02) |
| Recorded F minus M | −0.003 (−0.018, +0.013) | +0.91 (−4.60, +5.64) | −0.68 (−1.39, +0.05) |
| African American minus Caucasian | +0.003 (−0.028, +0.031) | −1.63 (−9.07, +7.66) | −0.29 (−1.34, +1.07) |

We fixed three contrasts for this follow-up after earlier exploratory inspection. They were not preregistered. The table gives pointwise percentile intervals from 4,000 bootstrap draws of whole hospitals with replacement (147 hospitals represented). Each draw recalculates the common queue capacity and group rates, conditional on fitted scores. The results file also gives Bonferroni-adjusted intervals for nine contrasts within each split (three comparisons × three metrics). In the patient holdout the adjusted age AUC interval is −0.140 to −0.045 and the adjusted age FPR interval is +5.72 to +9.75 pp. The remaining seven adjusted intervals include zero. This adjustment does not account for earlier exploratory choices or training uncertainty. We report other budgets, intersections and preprocessing variants descriptively. [Intervals and adjusted bounds](../results/review_budget/intervals.csv) · [Counts and rates](../results/review_budget/groups.csv)

**Capacity sensitivity.** At 5%, 10% and 20% capacity, respectively, the 80+ minus under-45 FPR gap was +1.93, +7.53 and +19.66 pp. The corresponding FNR gaps were +8.03, +7.04 and +2.07 pp. The recorded F-minus-M FNR gap was −1.67, +0.91 and +1.81 pp. Thus which gender appears to have the higher miss rate changes with the queue size. The uncertainty at 10% prevents a firm ranking. [All capacity contrasts](../results/review_budget/gaps.csv)

## 4(c) What might account for the age difference?

The 80+ group had higher observed mortality (13.3% versus 3.6%). That difference matters to a service prioritizing risk. FPR, however, uses survivors as its denominator. A difference in mortality prevalence alone does not determine the FPR gap. We therefore examined the **2,441 older and 2,576 younger survivors** in the primary holdout, keeping the original 10% queue fixed.

| Among eventual survivors | Ages 80+ | Under 45 |
|---|---:|---:|
| Recorded APACHE IVa hospital-death probability, median (quartiles) | 10% (5–18%) | 2% (1–4%) |
| APACHE probability unavailable or invalid | 10.3% | 12.9% |
| Mean fraction of 64 first-day measurement fields missing per encounter | 28.4% | 30.9% |
| Observed fraction flagged for review | 9.59% | 2.06% |
| Fraction flagged after standardizing the APACHE risk distribution | 6.32% | 5.13% |

For the last row, we calculated each age group's flag rate within the same six strata of recorded APACHE risk (up to 5%, then up to 10%, 20%, 50%, 100%, and unavailable). We averaged those rates using the pooled survivors’ stratum frequencies for both groups. The gap shrank from **7.53 to 1.19 percentage points**. A separate standardization by hospital changed the gap from **7.48 to 7.23 points** among survivors at the 127 hospitals represented in both groups. The two comparisons use separate weights. APACHE is a model-derived risk proxy and an input to the audited ensemble; it cannot isolate physiological severity independently of age. These descriptive estimates depend on the bands, and we did not estimate intervals for the standardized gaps. [Case mix by outcome](../results/review_budget/age_case_mix.csv) · [Standardization results](../results/review_budget/age_standardization.csv) · [Hospital composition](../results/review_budget/age_hospital_shares.csv)

The raw age FPR gap remained between +7.33 and +7.53 points across three APACHE input treatments. Removing the risk fields tests sensitivity to an input choice; reweighting survivors tests sensitivity to the mix of cases being compared. [APACHE baseline, calibration, cutoff sensitivity and hospital details](outcomes-appendix.md)

**Sensitivity to hospital selection.** We kept the pooled 10% rule and inner fitting protocol fixed and tested the original hospital split plus two consecutive outer seeds, refitting all four models for each new split. Each split held out 30 hospitals. The 80+ minus under-45 FPR gaps were **+7.97, +7.19 and +5.99 pp** (seeds 420–422). The corresponding FNR gaps were **−8.58, +5.87 and +3.93 pp**. These point estimates describe sensitivity to site selection within WiDS. The original split was previously inspected and shared 3,713 test encounters with the primary holdout. The two additional splits shared 5,382 and 2,640, respectively. None provides independent external confirmation. [Results for every seed and contrast](../results/review_budget/hospital_split_sensitivity.csv)

## Findings and Interpretation

Older survivors entered the simulated review queue more often. The direction recurred across hospital splits, but the magnitude depended strongly on the risk profiles being compared. An older survivor could have been an appropriate candidate for review given the information available on the first ICU day. Equalizing the proportion reviewed among eventual survivors would be a policy choice requiring evidence about clinical need, benefit and burden.

The audit supports lower within-group discrimination at ages 80+ in the primary holdout. It leaves the direction of age differences in missed deaths uncertain across hospitals. The survivor standardization addresses the flag-rate gap only. The mechanisms behind the AUC difference, the benefit patients would receive from review, and whether the observed allocation constitutes unfair treatment remain unresolved.
