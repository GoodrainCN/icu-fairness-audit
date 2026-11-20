# Section 4 research-design assessment and validation

The earlier 10% question made an audit-defined binary decision the starting point for claims about a continuous mortality predictor. The ranking and calibration evidence was usable as collected. Error rates observed at one capacity, even with intervals, could not establish how gaps behaved at other capacities. The missing experiment was whether the APACHE-standardized age FPR gap also shrank when the selection rule changed.

The revision puts subgroup discrimination and calibration first, calculates recall and precision for three equally hypothetical rules, and compares all three primary demographic contrasts at each capacity. Reference-scenario confusion counts and detailed intervals appear in the appendix. Reweighting uses the same survivor cohort, risk strata and pooled weights at each capacity. Only the global selection cutoff changes. The estimated standardized age gap changes sign at the most restrictive capacity. Its magnitude cannot be generalized from the 10% result.

## Checks

- Joined archived as-recorded scores to the original seed-420 holdout by encounter ID and independently reconstructed the three pooled queues. Counts, deaths, selected encounters, missed deaths, flagged survivors and AUC matched the archived results for all 48 cohort/group-by-capacity rows. Recall and precision use these counts.
- Checked all nine demographic contrast rows against the group estimates. Computed mean predicted risks from the original calibration bins weighted by bin size, retaining sparse groups and their warnings.
- Recomputed the standardized FPR contrasts independently from survivor-specific within-band flag means and pooled band shares. All three agreed with the new CSV, their crude contrasts agreed with the archived FPR gaps, and rerunning the pipeline left all previously tracked numerical outputs unchanged. Six existing tests passed.
- The Section 3 reproduction source identifies continuous mortality scores. The revised findings carry that definition into the implications for Section 5. The repository has no integrated, paginated final report or standalone English Section 5 to verify. Whole-report pagination against the ten-page limit remains an assembly check.

## Remaining uncertainty

FNR/FPR intervals are available at the 10% reference only. The 5% and 20% comparisons and all standardized gaps are descriptive. The scores and labels come from a retrospective study, mortality risk is only partially captured by APACHE, and internal hospital splits overlap across analyses. Review benefit, appropriate staffing and individual treatment effects were not measured.

[Revised analysis](outcomes.md) · [Pipeline](../src/age_case_mix.py) · [Three-capacity estimates](../results/review_budget/age_standardization_by_capacity.csv)
