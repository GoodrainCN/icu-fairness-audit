# Part 1 draft report plan

## Required deliverables
According to the supplied project instructions: draft PDF plus computation Colab notebook, due November 24, 2026 at 11:59 PM ET. Submit once per group on Gradescope and link all teammates.

- Complete Background and Input and Output sections.
- Provide detailed analysis plans for remaining sections.
- Include one early takeaway supported by preliminary exploration.
- Incorporate presentation feedback where appropriate.
- Cite the specific audited ADS, code, and data prominently.

## Planned content for this project
1. Background: clinical risk prediction purpose, intended decision support, stakeholders, source implementation, and fairness motivation.
2. Input and Output: data provenance, cohort and exclusions, outcome definition, feature types, missingness, demographic fields, preprocessing, and probability/threshold outputs.
3. Baseline: establish the original four-fold LightGBM system, parameter provenance, and the local single-model audit adaptation. Report both historical source validation and local held-out performance with their protocols. Use the local run as the reference for subsequent comparisons. See [baseline provenance and performance](../reports/baseline-provenance.md).
4. Analysis plan: patient-disjoint development/test split, training-only preprocessing, inner early stopping validation, demographic/intersectional support, error-rate gaps, calibration, threshold sensitivity, bootstrap intervals, repeated splits, and hospital-disjoint evaluation.
5. Preliminary finding: the local held-out baseline reaches ROC-AUC 0.9056 and detects 571 of 1,583 deaths at the 0.5 cutoff (36.07% recall, 73.58% precision). Explain this operating point before presenting any subgroup gaps or APACHE sensitivity results.
6. Limitations, next steps, source attribution, and accurate AI-use disclosure.

## Presentation
November 20 lab: 4–5 slides and a 3-minute elevator pitch, following the draft structure. Each member should be ready to present.

Use five slides: (1) question and original ADS, (2) inputs and output, (3) audit baseline, (4) preliminary finding, and (5) remaining analysis. On slide 3, show the inherited model settings, the audit choices, and the held-out metrics together. Keep the complete parameter table in the report. On slide 4, unpack the 1,012 missed deaths before moving to the subgroup analysis plan.

## Outstanding clarification
The supplied report structure is truncated after Background item 1(a). Obtain the remaining sections before finalizing the outline.
The supplied documents disagree on final dates: syllabus says final report December 18; project deliverable 5 says December 15, while its timeline says December 18. Video presentation is due December 18 in the project instructions. Confirm on Brightspace.
