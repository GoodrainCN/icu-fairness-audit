"""Supplementary evaluation of the audit ensemble on unseen WiDS hospitals."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.baseline import metrics, validate_data, write_json
from src.outcomes import SPLIT_SEED, THRESHOLDS, calibration_table, fit_ensemble


def run(data_path, output):
    data = pd.read_csv(data_path)
    validate_data(data)
    if data.hospital_id.isna().any() or not data.patient_id.is_unique:
        raise ValueError("This protocol requires known hospitals and one record per patient")
    partition = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SPLIT_SEED)
    train_idx, test_idx = next(partition.split(data, groups=data.hospital_id))
    development, test = data.iloc[train_idx], data.iloc[test_idx]
    assert set(development.hospital_id).isdisjoint(test.hospital_id)
    assert set(development.patient_id).isdisjoint(test.patient_id)
    scores, folds = fit_ensemble(development, test.drop(columns="hospital_death"),
                                "as-recorded")
    results = [dict(threshold=t, **metrics(test.hospital_death, scores, t))
               for t in THRESHOLDS]
    supported_sites = []
    for indices in test.groupby("hospital_id").indices.values():
        y = test.hospital_death.iloc[indices]
        if int(y.sum()) >= 20 and int((1 - y).sum()) >= 20:
            supported_sites.append(metrics(y, scores[indices], 0.50))
    output.mkdir(parents=True, exist_ok=True)
    cache = Path("data/processed")
    cache.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"encounter_id": test.encounter_id, "score": scores}).to_csv(
        cache / "scores-hospital.csv", index=False)
    pd.DataFrame(results).to_csv(output / "overall_thresholds.csv", index=False)
    calibration_table(test, scores).to_csv(output / "calibration.csv", index=False)
    write_json(output / "metrics.json", dict(
        seed=SPLIT_SEED, development_n=len(development), development_deaths=int(development.hospital_death.sum()),
        development_hospitals=development.hospital_id.nunique(), test_hospitals=test.hospital_id.nunique(),
        hospital_overlap=0, folds=folds, supported_sites=len(supported_sites),
        site_fnr_range=[min(row["fnr"] for row in supported_sites),
                        max(row["fnr"] for row in supported_sites)] if supported_sites else None,
        performance_at_0_50=results[-1],
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/training_v2.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/hospital_validation"))
    args = parser.parse_args()
    run(args.data, args.output)
