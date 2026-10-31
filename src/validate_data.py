"""Reproducible aggregate-only APACHE/missingness diagnostics; no data edits."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline import IMPUTE, Preprocessor, demographic_groups, patient_split, validate_data


def compare_runs(out):
    runs = {"as-recorded": "results/baseline",
            "negative-to-missing-bounded": "results/sensitivity/apache-bounded",
            "drop-probabilities": "results/sensitivity/apache-dropped"}
    rows = []
    for policy, folder in runs.items():
        path = Path(folder)
        if not (path / "overall_metrics.json").exists():
            continue
        m = json.loads((path / "overall_metrics.json").read_text())
        g = pd.read_csv(path / "subgroup_metrics.csv")
        for attr, table in g.groupby("attribute"):
            assert table.n.sum() == m["n"], attr
            for key in ["tn", "fp", "fn", "tp"]:
                assert table[key].sum() == m[key], (attr, key)
        gender = g[g.attribute == "gender"].set_index("group")
        row = {key: m[key] for key in ["roc_auc", "recall", "precision", "fnr", "fpr"]}
        row.update(policy=policy,
            female_minus_male_fnr=float(gender.loc["F", "fnr"] - gender.loc["M", "fnr"]),
            female_minus_male_fpr=float(gender.loc["F", "fpr"] - gender.loc["M", "fpr"]))
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(Path(out) / "policy_comparison.csv", index=False)


def main():
    path = Path("data/raw/training_v2.csv")
    df = pd.read_csv(path)
    validate_data(df)
    out = Path("results/data_validation")
    out.mkdir(parents=True, exist_ok=True)
    train, valid, test = patient_split(df, 420)
    partitions = dict(all=df, train=train, validation=valid, test=test)
    fields = [c for c in IMPUTE if c.startswith("apache_")]
    summary, subgroup, missingness, imputation = [], [], [], []
    for partition, data in partitions.items():
        groups = demographic_groups(data)
        for col in data:
            missingness.append(dict(partition=partition, feature=col, n=len(data),
                missing_n=int(data[col].isna().sum()), missing_fraction=float(data[col].isna().mean())))
        for col in fields:
            v = data[col]
            status = pd.Series(np.select([v.isna(), v < 0, v > 1],
                ["missing", "negative", "above_one"], default="valid_probability"), index=data.index)
            for state in ["missing", "negative", "above_one", "valid_probability"]:
                selected = data[status == state]
                summary.append(dict(partition=partition, feature=col, status=state,
                    n=len(selected), fraction=len(selected) / len(data),
                    deaths=int(selected.hospital_death.sum()),
                    death_rate=float(selected.hospital_death.mean()) if len(selected) else None))
            for attr, values in groups.items():
                for group in sorted(values.unique()):
                    mask = values == group
                    vv = v[mask]
                    invalid = vv.isna() | (vv < 0) | (vv > 1)
                    subgroup.append(dict(partition=partition, feature=col, attribute=attr,
                        group=str(group), n=int(mask.sum()), missing_n=int(vv.isna().sum()),
                        negative_n=int((vv < 0).sum()), invalid_or_missing_n=int(invalid.sum()),
                        invalid_or_missing_fraction=float(invalid.mean())))
    for policy in ["as-recorded", "negative-to-missing", "negative-to-missing-bounded", "drop-probabilities"]:
        prep = Preprocessor(policy).fit(train, 420)
        for name, data in [("train", train), ("validation", valid), ("test", test)]:
            transformed = prep.transform(data)
            for col in fields:
                if col in transformed:
                    v = transformed[col]
                    imputation.append(dict(policy=policy, partition=name, feature=col,
                        retained=True, minimum=float(v.min()), maximum=float(v.max()),
                        negative_after_processing=int((v < 0).sum()),
                        above_one_after_processing=int((v > 1).sum()),
                        missing_after_processing=int(v.isna().sum())))
                else:
                    imputation.append(dict(policy=policy, partition=name, feature=col, retained=False))
    pd.DataFrame(summary).to_csv(out / "apache_status.csv", index=False)
    pd.DataFrame(subgroup).to_csv(out / "apache_missingness_by_group.csv", index=False)
    pd.DataFrame(missingness).to_csv(out / "feature_missingness_by_split.csv", index=False)
    pd.DataFrame(imputation).to_csv(out / "imputation_diagnostics.csv", index=False)
    joint = pd.crosstab(df[fields[0]].fillna(-999).apply(lambda x: "NaN" if x == -999 else
        "negative" if x < 0 else "valid"), df[fields[1]].fillna(-999).apply(lambda x:
        "NaN" if x == -999 else "negative" if x < 0 else "valid"))
    joint.to_csv(out / "apache_joint_status.csv")
    manifest = dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        seed=420, rows=len(df), unique_patients=int(df.patient_id.nunique()),
        unique_hospitals=int(df.hospital_id.nunique()),
        negative_distinct_values={col: sorted(df.loc[df[col] < 0, col].unique().tolist()) for col in fields},
        data_modified=False, imputation_fit_partition="train_only",
        caution="Invalid-range diagnostics do not establish sentinel semantics or causation. "
                "Iterative regression is unconstrained; derived risk features can exceed [0,1].")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    compare_runs(out)
    print(json.dumps(manifest, indent=2))
    print(pd.DataFrame(summary).query("partition == 'all'").to_string(index=False))
    print(pd.DataFrame(imputation).to_string(index=False))


if __name__ == "__main__":
    main()
