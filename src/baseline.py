"""Audit adaptation of saminens/Women-in-Data-Science-2020, commit a19d3ba.

Preserves final LightGBM parameters and category consolidation. Deliberately
uses training-only preprocessing and a patient-disjoint holdout, not the
original competition's transductive preprocessing/four-fold ensemble.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

SOURCE = "https://github.com/saminens/Women-in-Data-Science-2020/tree/a19d3ba"
IMPUTE = ["age", "height", "weight", "apache_4a_hospital_death_prob",
          "apache_4a_icu_death_prob"]
DROP = ["hospital_death", "encounter_id", "patient_id", "hospital_id", "gender",
        "readmission_status", "icu_id", "urineoutput_apache", "icu_type"]
REPLACEMENTS = {
    "hospital_admit_source": {"Other ICU": "ICU", "ICU to SDU": "SDU",
        "Step-Down Unit (SDU)": "SDU", "Other Hospital": "Other",
        "Observation": "Recovery Room", "Acute Care/Floor": "Acute Care"},
    "icu_type": {"CCU-CTICU": "Grpd_CICU", "CTICU": "Grpd_CICU",
                 "Cardiac ICU": "Grpd_CICU"},
    "apache_2_bodysystem": {"Undefined diagnoses": "Undefined Diagnoses"},
}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def validate_data(df):
    required = set(["hospital_death", "encounter_id", "patient_id", "hospital_id",
                    "gender", "ethnicity"] + IMPUTE)
    missing = required - set(df)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["patient_id"].isna().any() or df["encounter_id"].isna().any():
        raise ValueError("Missing patient/encounter IDs; cannot guarantee disjointness")
    if df["encounter_id"].duplicated().any():
        raise ValueError("Duplicate encounter_id; investigate before splitting")
    if df["hospital_death"].isna().any() or set(df["hospital_death"].unique()) != {0, 1}:
        raise ValueError("Expected fully labeled binary hospital_death")


def patient_split(df, seed=420):
    # For repeated patients, stratify on any encounter death, then map back to
    # encounters. This stratifies patient labels, not exact encounter prevalence.
    labels = df.groupby("patient_id", sort=True)["hospital_death"].max()
    dev, test = train_test_split(labels.index.to_numpy(), test_size=0.2,
                                random_state=seed, stratify=labels.to_numpy())
    train, valid = train_test_split(dev, test_size=0.2, random_state=seed + 1,
                                   stratify=labels.loc[dev].to_numpy())
    parts = [df[df.patient_id.isin(ids)].copy() for ids in (train, valid, test)]
    patient_sets = [set(p.patient_id) for p in parts]
    assert not any(patient_sets[i] & patient_sets[j]
                   for i in range(3) for j in range(i))
    assert sum(map(len, parts)) == len(df)
    for p in parts:
        if p.hospital_death.nunique() != 2:
            raise ValueError("Both outcomes required in each split; insufficient support")
    return parts


class Preprocessor:
    """Fit all learned transformations only on the training partition."""

    def __init__(self, apache_policy="as-recorded"):
        if apache_policy not in {"as-recorded", "negative-to-missing", "negative-to-missing-bounded", "drop-probabilities"}:
            raise ValueError("Unknown APACHE policy")
        self.apache_policy = apache_policy

    def base(self, df):
        x = df.drop(columns=DROP, errors="ignore").copy()
        probability_cols = [c for c in IMPUTE if c.startswith("apache_") and c in x]
        if self.apache_policy in {"negative-to-missing", "negative-to-missing-bounded"}:
            x[probability_cols] = x[probability_cols].mask(x[probability_cols] < 0)
        elif self.apache_policy == "drop-probabilities":
            x = x.drop(columns=probability_cols)
        for col, mapping in REPLACEMENTS.items():
            if col in x:
                x[col] = x[col].replace(mapping)
        return x

    def fit(self, df, seed):
        x = self.base(df)
        self.empty = [c for c in x if x[c].isna().all()]
        x = x.drop(columns=self.empty)
        self.impute_cols = [c for c in IMPUTE if c in x]
        bounds = {}
        if self.apache_policy == "negative-to-missing-bounded":
            bounds = dict(min_value=np.array([0 if c.startswith("apache_") else -np.inf
                                              for c in self.impute_cols]),
                          max_value=np.array([1 if c.startswith("apache_") else np.inf
                                              for c in self.impute_cols]))
        self.imputer = IterativeImputer(random_state=seed, max_iter=10, **bounds)
        if self.impute_cols:
            x[self.impute_cols] = self.imputer.fit_transform(x[self.impute_cols])
        self.categories = {c: sorted(x[c].dropna().astype(str).unique().tolist())
                           for c in x.select_dtypes(include=["object", "category", "string"])}
        self.columns = x.columns.tolist()
        encoded = self.encode(x)
        corr = encoded.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.correlated = [c for c in upper if (upper[c] == 1.0).any()]
        self.features = [c for c in self.columns if c not in self.correlated]
        return self

    def encode(self, x):
        x = x.copy()
        for c, categories in self.categories.items():
            # Unseen and missing values both receive code -1, native missing in
            # LightGBM. No vocabulary is learned from validation or test data.
            x[c] = pd.Categorical(x[c].astype("string"), categories=categories)
        return x

    def transform(self, df):
        x = self.base(df).drop(columns=self.empty).reindex(columns=self.columns)
        if self.impute_cols:
            x[self.impute_cols] = self.imputer.transform(x[self.impute_cols])
        return self.encode(x)[self.features]


def metrics(y, probability, threshold):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(probability) >= threshold
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    def ratio(a, b):
        return float(a / b) if b else None
    return dict(n=len(y), deaths=int(y.sum()), death_rate=float(y.mean()),
                roc_auc=float(roc_auc_score(y, probability)) if len(set(y)) == 2 else None,
                accuracy=ratio(tp + tn, len(y)), precision=ratio(tp, tp + fp),
                recall=ratio(tp, tp + fn), fnr=ratio(fn, tp + fn),
                fpr=ratio(fp, fp + tn), f1=ratio(2 * tp, 2 * tp + fp + fn),
                mean_predicted_risk=float(np.mean(probability)),
                observed_minus_predicted=float(y.mean() - np.mean(probability)),
                tn=tn, fp=fp, fn=fn, tp=tp)


def demographic_groups(df):
    # Fixed exploratory age bins; not chosen to maximize observed disparities.
    age = pd.cut(df.age, [0, 45, 65, 80, np.inf], right=False,
                 labels=["under_45", "45_to_64", "65_to_79", "80_plus"])
    groups = pd.DataFrame(index=df.index)
    groups["age"] = age.astype("string").fillna("Missing")
    groups["gender"] = df.gender.astype("string").fillna("Missing")
    groups["ethnicity"] = df.ethnicity.astype("string").fillna("Missing")
    return groups


def subgroup_table(test, probability, threshold):
    rows = []
    for attr, values in demographic_groups(test).items():
        for group in sorted(values.unique()):
            mask = (values == group).to_numpy(dtype=bool)
            row = metrics(test.hospital_death.to_numpy()[mask], probability[mask], threshold)
            row.update(attribute=attr, group=str(group),
                       low_outcome_support=(row["deaths"] < 20 or row["n"] - row["deaths"] < 20))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_confusion(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = np.array([[result["tn"], result["fp"]], [result["fn"], result["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(a, cmap="Blues")
    for (i, j), value in np.ndenumerate(a):
        ax.text(j, i, str(value), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Survival", "Death"],
           yticklabels=["Survival", "Death"], xlabel="Predicted", ylabel="Observed",
           title="Held-out test confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def synthetic_data(seed=420, patients=600):
    """Artificial data for software checks only; never research evidence."""
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(patients), 2)
    n = len(ids)
    age = rng.uniform(18, 90, n)
    risk = 1 / (1 + np.exp(-(-4 + age / 30 + rng.normal(size=n))))
    df = pd.DataFrame(dict(encounter_id=np.arange(n), patient_id=ids,
        hospital_id=rng.integers(0, 8, n), hospital_death=rng.binomial(1, risk),
        age=age, height=rng.normal(170, 10, n), weight=rng.normal(75, 15, n),
        apache_4a_hospital_death_prob=risk, apache_4a_icu_death_prob=risk * 0.8,
        gender=rng.choice(["M", "F", None], n), ethnicity=rng.choice(["A", "B", None], n),
        hospital_admit_source=rng.choice(["Other ICU", "ICU", "Floor"], n),
        icu_id=rng.integers(0, 10, n), urineoutput_apache=rng.uniform(0, 100, n),
        icu_type="MICU", readmission_status=0, all_missing=np.nan))
    df.loc[rng.random(n) < 0.1, "age"] = np.nan
    return df


def run(args):
    source_path = Path(args.data)
    if not args.smoke_test and not source_path.is_file():
        raise SystemExit(f"Data not found: {source_path}\nPlace the official training_v2.csv "
                         "under data/raw/ after completing the official access process. "
                         "No real-data training has been performed.")
    if args.smoke_test:
        out = Path("results/smoke")
        df = synthetic_data(args.seed)
        rounds = min(args.rounds, 30)
        data_hash = None
    else:
        default_output = ("results/baseline" if args.apache_policy == "as-recorded"
                          else "results/sensitivity/" + args.apache_policy)
        out = Path(args.output or default_output)
        df = pd.read_csv(source_path)
        data_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        rounds = args.rounds
    validate_data(df)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'SYNTHETIC SOFTWARE CHECK' if args.smoke_test else 'REAL DATA'}: "
          f"{len(df)} encounters; {df.patient_id.nunique()} patients", flush=True)
    train, valid, test = patient_split(df, args.seed)
    print("Train/validation/test sizes:", *(len(p) for p in (train, valid, test)), flush=True)
    prep = Preprocessor(args.apache_policy).fit(train, args.seed)
    x_train, x_valid = prep.transform(train), prep.transform(valid)
    params = dict(boosting="gbdt", objective="binary", metric="auc", num_leaves=15,
        min_data_in_leaf=90, learning_rate=0.01, max_depth=5, scale_pos_weight=1,
        feature_fraction=0.1, bagging_freq=1, bagging_fraction=0.75, use_missing=True,
        num_threads=4, seed=args.seed, deterministic=True, force_col_wise=True, verbosity=-1)
    model = lgb.train(params, lgb.Dataset(x_train, label=train.hospital_death),
        num_boost_round=rounds,
        valid_sets=[lgb.Dataset(x_valid, label=valid.hospital_death)],
        callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(200)])
    # Test transformed and evaluated only after early stopping is complete.
    probability = model.predict(prep.transform(test), num_iteration=model.best_iteration)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid probability output")
    result = metrics(test.hospital_death, probability, args.threshold)
    write_json(out / "overall_metrics.json", result)
    subgroup_table(test, probability, args.threshold).to_csv(out / "subgroup_metrics.csv", index=False)
    split_summary = [{"partition": name, "encounters": len(p),
        "patients": int(p.patient_id.nunique()), "deaths": int(p.hospital_death.sum()),
        "death_rate": float(p.hospital_death.mean())}
        for name, p in zip(["train", "validation", "test"], [train, valid, test])]
    pd.DataFrame(split_summary).to_csv(out / "split_summary.csv", index=False)
    profile = []
    for attr, values in demographic_groups(df).items():
        for group in sorted(values.unique()):
            subset = df.loc[values == group]
            profile.append(dict(attribute=attr, group=str(group), n=len(subset),
                                deaths=int(subset.hospital_death.sum())))
    pd.DataFrame(profile).to_csv(out / "cohort_demographics.csv", index=False)
    pd.DataFrame({"feature": df.columns, "missing_count": df.isna().sum().to_numpy(),
                  "missing_fraction": df.isna().mean().to_numpy()}).to_csv(out / "missingness.csv", index=False)
    pd.DataFrame({"feature": prep.features, "gain": model.feature_importance("gain")}).sort_values(
        "gain", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    manifest = dict(run_time_utc=datetime.now(timezone.utc).isoformat(), source=SOURCE,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mode="synthetic_software_check" if args.smoke_test else "real_data_audit_adaptation",
        input_sha256=data_hash, seed=args.seed, threshold=args.threshold,
        apache_policy=args.apache_policy,
        rounds_limit=rounds, best_iteration=model.best_iteration, parameters=params,
        patient_overlap=0, split_method="patient-level any-death stratification; 64/16/20 patients",
        feature_count=len(prep.features), features=prep.features,
        dropped_all_missing=prep.empty, dropped_perfect_correlation=prep.correlated,
        imputation_columns=prep.impute_cols, preprocessing_fit_partition="train_only",
        python=platform.python_version(), versions={name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "lightgbm", "matplotlib"]},
        limitations=["Single-model adaptation, not the original four-fold competition ensemble.",
            "Original category/imputation behavior changed to prevent preprocessing leakage.",
            "APACHE policy is recorded; sensitivity runs do not establish a preferred model.",
            "No confidence intervals, hospital-disjoint results or causal fairness conclusions yet."])
    write_json(out / "run_manifest.json", manifest)
    plot_confusion(result, out / "confusion_matrix.png")
    if args.smoke_test:
        (out / "SYNTHETIC_ONLY.txt").write_text("Artificial data. Software validation only. Not research results.\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    print(f"Outputs: {out}; no encounter-level predictions or data exported", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/training_v2.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--rounds", type=int, default=7000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--apache-policy", default="as-recorded",
                        choices=["as-recorded", "negative-to-missing", "negative-to-missing-bounded", "drop-probabilities"])
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.rounds < 1:
        parser.error("threshold must be between 0 and 1; rounds must be positive")
    run(args)


if __name__ == "__main__":
    main()
