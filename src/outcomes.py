"""Section 4 audit of the WiDS four-model ensemble on a locked holdout."""

import argparse
import hashlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.baseline import (
    Preprocessor, demographic_groups, metrics, patient_split, subgroup_table,
    validate_data, write_json,
)
from src.fairness import MIN_OUTCOMES, REFERENCES, RESAMPLES, SEED, analyze
from src.source_ensemble import PARAMETERS, SOURCE


POLICIES = ("as-recorded", "negative-to-missing-bounded", "drop-probabilities")
THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50)
SPLIT_SEED = 420


def fit_ensemble(development, test, policy):
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SPLIT_SEED)
    scores = np.zeros(len(test))
    folds = []
    for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(development, development.hospital_death), start=1):
        train = development.iloc[train_idx]
        valid = development.iloc[valid_idx]
        prep = Preprocessor(policy).fit(train, SPLIT_SEED)
        x_train, x_valid = prep.transform(train), prep.transform(valid)
        train_set = lgb.Dataset(x_train, label=train.hospital_death, categorical_feature=[])
        valid_set = lgb.Dataset(x_valid, label=valid.hospital_death, reference=train_set,
                                categorical_feature=[])
        model = lgb.train(
            {**PARAMETERS, "verbosity": -1}, train_set, num_boost_round=7000,
            valid_sets=[valid_set], callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        validation_auc = roc_auc_score(valid.hospital_death, model.predict(x_valid))
        scores += model.predict(prep.transform(test)) / splitter.n_splits
        folds.append(dict(fold=fold, train_n=len(train), validation_n=len(valid),
                          validation_auc=validation_auc, best_iteration=model.best_iteration,
                          features=len(prep.features)))
        print(policy, folds[-1], flush=True)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid ensemble probability")
    return scores, folds


def calibration_table(test, scores):
    bins = [0, 0.05, 0.10, 0.20, 0.30, 0.50, 1]
    groups = demographic_groups(test).assign(overall="all")
    rows = []
    for attribute, values in groups.items():
        for group in sorted(values.unique()):
            mask = (values == group).to_numpy(dtype=bool)
            cells = pd.DataFrame({"label": test.hospital_death.to_numpy()[mask],
                                  "score": scores[mask]})
            cells["risk_bin"] = pd.cut(cells.score, bins, include_lowest=True)
            for risk_bin, data in cells.groupby("risk_bin", observed=True):
                rows.append(dict(attribute=attribute, group=group, risk_bin=str(risk_bin),
                                 n=len(data), deaths=int(data.label.sum()),
                                 observed=data.label.mean(), predicted=data.score.mean()))
    return pd.DataFrame(rows)


def run(data_path, output, policy):
    data = pd.read_csv(data_path)
    validate_data(data)
    if not data.patient_id.is_unique:
        raise ValueError("Protocol requires one encounter per patient for inner CV and bootstrap")
    fit, early_stop, test = patient_split(data, SPLIT_SEED)
    # Preserve the earlier audit's exact outer holdout. Inner folds use the
    # remaining rows in their original file order, as in the source notebook.
    development = pd.concat([fit, early_stop]).sort_index()
    assert not set(development.patient_id) & set(test.patient_id)
    scores, folds = fit_ensemble(development, test.drop(columns="hospital_death"), policy)
    overall = [dict(threshold=t, **metrics(test.hospital_death, scores, t)) for t in THRESHOLDS]
    primary = overall[-1]
    primary["brier"] = brier_score_loss(test.hospital_death, scores)
    subgroups = pd.concat(
        [subgroup_table(test, scores, t).assign(threshold=t) for t in THRESHOLDS],
        ignore_index=True,
    )
    gaps = analyze(subgroups[subgroups.threshold.eq(0.50)], np.random.default_rng(SEED))
    output = output / policy
    output.mkdir(parents=True, exist_ok=True)
    cache = Path("data/processed")
    cache.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"encounter_id": test.encounter_id, "score": scores}).to_csv(
        cache / f"scores-{policy}.csv", index=False)
    pd.DataFrame(overall).to_csv(output / "overall_thresholds.csv", index=False)
    subgroups.to_csv(output / "subgroup_thresholds.csv", index=False)
    gaps.to_csv(output / "disparity_intervals.csv", index=False)
    calibration_table(test, scores).to_csv(output / "calibration.csv", index=False)
    write_json(output / "metrics.json", dict(
        source=SOURCE, input_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        holdout_id_sha256=hashlib.sha256(test.patient_id.to_numpy(dtype="<i8").tobytes()).hexdigest(),
        preprocessing="fit within each inner training fold using audit Preprocessor",
        package_versions={"lightgbm": lgb.__version__, "numpy": np.__version__,
                          "pandas": pd.__version__},
        seed=SPLIT_SEED, parameters=PARAMETERS, rounds_limit=7000, early_stopping_rounds=100,
        apache_policy=policy, thresholds=THRESHOLDS, development_n=len(development),
        references=REFERENCES, bootstrap_resamples=RESAMPLES, bootstrap_seed=SEED,
        min_outcomes_for_interval=MIN_OUTCOMES, folds=folds, overall=primary,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/training_v2.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/outcomes"))
    parser.add_argument("--policy", choices=POLICIES, default=POLICIES[0])
    args = parser.parse_args()
    run(args.data, args.output, args.policy)
