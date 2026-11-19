"""Re-run the final WiDS source ensemble (a19d3ba, notebook cell index 54)."""

import argparse
import hashlib
from pathlib import Path
from statistics import mean

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.baseline import IMPUTE, REPLACEMENTS, SOURCE, validate_data, write_json


PARAMETERS = {
    "boosting": "gbdt", "objective": "binary", "metric": "auc",
    "num_leaves": 15, "min_data_in_leaf": 90, "learning_rate": 0.01,
    "max_depth": 5, "scale_pos_weight": 1, "feature_fraction": 0.1,
    "bagging_freq": 1, "bagging_fraction": 0.75, "use_missing": True,
    "num_threads": 4,
}
FINAL_DROPS = ["icu_id", "urineoutput_apache", "icu_type"]


def prepare(labeled, competition_test):
    excluded = ["readmission_status", "hospital_death", "patient_id",
                "hospital_id", "gender"]
    train = labeled.drop(columns=excluded).replace(REPLACEMENTS)
    test = (competition_test.drop(columns=excluded, errors="ignore").replace(REPLACEMENTS)
            if competition_test is not None else train.iloc[:0].copy())
    # The original notebook imputes each partition separately before CV.
    train[IMPUTE] = IterativeImputer().fit_transform(train[IMPUTE])
    if len(test):
        test[IMPUTE] = IterativeImputer().fit_transform(test[IMPUTE])
    for column in train.select_dtypes(include="object"):
        train_values = train[column].astype(str)
        test_values = test[column].astype(str)
        encoder = LabelEncoder().fit(pd.concat([train_values, test_values]))
        train[column] = pd.Categorical(encoder.transform(train_values) + 1)
        test[column] = pd.Categorical(encoder.transform(test_values) + 1)

    # Cell 23 uses numeric correlation; current pandas needs it explicitly.
    correlation = train.drop(columns=["encounter_id", "icu_id"]).corr(numeric_only=True)
    correlated = [
        column for index, column in enumerate(correlation.columns)
        if correlation.iloc[index, :index].abs().eq(1).any()
    ]
    drops = ["encounter_id", *dict.fromkeys([*correlated, *FINAL_DROPS])]
    return train.drop(columns=drops), test.drop(columns=drops), correlated


def run(labeled_path, test_path, output):
    labeled = pd.read_csv(labeled_path)
    validate_data(labeled)
    competition_test = pd.read_csv(test_path) if test_path else None
    x, x_test, correlated = prepare(labeled, competition_test)
    y = labeled["hospital_death"]
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=420)
    folds = []
    oof = np.full(len(y), np.nan)
    ensemble = np.zeros(len(x_test))
    for number, (fit, valid) in enumerate(splitter.split(x, y), start=1):
        print(f"Fold {number}/4", flush=True)
        train = lgb.Dataset(x.iloc[fit], label=y.iloc[fit], categorical_feature=[])
        validation = lgb.Dataset(x.iloc[valid], label=y.iloc[valid], reference=train,
                                 categorical_feature=[])
        # Cell 52 checks object dtype after encoding converted those fields to
        # category dtype, so the final source call passes an empty index list.
        model = lgb.train(
            PARAMETERS, train, num_boost_round=7000, valid_sets=[validation],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
        )
        oof[valid] = np.clip(model.predict(x.iloc[valid]), 0, 1)
        if len(ensemble):
            ensemble += np.clip(model.predict(x_test), 0, 1) / splitter.n_splits
        fold = dict(fold=number, validation_n=len(valid),
                    best_iteration=model.best_iteration,
                    roc_auc=roc_auc_score(y.iloc[valid], oof[valid]))
        print(fold, flush=True)
        folds.append(fold)

    result = dict(
        source=SOURCE, source_final_cell_index=54,
        input_sha256=hashlib.sha256(labeled_path.read_bytes()).hexdigest(),
        labeled_encounters=len(y), deaths=int(y.sum()),
        competition_test_encounters_scored=len(ensemble),
        features=x.shape[1], correlation_drops=correlated,
        parameters=PARAMETERS, seed=420, rounds_limit=7000, early_stopping_rounds=100,
        folds=folds, mean_fold_auc=mean(fold["roc_auc"] for fold in folds),
        pooled_out_of_fold_auc=roc_auc_score(y, oof),
        versions={"pandas": pd.__version__, "numpy": np.__version__,
                  "sklearn": sklearn.__version__, "lightgbm": lgb.__version__},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(f"Mean fold AUC: {result['mean_fold_auc']:.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/training_v2.csv"))
    parser.add_argument("--competition-test", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("results/source_ensemble/cross_validation.json"))
    args = parser.parse_args()
    run(args.data, args.competition_test, args.output)
