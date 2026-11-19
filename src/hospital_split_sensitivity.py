"""Repeat the whole-hospital split while holding the ensemble and review policy fixed."""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.baseline import patient_split, validate_data
from src.outcomes import fit_ensemble, SPLIT_SEED
from src.review_budget import OUT, evaluate, ranked


def main():
    raw = pd.read_csv("data/raw/training_v2.csv")
    validate_data(raw)
    _, _, primary = patient_split(raw, SPLIT_SEED)
    assert raw.patient_id.is_unique and raw.hospital_id.notna().all()
    rows = []
    for seed in (420, 421, 422):
        split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
        train, hold = next(split.split(raw, groups=raw.hospital_id))
        development, test = raw.iloc[train], raw.iloc[hold]
        assert set(test.hospital_id).isdisjoint(development.hospital_id)
        if seed == SPLIT_SEED:
            scores = pd.read_csv("data/processed/scores-hospital.csv")
            assert set(scores.encounter_id) == set(test.encounter_id)
            scored = test.merge(scores, on="encounter_id", validate="one_to_one")
        else:
            scores, _ = fit_ensemble(development, test.drop(columns="hospital_death"), "as-recorded")
            scored = test.assign(score=scores)
        gaps = []
        evaluate(ranked(scored), "repeat", [], gaps, [])
        for g in gaps:
            if g["budget"] == 0.1:
                rows.append(dict(seed=seed, n=len(test), deaths=int(test.hospital_death.sum()),
                                 hospitals=test.hospital_id.nunique(),
                                 primary_overlap=test.encounter_id.isin(primary.encounter_id).sum(), **g))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT/"hospital_split_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
