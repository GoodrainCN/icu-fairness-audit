"""Describe age-group composition and standardize survivor flag rates one factor at a time."""
import numpy as np
import pandas as pd

from src.baseline import demographic_groups, patient_split
from src.outcomes import SPLIT_SEED
from src.review_budget import OUT, ranked

AGE_GROUPS = ["80_plus", "under_45"]


def standardize(survivors, factor):
    counts = survivors.groupby([factor, "age_group"], observed=True).flag.agg(["size", "mean"])
    n = counts["size"].unstack().reindex(columns=AGE_GROUPS).dropna()
    rates = counts["mean"].unstack().loc[n.index, AGE_GROUPS]
    pooled = n.sum(axis=1) / n.sum().sum()
    standardized = rates.mul(pooled, axis=0).sum()
    crude = (rates*n).sum()/n.sum()
    return dict(factor=factor, shared_strata=len(n), older_n=int(n.iloc[:,0].sum()),
                younger_n=int(n.iloc[:,1].sum()), crude_gap=crude.iloc[0]-crude.iloc[1],
                older_standardized=standardized.iloc[0], younger_standardized=standardized.iloc[1],
                standardized_gap=standardized.iloc[0]-standardized.iloc[1])


def main():
    raw = pd.read_csv("data/raw/training_v2.csv")
    _, _, test = patient_split(raw, SPLIT_SEED)
    scores = pd.read_csv("data/processed/scores-as-recorded.csv")
    assert set(scores.encounter_id) == set(test.encounter_id)
    data = ranked(test.merge(scores, on="encounter_id", validate="one_to_one"))
    data["flag"] = np.arange(len(data)) < len(data)//10
    data["age_group"] = demographic_groups(data).age
    data = data[data.age_group.isin(AGE_GROUPS)].copy()
    fields = [c for c in raw if c.startswith("d1_")]
    data["missing_fraction"] = data[fields].isna().mean(axis=1)
    p = data.apache_4a_hospital_death_prob
    data["risk"] = p.where(p.between(0,1))
    data["risk_band"] = pd.cut(data.risk, [0,.05,.10,.20,.50,1], include_lowest=True
                               ).astype("string").fillna("Unavailable")
    rows = []
    distributions = []
    for outcome, sub in [("all", data), ("survivors", data[data.hospital_death.eq(0)])]:
        shares = pd.crosstab(sub.hospital_id, sub.age_group, normalize="columns")
        distributions.append(shares.assign(outcome=outcome).reset_index())
        distance = (shares[AGE_GROUPS[0]]-shares[AGE_GROUPS[1]]).abs().sum()/2
        for group, g in sub.groupby("age_group"):
            rows.append(dict(outcome=outcome, group=group, n=len(g), deaths=g.hospital_death.sum(),
                             risk_median=g.risk.median(), risk_q1=g.risk.quantile(.25),
                             risk_q3=g.risk.quantile(.75), risk_unavailable=g.risk.isna().mean(),
                             measurement_fields=len(fields), mean_missing=g.missing_fraction.mean(),
                             hospitals=g.hospital_id.nunique(), hospital_total_variation=distance))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT/"age_case_mix.csv", index=False)
    pd.concat(distributions).to_csv(OUT/"age_hospital_shares.csv", index=False)
    survivors = data[data.hospital_death.eq(0)]
    bands = survivors.groupby(["risk_band", "age_group"]).flag.agg(n="size", flagged="sum", fpr="mean")
    bands.to_csv(OUT/"age_survivor_risk_bands.csv")
    pd.DataFrame([standardize(survivors, f) for f in ["risk_band","hospital_id"]]).to_csv(
        OUT/"age_standardization.csv", index=False)


if __name__ == "__main__":
    main()
