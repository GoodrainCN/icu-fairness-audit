"""Compare subgroup ranking and errors at a shared, finite review capacity."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.baseline import demographic_groups, patient_split, validate_data
from src.outcomes import POLICIES, SPLIT_SEED, calibration_table

OUT = Path("results/review_budget")
BUDGETS = (0.05, 0.10, 0.20)
PAIRS = {"age": ("80_plus", "under_45"), "gender": ("F", "M"),
         "ethnicity": ("African American", "Caucasian")}
REPS = 4000


def ranked(data):
    return data.sort_values(["score", "encounter_id"], ascending=[False, True]).reset_index(drop=True)


def measures(data, flags):
    y = data.hospital_death.to_numpy()
    deaths = int(y.sum())
    tp = int(flags[y == 1].sum())
    fp = int(flags[y == 0].sum())
    return dict(n=len(data), deaths=deaths, reviewed=tp+fp, missed=deaths-tp, fp=fp,
                auc=roc_auc_score(y, data.score) if 0 < deaths < len(y) else np.nan,
                fnr=1-tp/deaths if deaths else np.nan,
                fpr=fp/(len(y)-deaths) if deaths < len(y) else np.nan,
                brier=np.mean((data.score-y)**2))


def evaluate(data, model, rows, gaps, sites):
    groups = demographic_groups(data).assign(overall="all")
    groups["age_gender"] = groups.age + "/" + groups.gender
    for budget in BUDGETS:
        flags = np.arange(len(data)) < int(np.floor(len(data)*budget))
        for attr, values in groups.items():
            local = {}
            for group, ids in values.groupby(values).indices.items():
                obs = measures(data.iloc[ids], flags[ids])
                local[group] = obs
                if attr == "age_gender" and min(obs["deaths"], obs["n"]-obs["deaths"]) < 20:
                    continue
                rows.append(dict(model=model, budget=budget, attribute=attr, group=group, **obs))
            if attr in PAIRS:
                a, b = PAIRS[attr]
                gaps.append(dict(model=model, budget=budget, attribute=attr, group=a, reference=b,
                                 **{k: local[a][k]-local[b][k] for k in ("auc", "fnr", "fpr")}))
        if model == "hospital" and budget == 0.10:
            for site, ids in data.groupby("hospital_id").indices.items():
                sites.append(dict(hospital_id=site, **measures(data.iloc[ids], flags[ids])))


def cluster_weights(data):
    site = pd.factorize(data.hospital_id)[0]
    n = site.max()+1
    rng = np.random.default_rng(420)
    for _ in range(REPS):
        yield rng.multinomial(n, np.full(n, 1/n))[site]


def auc_kernel(data):
    # Rows are in descending score order. Count concordant pairs and half of tied pairs.
    scores = data.score.to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(scores))+1]
    y = data.hospital_death.to_numpy()

    def weighted(weights):
        pos = np.add.reduceat(weights*y, starts)
        neg = np.add.reduceat(weights*(1-y), starts)
        denom = pos.sum()*neg.sum()
        if not denom:
            return np.nan
        return np.dot(neg, np.cumsum(pos)-0.5*pos)/denom
    return weighted


def intervals(data, model):
    groups = demographic_groups(data)
    kernels = []
    for attr, pair in PAIRS.items():
        for group in pair:
            idx = np.flatnonzero(groups[attr].eq(group).to_numpy(dtype=bool))
            kernels.append((idx, data.hospital_death.to_numpy()[idx], auc_kernel(data.iloc[idx])))
    def contrasts(w):
        # Reallocate exactly floor(10% of the resampled cohort), including duplicate rows.
        slots = w.sum()//10
        selected = np.minimum(w, np.maximum(0, slots - (np.cumsum(w)-w)))
        stats = []
        for idx, y, auc in kernels:
            weights, flags = w[idx], selected[idx]
            pos, neg = weights@y, weights@(1-y)
            stats.append([auc(weights), 1-flags@y/pos if pos else np.nan,
                          flags@(1-y)/neg if neg else np.nan])
        s = np.array(stats)
        return s[::2]-s[1::2]
    point = contrasts(np.ones(len(data), dtype=int))
    reps = np.array([contrasts(w) for w in cluster_weights(data)])
    rows = []
    for i, attr in enumerate(PAIRS):
        for j, metric in enumerate(("auc", "fnr", "fpr")):
            samples = reps[:, i, j]
            samples = samples[np.isfinite(samples)]
            lo, hi, adjlo, adjhi = np.quantile(samples, [.025, .975, .025/9, 1-.025/9])
            rows.append(dict(model=model, attribute=attr, metric=metric, gap=point[i,j],
                             lo=lo, hi=hi, family_lo=adjlo, family_hi=adjhi, valid_reps=len(samples)))
    return rows


def calibration(data):
    groups = demographic_groups(data)
    summary = []
    for attr, pair in PAIRS.items():
        for group in pair:
            sub = data.loc[groups[attr].eq(group)]
            p = sub.score.clip(1e-10, 1-1e-10)
            logit = np.log(p/(1-p)).to_numpy().reshape(-1,1)
            fit = LogisticRegression(penalty=None, tol=1e-10).fit(logit, sub.hospital_death)
            summary.append(dict(attribute=attr, group=group, intercept=fit.intercept_[0],
                                slope=fit.coef_[0,0], brier=np.mean((p-sub.hospital_death)**2)))
    pd.DataFrame(summary).to_csv(OUT/"calibration_summary.csv", index=False)
    bins = calibration_table(data, data.score.to_numpy())
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for col, (attr, pair) in enumerate(PAIRS.items()):
        for ax, group in zip(axes[:, col], pair):
            b = bins[bins.attribute.eq(attr) & bins.group.eq(group)]
            ax.plot([0,1], [0,1], color="gray", linestyle=":")
            ax.plot(b.predicted, b.observed, "o-", color="#216075")
            # Label every bin; alternate offsets to separate the low-risk bins.
            for i, row in enumerate(b.itertuples()):
                ax.annotate(f"{row.n:,}", (row.predicted, row.observed),
                            xytext=(4, -12 if i % 2 else 7),
                            textcoords="offset points", fontsize=8)
            ax.set(xlim=(-.02,.93), ylim=(-.04,.93),
                   title=f"{attr.title()} · {group.replace('_',' ')}")
    for ax in axes[-1]:
        ax.set_xlabel("Mean predicted risk")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed in-hospital mortality")
    fig.suptitle("Calibration by risk bin  |  Labels give encounter counts")
    fig.tight_layout()
    fig.savefig(OUT/"calibration.png", dpi=160)
    plt.close(fig)
    bins.to_csv(OUT/"calibration_bins.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv("data/raw/training_v2.csv")
    validate_data(raw)
    assert raw.patient_id.is_unique
    train, valid, test = patient_split(raw, SPLIT_SEED)
    split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SPLIT_SEED)
    dev, hold = next(split.split(raw, groups=raw.hospital_id))
    assert set(raw.patient_id.iloc[dev]).isdisjoint(raw.patient_id.iloc[hold])
    assert set(raw.hospital_id.iloc[dev]).isdisjoint(raw.hospital_id.iloc[hold])
    rows, gaps, cis, sites = [], [], [], []
    for model in (*POLICIES, "hospital"):
        cohort = raw.iloc[hold] if model == "hospital" else test
        scores = pd.read_csv(f"data/processed/scores-{model}.csv")
        assert set(cohort.encounter_id) == set(scores.encounter_id)
        data = ranked(cohort.merge(scores, on="encounter_id", validate="one_to_one"))
        archived = Path("results/hospital_validation" if model == "hospital" else f"results/outcomes/{model}")
        auc = pd.read_csv(archived/"overall_thresholds.csv").roc_auc.iloc[0]
        assert abs(roc_auc_score(data.hospital_death, data.score)-auc) < 1e-10
        evaluate(data, model, rows, gaps, sites)
        if model in ("as-recorded","hospital"):
            cis.extend(intervals(data, model))
        if model == "as-recorded":
            calibration(data)
            valid_score = data.apache_4a_hospital_death_prob.between(0,1)
            shared = data[valid_score].copy()
            evaluate(shared, "ensemble-common", rows, gaps, sites)
            apache = ranked(shared.assign(score=shared.apache_4a_hospital_death_prob))
            evaluate(apache, "apache-common", rows, gaps, sites)
            ens_auc, ap_auc = auc_kernel(shared), auc_kernel(apache)
            order = shared.encounter_id.reset_index(drop=True)
            reindex = pd.Index(order).get_indexer(apache.encounter_id)
            diffs = [ens_auc(w)-ap_auc(w[reindex]) for w in cluster_weights(shared)]
            lo, hi = np.quantile(diffs,[.025,.975])
            pd.DataFrame([dict(n=len(shared), excluded=len(test)-len(shared),
                               deaths=int(shared.hospital_death.sum()),
                               auc_difference=roc_auc_score(shared.hospital_death, shared.score)
                               -roc_auc_score(apache.hospital_death, apache.score),
                               lo=lo, hi=hi)]).to_csv(OUT/"baseline_comparison.csv", index=False)
    for name, table in [("groups",rows), ("gaps",gaps), ("intervals",cis), ("sites",sites)]:
        pd.DataFrame(table).to_csv(OUT/f"{name}.csv",index=False)
    print("Encounters, unique patients, hospitals",len(raw),raw.patient_id.nunique(),raw.hospital_id.nunique())


if __name__ == "__main__":
    main()
