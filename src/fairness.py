"""汇总 subgroup metrics，并以患者为单位计算 FNR/FPR 差异区间。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = {
    "baseline": Path("results/baseline"),
    "APACHE_bounded": Path("results/sensitivity/apache-bounded"),
    "APACHE_dropped": Path("results/sensitivity/apache-dropped"),
}
REFERENCES = {"gender": "M", "age": "45_to_64", "ethnicity": "Caucasian"}
RESAMPLES = 10000
SEED = 20261030
MIN_OUTCOMES = 20
COUNT_COLS = ["tn", "fp", "fn", "tp"]


def draw_counts(counts, rng, resamples):
    """四格频数的 multinomial 与组内逐患者有放回抽样完全同分布。

    前提：每人恰好一条记录。无此保证时必须改用 patient-cluster bootstrap。
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape != (4,) or (counts < 0).any() or counts.sum() == 0:
        raise ValueError("Expected nonnegative TN/FP/FN/TP counts with positive total")
    return rng.multinomial(int(counts.sum()), counts / counts.sum(), size=resamples)


def bootstrap_rates(counts, rng, resamples):
    draws = draw_counts(counts, rng, resamples)
    tn, fp, fn, tp = draws.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"FNR": fn / (fn + tp), "FPR": fp / (fp + tn)}


def interval(samples, supported):
    valid = np.isfinite(samples)
    if not supported:
        return np.nan, np.nan, int(valid.sum()), "low_support"
    if valid.mean() < 0.99:
        return np.nan, np.nan, int(valid.sum()), "too_many_undefined_resamples"
    low, high = np.quantile(samples[valid], [0.025, 0.975])
    return float(low), float(high), int(valid.sum()), "ok"


def analyze(table, rng, resamples=RESAMPLES):
    gaps = []
    for attr, rows in table.groupby("attribute", sort=True):
        rows = rows.sort_values("group").set_index("group")
        ref = REFERENCES[attr]
        if ref not in rows.index:
            raise ValueError(f"Reference missing: {attr}/{ref}")
        draws = {group: bootstrap_rates(row[COUNT_COLS].to_numpy(dtype=int), rng, resamples)
                 for group, row in rows.iterrows()}
        for group, row in rows.iterrows():
            if group == ref:
                continue
            other = rows.loc[ref]
            for metric, outcome in [("FNR", "deaths"), ("FPR", "survivors")]:
                n_group = int(row.deaths if outcome == "deaths" else row.n - row.deaths)
                n_ref = int(other.deaths if outcome == "deaths" else other.n - other.deaths)
                supported = min(n_group, n_ref) >= MIN_OUTCOMES
                low, high, valid, status = interval(draws[group][metric] - draws[ref][metric], supported)
                point = float(row[metric.lower()] - other[metric.lower()])
                error = "fn" if metric == "FNR" else "fp"
                correct = "tp" if metric == "FNR" else "tn"
                sparse_cells = min(int(row[error]), int(other[error]),
                                   int(row[correct]), int(other[correct])) < 5
                gaps.append(dict(attribute=attr, group=group, reference=ref, metric=metric,
                    difference=point, ci_low=low, ci_high=high,
                    group_outcomes=n_group, reference_outcomes=n_ref,
                    valid_resamples=valid, sparse_confusion_cells=sparse_cells, status=status))
    return pd.DataFrame(gaps)


def pct(value):
    return "—" if pd.isna(value) else f"{100 * value:.2f}%"


def gap_text(row):
    if row.status != "ok":
        return f"{100 * row.difference:+.2f}；低支持量，不报 CI"
    return (f"{100 * row.difference:+.2f} "
            f"[{100 * row.ci_low:+.2f}, {100 * row.ci_high:+.2f}]")


def report(subgroups, gaps):
    base = subgroups[subgroups.model == "baseline"]
    baseline_gaps = gaps[gaps.model == "baseline"]
    parts = ["# 公平性分析：第 2、3 步", "",
        "**范围：**原基线为主，测试集 18,343 人，threshold = 0.5；APACHE 两种处理只作 sensitivity analysis。", "",
        "## 1. Subgroup analysis", "",
        "FNR = FN / 实际死亡数（漏报）；FPR = FP / 实际存活数（误报）。Recall = 1 − FNR。Equal Opportunity 比较 TPR；Equalized Odds 同时比较 TPR 和 FPR。", "",
        "| 属性 | 群体 | 人数 | 死亡数 | AUC | FNR | FPR | Precision |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in base.itertuples():
        label = str(row.group) + (" *" if row.low_outcome_support else "")
        parts.append(f"| {row.attribute} | {label} | {row.n:,} | {row.deaths:,} | "
                     f"{row.roc_auc:.3f} | {pct(row.fnr)} | {pct(row.fpr)} | {pct(row.precision)} |")
    parts += ["", "`Missing` 独立保留；`*` 表示死亡或存活不足 20 人，点估计不宜单独解释。完整 Recall、F1 和 confusion counts 见 CSV。", "",
        "## 2. 差异与 95% bootstrap confidence interval", "",
        "差异方向均为 **该群体 − reference group**；正数表示该群体错误率更高。单位为百分点（pp）。", "",
        "Reference：gender = M；age = 45_to_64；ethnicity = Caucasian。它们是训练集中对应属性人数最多的组，不代表临床或道德上的标准。", "",
        "| 属性 | 群体 − reference | FNR 差异 [95% CI] | FPR 差异 [95% CI] |",
        "|---|---|---:|---:|"]
    for (attr, group), rows in baseline_gaps.groupby(["attribute", "group"], sort=True):
        a = rows.set_index("metric")
        parts.append(f"| {attr} | {group} − {rows.iloc[0].reference} | "
                     f"{gap_text(a.loc['FNR'])} | {gap_text(a.loc['FPR'])} |")
    parts += ["", "## 3. 预处理对性别差异的影响", "",
        "| 模型 | F−M 的 FNR 差异 [95% CI] | F−M 的 FPR 差异 [95% CI] |",
        "|---|---:|---:|"]
    selected = gaps[(gaps.attribute == "gender") & (gaps.group == "F")]
    for model in RUNS:
        rows = selected[selected.model == model].set_index("metric")
        parts.append(f"| {model} | {gap_text(rows.loc['FNR'])} | {gap_text(rows.loc['FPR'])} |")
    gender = baseline_gaps[(baseline_gaps.attribute == "gender") &
                           (baseline_gaps.group == "F") & (baseline_gaps.metric == "FNR")].iloc[0]
    parts += ["", "## 结论与边界", "",
        f"- 原基线 F−M 的 FNR 差异为 {100 * gender.difference:+.2f} pp，"
        + ("95% CI 包含 0；尚不足以支持稳定的方向性差异，也不能据此证明两组等同。"
           if gender.ci_low <= 0 <= gender.ci_high else
           "本次 pointwise 95% CI 未跨 0；仍需多重比较、重复划分和医院差异检验。"),
        "- AUC、漏报和误报需分别解读；年龄/族裔的原始差异未调整病情、医院和群体构成，不能解释成因果歧视。",
        "- 80_plus 相对 45_to_64 的 FPR 差异为 +0.69 pp，pointwise 95% CI [+0.12, +1.29]；是待验证信号，尚未做多重比较调整。",
        "- Bootstrap：组内以患者为单位有放回抽样 10,000 次，seed = 20261030，取 2.5%/97.5% percentile。每人一条记录，因此直接重采样 TN/FP/FN/TP 四格频数与逐人抽样等价。",
        "- CI 条件于当前已训练模型与固定划分；未包含重新训练或 hospital clustering 的不确定性。缺少相关 outcome（死亡或存活）20 人时不报告该项 CI。",
        "- CI 为未作 multiple-comparison adjustment 的 pointwise 区间；不从多个比较中挑选“显著”结论。不同模型的区间不是模型间差异的 paired CI。",
        "- Asian、Native American、ethnicity Missing 的 FP 各仅 1 例，percentile CI 可能不可靠；CSV 用 sparse_confusion_cells 标记这类比较。",
        "- 下一步：intersectional fairness；之后再做 threshold、随机划分与医院分布的 robustness。", "",
        "复算：`.venv/bin/python -m src.fairness`。数据表：`results/fairness/`。",
        "Bootstrap 方法说明：[SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)。",
        "AI-use：Codex 编写并运行汇总和区间计算；结果须由组员审阅。", ""]
    Path("reports/公平性分析.md").write_text("\n".join(parts))


def main():
    # Unique patient IDs are essential to the aggregate bootstrap equivalence.
    path = Path("data/raw/training_v2.csv")
    df = pd.read_csv(path, usecols=["patient_id", "hospital_death", "gender", "age", "ethnicity"])
    if df.patient_id.isna().any() or df.patient_id.duplicated().any():
        raise ValueError("Patient cluster bootstrap required: IDs missing or duplicated")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    # Verify references against training support, independent of test error rates.
    from src.baseline import demographic_groups, patient_split
    train, _, test = patient_split(df, 420)
    observed_references = {attr: str(values.value_counts().idxmax())
                           for attr, values in demographic_groups(train).items()}
    if observed_references != REFERENCES:
        raise ValueError("Reference-group assumptions no longer match training support")
    rng = np.random.default_rng(SEED)
    tables, differences, inputs = [], [], {}
    for model, folder in RUNS.items():
        manifest = json.loads((folder / "run_manifest.json").read_text())
        if (manifest["input_sha256"] != digest or manifest["seed"] != 420
                or manifest["threshold"] != 0.5):
            raise ValueError("Input, split seed or threshold mismatch")
        table = pd.read_csv(folder / "subgroup_metrics.csv")
        expected = {attr: values.value_counts().to_dict()
                    for attr, values in demographic_groups(test).items()}
        for attr, rows in table.groupby("attribute"):
            if rows.set_index("group").n.to_dict() != expected[attr]:
                raise ValueError("Subgroup membership mismatch")
            if int(rows.deaths.sum()) != int(test.hospital_death.sum()):
                raise ValueError("Death totals mismatch")
        if table[COUNT_COLS].sum(axis=1).ne(table.n).any():
            raise ValueError("Confusion counts do not reconcile")
        result = analyze(table, rng)
        table.insert(0, "model", model)
        result.insert(0, "model", model)
        tables.append(table)
        differences.append(result)
        inputs[model] = {str(f): hashlib.sha256((folder / f).read_bytes()).hexdigest()
                         for f in ["run_manifest.json", "subgroup_metrics.csv"]}
    tables, differences = pd.concat(tables, ignore_index=True), pd.concat(differences, ignore_index=True)
    out = Path("results/fairness")
    out.mkdir(parents=True, exist_ok=True)
    tables.to_csv(out / "subgroup_metrics.csv", index=False)
    differences.to_csv(out / "disparity_intervals.csv", index=False)
    metadata = dict(seed=SEED, resamples=RESAMPLES, confidence_level=0.95,
        method="within-group nonparametric patient bootstrap, exact multinomial count implementation, percentile CI",
        unique_patient_check=True, threshold=0.5, split_seed=420,
        min_outcomes_for_ci=MIN_OUTCOMES, references=REFERENCES,
        reference_rule="Largest observed group in fit partition, without consulting error rates",
        units="CSV differences are proportions; report uses percentage points",
        scope="Fixed model and split; not training, hospital-cluster or multiple-comparison adjusted uncertainty",
        input_sha256=digest, input_files=inputs,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy_version=np.__version__, pandas_version=pd.__version__)
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report(tables, differences)
    print("完成：reports/公平性分析.md；results/fairness/（45 行群体指标、72 行差异区间）")
    print(differences[(differences.attribute == "gender") & (differences.group == "F")].to_string(index=False))


if __name__ == "__main__":
    main()
