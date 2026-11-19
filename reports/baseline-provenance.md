# Baseline provenance and performance

## Report text

The system under audit predicts death during the hospitalization using information from the first day of an ICU stay. The reference implementation is saminens’ WiDS Datathon 2020 notebook at revision `a19d3ba`. Its final version averages predictions from four LightGBM models trained with stratified cross-validation. Its archived final-model fold AUCs are 0.91123, 0.90069, 0.90559, and 0.90561, averaging approximately 0.90578 from the displayed scores. The notebook’s printed mean of 0.90659 reads the earlier model’s accumulator. These are historical validation results from the source notebook. [Original implementation](https://github.com/saminens/Women-in-Data-Science-2020/blob/a19d3ba/WomenInDS.ipynb)

Our audit baseline uses one LightGBM model trained on the labeled WiDS data, with separate patient partitions for fitting (58,696 encounters), early stopping (14,674), and final evaluation (18,343). We fit preprocessing on the fitting partition, retain the source model’s main hyperparameters, and preserve the two APACHE mortality probability fields as recorded for this baseline. The evaluation threshold of 0.5 is an audit setting. The test set contains 1,583 deaths (8.63%). [Data](https://www.kaggle.com/competitions/widsdatathon2020/data) · [Implementation notes](../docs/reproduction-notes.md) · [Run manifest](../results/baseline/run_manifest.json)

| Parameter or choice | Value | Provenance |
|---|---|---|
| Algorithm and learning objective | LightGBM GBDT, binary objective, validation ROC-AUC | Source notebook, cell index 32, reused by final model in cell index 54 |
| Tree size and learning rate | `num_leaves=15`, `min_data_in_leaf=90`, `max_depth=5`, `learning_rate=0.01` | Same source configuration |
| Sampling and class weight | `feature_fraction=0.10`, `bagging_fraction=0.75`, `bagging_freq=1`, `scale_pos_weight=1` | Same source configuration |
| Missing-value handling and CPU threads | `use_missing=true`, four threads | Same source values; audit uses `num_threads` in place of `nthread` |
| Boosting limit and stopping rule | Up to 7,000 rounds, 100 rounds of patience | Source notebook; audit applies the rule on its own validation partition |
| Partition and reproducibility settings | Patient-level 64/16/20 split, outer seed 420, inner seed 421, model and imputer seed 420, deterministic column-wise CPU training, `verbosity=-1` | Audit implementation choices. The source also uses 420 for its own four-fold split |
| Fitted number of boosting rounds | 3,103 | Selected by validation AUC in the recorded local run |
| Positive prediction cutoff | Score ≥ 0.5 | Audit setting fixed for the reported evaluation; a clinical cutoff would require its own justification |

The source notebook supplies the inherited values. Its archived code establishes the configuration used, while the search or selection history behind those values remains unverified.

| Held-out metric | Local audit baseline |
|---|---:|
| ROC-AUC | 0.9056 |
| Recall at 0.5 | 36.07% (571/1,583) |
| Precision at 0.5 | 73.58% (571/776) |
| False-negative rate at 0.5 | 63.93% (1,012/1,583) |
| False-positive rate at 0.5 | 1.22% (205/16,760) |
| Accuracy at 0.5 | 93.37% |

The test confusion counts are 571 true positives, 1,012 false negatives, 205 false positives, and 16,555 true negatives. An always-survive rule on these same encounters would achieve 91.37% accuracy and 0% recall. This arithmetic reference shows how class imbalance affects accuracy. At the tested cutoff the LightGBM model detected about 36% of observed deaths despite a ROC-AUC of about 0.906, which summarizes ranking across thresholds. [Recorded test metrics](../results/baseline/overall_metrics.json)

Use this local run as the reference for the APACHE sensitivity variants evaluated on the same holdout. Identify the threshold and group denominator with each subgroup error rate, and pair disparity estimates with their uncertainty where available. The source four-fold AUC and the local held-out AUC describe different evaluation protocols. A direct claim of improvement over the source would require a matched evaluation.

## 中文讲解

- **先说原模型是谁。** 原作者用四个 LightGBM 模型做集成。原 notebook 存着四个验证折的表现，最后一版按四折输出重新算的 AUC 约为 0.90578。它自己打印的 0.90659 取错了变量。
- **再讲参数哪里来的。** 叶子数、树深、学习率、抽样比例和早停规则继承自原代码。我们定了新的切分方式、可复现设置和 0.5 分类阈值。3,103 轮由本地验证集选出。这三类数值的来源分别是原代码、审计设计和验证数据。
- **先交代总体性能，再讲公平性。** 同一测试集有 18,343 次就诊，其中 1,583 次住院死亡。模型在 0.5 阈值识别到 571 例，漏掉 1,012 例。AUC 衡量模型把死亡病例排在更高分位置的能力，召回率回答选定阈值实际识别出了多少。后面的分组漏报率和 APACHE 对照都要从这组数字讲起。
