# 项目待办

看板：https://github.com/users/GoodrainCN/projects/1

P0：交付关键路径；P1：核心分析。所有年份为 2026。除注明的课程截止日期外，日期均为建议内部目标。负责人暂未分配，全体成员共同参与并交叉审阅。

## 当前分析顺序（2026-10-30 更新）

1. Data validation：APACHE 负值、缺失模式、插补方法与预处理敏感性。
2. Fairness metrics：已完成三个模型的 gender、age、ethnicity 汇总；见 `reports/公平性分析.md`。
3. Statistical uncertainty：已完成 10,000 次组内患者 bootstrap，生成 FNR/FPR 差异的 pointwise 95% CI；小样本按规则不报区间。
4. Intersectional fairness：年龄×记录性别、记录性别×族裔；保留低支持量标记。
5. Robustness：预先规定阈值网格、多随机划分、医院不重叠评估；报告全部结果，不按测试集挑方案。

原基线保留作为对照；第 2、3 步已完成本地计算，未复用 proposal 的旧区间。下一步为交叉群体分析；各步骤先本地验证，再同步合作仓库。

## 1. [P0 · 11/02] 确认完整评分要求与最终截止日期

Issue: https://github.com/GoodrainCN/icu-fairness-audit/issues/1

完成标准：
- [ ] 获取报告结构 1(a) 之后的完整内容及评分 rubric。
- [ ] 在 Brightspace 核实 final submission 是 12/15 还是 12/18，记录出处。
- [ ] 确认 Part 1：11/20 Lab 汇报、11/24 23:59 ET 草稿。
- [ ] 确认课程已采用的 AI-use disclosure 格式。
依赖：无。日期为内部目标；课程截止日期以官方确认为准。

## 2. [P0 · 11/04] 获取数据、数据字典与固定版本源代码

Issue: https://github.com/GoodrainCN/icu-fairness-audit/issues/2

完成标准：
- [ ] 从官方来源取得 WiDS 2020 labeled training data 和数据字典，记录来源、版本、行数与校验值。
- [ ] 获取 saminens/Women-in-Data-Science-2020，核实 proposal 引用 commit a19d3ba。
- [ ] 数据仅保存 data/raw；不上传患者级数据或凭据。
- [ ] 明确原作者代码与本组新增代码。
依赖：任务 1 的相关要求确认。

## 3. [P0 · 11/06] 建立可复现环境与 Colab 运行入口

Issue: https://github.com/GoodrainCN/icu-fairness-audit/issues/3

完成标准：
- [ ] 记录依赖版本、随机种子及运行步骤。
- [ ] 创建 Colab notebook，说明数据如何由使用者提供。
- [ ] 最小训练流程可运行，记录与原方案的兼容性修改。
依赖：数据与源代码获取。

## 4. [P1 · 11/09] 数据画像、群体定义与输入输出说明

Issue: https://github.com/GoodrainCN/icu-fairness-audit/issues/4

完成标准：
- [ ] 核实 hospital_death 含义、预测时点、特征可用时间及潜在泄漏。
- [ ] 检查 patient/hospital ID、重复记录、缺失和类别编码。
- [ ] 生成总体及年龄/记录性别/族裔的样本量、死亡数、死亡率和缺失汇总。
- [ ] 预先确定年龄分箱、交叉群体和低支持量处理规则，缺失人口属性单列。
- [ ] 清楚描述输入特征、输出风险概率和阈值分类。
依赖：数据与字典。

## 5. [P0 · 11/12] 复现患者不重叠的 LightGBM 基线

Issue: https://github.com/GoodrainCN/icu-fairness-audit/issues/5

完成标准：
- [ ] 验证患者跨开发/测试集不重叠；若数据不支持此保证，明确限制。
- [ ] 预处理只在训练数据拟合；内部验证用于 early stopping；测试集保留到评估。
- [ ] 输出 AUC、accuracy、precision、recall、F1 和混淆矩阵。
- [ ] 对照 proposal 的 AUC 0.904 等历史结果，解释差异，不强求一致。
依赖：环境、数据画像。

## 6. [P1 · 11/15] 形成一条有证据的初步公平性发现

Issue: https://github.com/GoodrainCN/icu-fairness-audit/issues/6

完成标准：
- [ ] 输出基本人口群体的支持量、TPR/FNR、FPR、precision 和可计算的 AUC。
- [ ] 对主要群体差异给出适当 bootstrap 区间，并说明重采样单位。
- [ ] 选出一张图/表和一条有边界的 early takeaway。
- [ ] 区分低召回率、统计不确定性和公平性结论；区间包含零时不宣称稳定差异。
依赖：基线与群体定义。

## 7. [P1 · 11/17] 写完整 Background、Input and Output 与后续审计计划

Issue: https://github.com/GoodrainCN/icu-fairness-audit/issues/7

完成标准：
- [ ] Background：ADS 目的、具体实现、利益相关者、潜在影响及选题动机。
- [ ] Input/Output：数据来源、队列、特征、缺失、标签、预处理和输出。
- [ ] 对完整 rubric 的其余章节写具体方法、指标、产物和限制。
- [ ] 加入有 notebook 支撑的初步发现，明确来源和本组贡献。
依赖：完整 rubric、数据画像、初步发现。

## 8. [P0 · 11/20] 完成 Part 1：4–5 页幻灯片与 3 分钟 Lab 汇报

Issue: https://github.com/GoodrainCN/icu-fairness-audit/issues/8

完成标准：
- [ ] 按报告结构覆盖 Background、Input/Output、详细分析计划和一条 early takeaway。
- [ ] 准备 4–5 slides，排练 3 分钟 elevator pitch。
- [ ] 每位组员都能讲解，记录同学和老师反馈。
依赖：Part 1 草稿内容。官方安排：11/20 Lab；内部建议 11/19 完成排练。

## 9. [P0 · 11/24] 提交 Part 1 PDF 与 Colab notebook

Issue: https://github.com/GoodrainCN/icu-fairness-audit/issues/9

完成标准：
- [ ] 纳入 11/20 汇报反馈。
- [ ] 从干净环境运行 notebook，图表与报告数值一致。
- [ ] 补齐代码/数据引用、限制及真实 AI-use disclosure。
- [ ] 检查 PDF 与 Colab 可访问性，组内交叉审阅。
- [ ] 在 Gradescope 每组提交一次并关联全体组员，保存提交回执。
依赖：汇报反馈、草稿、可复现计算。官方截止：11/24 23:59 ET；建议 11/23 完成。

## 10. [P1 · 12/01] 完成交叉群体、校准和阈值敏感性审计

Issue: https://github.com/GoodrainCN/icu-fairness-audit/issues/10

完成标准：
- [ ] 比较年龄×性别、性别×族裔并呈现支持量与区间。
- [ ] 生成群体校准图和 observed-minus-predicted mortality。
- [ ] 分析阈值变化对 TPR/FNR/FPR/precision 与公平性差异的影响；不在测试集选择最优阈值。
- [ ] 解释 Equal Opportunity / Equalized Odds 的情境意义和取舍。
依赖：Part 1 基线；按反馈调整。

## 11. [P1 · 12/05] 验证重复划分稳定性与跨医院泛化

Issue: https://github.com/GoodrainCN/icu-fairness-audit/issues/11

完成标准：
- [ ] 用预先规定的多个随机种子评估差异稳定性。
- [ ] 实现医院不重叠评估并核查患者泄漏；若字段不支持，记录替代方案和限制。
- [ ] 比较常规测试与未见医院测试的总体/群体表现。
- [ ] 报告置信区间、支持量和外推限制，避免将相关性解释为原因。
依赖：基线、群体审计。

## 12. [P0 · 12/12 内部目标] 完成最终报告、notebook 与幻灯片

Issue: https://github.com/GoodrainCN/icu-fairness-audit/issues/12

完成标准：
- [ ] 按完整 rubric 完成各章节，正文最多 10 页，附录可选。
- [ ] 汇总结果、利益相关者影响、性能与公平性取舍、局限和建议。
- [ ] 从干净 Colab 运行，固定版本与种子，核对全部图表。
- [ ] 清楚标明原作者内容、本组新增分析与 AI 使用。
- [ ] PDF 报告、notebook 和 slides 完成组内审阅。
依赖：扩展审计与稳定性验证。为兼容说明中的 12/15 截止，采用 12/12 内部目标；实际提交日待任务 1 确认。

## 13. [P0 · 12/18] 录制 5 分钟视频并完成最终提交与贡献说明

Issue: https://github.com/GoodrainCN/icu-fairness-audit/issues/13

完成标准：
- [ ] 录制不超过 5 分钟的 findings presentation，不加速视频。
- [ ] 按官方确认日期提交最终 PDF、Colab/implementation 与 slides（说明正文写 12/15，须确认）。
- [ ] 12/18 23:59 ET 前按要求提交视频与 slides，核对访问权限。
- [ ] 每位成员单独提交自己和队友的贡献说明。
- [ ] 保存回执、检查全体成员已关联。
依赖：最终材料、截止日期确认。
