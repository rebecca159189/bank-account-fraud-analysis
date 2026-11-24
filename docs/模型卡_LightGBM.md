# 模型卡：LightGBM 开户欺诈评分模型（BAF 学习项目）

> 由 `src/coursework.py` 自动生成，数字全部来自冻结产物，请勿手工改。对应 CA6003 第 5 周的"模型卡""血缘与溯源"。

## 1 基本信息

- 用途：个人学习项目，用于练习欺诈识别建模、阈值选择、解释、公平性与漂移检查。
- 模型：LightGBM 梯度提升树，候选 `LGBM_leaves31`，超参数 {"num_leaves": 31, "reg_lambda": 10.0, "best_iteration": 263, "scale_pos_weight": 96.53269537480064}。
- 随机种子 42；训练月 [0, 1, 2, 3, 4, 5]，验证月 [6]，测试月 [7]。
- 选择规则：maximize validation recall under FPR<=0.05; AP breaks ties。冻结阈值 0.8146（在验证月按 FPR ≤ 5% 确定）。
- **不适用于**：真实的开户审批、任何会影响真实申请人的决定。数据是合成的，模型没有经过任何生产环境验证。

## 2 数据

- BAF Base（Feedzai 发布的合成开户申请数据）：1,000,000 行，欺诈 11,029 笔（1.10%）。
- 原始文件 SHA-256：`7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`。
- 缺失值：5 个字段用 -1、1 个字段用负数表示缺失（官方说明），建模前还原为缺失并加标记列。
- 字段说明与契约核对见 [特征说明_字段字典.md](特征说明_字段字典.md)。

## 3 测试月表现（冻结阈值，月 7）

| 指标 | 数值 |
| --- | --- |
| 召回率 | 53.29%（95% 区间 50.70%–55.87%） |
| 误报率 FPR | 4.38%（验证月目标 ≤ 5%） |
| 精确率 | 15.42% |
| AP | 0.210（随机基线约 0.015） |
| ROC-AUC | 0.893 |
| 告警占比 | 5.10% |
| 相同审核量（5,463 笔）下召回率 | 54.90% |

注意：冻结阈值口径与"相同审核量"口径不要混在一句话里比较。

## 4 公平性（按年龄分组，测试月）

| 组 | 样本数 | 欺诈数 | 误报率 | 召回率 |
| --- | --- | --- | --- | --- |
| age<50 | 84,176 | 1,010 | 3.52% | 46.73% |
| age>=50 | 12,667 | 418 | 10.16% | 69.14% |

相同审核量下，年长组与年轻组的误报率之比（审核量对齐后再比，避免把审核量变化混进来）：

| 口径 | 年长组 FPR | 年轻组 FPR | 比值 |
| --- | --- | --- | --- |
| 主模型 | 11.21% | 3.98% | 2.82 倍 |
| 去掉年龄特征的重训模型 | 9.73% | 4.18% | 2.33 倍 |

去掉年龄特征后差距缩小但没有消失，因为其余特征仍可推断年龄（见教程第 13 章）。

## 5 已知局限

- 数据为合成数据，结论不能直接外推到真实银行。
- 结果主要来自一个测试月；4 个窗口的滚动回测是事后补充的检查，不是预先设定的评估方案。
- AP 只有约 0.2：欺诈本身占比很低，模型是帮助排序审核队列，不是自动拒绝。
- 窗口计数类特征（velocity_*）逐月漂移，携带时间信息，未来月的阈值可能迁移。
- 规则化（决策树蒸馏）会损失效果：相同审核量下规则的召回率低于原模型。

## 6 血缘与溯源

- 代码流水线：`run_analysis.py` 依次运行 eda → train → evaluate → explain → extensions → rolling_backtest → coursework → feature_study → verify → report。
- 冻结产物的 SHA-256（生成本卡时重新计算）：

| 文件 | SHA-256 |
| --- | --- |
| `outputs/models/frozen_manifest.json` | `ed72dd01b8e39922…` |
| `outputs/models/lightgbm.joblib` | `c47e4bb0d025d7ee…` |
| `outputs/models/distilled_tree.joblib` | `240dcaf6faf1a5de…` |
| `outputs/models/preprocessor.joblib` | `64bbec37842955cf…` |
| `outputs/models/schema.json` | `cccd254834f339db…` |
| `outputs/models/logistic_regression.joblib` | `16a4dada8a542d2d…` |
| `outputs/models/mlp.joblib` | `c929f1aef361730b…` |
| `outputs/models/isolation_forest.joblib` | `861c53d882ceb51e…` |
| `outputs/evaluation.json` | `33afdade6d83e74a…` |
| `outputs/explanation.json` | `bf3ca392780c0cbb…` |
| `outputs/predictions/validation.parquet` | `c403c46e9434f7ca…` |
| `outputs/predictions/test.parquet` | `04ba29e5024f3d37…` |

- 运行环境：Windows-11-10.0.26300-SP0。
