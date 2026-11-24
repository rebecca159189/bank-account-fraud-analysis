# BAF 银行开户欺诈分析

[English](README.md)

对 **Bank Account Fraud（BAF）Base 版本的全部 1,000,000 笔申请**进行可复核的数据分析。项目围绕数据检查、时间切分、三个监督模型、特征分析与特征工程、模型转规则、年龄公平性与时间漂移展开。

数据是 Feedzai 以匿名真实银行开户申请为基础生成的**合成数据**，不是实际客户明细，也不是信用卡交易流水。只使用 Base，未混合另外 5 个偏差变体。

## 直接查看结果

- 完整中文报告：[`outputs/BAF_分析报告.html`](outputs/BAF_分析报告.html)。用浏览器打开；图像已经内嵌，复制单个 HTML 也能查看正文、数值与图表。表格可横向滚动，折叠项包含明细。
- 最终指标、置信区间、同审核量比较、分组和逐月结果：[`outputs/evaluation.json`](outputs/evaluation.json)。
- SHAP 排名、全部蒸馏叶子规则、学生与教师的保真度：[`outputs/explanation.json`](outputs/explanation.json)。
- 数据审计摘要：[`outputs/eda_summary.json`](outputs/eda_summary.json)；完整 CSV 表位于 [`outputs/tables/`](outputs/tables/)。
- 测试前冻结的模型配置、阈值、输入哈希和验证集入选结果：[`outputs/models/frozen_manifest.json`](outputs/models/frozen_manifest.json)。
- 事后补充分析（规则复杂度、年龄差异对照、velocity 漂移）：[`outputs/extensions.json`](outputs/extensions.json)，报告第 07 节。
- 特征分析与特征工程（WOE/IV 与 PSI 筛选、三组衍生特征与滚动窗口对照、置换重要性，事后探索，不改冻结结果）：[`outputs/feature_study.json`](outputs/feature_study.json)，报告第 04 节；表 `outputs/tables/fs_*.csv`；教程第 10 章。
- 课程方法补充（偏度/离群值/VIF/随机切分对照/SMOTE/模型卡，事后探索，不改冻结结果）：[`outputs/coursework.json`](outputs/coursework.json)，报告第 09 节；表 `outputs/tables/cw_*.csv`，图 `outputs/figures/cw_*.png`，文档 `docs/模型卡_LightGBM.md`、`docs/特征说明_字段字典.md`；教程见第 2.4、3.5、4.4–4.5、5.4、14.6 节。
- 滚动时间回测：[`outputs/rolling_backtest.json`](outputs/rolling_backtest.json)，报告第 08 节；分窗指标、年龄组结果和配对差值见 `outputs/tables/rolling_*.csv`。
- **从零开始的学习教程**：[`docs/从零学习_BAF欺诈分析.md`](docs/从零学习_BAF欺诈分析.md)。逐个概念讲清本项目每一步为什么这样做、代码在哪里、数字怎么读。

报告中的具体数值来自这些保存的结果文件，不含手填示例数值。报告内的追溯链接需要保留项目目录；上游结果改变后需重新生成报告。

## 分析协议

| 阶段 | 原始 `month` | 报告月份 | 用途 |
| --- | --- | --- | --- |
| 训练 | 0–5 | 第 1–6 月 | 拟合模型及所有转换 |
| 验证 | 6 | 第 7 月 | 选有限超参数、早停、阈值和规则树配置 |
| 测试 | 7 | 第 8 月 | 冻结后评分一次，不再修改方案 |

`month` 是相对索引，不是日历月；从预测输入中排除。主模型明确**包含年龄**。填补、标准化、类别词表及常量筛选只拟合训练数据。

若申请时点在预测时可得，`month` 本身不等于标签泄漏；这里排除它以避免时间索引捷径。随机切分混入未来、全量拟合预处理或窗口特征包含未来事件，才是应防范的泄漏路径。公开数据缺少月份内时间戳和实体 ID，无法逐条审计原始窗口特征是否严格因果。

1. **数据检查**：核对 100 万行、32 列、重复与常量字段，识别字段编码的缺失，查看各月欺诈率和特征分布。`intended_balcon_amount` 的所有负值视为缺失；另外 5 个文档约定字段将 `-1` 转为缺失。保留 `velocity_6h`、`credit_risk_score` 的合法负值，保留缺失指示。
2. **监督基准**：类别加权逻辑回归、LightGBM、类别加权 MLP。主 PR-AUC 定义为 Average Precision（AP）。以验证月 **FPR ≤ 5% 下的召回率**选模型，AP 用于并列判定。
3. **阈值迁移**：阈值仅由验证月选择并冻结，`score >= threshold` 发出告警；同分数整组处理。测试 FPR 为实测值，允许偏离 5%。FPR 的分母是正常申请，和“所有申请中 5% 被告警”不同。
4. **特征分析与特征工程**（`src/feature_study.py`，事后探索）：训练月 WOE/IV 与 PSI，按“IV < 0.02 或 PSI > 0.25”规则删字段重训；速度比率、身份一致性、类别频率/目标编码三组衍生特征（目标编码交叉拟合），月 6 选择、月 7 报告，并在 4 个滚动窗口做配对 bootstrap 对照；置换重要性与 TreeSHAP、IV 排名互相核对。另有一个 Isolation Forest 无监督对照，结果接近随机，只在报告第 03 节用一句话说明。
5. **模型转规则**：SHAP 解释 LightGBM；深度 3 或 4、最多 5 叶的回归树拟合教师的训练月 **log-odds**，不是直接拟合欺诈标签。配置在验证月选定。固定审核预算来自验证月 LightGBM 告警率；各模型在目标月精确选同样的 K 笔，叶子分数并列用 seed=42 的行 ID 哈希抽签，完全不看目标月标签。
6. **公平与漂移**：相同全局阈值下，比较 `age < 50` 与 `age >= 50` 的 FPR、分母、差值、比值和 Wilson 95% 区间；另做 50 岁边界敏感性分析。年龄为公开数据的十年档。第 1–6 月指标是训练内，第 7 月是验证，第 8 月才是最终测试，不能据此声称八次独立时间回测。

7. **事后补充分析**（`src/extensions.py`，在看过测试月之后追加，属于探索性分析）：
   - 规则复杂度：比较叶子数 5→64 的 8 组 CART 候选，以验证月召回达到教师 90% 为目标；本组没有候选达标，按约定选验证召回最高者，结论不推广到全部规则学习方法；
   - 年龄差异：两组基础欺诈率、去掉年龄的对照模型、其余字段推断年龄的能力、（分组阈值只作一句合规说明）；
   - velocity 漂移：`velocity_*` 的月中位数总体下降，携带较强的时间信息，去掉后重训比较。
   本阶段只读取冻结文件，运行前后核对其 SHA-256；同配置重训 LightGBM 先复现冻结模型，再做去特征对比。

8. **滚动时间回测**（看过原测试结果后追加的稳健性检验）：

   | 窗口 | 训练 `month` | 验证 `month` | 测试 `month` |
   | --- | --- | --- | --- |
   | 1 | 0–2 | 3 | 4 |
   | 2 | 0–3 | 4 | 5 |
   | 3 | 0–4 | 5 | 6 |
   | 4 | 0–5 | 6 | 7 |

   每窗重新拟合全部预处理，并从原来的 3 组 LightGBM 候选中按该窗验证结果选择基线，不把最后一窗的胜者配置回填到过去。去掉年龄、去掉短期 velocity、去掉全部 velocity 的 3 个消融模型采用该窗入选基线配置，各自只用验证月早停和选 FPR≤5% 的阈值。同 K 比较统一审核当窗申请数的 **5%**；这与原主实验沿用验证告警率的 **5.64%** 容量不同，不能直接拼接数值。逐窗报告结果和相对基线的差值，不依据这 4 个测试月再挑选新赢家。

   训练区间彼此重叠，月份只有 4 个，且分析方案是在看过原结果后提出；这增加了时间稳健性的证据，但不能充当 4 次全新、独立的确认实验。

主实验的预先约定见 [`protocol.json`](protocol.json)，第 07、08 节属于随后追加的探索。LightGBM 候选是 3 组配对配置 (15,5)、(31,10)、(63,20)，不是 3×3 网格（2026-10-18 已更正描述，代码未改）。本项目阈值协议与论文 / 官方 notebook 并非完全一致，因此属于独立分析，不声称精确复现论文数值。

类别加权模型的分数**未经概率校准**，不能直接解释为实际欺诈概率。SHAP 贡献和蒸馏叶子分数采用教师原始 log-odds 尺度，也不是概率。

**滚动回测的实际结果：** 四窗基线冻结阈值召回为 **45.29%–56.75%**、实际 FPR 为 **3.23%–6.12%**；统一 5% 总审核量后的召回为 **49.86%–53.31%**。去年龄模型在四个月都缩小了年龄组 FPR 差，但同 K 召回前三月点估计下降、第 8 月略升，四个差值区间都包含 0，不能称作跨月份无代价。去掉全部 velocity 的同 K 召回只有一个月上升、三个月下降，没有稳定改善。

原第 8 月约 5.64% 审核量下，去掉全部 velocity 的召回差是 +1.05 个百分点；改为统一 5% 后是 −0.14 个百分点（95% 区间 −1.54 至 +1.19），AP 差仍为 +0.60 个百分点。模型分数一致，变化来自审核容量与排名截止位置；AP 提高不保证每个预算下召回都提高。原单月表格保留，与报告第 08 节、教程第 14.5 节交叉解释。

**回测图的读法：** 左上召回率纵轴固定为 **0–100%**，呈现绝对水平；细小差别请读数值表和右下的百分点差及 95% 区间。右上 FPR、左下年龄组 FPR 差都从 0 起；右下“消融 − 基线”的召回差保留正负、围绕 0 阅读。图中四窗仍是事后稳健性检验，训练区间相互重叠，不能当作四次独立确认。

## 复现运行

以下命令在本目录的 PowerShell 中执行。项目使用现有 `.venv`；原始 CSV 放在 `data/raw/Base.csv`，不会被清洗过程覆盖。

在新机器上先使用 Python 3.12 建立环境并安装记录的依赖：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

一键运行：自动下载缺失数据并跳过已完成阶段，校验结果后生成报告和笔记本。

```powershell
.\.venv\Scripts\python.exe run_analysis.py
```

若需从头重训并覆盖分析输出，用 `run_analysis.py --retrain`；不需要笔记本时可加 `--skip-notebook`。

也可按顺序单独执行：

```powershell
.\.venv\Scripts\python.exe -m src.eda
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m src.evaluate
.\.venv\Scripts\python.exe -m src.explain
.\.venv\Scripts\python.exe -m src.extensions
.\.venv\Scripts\python.exe -m src.rolling_backtest
.\.venv\Scripts\python.exe -m src.coursework
.\.venv\Scripts\python.exe -m src.feature_study
.\.venv\Scripts\python.exe -m src.report --require-complete
.\.venv\Scripts\python.exe -m src.verify
```

`train` 不对测试月预测，完成时写入冻结清单；`evaluate` 消费冻结模型后再评分测试月。训练完成前不要运行测试评价。重跑训练后，应顺序重跑评价、解释、扩展、回测和报告，避免混用不同训练轮次的文件。

只重新生成报告：

```powershell
.\.venv\Scripts\python.exe -m src.report --require-complete
```

需要等上游输出时可用 `--wait-seconds 60`。不带 `--require-complete` 时，缺失章节明确显示“等待结果”，不会编造数值；最终交付请使用完整检查参数。`src.report` 会调用 `src/rolling_plot.py`，读取已保存的回测结果重新绘图后嵌入 HTML；它不会重训、重新选阈值或改动分数和指标。绘图模块与被协议哈希记录的回测实验代码分开，报告阶段使用项目现有的绘图依赖。

训练需要数分钟至更长时间，取决于硬件。主要依赖为 pandas、NumPy、scikit-learn、LightGBM、SciPy、Matplotlib、PyArrow、joblib；SHAP 贡献直接通过 **LightGBM 内置的精确 TreeSHAP** 计算，无需额外安装 `shap` 包。依赖版本见 [`requirements.txt`](requirements.txt)，实际运行版本记录在分析输出中。数据表和指标用全量数据，SHAP 从验证月以 seed=42 无放回抽取 5,000 笔解释样本，不能把解释抽样误认为模型训练抽样。

## 文件结构

```text
bank_fraud_analysis/
├── data/
│   ├── raw/Base.csv                # 原始官方 Base 数据（不随仓库发布，运行 scripts/download_data.py 下载）
│   └── processed/Base.parquet      # 保留原始值的本地读取缓存
├── references/                    # 来源 URL 与哈希（research_notes.json）；论文、datasheet 原件不随仓库发布
├── notebooks/BAF_analysis.ipynb    # 可打开查看的分析笔记本
├── docs/从零学习_BAF欺诈分析.md     # 从零开始的学习教程
├── src/
│   ├── common.py                  # 训练期转换、指标、阈值、同容量排序
│   ├── eda.py                     # 全量数据审计与描述图表
│   ├── train.py                   # 训练、验证、冻结清单
│   ├── evaluate.py                # 冻结模型的最终评价
│   ├── explain.py                 # SHAP、蒸馏规则与保真度
│   ├── extensions.py              # 事后补充分析：规则复杂度、年龄对照、漂移消融
│   ├── rolling_backtest.py        # 4 个扩展训练期窗口的事后稳健性检验
│   ├── rolling_plot.py            # 只读保存结果重绘回测图，不训练或改指标
│   ├── coursework.py              # 课程方法补充（6 部分，按部分缓存，事后探索）
│   ├── coursework_verify.py       # 补充结果的独立核对
│   ├── coursework_report.py       # 报告第 09 节
│   ├── feature_study.py           # 特征分析与特征工程（4 部分，按部分缓存，事后探索）
│   ├── feature_verify.py          # 特征结果的独立核对
│   ├── feature_report.py          # 报告第 04 节
│   ├── report.py                  # 自包含中文 HTML 报告
│   ├── verify.py                  # 已保存结果与协议的一致性检查
│   └── notebook.py                # 生成并执行分析笔记本
├── outputs/
│   ├── BAF_分析报告.html
│   ├── eda_summary.json
│   ├── evaluation.json
│   ├── explanation.json
│   ├── extensions.json            # 补充分析结果
│   ├── rolling_backtest.json      # 4 个窗口的事后时间稳健性检验
│   ├── models/                    # 拟合模型、转换器、schema、冻结清单
│   ├── predictions/               # 验证 / 测试分数，用于复核
│   ├── figures/                   # 独立 PNG 图
│   └── tables/                    # CSV 数值表
├── protocol.json
└── run_analysis.py                # 依次运行分析阶段
```

## 如何使用结论

报告可用于解释模型选择、人工审核容量、规则简化的损失和年龄组误报差异。规则树表示“风险分层 + 边界层抽样审核”；大量叶子并列时，纯规则阈值不一定能恰好产生固定数量的告警。SHAP 是模型解释，不是因果证据；年龄输入删除与否都不能单独保证公平。

主实验只有一次最终时间留出；后续补充 4 个训练期相互重叠的滚动窗口，仍是有限月份、有限候选和固定种子的事后稳健性检验，无法推断长期生产效果。Wilson 区间与固定预测的配对分层 bootstrap 不覆盖重新训练或超参数选择的全部不确定性。合成数据没有实际调查成本、损失金额、客户实体关联或月份内事件顺序，不能据此计算真实节省金额，亦不能重建实时监控窗口。

## 数据来源与引用

- [官方 Kaggle BAF 数据集](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
- [Feedzai 官方仓库](https://github.com/feedzai/bank-account-fraud)
- [官方 datasheet](https://raw.githubusercontent.com/feedzai/bank-account-fraud/main/documents/datasheet.pdf)
- [NeurIPS 2022 论文](https://papers.neurips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html)

Sérgio Jesus et al. (2022). *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*. Advances in Neural Information Processing Systems.

```bibtex
@article{jesusTurningTablesBiased2022,
  title={Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation},
  author={Jesus, S{\'e}rgio and Pombal, Jos{\'e} and Alves, Duarte and Cruz, Andr{\'e} and Saleiro, Pedro and Ribeiro, Rita P. and Gama, Jo{\~a}o and Bizarro, Pedro},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}
```

原始资料的 URL、哈希和字段解释见 [`references/research_notes.json`](references/research_notes.json)。原始数据按官方 **CC BY-NC-ND 4.0** 条款保存并保留归属；本项目不重新发布修改后的原始数据。
