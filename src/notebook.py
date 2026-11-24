"""Create and execute a readable notebook from the finalized analysis artifacts."""
from pathlib import Path
import json
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.manager import KernelManager
from .common import ROOT


def project_kernel(spec_dir):
    """A kernel that runs on the interpreter executing this script (the project .venv).

    kernel_name='python3' resolves through the user's Jupyter config and can silently
    pick another environment; a temporary spec pins sys.executable instead.
    """
    spec = Path(spec_dir)/'baf-project'; spec.mkdir()
    (spec/'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'BAF project interpreter', 'language': 'python'}), encoding='utf-8')
    return KernelManager(kernel_name='baf-project', kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(spec_dir)]))

def run():
    nbf=nbformat.v4
    cells=[]
    def md(text): cells.append(nbf.new_markdown_cell(text))
    def code(text): cells.append(nbf.new_code_cell(text))
    md('# 银行开户欺诈分析：BAF Base\n\n全量 1,000,000 条记录。完成数据检查、时间划分、三种监督模型、无监督基线、SHAP 与规则蒸馏、公平性与漂移，以及事后补充分析和四窗口滚动时间回测。\n\n此 Notebook 展示已冻结实验的实际产物；重复运行只读取结果，不再选择模型或阈值。完整重跑：在项目根目录执行 `python run_analysis.py --retrain`。')
    code('''from pathlib import Path
import json
import sys
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd()
if not (ROOT / "outputs").exists():
    ROOT = ROOT.parent
OUT = ROOT / "outputs"
evaluation = json.loads((OUT / "evaluation.json").read_text(encoding="utf-8"))
explanation = json.loads((OUT / "explanation.json").read_text(encoding="utf-8"))
def table(name):
    display(pd.read_csv(OUT / "tables" / name))
def figure(name):
    display(Image(filename=str(OUT / "figures" / name)))
print("执行解释器：", sys.executable)
print("模型选择仅使用 month=6 验证集。监督模型验证胜者：", evaluation["manifest"]["supervised_winner"])
print("最终测试时间：", evaluation["evaluation_time_singapore"])
''')
    md('## 1. 数据检查\n\n这是基于匿名真实银行开户数据生成的合成基准。原始 CSV 没有原生空值，但存在官方定义的缺失哨兵。五列的 -1 和 intended_balcon_amount 的所有负数设为缺失；velocity_6h 和 credit_risk_score 的合法负数保留。数值填补、标准化和类别编码均只在训练期拟合。')
    code('table("eda_schema.csv")\ntable("eda_split_summary.csv")\nfigure("eda_missing_values.png")\nfigure("eda_monthly_volume_fraud.png")')
    md('## 2. 按时间划分\n\n原始 month=0–7 对应第 1–8 月。0–5 训练，6 验证，7 最终测试。month 只负责切分并从预测特征中排除；训练期常量字段也剔除。年龄保留在主模型中，公平性另作审计。\n\n不以随机拆分替代时间拆分；不在最终测试月调参、调阈值或改规则。')
    code('print(json.dumps(evaluation["manifest"]["split"], indent=2))\nprint(evaluation["manifest"]["selection"])\ntable("validation_candidates.csv")')
    md('## 3. 三种监督模型\n\n逻辑回归、LightGBM、MLP 使用类别权重。PR-AUC 主口径为 Average Precision（AP）；梯形积分口径另行导出。每个模型在验证月选取满足 FPR≤5% 的最低可行整分数阈值，冻结后应用于测试月，因此测试 FPR 不保证恰好 5%。\n\nFPR=FP/(FP+TN)，总告警率=(TP+FP)/N，两者含义不同。')
    code('table("validation_metrics.csv")\ntable("test_metrics.csv")\nfigure("benchmark.png")\nfigure("pr_curves.png")')
    md('## 4. 无监督对照\n\nIsolation Forest 使用预先指定的一套参数，仅在训练期特征上拟合，不使用欺诈标签。为与监督模型的业务约束一致，其报警阈值仍使用验证标签校准，所以这是“无标签拟合 + 有标签阈值校准”的基线。')
    code('display(pd.DataFrame(evaluation["test_results"]).query("model == \'isolation_forest\'"))')
    md('## 5. SHAP 与规则蒸馏\n\n对验证月随机抽取 5,000 条记录计算原生精确 TreeSHAP，并验证贡献之和等于模型原始 log-odds。SHAP 描述模型如何使用特征，不表示因果影响。\n\n学生树拟合训练集上的 LightGBM log-odds，深度候选为 3/4，最多五个叶节点。下面五条互斥路径是风险分档，按学生预测风险排序。显示阈值已回到原字段单位，数字经过舍入；精确执行以已保存模型为准。缺失值需先按训练中位数填补。')
    code('table("shap_importance.csv")\nfigure("shap_importance.png")\nfigure("shap_beeswarm.png")\ntable("distilled_rules.csv")\nfigure("distilled_tree.png")')
    md('### 相同告警量\n\n用验证集 LightGBM 告警比例冻结审核预算，在测试月对所有方法审核完全相同的 K 条。排序只使用测试特征分数；边界并列分数使用 seed=42 的行哈希分配，标签只用于事后统计。浅层树整叶报警粒度较粗，满足整分数 FPR≤5% 的冻结阈值可能一条也不报；不能把该退化结果解释为公平性改善。严格同 K 结果展示其风险排序的实际取舍。')
    code('table("same_budget.csv")\nprint(json.dumps(evaluation["rules_comparison"], ensure_ascii=False, indent=2))\nfigure("equal_budget.png")\nprint("学生树对教师的拟合：", explanation["student_fidelity"])')
    md('## 6. 公平性与时间漂移\n\n使用同一个全局冻结阈值，比较 age<50 与 age≥50 的正常申请误报率及 Wilson 95% 区间。年龄已按十岁取整；组间统计差异不单独证明歧视原因。另导出 >50/≤50 边界敏感性结果。\n\n月 0–5 的召回来自训练内预测，有乐观偏差；月 6 为验证，月 7 为单次最终测试。图能显示实际变化，但不能把八个月都称为独立跨期验证。')
    code('table("fairness.csv")\ntable("fairness_gaps.csv")\nfigure("fairness.png")\ntable("monthly_recall.csv")\nfigure("monthly_recall.png")\nfigure("eda_monthly_psi.png")')
    md('## 7. 补充分析（事后探索）\n\n这一节在看过第 8 月结果之后追加，沿用相同纪律：新模型只在月 0–5 拟合，早停、阈值和推荐规则数只看月 6，月 7 只报告；冻结文件的哈希在运行前后一致。\n\n- **A. 规则复杂度**：叶子数 5→64，看整叶规则何时摆脱零告警、多少条规则接近教师。\n- **B. 年龄差异**：两组基础欺诈率、去掉年龄的对照模型、其余字段能否猜出年龄、分组阈值的代价。\n- **C. velocity 漂移**：velocity_* 总体随月份下降，可能携带时间信息；去掉它们后重训对比。')
    code('''extensions = json.loads((OUT / "extensions.json").read_text(encoding="utf-8"))
print("冻结文件未改变：", extensions["frozen_artifacts_unchanged"], "| 同配置重训复现：", extensions["reproduction"])
table("ext_rule_complexity.csv")
figure("ext_rule_complexity.png")
print("推荐叶子数（验证月选择）：", extensions["rules"]["chosen_leaves"])
table("ext_rules_recommended.csv")''')
    code('''table("ext_fairness_context.csv")
figure("ext_fairness_context.png")
print(json.dumps(extensions["fairness"]["age_proxy"], ensure_ascii=False, indent=2))''')
    code('''table("ext_velocity_monthly.csv")
table("ext_feature_ablation.csv")
figure("ext_drift_ablation.png")''')
    md('## 8. 滚动时间回测（事后稳健性检验）\n\n测试第5–8月：例如第1–3月训练、第4月验证、第5月测试，然后依次前移。每个窗口只用自己的过去数据重新拟合类别映射、选择三组LightGBM配置、早停和阈值。三个消融模型沿用本窗基线配置，各自早停。\n\n这里统一审核每月5%的申请，K=floor(0.05*N)，区别于主实验从验证月得到的约5.64%。测试FPR可以偏离验证时的5%约束。全部月份此前已被查看，因此是事后稳健性检验，不是四份新的独立测试；四窗训练集重叠，跨月最小—最大范围也不是置信区间。')
    code('''rolling = json.loads((OUT / "rolling_backtest.json").read_text(encoding="utf-8"))
print("冻结主实验未改变：", rolling["frozen_artifacts_unchanged"])
table("rolling_windows.csv")
display(pd.read_csv(OUT / "tables" / "rolling_metrics.csv").query("split == 'test'"))
figure("rolling_backtest.png")''')
    code('''table("rolling_differences.csv")
table("rolling_fairness.csv")
print(json.dumps(rolling["summary"], ensure_ascii=False, indent=2))
print("区间：固定模型、阈值及告警标记的年龄组×标签分层配对bootstrap，500次；不覆盖重训、阈值估计和未来月份的不确定性。")''')
    md('## 核对与资料\n\n实验仅使用 Base，不使用五个偏差变体。有限候选和单次划分无法给出跨数据集普遍优劣结论；类别加权分数未经概率校准，不能当作真实欺诈概率。阈值需随业务审核量和漂移重新验证。\n\n[Feedzai 官方项目](https://github.com/feedzai/bank-account-fraud) · [官方 Kaggle 数据](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) · [官方 datasheet](https://raw.githubusercontent.com/feedzai/bank-account-fraud/main/documents/datasheet.pdf) · [NeurIPS 2022 论文](https://arxiv.org/abs/2211.13358)')
    code('verification = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))\nprint("独立一致性核对：", verification["passed"], "检查数量：", verification["count"])')
    nb=nbf.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'}})
    target=ROOT/'notebooks/BAF_analysis.ipynb'
    nbformat.write(nb,target)
    with tempfile.TemporaryDirectory(dir=ROOT/'notebooks') as spec_dir:
        client=NotebookClient(nb,timeout=180,km=project_kernel(spec_dir),resources={'metadata':{'path':str(ROOT)}})
        client.execute()
    nbformat.write(nb,target)
    assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]))
    print('Executed notebook:',target)

if __name__=='__main__': run()
