"""Build a self-contained Chinese HTML report from saved analysis outputs.

No models are fitted and no thresholds are selected by this module. Missing
upstream outputs are displayed as pending, never replaced with example numbers.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone, timedelta
from html import escape
import json
import math
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIGURES = OUT / "figures"
REPORT = OUT / "BAF_分析报告.html"
MODEL_NAMES = {
    "logistic_regression": "逻辑回归（类别权重）",
    "lightgbm": "LightGBM",
    "mlp": "MLP（类别权重）",
    "isolation_forest": "Isolation Forest",
    "distilled_tree": "蒸馏规则树",
}
SPLIT_NAMES = {
    "all": "全量", "train": "训练", "train_in_sample": "训练内",
    "validation": "验证", "test": "测试",
}


def read_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError) as error:
        print(f"暂不能读取 {path.name}: {error}", flush=True)
        return {} if default is None else default


def e(value) -> str:
    return escape(str(value), quote=True)


def number(value, digits=0) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
        if not math.isfinite(value):
            return "—"
        return f"{value:,.{digits}f}"
    except (TypeError, ValueError):
        return e(value)


def pct(value, digits=2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.{digits}f}%" if math.isfinite(float(value)) else "—"
    except (TypeError, ValueError):
        return "—"


def pp(value) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:+.2f} 个百分点"


def scientific(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3e}" if math.isfinite(float(value)) else "—"


def model_name(value) -> str:
    return e(MODEL_NAMES.get(value, value or "—"))


def split_name(value) -> str:
    return e(SPLIT_NAMES.get(value, value or "—"))


def interval(row, metric: str) -> str:
    ci = row.get(metric + "_ci")
    if isinstance(ci, dict):
        low, high = ci.get("low"), ci.get("high")
    elif isinstance(ci, (list, tuple)) and len(ci) == 2:
        low, high = ci
    else:
        low, high = row.get(metric + "_ci_low"), row.get(metric + "_ci_high")
    digits = 4 if metric == "fpr" and high is not None and 0 < high < .001 else 2
    return f"{pct(low, digits)} – {pct(high, digits)}" if low is not None and high is not None else "—"


def pending(message="等待上游分析结果；当前不填入估算或示例数值。") -> str:
    return f'<p class="pending">{e(message)}</p>'


def table(headers, rows, caption="", numeric_from=1) -> str:
    if not rows:
        return pending()
    head = "".join(f'<th scope="col">{e(h)}</th>' for h in headers)
    body = "".join(
        "<tr>" + "".join(
            f'<td class="{"num" if i >= numeric_from else "text"}">{cell}</td>'
            for i, cell in enumerate(row)
        ) + "</tr>" for row in rows
    )
    cap = f"<caption>{e(caption)}</caption>" if caption else ""
    return f'<div class="table-wrap" tabindex="0"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(name, caption, alt=None) -> str:
    """Only embed an existing image; never create a broken placeholder image."""
    paths = sorted(FIGURES.glob(name))
    if not paths:
        return ""
    path = paths[0]
    pixels = path.read_bytes()
    width, height = struct.unpack(">II", pixels[16:24])
    encoded = base64.b64encode(pixels).decode("ascii")
    return (f'<figure><img loading="lazy" width="{width}" height="{height}" '
            f'src="data:image/png;base64,{encoded}" alt="{e(alt or caption)}">'
            f'<figcaption>{e(caption)} <span class="source">来源：figures/{e(path.name)}</span>'
            '</figcaption></figure>')


def detail(title, content) -> str:
    return f'<details><summary>{e(title)}</summary>{content}</details>'


def metric_table(records, title="") -> str:
    return table(
        ["模型", "AP（主 PR-AUC）", "召回率", "召回率 95% CI", "FPR", "FPR 95% CI",
         "精确率", "总告警率", "告警数", "TP", "FP", "阈值"],
        [[model_name(r.get("model")), number(r.get("pr_auc_ap"), 4), pct(r.get("recall")),
          interval(r, "recall"), pct(r.get("fpr")), interval(r, "fpr"), pct(r.get("precision")),
          pct(r.get("alert_rate")), number(r.get("alerts")), number(r.get("tp")),
          number(r.get("fp")), number(r.get("threshold"), 6)] for r in records], title)


def ci_pp(ci) -> str:
    if not ci or None in ci:
        return "—"
    return f"{ci[0] * 100:+.2f} 至 {ci[1] * 100:+.2f} 个百分点"


def significance(ci) -> str:
    if not ci or None in ci:
        return ""
    return "区间不含 0，差异在固定预测下稳定。" if (ci[0] > 0 or ci[1] < 0) else "区间包含 0，在这次测试月中不能确认存在差异。"


def extensions_section(ext: dict) -> str:
    """Section 07: post-hoc supplementary analyses read from outputs/extensions.json."""
    if not ext:
        return pending("补充分析尚未运行：执行 python -m src.extensions 后重新生成报告。")
    rules = ext.get("rules", {})
    curve = ext.get("rule_curve", [])
    chosen = rules.get("chosen_leaves")
    ct, tt, ts = rules.get("chosen_test", {}), rules.get("teacher_test_frozen", {}), rules.get("teacher_test_same_budget", {})
    rep = ext.get("reproduction", {})
    intro = (
        '<div class="finding"><strong>这一节是看过第 8 月结果之后追加的探索性分析。</strong>'
        '它沿用同样的纪律：新模型只在第 1–6 月拟合，所有新选择（早停、阈值、推荐规则数）只看第 7 月，第 8 月只报告。'
        '但因为问题本身是在看到测试结果后提出的，这里的结论用于理解和后续实验设计，不用来替换已冻结的主模型。'
        f'冻结文件的 SHA-256 在本阶段前后一致：{"是" if ext.get("frozen_artifacts_unchanged") else "否"}。</div>'
    )
    repro = (f'<p class="note">复现检查：按 <code>train.py</code> 相同配置重训 LightGBM，最佳迭代 {number(rep.get("best_iteration"))}'
             f'（冻结模型 {number(rep.get("frozen_best_iteration"))}），测试分数最大绝对差 {scientific(rep.get("max_abs_test_score_diff"))}，'
             f'{"与冻结模型一致，排除了参照模型未复现造成的差异；消融结果反映去掉字段后重新拟合、早停和阈值选择的整体变化。" if rep.get("reproduced") else "未能逐位复现，下方对比需同时参考“同配置重训”一行。"}</p>')

    # A. rules
    curve_rows = [[number(r.get("leaves")), number(r.get("depth")), number(r.get("alerted_leaves")),
                   number(r.get("mean_conditions_alerted"), 1), pct(r.get("val_whole_recall")),
                   pct(r.get("test_whole_recall")), pct(r.get("test_whole_fpr")), pct(r.get("test_same_budget_recall")),
                   number(r.get("test_spearman"), 3)] for r in curve]
    zero = [r.get("leaves") for r in curve if not r.get("val_whole_alerts")]
    target = 0.9 * (rules.get("teacher_validation_recall") or 0)
    best_val = max((r.get("val_whole_recall") or 0 for r in curve), default=0)
    if curve and best_val >= target:
        choice_text = (f'验证月规则：取整叶召回达到教师验证召回 90%（{pct(target)}）的最少叶子数，'
                       f'选出 <strong>{number(chosen)} 个叶子</strong>。')
    else:
        choice_text = (f'验证月里<strong>没有任何叶子数</strong>达到教师验证召回的 90%（{pct(target)}），最高只有 {pct(best_val)}；'
                       f'按预设的退路取验证整叶召回最高者，即 <strong>{number(chosen)} 个叶子</strong>。'
                       '本次 8 组 CART 候选都未达到这个目标；这不证明所有不超过 64 个叶子的树或其他规则学习方法都无法逼近教师。')
    first_nonzero = next((r.get("leaves") for r in curve if r.get("val_whole_alerts")), None)
    rule_rows = [[e(r.get("rule_id")), "告警" if r.get("alert") else "放行",
                  f'<div class="rule">{e(r.get("path"))}</div>', number(r.get("teacher_log_odds"), 3),
                  number(r.get("test_rows")), number(r.get("test_fraud")), pct(r.get("test_fraud_rate"))]
                 for r in ext.get("recommended_rules", [])]
    part_a = (
        '<h3>A. 需要多少条规则，才不再“零告警”？</h3>'
        '<p>第 05 节的 5 叶规则树在 FPR≤5% 约束下一个也不报：风险最高的叶子太大，整块打开就超过误报上限。'
        '这里把叶子数从 5 增加到 64，每棵树仍然拟合训练月 LightGBM 的 log-odds，并用两种口径评估：'
        '<strong>整叶规则</strong>（验证月按 FPR≤5% 选出要打开的叶子，叶内全部告警，可表达为确定性条件规则），'
        '以及<strong>严格同 K</strong>（与第 05 节一致，边界叶子按行哈希抽样）。</p>'
        + (f'<p>验证月整叶规则零告警的叶子数：{e("、".join(str(x) for x in zero)) or "无"}；'
           f'从 {number(first_nonzero)} 个叶子开始可以在约束内告警。</p>'
           '<p class="note">为什么叶子加了很多仍然零告警：回归树按全体训练样本的平方误差下降量选择分裂，'
           '这一目标与高风险端的审核容量并不一致。一个可能机制是新增分裂更多改善低风险区的拟合，'
           '而最高分叶子仍过大；本次没有逐次分裂误差分解，不能把这个机制当作已验证的原因。'
           '直接观察到的是：部分候选的最高分叶子在验证月覆盖的正常申请超过 5%，所以按分数阈值整叶告警不可行。'
           '下一轮可以检验是否将分裂集中在高风险边界附近能改善效果（例如只在教师高分区域再拟合规则、对高分样本加权，或使用规则列表方法），'
           '这些可以作为下一轮实验，在新的月份上检验。</p>' if curve else "")
        + table(["叶子数", "深度", "告警叶子", "告警规则平均条件数", "验证整叶召回", "测试整叶召回", "测试整叶 FPR",
                 "测试同 K 召回", "测试 Spearman"], curve_rows, "规则复杂度与效果（整叶 = 确定性条件规则；同 K = 与教师同审核量）")
        + figure("ext_rule_complexity.png", "叶子数与召回率。点线为 LightGBM 教师在测试月的两种口径；竖线为验证月选出的推荐规则数。")
        + (f'<div class="finding">{choice_text}'
           f'测试月整叶规则召回 {pct(ct.get("recall"))}（95% CI {interval(ct, "recall")}），FPR {pct(ct.get("fpr"))}，告警 {number(ct.get("alerts"))} 笔；'
           f'教师冻结阈值召回 {pct(tt.get("recall"))}，FPR {pct(tt.get("fpr"))}。同 K={number(rules.get("test_k"))} 时教师减规则的召回差为 '
           f'{pp(rules.get("same_budget_recall_gap_teacher_minus_rules"))}，bootstrap 95% 区间 {ci_pp(rules.get("same_budget_gap_bootstrap95"))}。'
           f'{significance(rules.get("same_budget_gap_bootstrap95"))}</div>' if chosen else "")
        + detail(f"查看推荐的 {chosen} 条规则（按风险排序）", table(["规则", "动作", "条件（同时成立）", "教师 log-odds", "测试覆盖", "测试欺诈", "测试欺诈率"], rule_rows, numeric_from=3))
        + '<p class="note">本组候选增加叶子数时，全体分数的拟合总体更接近教师，但同审核量召回并非逐项上升，条件也可能更长、更难人工维护。整叶规则在测试月的 FPR 会随人群变化偏离 5%，'
          '这和模型阈值迁移是同一个问题。数值字段缺失时，规则按训练中位数判断，这一点已写进条件文本。</p>'
    )

    # B. fairness
    fair = ext.get("fairness", {})
    ctx = [r for r in ext.get("fairness_context", []) if r.get("split") == "test"]
    setting_names = {"global_threshold": "主模型 · 冻结全局阈值", "no_age_model": "去掉年龄重训 · 冻结全局阈值",
                     "main_same_k": "主模型 · 同审核量 K", "no_age_same_k": "去掉年龄重训 · 同审核量 K",
                     "group_thresholds": "主模型 · 分组阈值"}
    order = list(setting_names)
    ctx = [r for r in ctx if r.get("setting") in ("global_threshold", "main_same_k", "no_age_same_k")]
    ctx = sorted(ctx, key=lambda r: (order.index(r.get("setting")) if r.get("setting") in order else 99, r.get("group", "")))
    ctx_rows = [[e(setting_names.get(r.get("setting"), r.get("setting"))), e(r.get("group")), number(r.get("n")),
                 pct(r.get("base_fraud_rate")), pct(r.get("fpr")), interval(r, "fpr"), pct(r.get("recall")),
                 pct(r.get("precision")), pct(r.get("alert_rate"))] for r in ctx]
    g = {r.get("group"): r for r in ctx if r.get("setting") == "global_threshold"}
    young, old = g.get("age<50", {}), g.get("age>=50", {})
    overall = fair.get("test_overall", {})
    ov_rows = [[e(setting_names.get(k, k)), pct(v.get("recall")), pct(v.get("fpr")), pct(v.get("precision")),
                number(v.get("alerts")), number(v.get("fpr_ratio"), 2), pp(v.get("fpr_gap"))] for k, v in overall.items()
               if k in ("main_same_k", "no_age_same_k")]
    ov = {k: v for k, v in overall.items()}
    same_k_text = ""
    if "main_same_k" in ov and "no_age_same_k" in ov:
        a, b = ov["main_same_k"], ov["no_age_same_k"]
        same_k_text = (f'<p><strong>比较时要先对齐审核量。</strong>去掉年龄的模型在冻结阈值下总共只告警 {number(ov.get("no_age_model", {}).get("alerts"))} 笔，'
                       f'主模型告警 {number(ov.get("global_threshold", {}).get("alerts"))} 笔。总告警减少并不保证每个年龄组的 FPR 都下降；'
                       '审核容量和分数排序同时变化，会混合两者的影响。'
                       f'所以再让两个模型都只审核同样的 K={number(ext.get("test_k"))} 笔：主模型年长/年轻 FPR 比为 {number(a.get("fpr_ratio"), 2)}，'
                       f'去掉年龄后为 {number(b.get("fpr_ratio"), 2)}（差值 {pp(a.get("fpr_gap"))} → {pp(b.get("fpr_gap"))}）。'
                       '同审核量下的变化更接近“去掉年龄”本身的作用，但它仍是单个测试月、事后提出的观察。</p>')
    proxy = fair.get("age_proxy", {})
    auc = proxy.get("roc_auc", {})
    top = "、".join(f"<code>{e(k)}</code>" for k in list(proxy.get("top_gain_features", {}))[:3])
    na = next((r for r in ext.get("ablation", []) if r.get("variant") == "no_age"), {})
    base_ratio = (old.get("base_fraud_rate") / young.get("base_fraud_rate")) if young.get("base_fraud_rate") else None
    part_b = (
        '<h3>B. 年龄差异：先看基础率，再看对照实验</h3>'
        + (f'<p>测试月里，较年长组本身的欺诈率是 {pct(old.get("base_fraud_rate"))}，较年轻组是 {pct(young.get("base_fraud_rate"))}，'
           f'相差约 {number(base_ratio, 2)} 倍。用同一个阈值时，年长组更多人分数偏高，FPR 更高（{pct(old.get("fpr"))} 对 {pct(young.get("fpr"))}），'
           f'但年长组告警的精确率也更高（{pct(old.get("precision"))} 对 {pct(young.get("precision"))}）。'
           '这里有一个恒等式：FPR = 基础率几率 × (1 − 精确率)/精确率 × 召回率，其中基础率几率 = p/(1−p)。'
           '对于召回非零、精确率严格介于 0 与 1 的非退化分类器，基础率不同时，FPR、漏报率、精确率三者不可能同时在两组相等；'
           '但它<strong>并不说明</strong>当前的 FPR 差距是必然的。只要允许两组召回率不同，FPR 和精确率可以同时相等。'
           '所以基础率差异与观察到的 FPR 差距相容，却不能单独解释它的大小，模型、特征和阈值同样起作用。</p>' if g else "")
        + table(["方案", "年龄组", "人数", "基础欺诈率", "FPR", "FPR 95% CI", "召回率", "精确率", "告警率"], ctx_rows,
                "第 8 月：全局阈值（看基础率）与同审核量 K 下的分组表现")
        + table(["方案", "总召回率", "总 FPR", "总精确率", "告警数", "FPR 比（年长/年轻）", "FPR 差"], ov_rows, "第 8 月同审核量 K 下的整体表现与组间 FPR 差异")
        + same_k_text
        + (f'<p><strong>去掉年龄不等于去掉年龄信息。</strong>只用其余输入字段训练一个“猜年龄是否 ≥50”的模型，'
           f'验证月 ROC-AUC {number(auc.get("validation"), 3)}，测试月 {number(auc.get("test"), 3)}（0.5 为瞎猜），'
           f'贡献最大的字段是 {top}。去掉年龄重训后，测试 AP 变化 {pp(na.get("ap_diff_vs_main"))}'
           f'（95% 区间 {ci_pp(na.get("ap_diff_ci95"))}），同 K 召回变化 {pp(na.get("same_budget_recall_diff_vs_main"))}'
           f'（95% 区间 {ci_pp(na.get("same_budget_diff_ci95"))}）。两个区间都包含 0：本次<strong>未观察到明确的性能损失</strong>，'
           '但这不等于证明两个模型效果相同。其余字段能推断年龄，说明模型仍可能间接利用年龄信息，这是剩余差距的一个可能来源，'
           '但不能据此认定剩余差距就由它造成；与年龄相关的合理风险差异同样可能。</p>' if proxy else "")
        + '<p class="note">按年龄分组设不同阈值虽能压平 FPR，但决策直接使用受保护属性，合规上不可取，本报告不作为方案讨论。</p>'
    )

    # C. drift
    vel = ext.get("velocity", {})
    med = vel.get("monthly_medians", {})
    rho = vel.get("spearman_with_month", {})
    abl = ext.get("ablation", [])
    variant_names = {"reference_refit": "同配置重训（全部特征）", "no_age": "去掉 customer_age",
                     "no_velocity_short": "去掉 velocity_6h / 24h", "no_velocity_all": "去掉全部 velocity_*"}
    abl_rows = [[e(variant_names.get(r.get("variant"), r.get("variant"))), number(r.get("best_iteration")),
                 number(r.get("val_ap"), 4), number(r.get("test_ap"), 4), pct(r.get("test_recall")), pct(r.get("test_fpr")),
                 pp(r.get("test_fpr_shift_from_target")), pct(r.get("test_same_budget_recall")),
                 pp(r.get("same_budget_recall_diff_vs_main")) if r.get("variant") != "reference_refit" else "—",
                 ci_pp(r.get("same_budget_diff_ci95")) if r.get("variant") != "reference_refit" else "—"] for r in abl]
    v24 = med.get("velocity_24h", [])
    psi_all = vel.get("test_month_psi_all_fields", [])
    psi24 = next((r.get("psi") for r in ext.get("velocity_monthly", []) if r.get("feature") == "velocity_24h" and r.get("month") == 7), None)
    psi4w = next((r.get("psi") for r in ext.get("velocity_monthly", []) if r.get("feature") == "velocity_4w" and r.get("month") == 7), None)
    short = next((r for r in abl if r.get("variant") == "no_velocity_short"), {})
    part_c = (
        '<h3>C. velocity 字段：被排除的 month 从侧门回来了</h3>'
        + (f'<p><code>velocity_24h</code> 的月中位数从第 1 月的 {number(v24[0])} 一路降到第 8 月的 {number(v24[-1])}；'
           f'它与月份索引的 Spearman 相关为 {number(rho.get("velocity_24h"), 3)}，<code>velocity_6h</code> 为 {number(rho.get("velocity_6h"), 3)}，'
           f'<code>velocity_4w</code> 为 {number(rho.get("velocity_4w"), 3)}'
           + (f'（第 8 月相对训练期的 PSI：velocity_4w 为 {number(psi4w, 2)}，velocity_24h 为 {number(psi24, 2)}；EDA 表原先没有计算 velocity_4w）' if psi4w is not None else '')
           + '。这些字段描述的是申请提交前一段时间内整个系统平均每小时收到多少申请，'
           '主要受全系统申请量驱动，会随月份整体下移，因此带有很强的时间信息。模型虽然没有看到 month，仍可能通过它们间接学到“这是哪段时间”。</p>' if v24 else "")
        + (('<p>按 EDA 的同一方法补算全部 ' + str(len(psi_all)) + ' 个输入字段的第 8 月 PSI，最高的五个是：'
            + '、'.join(f'<code>{e(r.get("feature"))}</code>（{number(r.get("psi"), 2)}）' for r in psi_all[:5])
            + '。原 EDA 只计算了 13 个字段，漏掉了其中几个；排在前面的都是“过去一段时间内的计数/速率”类窗口字段，'
              '它们和 velocity 一样容易携带时间信息。</p>') if psi_all else "")
        + figure("ext_drift_ablation.png", "左：velocity 字段月中位数相对第 1 月的比例（灰底为训练月）；右：去掉不同字段重训后的测试 AP、同 K 召回与实际 FPR。")
        + table(["模型", "最佳迭代", "验证 AP", "测试 AP", "测试召回", "测试 FPR", "FPR 偏离 5%", "测试同 K 召回", "同 K 召回差（对主模型）", "差值 95% CI"],
                abl_rows, "LightGBM 去掉指定字段重训：超参数与主模型相同，早停和阈值仍只看验证月")
        + (f'<p>去掉 <code>velocity_6h / 24h</code> 后，同 K 召回变化 {pp(short.get("same_budget_recall_diff_vs_main"))}'
           f'（95% 区间 {ci_pp(short.get("same_budget_diff_ci95"))}），测试 FPR 偏离 5% 的幅度从主模型重训的 '
           f'{pp(abl[0].get("test_fpr_shift_from_target")) if abl else "—"} 变为 {pp(short.get("test_fpr_shift_from_target"))}。'
           f'{significance(short.get("same_budget_diff_ci95"))}'
           '是否在生产中保留这类字段，要看后续多个月份的滚动回测，单月结果只能提示方向。</p>' if short else "")
    )
    cross_link = (
        '<p class="note">本节同 K 使用原主实验约 5.64% 的总审核容量，结论限于第 8 月。'
        '第 <a href="#rolling">08 节</a>另用统一 5% 审核容量做四窗回测；'
        '它检验这些探索方向是否跨月延续，保留本节原数值，也不将单月“未观察到明确损失”推广为跨月份无代价。</p>'
    )
    return intro + repro + part_a + part_b + part_c + cross_link


def rolling_section(rolling: dict, extensions: dict | None = None) -> str:
    """Present saved expanding-window results without fitting or selecting models."""
    if not rolling:
        return pending("滚动回测结果尚未生成：执行 python -m src.rolling_backtest 后重新生成报告。")
    windows = rolling.get("windows", [])
    metrics = [r for r in rolling.get("metrics", []) if r.get("split") == "test"]
    fairness = rolling.get("fairness", [])
    differences = rolling.get("differences", [])
    variants = {
        "baseline": "该窗验证入选基线", "no_age": "去掉年龄",
        "no_velocity_short": "去掉 velocity_6h / 24h", "no_velocity_all": "去掉全部 velocity",
    }
    window_index = {r.get("window_id"): r for r in windows}
    metric_index = {(r.get("window_id"), r.get("variant"), r.get("policy")): r for r in metrics}
    fairness_index = {(r.get("window_id"), r.get("variant"), r.get("policy")): r for r in fairness}

    def month_label(row):
        month = row.get("test_month", window_index.get(row.get("window_id"), {}).get("test_month"))
        return f'第 {int(month) + 1} 月' if month is not None else e(row.get("window_id", "—"))

    def difference_interval(row, metric):
        low = row.get(metric + "_ci_low", row.get(metric + "_diff_ci_low"))
        high = row.get(metric + "_ci_high", row.get(metric + "_diff_ci_high"))
        return [low, high] if low is not None and high is not None else None

    def difference_cell(row, metric):
        point = row.get(metric + "_diff")
        return f'{pp(point)}<br><small>95% CI {ci_pp(difference_interval(row, metric))}</small>'

    frozen_baseline = [r for r in metrics if r.get("variant") == "baseline" and r.get("policy") == "frozen_threshold"]
    fixed_baseline = [r for r in metrics if r.get("variant") == "baseline" and r.get("policy") == "fixed_review_5pct"]
    range_text = lambda records, key: (
        f'{pct(min(r[key] for r in records))}–{pct(max(r[key] for r in records))}' if records else "—"
    )
    intro = (
        '<div class="finding"><strong>这是在看过原主实验结果后追加的时间稳健性检验。</strong>'
        '每窗只用当时训练期拟合、验证期选择，再报告下一月结果；'
        '但分析问题是在既有数据上提出的，因此不能把这些月份视为全新、独立的确认实验。'
        '本节不依据四个测试月改选主模型，也不替换第 03 节的冻结结果。</div>'
        '<p>每个窗口都从原来的三组 LightGBM 配对配置（15,5）、（31,10）、（63,20）重新选择基线，'
        '按该窗验证月 FPR≤5% 下的召回率排序、AP 打破并列；不会将最后一窗的胜者配置回填到过去。'
        '三个消融沿用本窗入选基线配置，各自只用验证月早停、选阈值。预处理也重新只在当窗训练期拟合。</p>'
        '<p><strong>本节统一审核测试月申请数的 5%</strong>，即 <code>K=floor(0.05×N)</code>。'
        '这是总审核容量；它与验证 FPR≤5% 是两种约束，也与第 05、07 节的约 5.64% 容量不同，'
        '数值不能直接拼接。冻结阈值策略沿用验证阈值；同 K 策略按当月分数排序，不用测试标签确定容量或处理并列。</p>'
    )
    window_rows = []
    for row in windows:
        train_months = row.get("train_months", [])
        train_label = '、'.join(str(int(m) + 1) for m in train_months)
        window_rows.append([
            month_label(row), e(train_label), number(row.get("validation_month") + 1) if row.get("validation_month") is not None else "—",
            number(row.get("train_n")), number(row.get("validation_n")), number(row.get("test_n")),
            number(row.get("test_fraud")), number(row.get("k")),
            f'{number(row.get("selected_num_leaves"))} / {number(row.get("selected_reg_lambda"))}',
            number(row.get("selected_best_iteration")),
        ])
    outcome = (
        f'<p>本次完成 {number(len(windows))} 个窗口。各窗基线在冻结验证阈值下，'
        f'测试召回率范围为 {range_text(frozen_baseline, "recall")}，实际 FPR 范围为 '
        f'{range_text(frozen_baseline, "fpr")}；固定 5% 总审核容量时，召回范围为 '
        f'{range_text(fixed_baseline, "recall")}。这些范围描述已有月份的波动，不是置信区间，不能给出长期表现保证。</p>'
        '<p>读逐月变化时，同时看冻结阈值与同 K 结果：前者会受分数分布和审核量变化影响，'
        '后者对齐了容量，更侧重分数排序。冻结召回下降并不自动说明排序能力受损；'
        '每窗的训练数据、入选配置和测试人群也不同，不能只凭一条召回曲线识别性能变化的原因。</p>'
    )
    direction_rows = []
    direction_notes = []
    for variant in list(variants)[1:]:
        records = [r for r in differences if r.get("variant") == variant]
        positive_recall = sum((r.get("recall_topk_diff") or 0) > 0 for r in records)
        negative_recall = sum((r.get("recall_topk_diff") or 0) < 0 for r in records)
        positive_ap = sum((r.get("ap_diff") or 0) > 0 for r in records)
        negative_ap = sum((r.get("ap_diff") or 0) < 0 for r in records)
        positive_ci = sum((difference_interval(r, "recall_topk") or [0, 0])[0] > 0 for r in records)
        negative_ci = sum((difference_interval(r, "recall_topk") or [0, 0])[1] < 0 for r in records)
        smaller_age_gap = 0
        for row in windows:
            base = fairness_index.get((row.get("window_id"), "baseline", "fixed_review_5pct"), {})
            other = fairness_index.get((row.get("window_id"), variant, "fixed_review_5pct"), {})
            if base.get("fpr_gap") is not None and other.get("fpr_gap") is not None:
                smaller_age_gap += abs(other["fpr_gap"]) < abs(base["fpr_gap"])
        direction_rows.append([
            e(variants[variant]), f'{positive_ap} / {negative_ap}', f'{positive_recall} / {negative_recall}',
            f'{positive_ci} / {negative_ci}',
            f'{smaller_age_gap} / {len(windows)}',
        ])
        if variant == "no_age":
            direction_notes.append(
                f'<p><strong>去掉年龄：</strong>{smaller_age_gap}/{len(windows)} 个窗口的组间 FPR 绝对差缩小；'
                f'同 K 召回点估计在 {positive_recall} 窗上升、{negative_recall} 窗下降。'
                + ('四窗的同 K 召回差区间都包含 0，这不证明跨月无损或模型等效。'
                   if not positive_ci and not negative_ci and len(records) == 4 else '')
                + '误报差异与识别能力需要一起看，不能把原第 8 月“未观察到明确损失”推广为跨月份无代价。</p>'
            )
        elif variant == "no_velocity_all":
            direction_notes.append(
                f'<p><strong>去掉全部 velocity：</strong>同 K 召回点估计在 {positive_recall} 窗上升、'
                f'{negative_recall} 窗下降；AP 点估计在 {positive_ap} 窗上升、{negative_ap} 窗下降。'
                + ('不同月份方向不一致，原单月改善迹象没有稳定延续。' if positive_recall and negative_recall
                   else '这些是有限月份中的观察方向，不能代替新时间数据上的确认。')
                + '</p>'
            )
    previous_all = next((r for r in (extensions or {}).get("ablation", []) if r.get("variant") == "no_velocity_all"), {})
    rolling_all_last = next((r for r in differences if r.get("variant") == "no_velocity_all" and r.get("test_month") == 7), {})
    budget_example = ""
    if previous_all and rolling_all_last:
        budget_example = (
            '<div class="definition"><p><strong>同一排序，换审核容量，结论也可能改变。</strong>'
            '原第 8 月约 5.64% 审核量下，去掉全部 velocity 的召回差为 '
            f'{pp(previous_all.get("same_budget_recall_diff_vs_main"))}；本节同一月固定 5% 审核量时为 '
            f'{pp(rolling_all_last.get("recall_topk_diff"))}（95% CI {ci_pp(difference_interval(rolling_all_last, "recall_topk"))}）。'
            f'AP 差仍为 {pp(rolling_all_last.get("ap_diff"))}。两组第 8 月模型分数与原实验一致，'
            '改变的是容量和排名截止位置，原表因此保留。这不是前后数据冲突：AP 概括整条排序曲线，'
            'AP 提高不保证任意审核预算的召回都提高。</p></div>'
        )
    metric_rows = []
    for window in windows:
        for variant, label in variants.items():
            frozen = metric_index.get((window.get("window_id"), variant, "frozen_threshold"), {})
            fixed = metric_index.get((window.get("window_id"), variant, "fixed_review_5pct"), {})
            metric_rows.append([
                month_label(window), e(label), number(frozen.get("pr_auc_ap"), 4),
                pct(frozen.get("recall")), interval(frozen, "recall"), pct(frozen.get("fpr")),
                number(frozen.get("alerts")), pct(fixed.get("recall")), pct(fixed.get("fpr")), number(fixed.get("alerts")),
            ])
    difference_rows = [[
        month_label(row), e(variants.get(row.get("variant"), row.get("variant"))),
        difference_cell(row, "ap"), difference_cell(row, "recall_frozen"),
        difference_cell(row, "recall_topk"), difference_cell(row, "fpr_gap_topk"),
    ] for row in differences]
    fairness_rows = []
    for window in windows:
        for variant, label in variants.items():
            row = fairness_index.get((window.get("window_id"), variant, "fixed_review_5pct"), {})
            fairness_rows.append([
                month_label(window), e(label), number(row.get("young_normal")), number(row.get("older_normal")),
                pct(row.get("fpr_younger")), pct(row.get("fpr_older")),
                pp(row.get("fpr_gap")), number(row.get("fpr_ratio"), 2),
                pct(row.get("young_recall")), pct(row.get("older_recall")),
            ])
    return (
        intro
        + table(["测试月", "训练月（报告编号）", "验证月", "训练数", "验证数", "测试数", "测试欺诈数", "同 K 容量", "入选 leaves / λ", "最佳迭代"],
                window_rows, "扩展训练期的 4 个窗口；报告月份 = 原始 month + 1", numeric_from=2)
        + figure("rolling_backtest.png", "逐月回测结果：所有同 K 指标固定 5% 总审核容量。左上召回率采用完整的 0%–100% 纵轴，标注基线逐月值；细小差异请结合表格及右下的百分点差值与区间阅读。右上 FPR 从 0 起，左下年龄 FPR 差包含 0；右下展示正负差值。各窗口按自己的验证月选基线，但训练区间重叠，属于事后稳健性检验。")
        + outcome
        + '<h3>先看方向是否重复，再看每月差值</h3>'
        + table(["消融", "AP 上升 / 下降窗口数", "同 K 召回上升 / 下降窗口数", "召回差区间全 >0 / 全 <0 窗口数", "组间 FPR 绝对差缩小的窗口"],
                direction_rows, "升降均为相对当窗基线；等于 0 的点估计不计入升降。计数不是显著性投票，不用于选择赢家")
        + ''.join(direction_notes)
        + budget_example
        + '<p>差值都按“消融 − 当窗基线”计算。本节按年龄组×标签分为四层，'
        f'配对 bootstrap 重抽样 {number(rolling.get("protocol", {}).get("bootstrap_reps"))} 次，每层保留原观测数量；'
        '模型、验证阈值和告警名单固定，不在重抽样中重新选择 Top-K。'
        '区间条件于已观察到的组别规模与基础率，不包含模型重训、阈值估计或跨月分布波动。'
        '召回差或 AP 差为正表示该点估计较高；'
        'FPR 差的变化是有方向的“年长 − 年轻”差变化，负值本身不一定代表更公平，'
        '需要结合原差值和绝对差一起看。区间含 0 不证明模型等效；区间不含 0 也不覆盖重新训练或多重比较的不确定性。</p>'
        + table(["测试月", "模型", "测试 AP", "冻结阈值召回", "召回 95% CI", "冻结阈值 FPR", "冻结告警数", "同 K 召回", "同 K FPR", "同 K 告警数"],
                metric_rows, "两种策略分列；5% FPR 验证阈值迁移后不保证测试仍为 5%", numeric_from=2)
        + detail("查看消融相对基线的逐窗配对差值与区间", table(
            ["测试月", "消融", "AP 差", "冻结阈值召回差", "同 K 召回差", "同 K 组间 FPR 差的变化"],
            difference_rows, "百分点差及年龄组×标签分层的 95% 配对 bootstrap 区间；固定名单、不重新选 Top-K、未作多重比较校正", numeric_from=2))
        + detail("查看统一 5% 审核容量下的年龄组表现", table(
            ["测试月", "模型", "年轻正常数", "年长正常数", "年轻 FPR", "年长 FPR", "FPR 差（年长 − 年轻）", "FPR 比（年长 / 年轻）", "年轻召回", "年长召回"],
            fairness_rows, "年龄边界沿用 <50 / ≥50；两个组共享本模型的全局排序和总容量", numeric_from=2))
        + '<p class="note">四个测试月份不同，但训练数据相互重叠，月份间可能相关；'
        '只有一个随机种子，且仍是同一个合成数据集。不能把四窗合并当作四次独立确认，'
        '也不能用点估计升降反推某个字段造成漂移或歧视。后续应在新时间数据上检验预先写明的选择方案。</p>'
        '<p class="note">同 K 使用整个测试月的无标签分数做批量排序，是回顾性容量对齐；'
        '公开数据缺少逐日时间戳，本项目也未建模到达流量，因此它不直接等同于逐日实时审核策略。</p>'
        + '<p>复核：<a href="rolling_backtest.json">回测完整结果</a>、'
        '<a href="tables/rolling_windows.csv">窗口与选择</a>、<a href="tables/rolling_metrics.csv">性能指标</a>、'
        '<a href="tables/rolling_fairness.csv">年龄组指标</a>、<a href="tables/rolling_differences.csv">配对差值</a>。</p>'
    )


def build_report(eda: dict, evaluation: dict, explanation: dict, protocol: dict,
                 extensions: dict | None = None, rolling: dict | None = None,
                 coursework: dict | None = None, features: dict | None = None) -> str:
    manifest = evaluation.get("manifest", {}) or read_json(OUT / "models/frozen_manifest.json")
    test = evaluation.get("test_results", [])
    validation = evaluation.get("validation_results", [])
    budget = evaluation.get("same_budget_results", [])
    fairness = evaluation.get("fairness", [])
    gaps = evaluation.get("fairness_gaps", [])
    monthly = evaluation.get("monthly", [])
    rules = explanation.get("rules", [])
    importance = explanation.get("shap_importance", [])
    fidelity = explanation.get("student_fidelity", {})
    comparison = evaluation.get("rules_comparison", {})
    winner = manifest.get("supervised_winner")
    test_by_model = {r.get("model"): r for r in test}
    best = test_by_model.get(winner, {})
    lgbm = test_by_model.get("lightgbm", {})
    fraud = eda.get("fraud", {})
    generated = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M（新加坡时间）")
    needed = [("eda_summary.json", eda), ("evaluation.json", evaluation), ("explanation.json", explanation),
              ("extensions.json", extensions), ("rolling_backtest.json", rolling)]
    missing = [name for name, data in needed if not data]
    complete = not missing
    status = "完整分析结果" if complete else "分析进行中 · 部分结果待生成"
    sections = []

    cards = [
        ("全量申请", number(eda.get("rows")), "Base 版本 · 8 个相对月份"),
        ("欺诈占比", pct(fraud.get("fraud_rate"), 3), f'{number(fraud.get("fraud_count"))} 笔标记为欺诈'),
        ("最终测试召回率", pct(best.get("recall")), MODEL_NAMES.get(winner, "等待验证集模型选择")),
        ("最终测试 FPR", pct(best.get("fpr")), "验证月 5% FPR 阈值直接迁移"),
    ]
    cards_html = '<div class="cards">' + "".join(
        f'<div class="card"><span>{e(label)}</span><strong>{value}</strong><small>{e(note)}</small></div>'
        for label, value, note in cards
    ) + "</div>"
    intro = (
        '<p class="lead">用全量 BAF Base 数据检验：在有限误报下，监督模型能抓到多少欺诈；'
        '能否压缩成少量可读规则；不同年龄组和后续月份的表现是否稳定。</p>'
        '<p>分析使用<strong>基于真实银行开户申请数据生成的合成数据</strong>，并非真实客户明细，'
        '也不是银行卡交易流水。仅分析 Base 版本，未混合其余 5 个偏差变体。</p>'
    )
    if missing:
        intro += pending("尚缺 " + "、".join(missing) + "。已有结果可阅读，完整结论需在分析结束后重新生成。")
    headline_items = []
    if best:
        headline_items.append(
            f'<strong>模型：</strong>验证月选出 {model_name(winner)}；第 8 月沿用冻结阈值，抓到 '
            f'{number(best.get("tp"))} 笔欺诈，召回率 {pct(best.get("recall"))}，'
            f'实际 FPR {pct(best.get("fpr"))}，总告警率 {pct(best.get("alert_rate"))}。'
        )
    if comparison:
        student_headline = test_by_model.get("distilled_tree", {})
        zero_alert_note = '5 叶规则的整叶阈值方案零告警；' if student_headline.get("alerts") == 0 else ''
        headline_items.append(
            f'<strong>规则：</strong>{zero_alert_note}同审核量 K={number(comparison.get("test_k"))} 时，'
            f'教师抓到 {number(comparison.get("teacher_tp"))} 笔，规则加边界抽样抓到 '
            f'{number(comparison.get("rules_tp"))} 笔，简化带来明显识别损失。'
        )
    age_headline = next((r for r in gaps if r.get("model") == "lightgbm" and r.get("split") == "test"), {})
    if age_headline:
        headline_items.append(
            '<strong>公平性：</strong>主模型较年长组的正常申请误报率是较年轻组的 '
            f'{number(age_headline.get("fpr_ratio_older_to_younger"), 2)} 倍；总体有效仍伴随年龄组差异，'
            '不能将该差异直接归因为年龄输入或某个代理变量。'
        )
    future_headline = {r.get("month"): r for r in monthly if r.get("model", "lightgbm") == "lightgbm"}
    if future_headline.get(6) and future_headline.get(7):
        val_headline = future_headline[6].get("recall")
        test_headline = future_headline[7].get("recall")
        trend_headline = '这两月未观察到召回下降' if test_headline >= val_headline else '测试月召回低于验证月'
        headline_items.append(
            '<strong>漂移：</strong>窗口特征携带明显时间信息；主模型验证、测试召回分别为 '
            f'{pct(val_headline)}、{pct(test_headline)}，{trend_headline}。'
            '更多窗口的事后稳健性检验见第 08 节，仍需新的数据检验长期效果。'
        )
    if headline_items:
        intro += '<div class="finding"><strong>先读这四点</strong><ul>' + ''.join(
            f'<li>{item}</li>' for item in headline_items
        ) + '</ul><p>以上主实验来自一次时间留出和合成基准，不构成生产效果承诺。</p></div>'
    sections.append(f'<section id="overview"><h2>分析概览</h2>{intro}{cards_html}</section>')

    split_rows = [[split_name(r.get("split")), e(r.get("months", "—")), number(r.get("rows")),
                   number(r.get("fraud_count")), number(r.get("legitimate_count")), pct(r.get("fraud_rate"), 3)]
                  for r in eda.get("split_summary", [])]
    missing_rows = [[f'<code>{e(r.get("column"))}</code>',
                     e(eda.get("missing_rules", {}).get(r.get("column"), "原生空值")),
                     number(r.get("effective_missing_count")), pct(r.get("effective_missing_fraction"))]
                    for r in eda.get("missing_by_feature", [])]
    psi_rows = [[f'<code>{e(r.get("feature"))}</code>', number(r.get("psi"), 4),
                 number(r.get("bins_including_missing"))] for r in eda.get("test_month_top_feature_psi", [])]
    data_section = (
        '<p>标签 <code>fraud_bool=1</code> 表示欺诈，<code>0</code> 表示正常申请。'
        '<code>month=0…7</code> 是连续 8 个月的相对索引，报告对应称为第 1…8 月；'
        '不把索引解释为具体日历月。所有统计使用完整 Base 数据。</p>'
        + table(["切分", "month 索引", "申请数", "欺诈数", "正常数", "欺诈率"], split_rows,
                "数据范围及时间切分（未经抽样）", numeric_from=2)
        + '<h3>空值不只存在于 NaN</h3>'
        + f'<p>原始原生空值单元格数为 {number(eda.get("native_missing_cells"))}，'
        f'字段约定编码的缺失单元格数为 {number(eda.get("sentinel_missing_cells_excluding_native_nulls"))}。'
        '只对官方 datasheet 明确规定的字段做转换，保留相应缺失指示变量。</p>'
        + table(["字段", "缺失判定", "缺失数", "缺失率"], missing_rows, "按字段定义识别的有效缺失", numeric_from=2)
        + '<p><code>intended_balcon_amount</code> 的<strong>所有负数</strong>都视为缺失；'
        '其余表中 5 个字段仅将 <code>-1</code> 视为缺失。'
        '<code>velocity_6h</code> 和 <code>credit_risk_score</code> 的合法负值保留，'
        '不做全表负值替换。官方文档的字段级缺失定义优先于笼统的“无缺失”说明。</p>'
        + figure("eda_missing_values.png", "按官方字段定义统计的缺失比例。缺失较多的字段保留缺失信息，而非直接删行。")
        + f'<p>全行重复数（含标签与月份）：{number(eda.get("duplicate_rows_including_label_and_month"))}。'
        '全量审计发现的常量字段：'
        + ("、".join(f'<code>{e(c)}</code>' for c in eda.get("constant_columns", [])) or "等待数据审计")
        + '。建模时由训练集独立识别并删除常量字段；审计信息不用于测试集调参。</p>'
        + '<h3>基础分布与时间变化</h3>'
        + figure("eda_monthly_volume_fraud.png", "每月申请量与欺诈率。全量标签汇总仅用于描述，不参与模型或阈值选择。")
        + figure("eda_monthly_feature_distributions.png", "按月观察关键字段分布，识别可能影响阈值迁移的人群变化。")
        + detail("查看第 8 月相对训练期的 PSI", '<p>分箱仅由训练月确定：类别字段与非缺失唯一值≤20的数值字段逐值分箱，另设未见值和缺失箱；'
                 '其余连续字段按训练十分位数分箱并单列缺失。每箱计数加入 0.5 伪计数后再归一化。'
                 'PSI 是描述性分布差异量，受分箱影响，不是显著性检验，也不直接证明模型失效。</p>'
                 + table(["字段", "PSI", "箱数（含缺失）"], psi_rows, "第 8 月相对第 1–6 月合并分布")
                 + figure("eda_monthly_psi.png", "各月 PSI 以训练期为参照；不将经验阈值当作显著性证据。"))
    )
    sections.append(f'<section id="data"><p class="eyebrow">01 · 数据检查</p><h2>先确定数据表达了什么</h2>{data_section}</section>')

    protocol_rows = [
        ["训练", "第 1–6 月（month=0…5）", "拟合模型、填补器、标准化、类别词表及常量筛选"],
        ["验证", "第 7 月（month=6）", "选有限超参数、早停、阈值与规则树配置，然后冻结"],
        ["测试", "第 8 月（month=7）", "冻结后评分一次；不回调阈值，不重新选模型"],
    ]
    candidate_rows = []
    for name, spec in manifest.get("models", {}).items():
        candidate_rows.append([model_name(name), e(spec.get("candidate", "—")),
                               '<code class="wrap">' + e(json.dumps(spec.get("params", {}), ensure_ascii=False)) + '</code>',
                               number(spec.get("threshold"), 6)])
    protocol_section = (
        table(["阶段", "时间范围", "用途"], [[e(c) for c in row] for row in protocol_rows], numeric_from=3)
        + '<p><code>month</code> 只用于切分，<strong>从预测输入中排除</strong>；'
        '年龄 <code>customer_age</code> 明确保留在主模型中。其他特征仍可能携带时间和年龄代理信息，'
        '删除 <code>month</code> 本身不能消除分布漂移。</p>'
        + '<p><code>month</code> 记录申请时点；若在预测时可得，它本身不等于标签泄漏。'
        '此处排除它以避免时间索引捷径。真正需要防范的是随机切分混入未来、用全量数据拟合预处理，'
        '以及窗口特征包含预测时点之后的信息。本项目按时间切分且只在训练期拟合转换；'
        '公开数据缺少月份内时间戳和实体 ID，无法逐条审计原始窗口构造是否严格因果。</p>'
        + '<p>逻辑回归、MLP、Isolation Forest 与规则树使用训练集数值中位数、缺失指示、'
        '标准化和类别 One-hot 编码；未知类别安全忽略。LightGBM 使用原生类别及缺失值处理。'
        '监督模型使用平衡类别权重或样本权重，未进行合成重采样。随机种子固定为 42。'
        '<strong>模型分数未经概率校准</strong>，即使输出处于 0–1 区间，也不直接代表实际欺诈概率。</p>'
        + '<h3>两个不同的业务约束</h3>'
        + '<div class="definition"><p><strong>FPR = FP / (FP + TN)</strong>：正常申请被误报的比例。'
        '验证集选取满足 FPR ≤ 5% 的最低可行阈值；相同分数整体保留，预测比较符统一为 <code>score ≥ threshold</code>。'
        '该阈值冻结后直接用于测试，因此测试 FPR 可以偏离 5%。</p>'
        '<p><strong>总告警率 = (TP + FP) / N</strong>：全部申请中需要审核的比例。'
        '它与 5% FPR 不等价。同审核量比较另用固定容量 Top-K，不用测试标签确定 K 或分数并列顺序。</p></div>'
        + '<p>主 PR-AUC 明确定义为 <strong>Average Precision（AP）</strong>，由 '
        '<code>average_precision_score</code> 计算。'
        '模型选择先比较验证月 5% FPR 约束下的召回率，再以 AP 打破并列。'
        'ROC-AUC 可追溯到 JSON，类别高度不平衡时不作为主选择指标。</p>'
        + detail("查看冻结配置与阈值", table(["模型", "入选候选", "冻结参数", "验证阈值"], candidate_rows,
                 "来源：outputs/models/frozen_manifest.json", numeric_from=3))
        + '<p class="note">本项目采用更贴近上线的验证阈值迁移协议。原论文及官方 notebook 的实验设计'
        '与本项目并非完全一致，因而此处是独立分析，不声称精确复现论文表格。</p>'
    )
    sections.append(f'<section id="protocol"><p class="eyebrow">02 · 时间切分</p><h2>先冻结决策，再打开测试月</h2>{protocol_section}</section>')

    supervised_keys = {"logistic_regression", "lightgbm", "mlp"}
    sup_test = [r for r in test if r.get("model") in supervised_keys]
    sup_val = [r for r in validation if r.get("model") in supervised_keys]
    mlp = test_by_model.get("mlp", {})
    comparison_note = ""
    if mlp and lgbm and mlp.get("recall", 0) > lgbm.get("recall", 0):
        comparison_note = (
            f'<p>MLP 的冻结阈值测试召回率为 {pct(mlp.get("recall"))}，'
            f'同时 FPR 为 {pct(mlp.get("fpr"))}，对应审核 {number(mlp.get("alerts"))} 笔；'
            f'LightGBM 的 FPR 为 {pct(lgbm.get("fpr"))}，审核 {number(lgbm.get("alerts"))} 笔。'
            '两者承担的误报与审核量不同，不能只凭原始召回率称 MLP 更优，也不据此更换验证集选出的模型。'
            '相同审核量的对比见规则章节。</p>'
        )
    if_row = next((r for r in test if r.get("model") == "isolation_forest"), {})
    if_note = (f'<p class="note">另做过一个无监督对照：Isolation Forest 只在训练期特征上拟合（不用标签），阈值同样按验证月 FPR ≤ 5% 校准。'
               f'测试月召回率 {pct(if_row.get("recall"))}、精确率 {pct(if_row.get("precision"))}，与随机告警相差不大，说明“异常”不等于“欺诈”，不再单独展开。</p>'
               if if_row else "")
    benchmark_section = (
        '<p>对比线性模型、梯度提升树和多层感知机。候选范围有限，结果反映本次预先约定的搜索预算，'
        '不能解释为这些算法的绝对上限。</p>'
        + ('<div class="finding">监督模型入选依据来自验证月：<strong>' + model_name(winner)
           + '</strong>。以下测试表用于报告冻结后的表现；不按测试分数重选冠军。</div>' if winner else pending())
        + '<h3>最终留出：第 8 月</h3>' + metric_table(sup_test, "冻结验证阈值在测试月的表现")
        + comparison_note
        + figure("benchmark.png", "模型对比：阈值在验证月选取并冻结。测试月实际 FPR 不强制等于 5%。")
        + figure("pr_curves.png", "测试月 PR 曲线，配合 AP 衡量排序能力；曲线用于评价，不用于重新选阈值。")
        + detail("查看验证月结果", metric_table(sup_val, "第 7 月：用于配置和阈值选择，不能当作独立泛化估计"))
        + if_note
        + '<p class="note">召回率和 FPR 的 95% 区间为 Wilson 二项比例区间，条件于当前固定预测。'
        '它们不覆盖重新训练、超参数选择、月份更替或实体相关性带来的全部不确定性。</p>'
    )
    sections.append(f'<section id="supervised"><p class="eyebrow">03 · 监督模型对比</p><h2>在可控误报下识别欺诈</h2>{benchmark_section}</section>')

    from .feature_report import feature_section
    sections.append('<section id="features"><p class="eyebrow">04 · 特征分析与特征工程（事后探索）</p>'
                    '<h2>哪些字段有用、哪些会漂移，造新特征能不能更好</h2>' + feature_section(features or {}) + '</section>')

    importance_rows = [[str(i), f'<code>{e(r.get("feature"))}</code>', number(r.get("mean_abs_shap"), 5)]
                       for i, r in enumerate(importance[:20], 1)]
    rule_rows = []
    for rule in rules:
        path = rule.get("path", "—")
        if isinstance(path, list):
            path = " 且 ".join(str(item) for item in path)
        rule_rows.append([
            e(rule.get("rule_id", "—")), f'<div class="rule">{e(path)}</div>',
            number(rule.get("teacher_log_odds"), 4), number(rule.get("train_rows")),
            number(rule.get("validation_rows")), number(rule.get("test_rows")),
            number(rule.get("test_fraud")), pct(rule.get("test_fraud_rate")),
        ])
    budget_rows = [[model_name(r.get("model")), split_name(r.get("split", "test")), number(r.get("n")),
                    number(r.get("k", r.get("alerts"))), number(r.get("tp")), number(r.get("fp")),
                    pct(r.get("recall")), pct(r.get("precision")), pct(r.get("alert_rate"))] for r in budget]
    compare_sentence = ""
    if comparison:
        compare_sentence = (
            f'<div class="finding">同一测试审核容量 <strong>K={number(comparison.get("test_k"))}</strong>：'
            f'LightGBM 抓到 {number(comparison.get("teacher_tp"))} 笔，规则树抓到 '
            f'{number(comparison.get("rules_tp"))} 笔；模型减规则的抓获数差为 '
            f'{number(comparison.get("caught_difference"))} 笔，召回率差为 '
            f'{pp(comparison.get("recall_gap"))}。</div>'
        )
        cutoff_n = comparison.get("rules_cutoff_tie_count")
        selected_n = comparison.get("rules_selected_from_cutoff_tie")
        if cutoff_n is not None and selected_n is not None:
            compare_sentence += (
                f'<p>本次规则边界层有 {number(cutoff_n)} 笔同分申请，'
                f'其中通过固定行哈希选入 {number(selected_n)} 笔。'
                '同预算抓获数依赖这个已固定、与标签独立的容量决策。</p>'
            )
    bootstrap = comparison.get("bootstrap95")
    if isinstance(bootstrap, (list, tuple)) and len(bootstrap) == 2:
        compare_sentence += (f'<p>固定预测下的配对分层 bootstrap 95% 召回差区间：'
                             f'{pp(bootstrap[0])} 至 {pp(bootstrap[1])}。'
                             f'正类内配对重抽样 {number(comparison.get("bootstrap_replicates"))} 次；'
                             '该区间不包含重新训练和重新选模型的变化。</p>')
    elif isinstance(bootstrap, dict):
        compare_sentence += '<p>配对分层 bootstrap 结果：<code class="wrap">' + e(json.dumps(bootstrap, ensure_ascii=False)) + '</code>。固定预测下估计，不含重新训练不确定性。</p>'
    fidelity_rows = [["验证月 Spearman", number(fidelity.get("validation_spearman"), 4)],
                     ["测试月 Spearman", number(fidelity.get("test_spearman"), 4)]]
    if "validation_r2" in fidelity:
        fidelity_rows.append(["验证月 R²", number(fidelity["validation_r2"], 4)])
    if "test_r2" in fidelity:
        fidelity_rows.append(["测试月 R²", number(fidelity["test_r2"], 4)])
    student = test_by_model.get("distilled_tree", {})
    student_val = next((r for r in validation if r.get("model") == "distilled_tree"), {})
    degenerate = bool(student) and student.get("alerts") == 0 and student_val.get("alerts") == 0
    granularity_note = ""
    if degenerate:
        granularity_note = (
            '<div class="finding"><strong>规则容量粒度过粗：整体阈值方案退化为零告警。</strong>'
            '最高风险叶子在验证月覆盖的正常申请已超过 5%，若按整叶相同分数一起报警便违反 FPR 上限。'
            '因此满足 5% FPR 约束的冻结阈值高于最高叶子分数，验证和测试都不报警，测试召回率为 0。'
            '这说明少量规则不能提供足够细的审核容量控制；规则树在此作为可解释风险分层基线。'
            '下方严格同 K 方案通过边界层抽样，才使它与教师模型使用相同审核量。</div>'
            + detail("查看规则树的冻结阈值结果", metric_table([student], "整叶分数并列不拆分的冻结阈值结果"))
        )
    shap_findings = ""
    if importance:
        shap_findings = '<p>本次平均绝对贡献最大的字段为 ' + "、".join(
            f'<code>{e(r.get("feature"))}</code>' for r in importance[:3]
        ) + '；它们表明模型主要依赖哪些输入，并不证明这些特征造成欺诈。</p>'
    explanation_section = (
        '<h3>SHAP：解释 LightGBM 的判断依据</h3>'
        + f'<p>SHAP 从第 7 相对月（原始 <code>month=6</code>）验证集以 seed=42 固定无放回抽取 '
        f'{number(explanation.get("shap_sample_size"))} 个解释样本，'
        '基于 LightGBM 原生 TreeSHAP，贡献单位是原始 log-odds；'
        '排序按平均绝对 SHAP 值，表示对模型输出贡献的幅度，不表示变量的因果影响。'
        '相关字段之间的贡献可能分摊或替代，beeswarm 展示样本间方向与差异。'
        '该解释不用于在查看测试结果后重新训练或调参。</p>'
        + shap_findings
        + figure("shap_importance.png", "LightGBM 的 SHAP 全局重要性；数值尺度以解释输出和原始记录为准。")
        + figure("shap_beeswarm.png", "SHAP 样本贡献分布：特征取值颜色与贡献方向同时阅读，不作因果归因。")
        + detail("查看 SHAP 数值与一致性检查", table(["排序", "字段", "平均绝对 SHAP"], importance_rows, numeric_from=2)
                 + f'<p>SHAP 加和最大误差：{scientific(explanation.get("shap_additivity_max_error"))}。'
                 '完整字段排序见 <a href="explanation.json">explanation.json</a>。</p>')
        + '<h3>蒸馏：用最多 5 个叶子近似教师模型</h3>'
        + '<p>学生是深度 3 或 4、最多 5 个叶子的回归树，拟合目标是训练月 LightGBM 的 '
        '<strong>原始 log-odds 分数</strong>，不是 <code>fraud_bool</code>。'
        '类别加权教师未经校准，学生叶子分数也不能解释为实际欺诈概率。'
        '树结构拟合只看训练月；深度根据验证月同预算召回率选择，教师分数 Spearman 相关系数用于并列判定。'
        '规则的测试欺诈率是事后描述，不参与规则生成。</p>'
        + figure("distilled_tree.png", "蒸馏树的分支与叶子。完整原始条件可追溯到 explanation.json。")
        + table(["规则", "可读条件（各条件需同时成立）", "教师 log-odds", "训练覆盖", "验证覆盖", "测试覆盖", "测试欺诈数", "测试欺诈率"],
                rule_rows, "全部终端叶子规则；分数为叶子的平均教师输出", numeric_from=2)
        + granularity_note
        + detail("查看学生对教师的保真度", table(["指标", "结果"], [[e(x), y] for x, y in fidelity_rows])
                 + '<p>Spearman 衡量排序一致性，不等同于欺诈识别效果。少量叶子通常会损失教师模型的细粒度排序。</p>')
        + '<h3>同样的人工审核量，严格比较模型与规则</h3>'
        + '<p>先以验证月 LightGBM 的告警率固定预算比例，再为目标月计算 '
        '<code>K=floor(预算比例 × 该月申请数)</code>，各模型恰好选分数最高的 K 笔。'
        '规则叶子存在大量分数并列：并列时使用 <strong>seed=42 的行 ID 哈希抽签</strong>，'
        '既不使用测试标签，也不按源文件月份内顺序占位。此结果对应“规则分层 + 边界层抽样审核”，'
        '不是只靠规则阈值就能恰好产生 K 个告警。</p>'
        + compare_sentence
        + table(["模型", "切分", "申请数", "同一 K", "抓获 TP", "误报 FP", "召回率", "精确率", "告警率"],
                budget_rows, "同预算排序比较与冻结阈值比较是两个不同的评估问题", numeric_from=2)
        + figure("equal_budget.png", "固定相同告警数量后比较抓获数，避免规则与模型审核量不同导致的失真。")
        + '<p class="note">由于树输入经过训练集标准化和 One-hot 编码，规则解释应使用导出的原尺度条件；'
        '涉及缺失指示或填补值时须同时阅读其分支定义。收入为分位形式，不能把阈值解释成货币金额。'
        '类别代码（例如 <code>housing_status=BA</code>）保持匿名含义，不推断其对应的具体住房类型。'
        '<code>velocity_6h</code> 等字段表示窗口内平均每小时申请量，不能据此声称恢复了单客户实时计数规则。</p>'
    )
    explanation_section += '<p class="note">叶子数增加后规则能否摆脱零告警、需要多少条才接近教师，见 <a href="#extensions">第 07 节 A</a>。</p>'
    sections.append(f'<section id="rules"><p class="eyebrow">05 · 模型转规则</p><h2>解释复杂模型，再衡量简化的代价</h2>{explanation_section}</section>')

    fairness_test = [r for r in fairness if r.get("split") == "test"]
    fairness_val = [r for r in fairness if r.get("split") == "validation"]
    def fairness_table(records):
        def total(row, left, right):
            return row[left] + row[right] if row.get(left) is not None and row.get(right) is not None else None
        return table(["模型", "年龄组", "人数", "正常数", "FP", "FPR", "FPR 95% CI", "欺诈数", "TP", "召回率"],
                     [[model_name(r.get("model")), e(r.get("group", "—")), number(r.get("n")),
                       number(total(r, "tn", "fp")), number(r.get("fp")),
                       pct(r.get("fpr")), interval(r, "fpr"),
                       number(total(r, "tp", "fn")), number(r.get("tp")), pct(r.get("recall"))]
                      for r in records], "同一模型对两个年龄组使用同一个全局冻结阈值", numeric_from=2)
    gap_rows = [[model_name(r.get("model")), split_name(r.get("split")),
                 pp(r.get("fpr_gap_older_minus_younger")), number(r.get("fpr_ratio_older_to_younger"), 3)]
                for r in gaps]
    month_rows = [[model_name(r.get("model", "lightgbm")),
                  f'第 {int(r["month"]) + 1} 月（索引 {int(r["month"])}）', split_name(r.get("split")),
                  number(r.get("n")), number(r.get("pr_auc_ap"), 4), pct(r.get("recall")), interval(r, "recall"),
                  pct(r.get("fpr")), number(r.get("alerts"))] for r in monthly if r.get("month") is not None]
    lgbm_gap = next((r for r in gaps if r.get("model") == "lightgbm" and r.get("split") == "test"), {})
    fairness_note = ""
    if lgbm_gap:
        fairness_note += (
            f'<div class="finding">LightGBM 在测试月，较年长组 FPR 是较年轻组的 '
            f'<strong>{number(lgbm_gap.get("fpr_ratio_older_to_younger"), 2)} 倍</strong>，'
            f'差值为 {pp(lgbm_gap.get("fpr_gap_older_minus_younger"))}。'
            '整体 FPR 较低仍伴随明显组间差异，需要在后续策略评价中同时跟踪。</div>'
        )
    if degenerate:
        fairness_note += (
            '<p class="note">规则树的组间 FPR 差为 0，来自<strong>完全不报警、召回也为 0</strong>的退化策略；'
            '比值 0/0 无定义，以“—”显示。这不能解释为规则树更公平或更有用。'
            '同 K 抽样策略与这里的整体冻结阈值不同，公平性结论不可直接互换。</p>'
        )
    future = {r.get("month"): r for r in monthly if r.get("model", "lightgbm") == "lightgbm"}
    drift_note = ""
    if future.get(6) and future.get(7):
        va_recall, te_recall = future[6].get("recall"), future[7].get("recall")
        drift_note = (f'<p>LightGBM 第 7 月召回率为 {pct(va_recall)}，第 8 月为 {pct(te_recall)}。'
                      + ('这两个未来月之间未观察到召回率下降。' if te_recall >= va_recall else '测试月召回率低于验证月，但这一个相邻月差异不能证明持续下降。')
                      + '两个月基准率与阈值实际 FPR 也发生变化，不能仅凭这两个比例推断长期趋势。</p>')
    fairness_section = (
        '<h3>年龄分组：先把边界和分母说清楚</h3>'
        '<p>主分析比较 <strong>age &lt; 50</strong> 与 <strong>age ≥ 50</strong>。'
        '公开年龄取值按十年档处理，50 岁组边界代表公开年龄档，并非精确真实年龄。'
        '官方 datasheet / notebook 与论文正文对 50 岁边界存在差异，因此额外导出 '
        '<code>age ≤ 50</code> / <code>age &gt; 50</code> 的敏感性结果。</p>'
        '<p>主模型<strong>包含年龄</strong>，没有实施公平性约束或分组阈值修正。'
        '同一模型在两个组上使用完全相同的全局冻结阈值；FPR 的分母是该组正常申请数，'
        '不能用该组全部人数替代。误报差值定义为“较年长组 − 较年轻组”，比值定义为前者除以后者。</p>'
        + fairness_table(fairness_test)
        + figure("fairness.png", "测试月年龄组 FPR：同时观察差异大小、组内正常申请分母和 95% 区间。")
        + table(["模型", "切分", "FPR 差（年长 − 年轻）", "FPR 比（年长 / 年轻）"], gap_rows,
                "差值 0、比值 1 表示该指标上的组间一致", numeric_from=2)
        + fairness_note
        + detail("查看验证月年龄组指标", fairness_table(fairness_val))
        + detail("查看测试月 50 岁边界敏感性", fairness_table([r for r in evaluation.get("age_boundary_sensitivity", []) if r.get("split") == "test"]))
        + figure("eda_age_group_monthly.png", "每月年龄组组成与组内欺诈率，为组间 FPR 差异提供描述性背景。")
        + '<p>不同基础欺诈率、样本量和协变量分布会影响多种公平指标。单个 FPR 差异不能识别歧视的因果来源；'
        '删除年龄也不能保证消除代理变量影响。本报告测量差异，不把某个比值当作全面公平认证。</p>'
        + '<h3>时间漂移：分开训练内、验证和最终测试</h3>'
        + figure("monthly_recall.png", "冻结阈值下的逐月表现；第 1–6 月是训练内分数，第 7 月用于选择，第 8 月才是最终留出。")
        + table(["模型", "月份", "用途", "申请数", "AP", "召回率", "召回率 95% CI", "FPR", "告警数"],
                month_rows, "逐月指标；训练内的高分不构成未来表现证据", numeric_from=3)
        + drift_note
        + '<p class="note">第 1–6 月参与模型拟合，不能用它们与后两个月的差异证明“效果逐月下降”。'
        '第 7 月还参与模型选择；这里只有第 8 月一个最终时间留出。'
        '可以结合分布差异、基准率和冻结阈值后的实际 FPR 描述迁移表现，'
        '但需要更多未来月份和滚动时间回测，才能判断持续漂移趋势。</p>'
    )
    fairness_section += '<p class="note">两组基础欺诈率、去掉年龄的对照模型以及 velocity 字段的漂移检验，见 <a href="#extensions">第 07 节 B、C</a>。</p>'
    sections.append(f'<section id="fairness"><p class="eyebrow">06 · 公平性与漂移</p><h2>整体有效，不代表各组与各月都相同</h2>{fairness_section}</section>')

    sections.append('<section id="extensions"><p class="eyebrow">07 · 补充分析（事后探索）</p>'
                    '<h2>规则需要多少条、年龄差异从何而来、漂移字段该不该留</h2>' + extensions_section(extensions or {}) + '</section>')

    sections.append('<section id="rolling"><p class="eyebrow">08 · 滚动时间回测（事后稳健性检验）</p>'
                    '<h2>将同一比较协议放到四个后续月份</h2>' + rolling_section(rolling or {}, extensions or {}) + '</section>')

    from .coursework_report import coursework_section
    sections.append('<section id="coursework"><p class="eyebrow">09 · 课程方法补充（事后探索）</p>'
                    '<h2>把课上的数据处理与评估方法放到同一个冻结协议上</h2>' + coursework_section(coursework or {}) + '</section>')

    limitation_items = [
        "主实验只有一次最终时间留出；第 08 节补充的四个滚动窗口仍有重叠训练数据、有限月份和固定随机种子，未进行多随机种子重复实验。验证集同时承担调参与阈值选择，存在验证集选择乐观性。",
        "测试数据的整体与分组标签统计只用于描述和最终评价；不据此改动训练方案、预算比例、规则或阈值。",
        "本项目始终使用同一个合成基准；更多时间窗口不能将其变为真实客户验证。数据不包含真实调查成本、损失金额、客户实体链接和月份内事件顺序，不能计算真实经济收益，也不能还原实时窗口计数系统。",
        "SHAP 解释模型而非因果；蒸馏简化教师分数而非证明规则稳定。规则边界并列抽样是固定审核容量方案的一部分。",
        "Wilson 区间与固定预测 bootstrap 区间未覆盖模型重训、超参数选择以及长期部署分布变化。",
        "本地保存的原始数据、预测、模型与哈希支持复核；HTML 图表和数值是静态快照，应在上游输出更新后重新生成。",
        "第 07、08 节在看过原测试月后设计，属于探索和事后稳健性检验，不能当作全新数据上的独立确认，也不据此更换主模型。",
        "第 04 节的特征分析与特征工程是看过测试月后加做的：分箱、编码、筛选规则和早停只用月 0–6，月 7 只报告一次，结论不替换冻结模型。",
        "第 09 节的课程方法补充同样是事后探索：SMOTE 为普通 SMOTE（会对独热和缺失标记列也做插值），随机切分对照有意把月 0–6 混合，均不改变冻结模型与主结论。",
    ]
    links = [
        ("原始审计摘要", "eda_summary.json"), ("最终指标和置信区间", "evaluation.json"),
        ("SHAP 与全部规则", "explanation.json"), ("冻结模型与协议清单", "models/frozen_manifest.json"),
        ("补充分析结果", "extensions.json"),
        ("滚动时间回测", "rolling_backtest.json"),
        ("课程方法补充", "coursework.json"),
        ("特征分析与特征工程", "feature_study.json"),
        ("CSV 数据表目录", "tables/"), ("原始图目录", "figures/"),
    ]
    audit_links = "".join(f'<li><a href="{e(href)}">{e(label)}</a></li>' for label, href in links)
    sha = manifest.get("raw_sha256")
    audit = (f'<p>输入 Base.csv SHA-256：<code class="wrap">{e(sha)}</code></p>' if sha else "")
    notes = '<ul>' + "".join(f"<li>{e(x)}</li>" for x in limitation_items) + "</ul>"
    provenance = (
        '<p>下列本地链接以本报告所在的 <code>outputs</code> 目录为基准；图像已内嵌，'
        '单独复制 HTML 仍可阅读全部正文、数值和图表，追溯源码和数据则需保留项目目录。</p>'
        '<ul class="file-links">' + audit_links + '</ul>' + audit
        + '<p>数据与字段定义来自 '
        '<a href="https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022">官方 Kaggle 数据集</a>、'
        '<a href="https://github.com/feedzai/bank-account-fraud">Feedzai 官方仓库</a> 与 '
        '<a href="https://raw.githubusercontent.com/feedzai/bank-account-fraud/main/documents/datasheet.pdf">官方 datasheet</a>。'
        '研究依据：Jesus et al. (2022), '
        '<a href="https://papers.neurips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html">'
        '<em>Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation</em></a>，NeurIPS 2022。'
        '原始资料及下载校验保存在项目 <code>references/</code>。</p>'
        '<p>原始数据按官方 CC BY-NC-ND 4.0 条款保留来源；本项目不重新发布修改后的原始数据。'
        '报告和实验代码是本地分析产物。</p>'
    )
    sections.append(f'<section id="limits"><h2>适用边界与复核路径</h2>{notes}<h3>结果追溯</h3>{provenance}</section>')

    nav = [("overview", "概览"), ("data", "1 数据"), ("protocol", "2 时间切分"),
           ("supervised", "3 监督模型"), ("features", "4 特征工程"), ("rules", "5 规则蒸馏"),
           ("fairness", "6 公平与漂移"), ("extensions", "7 补充分析"),
           ("rolling", "8 滚动回测"), ("coursework", "9 课程方法"), ("limits", "限制与来源")]
    nav_html = "".join(f'<a href="#{key}">{e(label)}</a>' for key, label in nav)
    css = """
    :root{--ink:#162b3e;--muted:#536574;--line:#dce5ea;--accent:#167a7b;--wash:#edf6f5;--paper:#fff;--bg:#f4f7f9}
    *{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:82px}body{margin:0;color:var(--ink);background:var(--bg);font:16px/1.8 "Microsoft YaHei","PingFang SC",system-ui,sans-serif}a{color:#086e82;text-decoration-thickness:1px;text-underline-offset:3px}a:hover{color:#104c61}header{background:#15374a;color:#fff;padding:54px max(24px,calc((100vw - 1180px)/2)) 40px}header .kicker{font-size:13px;letter-spacing:.12em;color:#9fd9d3}h1{font-size:clamp(28px,4.5vw,43px);line-height:1.3;margin:12px 0 14px;font-weight:700}header p{margin:4px 0;color:#d9e8ed}header .meta{font-size:13px;color:#b7cfd8}nav{position:sticky;top:0;z-index:10;display:flex;gap:4px;overflow:auto;white-space:nowrap;background:#fff;border-bottom:1px solid var(--line);padding:10px max(16px,calc((100vw - 1180px)/2));box-shadow:0 2px 6px #162b3e08}nav a{padding:5px 11px;font-size:14px;text-decoration:none;border-radius:4px}nav a:hover{background:var(--wash)}main{max-width:1244px;margin:28px auto;padding:0 32px}section{background:var(--paper);padding:30px 34px;margin-bottom:22px;border:1px solid var(--line);border-radius:8px}h2{font-size:26px;line-height:1.45;margin:0 0 18px}h3{font-size:20px;margin:30px 0 12px}p{margin:12px 0}.lead{font-size:19px;line-height:1.85}.eyebrow{font-size:13px;font-weight:700;color:var(--accent);letter-spacing:.06em;margin:0 0 8px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0 6px}.card{border-top:3px solid var(--accent);background:#f4f8fa;padding:16px}.card span,.card small{display:block;color:var(--muted)}.card span{font-size:13px}.card strong{font-size:30px;line-height:1.55;display:block;font-variant-numeric:tabular-nums}.card small{font-size:12px;line-height:1.65}.finding{border-left:4px solid var(--accent);padding:15px 20px;background:var(--wash);margin:20px 0}.definition{background:#f4f8fa;border:1px solid var(--line);padding:8px 20px}.pending{color:#8c5814;background:#fff5e6;padding:12px 16px;border-left:3px solid #c99442}.note{font-size:14px;color:var(--muted);border-left:3px solid #b8cbd4;padding-left:14px}.table-wrap{width:100%;overflow-x:auto;border:1px solid var(--line);border-radius:4px;margin:18px 0;overscroll-behavior-x:contain}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.65}caption{text-align:left;padding:11px 14px;color:var(--muted);background:#f5f8fa;font-size:12px}th{background:#edf3f6;white-space:nowrap;text-align:left;font-weight:600;color:#2b4e62}th,td{padding:10px 13px;border-bottom:1px solid var(--line);vertical-align:top}tbody tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#fafcfd}.num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}.text{min-width:120px}.rule{min-width:380px;max-width:640px;white-space:normal;overflow-wrap:anywhere;font-family:Consolas,monospace;font-size:12px}code{font-family:Consolas,"SFMono-Regular",monospace;font-size:.88em;background:#edf3f6;padding:1px 4px;border-radius:3px}.wrap{white-space:normal;overflow-wrap:anywhere}figure{margin:24px 0;border:1px solid var(--line);background:#fff;border-radius:5px;overflow:hidden}figure img{width:100%;height:auto;display:block}figcaption{font-size:13px;color:var(--muted);padding:12px 16px;border-top:1px solid var(--line);line-height:1.7}.source{display:block;font-size:11px;color:#718491}details{border:1px solid var(--line);border-radius:4px;padding:12px 16px;margin:18px 0}summary{cursor:pointer;font-size:14px;font-weight:600;color:#315b70}details[open] summary{margin-bottom:14px}li{margin:8px 0}.file-links{display:grid;grid-template-columns:repeat(2,1fr);padding-left:22px}footer{max-width:1180px;margin:24px auto 40px;padding:0 24px;font-size:12px;color:var(--muted)}.print{float:right;background:transparent;border:1px solid #a4c2ce;border-radius:4px;color:#eaf3f7;padding:7px 13px;cursor:pointer;font:inherit;font-size:12px}
    @media(max-width:800px){header{padding:32px 22px}.cards{grid-template-columns:repeat(2,1fr)}main{padding:0 12px;margin:16px auto}section{padding:24px 20px}.file-links{grid-template-columns:1fr}h2{font-size:23px}body{font-size:15px}.card strong{font-size:26px}.print{display:none}}
    @media(max-width:420px){section{padding:20px 15px}.cards{gap:8px}.card{padding:12px}.card strong{font-size:23px}nav a{padding:5px 9px}}
    @media print{nav,.print{display:none}body{background:white;color:#111;font-size:10pt}header{padding:18px 0;background:white;color:#111}header p,header .kicker,header .meta{color:#555}main{max-width:none;margin:0;padding:0}section{padding:18px 0;border:0;border-radius:0;break-before:auto}h2,h3{break-after:avoid}figure,.card{break-inside:avoid}.table-wrap{overflow:visible}table{font-size:8pt}.num{white-space:normal}.rule{min-width:0}.cards{gap:8px}details{break-inside:avoid}details:not([open]){display:none}a{color:#222}footer{padding:0}.source{color:#555}}
    """
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="description" content="BAF Base 全量银行开户欺诈分析：时间留出、模型基准、规则蒸馏、公平性与漂移。">'
        '<title>BAF 银行开户欺诈分析报告</title><style>' + css + '</style></head><body>'
        '<header><button class="print" onclick="window.print()">打印 / 保存</button>'
        '<div class="kicker">BANK ACCOUNT FRAUD · BASE DATASET</div>'
        '<h1>银行开户欺诈：从模型到可读规则</h1>'
        '<p>全量数据审计 · 时间留出评估 · 同预算比较 · 年龄公平性</p>'
        f'<p class="meta">{e(status)} · 生成于 {e(generated)}</p></header>'
        '<nav aria-label="报告目录">' + nav_html + '</nav><main>' + "".join(sections) + '</main>'
        '<footer>基于本地保存的分析输出生成。所有图像已内嵌；表格可横向滚动，点开折叠项可查看详细数值。</footer>'
        '</body></html>'
    )


def run(wait_seconds=0, require_complete=False):
    required = [OUT / name for name in ["eda_summary.json", "evaluation.json", "explanation.json",
                                        "extensions.json", "rolling_backtest.json"]]
    deadline = time.monotonic() + max(0, wait_seconds)
    while not all(p.exists() for p in required) and time.monotonic() < deadline:
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    eda, evaluation, explanation, extensions, rolling = [read_json(path) for path in required]
    loaded = [eda, evaluation, explanation, extensions, rolling]
    coursework = read_json(OUT / "coursework.json")
    features = read_json(OUT / "feature_study.json")
    if require_complete and not all(loaded):
        missing = ", ".join(str(p.relative_to(ROOT)) for p, value in zip(required, loaded) if not value)
        raise SystemExit("完整报告所需结果尚未生成或无法读取：" + missing)
    OUT.mkdir(parents=True, exist_ok=True)
    if rolling:
        from .rolling_plot import render
        render(rolling)
    REPORT.write_text(build_report(eda, evaluation, explanation, read_json(ROOT / "protocol.json"), extensions, rolling, coursework, features), encoding="utf-8")
    print(f"报告已生成：{REPORT}", flush=True)
    if not all(loaded):
        print("部分上游结果尚缺；对应章节已明确标记等待，不使用示例数值。", flush=True)
    return REPORT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成自包含中文 BAF 分析报告")
    parser.add_argument("--wait-seconds", type=float, default=0, help="最多等待上游 JSON 生成的秒数")
    parser.add_argument("--require-complete", action="store_true", help="缺少必要结果时失败，不生成不完整报告")
    args = parser.parse_args()
    run(args.wait_seconds, args.require_complete)
