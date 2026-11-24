"""Section 09 of the HTML report: the course-method supplement (reads outputs/coursework.json)."""
from .report import table, figure, pending, pct, number, e, detail


def _scale_rows(rows):
    return [[e(r['label']), number(r['test_ap'], 4), pct(r['test_recall']), pct(r['test_fpr']),
             pct(r.get('test_same_budget_recall'))] for r in rows]


def coursework_section(cw: dict) -> str:
    parts = (cw or {}).get('parts', {})
    if not parts:
        return pending("课程方法补充尚未生成：执行 python -m src.coursework 后重新生成报告。")
    out = ['<p class="note">本节在看过原测试月之后设计，属于探索性补充：冻结的模型、阈值与第 01–08 节的数字都没有改动。'
           '所有重训只用月 0–5 拟合、月 6 选阈值，月 7 只用来报告（随机切分对照是例外：它把月 0–6 打乱）。'
           '每一小节标明对应的课程章节。</p>']
    # ---- A skew / scaling
    p = parts.get('skew_scale')
    if p:
        out.append('<h3>A 偏度、幂变换与缩放器（CA6003 第 2 周，CA6000 Topic 11）</h3>')
        out.append(f'<p>{number(p["n_continuous"])} 个连续字段中，{number(len(p["highly_skewed"]))} 个的 |偏度| 超过 {p["skew_limit"]:g}。'
                   '下表在逻辑回归上换不同缩放或变换（其余设置与冻结的逻辑回归相同），并让 LightGBM 对偏斜字段取 log1p。'
                   '看每一行与"冻结参考"的差，而不是绝对数值。</p>')
        out.append(table(['方法', '测试 AP', '召回率（冻结阈值）', 'FPR', '相同审核量召回率'], _scale_rows(p['rows']),
                         '月 7 结果；阈值均由月 6 的 FPR ≤ 5% 确定', numeric_from=1))
        out.append(figure('cw_skewness.png', '左：训练月各连续字段的偏度（红色为 |偏度|>1）；中右：偏度最大且可取对数的字段，变换前后对比。'))
    # ---- B outliers
    p = parts.get('outliers')
    if p:
        rl = p['row_level']
        out.append('<h3>B 离群值：保留还是删除（CA6003 第 1、4 周）</h3>')
        out.append(f'<p>训练月中 {pct(rl["share_rows_any_iqr3"])} 的行在某个字段上超出 3×IQR 围栏；这些行的欺诈率为 {pct(rl["fraud_rate_any_iqr3"])}，'
                   f'其余行为 {pct(rl["fraud_rate_none"])}（整体 {pct(rl["base_rate"])}）。离群值里的欺诈浓度是否更高，决定了该不该删。</p>')
        out.append(table(['方法', '测试 AP', '召回率（冻结阈值）', 'FPR', '相同审核量召回率'], _scale_rows(p['rows']), '月 7 结果', numeric_from=1))
        out.append(figure('cw_outlier_lift.png', '各字段 3×IQR 离群行内外的欺诈率之比（>1 表示离群行更像欺诈）。'))
    # ---- C distributions
    p = parts.get('distributions')
    if p:
        out.append('<h3>C 分布、相关与多重共线性（CA6002 2.1，CA6003 第 4 周）</h3>')
        out.append(table(['字段', '欺诈中位数', '正常中位数', '标准化均值差', 'KS 统计量'],
                         [[f'<code>{e(r["field"])}</code>', number(r['median_fraud'], 3), number(r['median_normal'], 3),
                           number(r['standardized_mean_diff'], 3), number(r['ks_statistic'], 3)] for r in p['top_ks']],
                         'KS 统计量最大的 5 个连续字段（训练月，欺诈 vs 正常）', numeric_from=1))
        pairs = p['strong_pairs']
        out.append(f'<p>连续字段两两 Spearman 相关中，|ρ| ≥ 0.7 的有 {number(len(pairs))} 对（共 {number(p["n_pairs"])} 对）；'
                   f'VIF 最大值 {number(p["vif_max"], 2)}，VIF > 5 的字段有 {number(len(p["vif_over_5"]))} 个。</p>')
        out.append(figure('cw_distribution_by_label.png', '欺诈与正常申请在 KS 最大的 6 个字段上的分布（小提琴图，已按 1–99 分位截断显示）。'))
        out.append(figure('cw_correlation_heatmap.png', '连续字段的 Spearman 相关热力图（20 万行训练样本）。'))
    # ---- D split
    p = parts.get('split')
    if p:
        rr = {('random' if '随机' in r['setting'] else 'time'): r for r in p['rows']}
        cv = p['cv']
        out.append('<h3>D 随机切分与时间切分（CA6001 第 1 章，CA6003 第 4 周）</h3>')
        if 'random' in rr and 'time' in rr:
            out.append(f'<p>同一套 LightGBM 配方：月 0–6 打乱后随机留出估计的 AP 为 {number(rr["random"]["est_ap"], 3)}，'
                       f'5 折交叉验证 {number(cv["ap_mean"], 3)} ± {number(cv["ap_std"], 3)}；时间切分用月 6 估计的 AP 为 {number(rr["time"]["est_ap"], 3)}；'
                       f'而月 7 的真实 AP 约为 {number(rr["time"]["month7_ap"], 3)}。两种估计都低于月 7，主要因为欺诈率逐月上升（AP 随基础率升高），'
                       '所以这里看不出“随机切分更乐观”；它提醒的是月与月之间的变化比切分方式影响更大，这正是项目坚持按时间评估的原因。</p>')
    # ---- G smote
    p = parts.get('smote')
    if p:
        out.append('<h3>E SMOTE、类别权重与不处理（CA6003 第 2–3 周）</h3>')
        out.append(table(['方法', '测试 AP', '召回率（冻结阈值）', 'FPR', '平均预测分数', '按 0.5 判定的准确率'],
                         [[e(r['label']), number(r['test_ap'], 4), pct(r['test_recall']), pct(r['test_fpr']),
                           number(r['test_mean_score'], 4), pct(r['test_accuracy_at_0_5'])] for r in p['rows']],
                         f'月 7 结果；真实欺诈率 {pct(p["rows"][0]["test_base_rate"])}。SMOTE 为普通 SMOTE，会对独热列和缺失标记列也做插值（不是 SMOTE-NC）', numeric_from=1))
        out.append(figure('cw_imbalance_methods.png', '左：排序质量（AP）；右：平均预测分数。平均分数远高于真实欺诈率，说明经过加权或过采样后分数不再是概率。'))
    # ---- H docs
    p = parts.get('docs')
    if p:
        out.append('<h3>F 模型卡、字段字典与数据契约（CA6003 第 5 周）</h3>')
        bad = p['contract_fields_with_violations']
        out.append('<p>已生成 <code>docs/模型卡_LightGBM.md</code> 与 <code>docs/特征说明_字段字典.md</code>；数字由冻结产物自动写入。'
                   f'契约核对：全部 100 万行中，{number(len(bad))} 个字段有取值超出官方数据说明给出的范围（共 {number(p["contract_total_violations"])} 个取值）'
                   + (f'：{", ".join(f"<code>{e(c)}</code>" for c in bad)}。越界幅度和原因需要先查清，再决定放宽契约还是当作脏数据；官方范围可能只是写得偏窄。' if bad else '，没有越界。') + '</p>')
    out.append('<p>复核：<a href="coursework.json">完整结果</a>、<a href="tables/">cw_*.csv 数据表</a>。</p>')
    return ''.join(out)
