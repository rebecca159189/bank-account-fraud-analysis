"""Section 04 of the HTML report: feature analysis and feature engineering (reads outputs/feature_study.json)."""
from .report import table, figure, pending, pct, number, e, detail


def _pp(v):
    return "—" if v is None else f"{100*v:+.2f} 个百分点"


def _ci(v, scale=100, unit=" 个百分点", digits=2):
    return "—" if not v else f"[{scale*v[0]:+.{digits}f}, {scale*v[1]:+.{digits}f}]{unit}"


def feature_section(fs: dict) -> str:
    parts = (fs or {}).get('parts', {})
    if not parts:
        return pending("特征分析尚未生成：执行 python -m src.feature_study 后重新生成报告。")
    out = ['<p class="note">本节在看过原测试月之后加做，属于探索：分箱、编码、筛选规则和早停只用月 0–6，'
           '月 7 只报告一次；冻结模型和第 03、05、06 节的数字都没有改动。</p>']
    p = parts.get('iv')
    if p:
        t = p['table']
        lv = p.get('level_counts', {})
        out.append('<h3>A. WOE / IV：每个字段单独有多少信号、会不会漂移</h3>')
        out.append('<p>每个字段在训练月（0–5 月）上分箱：连续字段按分位数分 10 箱，取值少的字段每个取值一箱，缺失单独一箱。'
                   '每箱的 <strong>WOE = ln(该箱欺诈占全部欺诈的比例 ÷ 该箱正常占全部正常的比例)</strong>，正值表示比平均更危险；'
                   '<strong>IV = Σ(欺诈占比 − 正常占比) × WOE</strong>，衡量字段单独区分欺诈的能力。'
                   '同一套分箱再算训练月与第 6 月之间的 PSI，衡量稳定性。</p>')
        out.append(f'<p>{number(p["n_fields"])} 个字段中：' + '、'.join(f'{k} {v} 个' for k, v in lv.items()) +
                   f'。按事先定下的规则（IV &lt; {p["iv_min"]:g} 或 PSI &gt; {p["psi_max"]:g} 列为剔除候选），候选字段为：'
                   + ('、'.join(f'<code>{e(f)}</code>' for f in p['dropped_by_rule']) or '无') + '。</p>')
        rows = [[f'<code>{e(r["field"])}</code>', number(r['iv'], 3), e(r['level']), number(r['psi_train_vs_month6'], 3),
                 e(r['riskiest_bin']), pct(r['riskiest_bin_fraud_rate']), '保留' if r['keep_by_rule'] else '剔除候选'] for r in t]
        out.append(table(['字段', 'IV', 'IV 等级', 'PSI（月 0–5 对月 6）', '风险最高的箱', '该箱欺诈率', '规则结论'], rows[:12],
                         'IV 最高的 12 个字段（训练月）', numeric_from=1))
        out.append(detail('查看全部字段', table(['字段', 'IV', 'IV 等级', 'PSI（月 0–5 对月 6）', '风险最高的箱', '该箱欺诈率', '规则结论'],
                                                 rows, numeric_from=1)))
        out.append(figure('fs_iv_psi.png', '左：各字段 IV（红色为规则剔除候选）；右：IV 与 PSI 的散点，右下方是“有信号又稳定”的字段。'))
        out.append(figure('fs_woe_top.png', 'IV 最高的 4 个字段：各箱 WOE（红色表示比平均更危险）。'))
        out.append('<p class="note">IV 是单变量指标：字段之间的组合作用和树模型能捕捉的交互看不到，所以 IV 低不等于对模型没用，'
                   '需要下面的重训实验来检验。IV 很高（&gt; 0.5）在信用评分里通常要核查是否泄漏；这里的类别字段是数据本身的强信号。</p>')
    p = parts.get('selection')
    if p:
        out.append('<h3>B. 按 IV / PSI 规则删字段，重训会怎样</h3>')
        d = p["reference_reproduces_frozen_max_abs"]
        out.append(f'<p>用冻结 LightGBM 的同一配方重训。“同配置重训”与冻结模型的分数最大差为 {d:.1e}，'
                   + ('说明重训流程逐位复现了冻结模型，下面的差别只来自删掉的字段。</p>' if d < 1e-9 else
                      '<strong>没有逐位复现冻结模型</strong>，下面的比较应以“同配置重训”为基线。</p>'))
        out.append(table(['方案', '字段数', '验证 AP', '测试 AP', '召回率（冻结阈值）', 'FPR', '同审核量召回率'],
                         [[e(r['label']), number(r.get('n_features')), number(r['val_ap'], 4), number(r['test_ap'], 4),
                           pct(r['test_recall']), pct(r['test_fpr']), pct(r.get('test_same_budget_recall'))] for r in p['rows']],
                         '阈值均由月 6 的 FPR ≤ 5% 确定；月 7 只报告', numeric_from=1))
    p = parts.get('derived')
    if p:
        gl = p['group_labels']
        out.append('<h3>C. 衍生特征：三组新特征能不能提升</h3>')
        out.append('<p>BAF 没有客户 ID 和月内时间戳，做不了“同一客户过去 N 天”这类历史聚合，只能在单条申请内部构造。三组共 '
                   f'{sum(len(v) for v in p["groups"].values())} 个：</p><ul>'
                   f'<li><strong>{e(gl["ratios"])}</strong>：<code>velocity_6h / velocity_24h</code>、<code>velocity_24h / velocity_4w</code>。原始 velocity 字段随月份明显漂移，比率可能抵消整体趋势。</li>'
                   f'<li><strong>{e(gl["consistency"])}</strong>：缺失字段个数、有效电话个数、免费邮箱 × 姓名邮箱不相似度、申请额度 / 收入、开户月数 / 年龄、两段地址月数之和、同生日邮箱数 / 同设备邮箱数。</li>'
                   f'<li><strong>{e(gl["encodings"])}</strong>：5 个类别字段的出现频率，以及平滑目标编码（平滑强度 {number(p["te_smoothing"])}；训练行用 {number(p["te_folds"])} 折交叉拟合，避免看到自己的标签）。</li></ul>')
        rows = []
        for r in p['rows']:
            rows.append([e(r['label']) + (' ✓' if r.get('chosen_on_month6') else ''), number(r.get('n_new')),
                         number(r['val_ap'], 4), pct(r['val_recall']), number(r['test_ap'], 4), pct(r['test_recall']),
                         pct(r.get('test_same_budget_recall')),
                         _ci(r.get('vs_reference_ap_diff_ci95'), 1, '', 4) if r['variant'] != 'reference' else '—'])
        out.append(table(['方案', '新增列数', '验证 AP', '验证召回率', '测试 AP', '测试召回率', '同审核量召回率', '测试 AP 差 95% 区间'], rows,
                         '✓ 为按项目规则（验证召回率优先、AP 破平）在月 6 选出的方案；区间为月 7 配对 bootstrap', numeric_from=1))
        roll = p.get('rolling', [])
        if roll:
            out.append('<p>为了不只看一个月，再把“基线”和事先定好的“加入全部三组”放进 4 个向前滚动的窗口（每个窗口用前面的月训练、下一个月早停、再下一个月测试）：</p>')
            out.append(table(['测试月', '基线 AP', '加入全部 AP', 'AP 差', 'AP 差 95% 区间', '同审核量召回率差', '召回率差 95% 区间'],
                             [[number(w['test_month']), number(w['reference_test_ap'], 4), number(w['all_test_ap'], 4),
                               number(w['ap_diff'], 4), _ci(w['ap_diff_ci95'], 1, '', 4), _pp(w['same_budget_recall_diff']),
                               _ci(w['same_budget_recall_diff_ci95'])] for w in roll],
                             '同审核量按每个测试月 5% 计；区间为配对 bootstrap，未包含重训带来的波动', numeric_from=1))
        out.append(figure('fs_derived.png', '左：各组衍生特征的验证 / 测试 AP；右：4 个滚动窗口中“加入全部 − 基线”的 AP 差与 95% 区间。'))
    p = parts.get('importance')
    if p:
        rho = p['spearman']
        out.append('<h3>D. 三种重要性排名互相核对</h3>')
        out.append('<p>同一个问题“哪个字段重要”有三种量法：<strong>置换重要性</strong>（在月 6 上把一个字段打乱，看冻结 LightGBM 的 AP 掉多少）、'
                   '<strong>TreeSHAP</strong>（第 05 节，平均贡献大小）、<strong>IV</strong>（单变量信号）。'
                   f'排名的 Spearman 相关：置换 vs SHAP {number(rho["perm_ap_drop_vs_shap_mean_abs"], 2)}，置换 vs IV {number(rho["perm_ap_drop_vs_iv"], 2)}，'
                   f'SHAP vs IV {number(rho["shap_mean_abs_vs_iv"], 2)}。三种方法前 10 名的交集：'
                   + ('、'.join(f'<code>{e(f)}</code>' for f in p['top10_all_three']) or '无') + '。</p>')
        out.append(table(['字段', '置换：AP 下降', '置换排名', 'SHAP 排名', 'IV 排名'],
                         [[f'<code>{e(r["field"])}</code>', number(r['perm_ap_drop'], 4), number(r['perm_rank']),
                           number(r['shap_rank']), number(r['iv_rank'])] for r in p['table'][:12]],
                         '置换重要性最高的 12 个字段', numeric_from=1))
        out.append(figure('fs_importance.png', '三种重要性并排：置换（月 6 AP 下降）、TreeSHAP、IV。'))
        out.append('<p class="note">三者回答的问题不同：IV 看单个字段，SHAP 看模型怎么用它，置换看拿掉它模型会损失多少。'
                   '相关字段之间会“互相顶替”，所以置换重要性会低估相关字段组中的每一个。</p>')
    out.append('<p>复核：<a href="feature_study.json">完整结果</a>、<a href="tables/">fs_*.csv 数据表</a>。</p>')
    return ''.join(out)
