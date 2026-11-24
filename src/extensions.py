"""Post-hoc supplementary analyses, added AFTER the frozen month-7 evaluation was seen.

What this stage answers
  A. Rule complexity: how many leaves does a distilled rule tree need before it
     stops degenerating and gets close to the LightGBM teacher?
  B. Age fairness in context: base fraud rates per group, a no-age control
     model, how much age a model can still infer from other fields, and
     per-group thresholds as a post-processing comparison.
  C. Velocity drift: velocity_* fields fall month after month (a hidden time
     index). Do models without them transfer better or worse?

Discipline (same as the main pipeline)
  * Frozen artifacts (models/, evaluation.json, explanation.json, predictions/
    test.parquet & validation.parquet) are only READ. Their SHA-256 is recorded
    before and after this stage so verify.py can prove nothing was modified.
  * Every new fit uses months 0-5 only; every new choice (early stopping,
    thresholds, recommended leaf count) uses month 6 only; month 7 is reported.
  * Because these questions were formulated after month-7 results were known,
    they are exploratory. They do not replace or re-select the frozen winner.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(key, '4')
import json
import argparse
import platform
import sys
import time
import warnings
from datetime import datetime, timezone, timedelta
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits
from .common import *
from .evaluate import style
from .eda import (calculate_psi, fit_psi_bins, clean_for_description,
                  PSI_LOW_CARDINALITY_MAX, CATEGORICAL as EDA_CATEGORICAL)

OUT = ROOT/'outputs'; MODELS = OUT/'models'; FIG = OUT/'figures'; TABLES = OUT/'tables'
EXT_MODELS = OUT/'extension_models'
FROZEN_FILES = ['outputs/models/frozen_manifest.json', 'outputs/models/lightgbm.joblib',
                'outputs/models/distilled_tree.joblib', 'outputs/models/preprocessor.joblib',
                'outputs/models/schema.json', 'outputs/models/logistic_regression.joblib',
                'outputs/models/mlp.joblib', 'outputs/models/isolation_forest.joblib',
                'outputs/evaluation.json', 'outputs/explanation.json',
                'outputs/predictions/validation.parquet', 'outputs/predictions/test.parquet']
LEAF_GRID = [5, 8, 12, 16, 24, 32, 48, 64]
RULE_TARGET = 0.90          # recommended rules: >=90% of the teacher's validation recall
BOOT = 500
VARIANTS = {
    'reference_refit': [],
    'no_age': ['customer_age'],
    'no_velocity_short': ['velocity_6h', 'velocity_24h'],
    'no_velocity_all': ['velocity_6h', 'velocity_24h', 'velocity_4w'],
}
VARIANT_NAMES = {'reference_refit': '同配置重训（含全部特征）', 'no_age': '去掉 customer_age',
                 'no_velocity_short': '去掉 velocity_6h / 24h', 'no_velocity_all': '去掉全部 velocity_*'}


def log(s): print(time.strftime('%H:%M:%S'), s, flush=True)


def strict_json(x):
    """NaN/inf are not valid JSON; store them as null so strict parsers accept the file."""
    if isinstance(x, dict): return {k: strict_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [strict_json(v) for v in x]
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer): return int(x)
    return x


def frozen_hashes():
    return {p: sha256(ROOT/p) for p in FROZEN_FILES}


def same_budget(scores, k, row_ids, y):
    flags = top_k(scores, k, row_ids)
    return flags, binary_metrics(y, flags)


def paired_recall_boot(a, b, y, seed=SEED, reps=BOOT):
    """Recall(a) - recall(b) on the same frauds; resample frauds with replacement."""
    pos = np.where(np.asarray(y) == 1)[0]
    d = a[pos].astype(float) - b[pos].astype(float)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(pos), size=(reps, len(pos)))].mean(axis=1)
    return float(d.mean()), np.quantile(boot, [.025, .975]).tolist()


def paired_ap_boot(sa, sb, y, seed=SEED, reps=BOOT):
    """AP(a) - AP(b); stratified bootstrap keeps the fraud/normal counts fixed."""
    y = np.asarray(y); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs.append(average_precision_score(y[idx], sa[idx]) - average_precision_score(y[idx], sb[idx]))
    return float(average_precision_score(y, sa) - average_precision_score(y, sb)), np.quantile(diffs, [.025, .975]).tolist()


def select_rule_size(table, teacher_validation_recall, target_fraction=RULE_TARGET):
    """Validation-only rule selection, with an explicit unmet-target fallback."""
    target_recall = float(target_fraction * teacher_validation_recall)
    eligible = table[table.val_whole_recall >= target_recall]
    target_met = len(eligible) > 0
    if target_met:
        selected = eligible.sort_values('leaves').iloc[0]
    else:
        selected = table.sort_values(['val_whole_recall', 'leaves'], ascending=[False, True]).iloc[0]
    chosen = int(selected['leaves'])
    achieved = float(selected['val_whole_recall'])
    metadata = dict(
        selection_rule='fewest leaves meeting the validation target; if none, highest validation recall, then fewest leaves',
        target_fraction_of_teacher=float(target_fraction), target_validation_recall=target_recall,
        target_met=bool(target_met), fallback_used=bool(not target_met),
        fallback_reason=None if target_met else 'No candidate reached the validation recall target within the declared leaf grid.',
        achieved_fraction_of_teacher=achieved/teacher_validation_recall if teacher_validation_recall else None,
        selection_split='validation month 6', test_used_for_selection=False)
    return chosen, metadata


def drift_tables(df):
    """Descriptive PSI refresh, independent of all model fitting and labels."""
    train_rows = (df.month <= 5).to_numpy()
    vel_cols = ['velocity_6h', 'velocity_24h', 'velocity_4w']
    grouped = df.groupby('month')[vel_cols]
    velocity = pd.concat([grouped.median().stack().rename('median'), grouped.mean().stack().rename('mean'),
                          grouped.quantile(.1).stack().rename('p10'), grouped.quantile(.9).stack().rename('p90')], axis=1)
    velocity = velocity.rename_axis(['month', 'feature']).reset_index()
    velocity['month'] = velocity.month.astype(int)
    psi = pd.DataFrame([dict(month=m, feature=f, psi=calculate_psi(df.loc[train_rows, f], df.loc[(df.month == m).to_numpy(), f], False)[0])
                        for f in vel_cols for m in range(8)])
    clean = clean_for_description(df.drop(columns=['fraud_bool']))
    for c in EDA_CATEGORICAL:
        clean[c] = clean[c].astype(object)
    fields = [c for c in clean.columns if c not in ('month', 'device_fraud_count')]
    test_rows = (clean.month == 7).to_numpy()
    all_psi_rows = []
    for feature in fields:
        reference = clean.loc[train_rows, feature]; current = clean.loc[test_rows, feature]
        value, bins = calculate_psi(reference, current, feature in EDA_CATEGORICAL)
        all_psi_rows.append(dict(month=7, feature=feature, psi=value, bins_including_missing=bins,
                                 binning=fit_psi_bins(reference, feature in EDA_CATEGORICAL)['kind']))
    psi_all = pd.DataFrame(all_psi_rows).sort_values('psi', ascending=False).reset_index(drop=True)
    eda_psi = pd.read_csv(TABLES/'eda_monthly_psi.csv', encoding='utf-8-sig')
    both = pd.concat([psi, psi_all[~psi_all.feature.str.startswith('velocity_')]]).merge(
        eda_psi[['month', 'feature', 'psi']], on=['month', 'feature'], suffixes=('', '_eda'))
    psi_check = dict(rows_compared=len(both), max_abs_diff=float((both.psi-both.psi_eda).abs().max()) if len(both) else None)
    velocity = velocity.merge(psi, on=['month', 'feature'], how='left')
    summary = dict(monthly_medians={f: velocity[velocity.feature == f].sort_values('month')['median'].round(1).tolist()
                                    for f in sorted(velocity.feature.unique())},
                   psi_consistency_with_eda=psi_check,
                   psi_method=dict(version=2, numeric_discrete_max_values=PSI_LOW_CARDINALITY_MAX,
                                   bin_fitting='training rows only', pseudocount=0.5,
                                   discrete='per-training-value bins plus unseen/missing bins', continuous='training deciles plus missing bin'),
                   test_month_psi_all_fields=psi_all.drop(columns='month').to_dict(orient='records'),
                   spearman_with_month={f: float(spearmanr(df.month, df[f]).statistic) for f in vel_cols})
    return velocity, summary


def refresh_descriptive():
    """Refresh PSI and selection metadata without training or changing predictions.

    Historical models that were never saved cannot be reconstructed from scores;
    this mode states that model persistence still requires a complete extension run.
    """
    before = frozen_hashes()
    output_path = OUT/'extensions.json'
    out = json.loads(output_path.read_text(encoding='utf-8'))
    df = load_data()
    velocity, velocity_summary = drift_tables(df)
    curve = pd.DataFrame(out['rule_curve'])
    chosen, selection = select_rule_size(curve, out['rules']['teacher_validation_recall'])
    if chosen != out['rules']['chosen_leaves']:
        raise ValueError('Metadata-only refresh would change the selected model; run a full extension stage instead.')
    out['rules'].update(selection)
    out['velocity'] = velocity_summary
    out['velocity_monthly'] = velocity.to_dict(orient='records')
    velocity.to_csv(TABLES/'ext_velocity_monthly.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(velocity_summary['test_month_psi_all_fields']).to_csv(TABLES/'ext_psi_all_fields_test.csv', index=False, encoding='utf-8-sig')
    after = frozen_hashes()
    if before != after:
        raise ValueError('A frozen artifact changed during the descriptive refresh.')
    out['descriptive_revision'] = dict(created_singapore=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                                       mode='PSI and selection metadata only; no fitting or prediction changes',
                                       frozen_hashes_before=before, frozen_hashes_after=after,
                                       model_persistence='available' if 'artifacts' in out else 'pending a complete extension run')
    save_json(strict_json(out), output_path)
    log('Descriptive refresh complete; original predictions and fitted models were unchanged.')


def fit_lgbm(xtr, ytr, xv, yv, num_leaves, reg_lambda):
    """Exactly the training call used in src/train.py, so the refit can be checked."""
    ratio = float((ytr == 0).sum()/(ytr == 1).sum())
    model = lgb.LGBMClassifier(n_estimators=900, learning_rate=.04, num_leaves=num_leaves,
        min_child_samples=150, reg_lambda=reg_lambda, colsample_bytree=.9, subsample=.9,
        subsample_freq=1, scale_pos_weight=ratio, n_jobs=4, deterministic=True,
        force_col_wise=True, random_state=SEED, verbosity=-1, metric='average_precision')
    with warnings.catch_warnings():
        # LightGBM 4.7 renamed eval_set -> eval_X/eval_y; behaviour is identical.
        warnings.simplefilter('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', message='.*eval_set.*')
        model.fit(xtr, ytr, eval_set=[(xv, yv)], eval_metric='average_precision',
                  callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False)])
    return model


def group_metrics(y, pred, age, split, label):
    rows = []
    for name, mask in [('age<50', age < 50), ('age>=50', age >= 50)]:
        m = binary_metrics(y[mask], pred[mask]); lo, hi = wilson(m['fp'], m['fp']+m['tn'])
        rows.append(dict(setting=label, split=split, group=name, base_fraud_rate=m['fraud']/m['n'],
                         **m, fpr_ci_low=lo, fpr_ci_high=hi))
    return rows


def fpr_ratio(rows):
    g = {r['group']: r for r in rows}
    young, old = g['age<50']['fpr'], g['age>=50']['fpr']
    return dict(fpr_older=old, fpr_younger=young, fpr_gap=old-young, fpr_ratio=old/young if young else None)


# ---------------------------------------------------------------- readable rules
def rule_translator(prep):
    features = list(prep.get_feature_names_out()); numeric = list(prep.transformers_[0][2])
    scaler = prep.named_transformers_['num'].named_steps['scale']
    medians = dict(zip(numeric, prep.named_transformers_['num'].named_steps['impute'].statistics_))

    def condition(fid, threshold, left):
        name = features[fid]
        if fid < len(numeric):
            t = threshold*scaler.scale_[fid] + scaler.mean_[fid]
            if name.endswith('__missing'):
                return name.removesuffix('__missing'), name.removesuffix('__missing') + (' 非缺失' if left else ' 缺失')
            text = f"{name} {'<=' if left else '>'} {t:.6g}"
            if name in MISSING_FIELDS:
                text += f'（缺失时按训练中位数 {medians[name]:.6g} 判断）'
            return name, text
        for c in CATEGORICAL:
            if name.startswith(c+'_'):
                return c, f"{c} {'不等于' if left else '等于'} {name[len(c)+1:]}"
        return name, f"{name} {'<=' if left else '>'} {threshold:.6g}"

    def paths(tree):
        t = tree.tree_; out = {}
        def walk(node, parts, feats):
            if t.children_left[node] == t.children_right[node]:
                out[node] = (parts, feats); return
            f, text_l = condition(t.feature[node], t.threshold[node], True)
            _, text_r = condition(t.feature[node], t.threshold[node], False)
            walk(t.children_left[node], parts+[text_l], feats | {f})
            walk(t.children_right[node], parts+[text_r], feats | {f})
        walk(0, [], frozenset())
        return out
    return paths


# ---------------------------------------------------------------- A. rule curve
def rule_complexity(ztr, zv, zt, teacher_train, teacher_val, teacher_test, yv, yt, ids_v, ids_t,
                    budget, teacher_val_recall, frozen_thr_lgbm, sv_teacher, st_teacher, paths_of):
    kv = int(np.floor(budget*len(yv))); kt = int(np.floor(budget*len(yt)))
    _, teacher_same_test = same_budget(st_teacher, kt, ids_t, yt)
    rows = []; trees = {}; validation_scores = {}; test_scores = {}
    for leaves in LEAF_GRID:
        start = time.perf_counter()
        tree = DecisionTreeRegressor(max_leaf_nodes=leaves, min_samples_leaf=.005, random_state=SEED)
        tree.fit(ztr, teacher_train); trees[leaves] = tree
        sv = tree.predict(zv); st = tree.predict(zt)
        validation_scores[leaves] = sv; test_scores[leaves] = st
        thr = threshold_at_fpr(yv, sv)                       # whole-leaf rule: no partial leaf
        whole_v = binary_metrics(yv, sv >= thr); whole_t = binary_metrics(yt, st >= thr)
        _, same_v = same_budget(sv, kv, ids_v, yv); flags_t, same_t = same_budget(st, kt, ids_t, yt)
        p = paths_of(tree); leaf_values = {n: tree.tree_.value[n, 0, 0] for n in p}
        alerted = [n for n in p if leaf_values[n] >= thr]
        conds = [len(p[n][0]) for n in alerted]
        rows.append(dict(leaves=leaves, actual_leaves=int(tree.get_n_leaves()), depth=int(tree.get_depth()),
            threshold=float(thr), alerted_leaves=len(alerted),
            mean_conditions_alerted=float(np.mean(conds)) if conds else 0.,
            max_conditions_alerted=int(max(conds)) if conds else 0,
            features_used=len(set().union(*[p[n][1] for n in p])),
            val_whole_recall=whole_v['recall'], val_whole_fpr=whole_v['fpr'], val_whole_alerts=whole_v['alerts'],
            test_whole_recall=whole_t['recall'], test_whole_fpr=whole_t['fpr'], test_whole_alerts=whole_t['alerts'],
            test_whole_precision=whole_t['precision'],
            val_same_budget_recall=same_v['recall'], test_same_budget_recall=same_t['recall'],
            test_same_budget_tp=same_t['tp'],
            val_spearman=float(spearmanr(teacher_val, sv).statistic), test_spearman=float(spearmanr(teacher_test, st).statistic),
            val_r2=float(r2_score(teacher_val, sv)), test_r2=float(r2_score(teacher_test, st)),
            test_ap=float(average_precision_score(yt, st)), fit_seconds=time.perf_counter()-start))
        log(f'rules leaves={leaves}: val whole-leaf recall={whole_v["recall"]:.3f}, test={whole_t["recall"]:.3f}, '
            f'test same-budget={same_t["recall"]:.3f}, spearman={rows[-1]["test_spearman"]:.3f}')
    table = pd.DataFrame(rows)
    # Validation-only choice: fewest leaves whose deployable whole-leaf recall reaches 90% of the teacher.
    chosen, selection = select_rule_size(table, teacher_val_recall)
    tree = trees[chosen]; sv = tree.predict(zv); st = tree.predict(zt); thr = threshold_at_fpr(yv, sv)
    p = paths_of(tree); leaf_t = tree.apply(zt); leaf_v = tree.apply(zv); leaf_tr = tree.apply(ztr)
    rules = []
    for rank, node in enumerate(sorted(p, key=lambda n: tree.tree_.value[n, 0, 0], reverse=True), 1):
        mt = leaf_t == node; mv = leaf_v == node
        rules.append(dict(rule_id=f'L{chosen}-R{rank}', teacher_log_odds=float(tree.tree_.value[node, 0, 0]),
            alert=bool(tree.tree_.value[node, 0, 0] >= thr), conditions=len(p[node][0]),
            path=' 且 '.join(p[node][0]), train_rows=int((leaf_tr == node).sum()),
            validation_rows=int(mv.sum()), validation_fraud_rate=float(yv[mv].mean()) if mv.any() else None,
            test_rows=int(mt.sum()), test_fraud=int(yt[mt].sum()),
            test_fraud_rate=float(yt[mt].mean()) if mt.any() else None))
    whole_t = binary_metrics(yt, st >= thr); whole_v = binary_metrics(yv, sv >= thr)
    teacher_whole_t = binary_metrics(yt, st_teacher >= frozen_thr_lgbm)
    flags_rule, _ = same_budget(st, kt, ids_t, yt); flags_teacher, _ = same_budget(st_teacher, kt, ids_t, yt)
    gap, ci = paired_recall_boot(flags_teacher, flags_rule, yt)
    summary = dict(leaf_grid=LEAF_GRID, **selection,
        teacher_validation_recall=teacher_val_recall, chosen_leaves=chosen, chosen_threshold=float(thr),
        chosen_validation=dict(whole_v, ap=float(average_precision_score(yv, sv))),
        chosen_test=dict(whole_t, ap=float(average_precision_score(yt, st)),
                         recall_ci=wilson(whole_t['tp'], whole_t['tp']+whole_t['fn']),
                         fpr_ci=wilson(whole_t['fp'], whole_t['fp']+whole_t['tn'])),
        teacher_test_frozen=teacher_whole_t, teacher_test_same_budget=teacher_same_test,
        same_budget_recall_gap_teacher_minus_rules=gap, same_budget_gap_bootstrap95=ci, test_k=kt,
        note='Whole-leaf rules alert every row in an alerted leaf; no row-hash sampling is needed for this operating point.')
    artifacts = dict(models=trees, chosen_model=tree, validation_scores=validation_scores, test_scores=test_scores)
    return table, pd.DataFrame(rules), summary, artifacts


# ---------------------------------------------------------------- figures
def plot_rules(table, summary):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(table.leaves, table.val_whole_recall, 'o--', color='#577590', label='Validation: whole-leaf rules (FPR<=5%)')
    ax.plot(table.leaves, table.test_whole_recall, 'o-', color='#147d92', label='Test: same whole-leaf rules')
    ax.plot(table.leaves, table.test_same_budget_recall, 's-', color='#c8862d', label=f'Test: exact K={summary["test_k"]:,} alerts')
    ax.axhline(summary['teacher_test_frozen']['recall'], color='#147d92', ls=':', lw=1.3, label='LightGBM teacher, test (frozen threshold)')
    ax.axhline(summary['teacher_test_same_budget']['recall'], color='#c8862d', ls=':', lw=1.3, label='LightGBM teacher, test (exact K)')
    ax.axvline(summary['chosen_leaves'], color='#999999', lw=1)
    ax.text(summary['chosen_leaves']*.97, .86, f'chosen on validation: {summary["chosen_leaves"]} leaves ', color='#555555', fontsize=9, ha='right')
    ax.set_xscale('log', base=2); ax.set_xticks(LEAF_GRID, [str(x) for x in LEAF_GRID])
    ax.set(xlabel='Number of leaves (rules)', ylabel='Recall', ylim=(0, .9),
           title='How many rules are needed? Recall vs. rule-tree size')
    ax.grid(alpha=.2); ax.legend(fontsize=8, loc='upper left', ncol=2, frameon=False); fig.tight_layout()
    fig.savefig(FIG/'ext_rule_complexity.png', dpi=170); plt.close(fig)


def plot_fairness(context, compare):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={'width_ratios': [1, 1.25]})
    t = context[(context.split == 'test') & (context.setting == 'global_threshold')].set_index('group')
    metrics_ = [('base_fraud_rate', 'Fraud base rate'), ('fpr', 'False positive rate'), ('precision', 'Precision'), ('recall', 'Recall')]
    pos = np.arange(len(metrics_)); w = .36
    for off, g, c in [(-w/2, 'age<50', '#577590'), (w/2, 'age>=50', '#c8862d')]:
        vals = [t.loc[g, m] for m, _ in metrics_]
        bars = axes[0].bar(pos+off, vals, w, color=c, label=g)
        for b, v in zip(bars, vals): axes[0].text(b.get_x()+b.get_width()/2, v+.008, f'{v:.1%}', ha='center', fontsize=8)
    axes[0].set_xticks(pos, [n for _, n in metrics_]); axes[0].set(title='Test month, LightGBM, one global threshold', ylim=(0, .85))
    axes[0].legend(); axes[0].grid(axis='y', alpha=.2)
    labels = list(compare.setting_label.unique()); pos = np.arange(len(labels))
    for off, g, c in [(-w/2, 'age<50', '#577590'), (w/2, 'age>=50', '#c8862d')]:
        s = compare[compare.group == g].set_index('setting_label').loc[labels]
        err = np.vstack([s.fpr-s.fpr_ci_low, s.fpr_ci_high-s.fpr])
        axes[1].bar(pos+off, s.fpr, w, color=c, yerr=err, capsize=3, label=g)
    axes[1].axhline(.05, color='#777777', ls='--', lw=1); axes[1].set_xticks(pos, labels, fontsize=8)
    axes[1].set(title='Test FPR by age group: five ways to decide', ylabel='FPR'); axes[1].grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(FIG/'ext_fairness_context.png', dpi=170); plt.close(fig)


def plot_drift(velocity, ablation):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for f, c in [('velocity_6h', '#c4584a'), ('velocity_24h', '#147d92'), ('velocity_4w', '#577590')]:
        s = velocity[velocity.feature == f].sort_values('month')
        if len(s): axes[0].plot(s.month, s['median']/s['median'].iloc[0], 'o-', color=c, label=f)
    axes[0].axvspan(-.3, 5.5, color='#eeeeee')
    axes[0].set(xlabel='Relative month index (grey = training months)', ylabel='Median / month-0 median', title='velocity_* drift with time (a hidden time index)')
    axes[0].legend(); axes[0].grid(alpha=.2)
    a = ablation.set_index('variant'); order = list(a.index); pos = np.arange(len(order)); w = .38
    axes[1].bar(pos-w/2, a.test_ap, w, color='#577590', label='Test AP')
    axes[1].bar(pos+w/2, a.test_same_budget_recall, w, color='#c8862d', label='Test recall at exact K')
    for i, v in enumerate(a.test_fpr): axes[1].text(i, max(a.test_same_budget_recall.iloc[i], a.test_ap.iloc[i])+.015, f'FPR {v:.2%}', ha='center', fontsize=8)
    axes[1].set_xticks(pos, [o.replace('_', '\n') for o in order], fontsize=8.5)
    axes[1].set(title='LightGBM refits without selected fields', ylim=(0, .75)); axes[1].legend(fontsize=8.5); axes[1].grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(FIG/'ext_drift_ablation.png', dpi=170); plt.close(fig)


# ---------------------------------------------------------------- main
def run():
    before = frozen_hashes()
    EXT_MODELS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((MODELS/'frozen_manifest.json').read_text(encoding='utf-8'))
    schema = json.loads((MODELS/'schema.json').read_text(encoding='utf-8'))
    assert sha256(ROOT/'data/raw/Base.csv') == manifest['raw_sha256']
    lg = manifest['models']['lightgbm']; params = lg['params']
    log('Loading data and frozen artifacts (read only)')
    df = load_data(); tr = (df.month <= 5).to_numpy(); va = (df.month == 6).to_numpy(); te = (df.month == 7).to_numpy()
    xtr, _ = clean_features(df.loc[tr], schema); xv, _ = clean_features(df.loc[va], schema); xt, _ = clean_features(df.loc[te], schema)
    ytr, yv, yt = (df.loc[m, 'fraud_bool'].to_numpy() for m in (tr, va, te))
    age_v = df.loc[va, 'customer_age'].to_numpy(); age_t = df.loc[te, 'customer_age'].to_numpy()
    ids_v = df.index[va].to_numpy(); ids_t = df.index[te].to_numpy()
    prep = joblib.load(MODELS/'preprocessor.joblib'); teacher = joblib.load(MODELS/'lightgbm.joblib')
    val_pred = pd.read_parquet(OUT/'predictions/validation.parquet'); test_pred = pd.read_parquet(OUT/'predictions/test.parquet')
    assert np.array_equal(val_pred.row_id.to_numpy(), ids_v) and np.array_equal(test_pred.row_id.to_numpy(), ids_t)
    sv_main = val_pred.lightgbm.to_numpy(float); st_main = test_pred.lightgbm.to_numpy(float)
    thr_main = lg['threshold']; budget = manifest['review_budget_rate']
    kv = int(np.floor(budget*len(yv))); kt = int(np.floor(budget*len(yt)))
    main_flags_t, main_same_t = same_budget(st_main, kt, ids_t, yt)
    out = dict(stage='post-hoc supplementary analyses', created_singapore=datetime.now(timezone(timedelta(hours=8))).isoformat(),
               discipline=['frozen artifacts read only; hashes checked before/after',
                           'fit on months 0-5, every choice on month 6, month 7 reported only',
                           'designed after month-7 results were seen: exploratory, does not re-select the frozen winner'],
               frozen_hashes_before=before, budget_rate=budget, validation_k=kv, test_k=kt)
    prediction_frames = {
        'validation': pd.DataFrame({'row_id': ids_v, 'fraud_bool': yv, 'customer_age': age_v, 'frozen_lightgbm': sv_main}),
        'test': pd.DataFrame({'row_id': ids_t, 'fraud_bool': yt, 'customer_age': age_t, 'frozen_lightgbm': st_main}),
    }
    saved_models = {}

    with threadpool_limits(limits=4):
        # ---------------- A. rule complexity curve
        log('A. Rule complexity curve')
        ztr = prep.transform(xtr).astype('float32'); zv = prep.transform(xv).astype('float32'); zt = prep.transform(xt).astype('float32')
        teacher_train = teacher.predict(xtr, raw_score=True); teacher_val = teacher.predict(xv, raw_score=True); teacher_test = teacher.predict(xt, raw_score=True)
        curve, rules, rule_summary, rule_artifacts = rule_complexity(ztr, zv, zt, teacher_train, teacher_val, teacher_test, yv, yt,
            ids_v, ids_t, budget, lg['validation']['recall'], thr_main, sv_main, st_main, rule_translator(prep))
        out['rules'] = rule_summary
        for leaves, tree in rule_artifacts['models'].items():
            name = f'rules_{leaves}'
            path = EXT_MODELS/f'{name}.joblib'; joblib.dump(tree, path)
            saved_models[name] = path.relative_to(ROOT).as_posix()
            threshold = float(curve.loc[curve.leaves == leaves, 'threshold'].iloc[0])
            for split, ids, capacity in [('validation', ids_v, kv), ('test', ids_t, kt)]:
                scores = np.asarray(rule_artifacts[split+'_scores'][leaves], dtype=float)
                prediction_frames[split][name] = scores
                prediction_frames[split][name+'__alert_frozen'] = scores >= threshold
                prediction_frames[split][name+'__alert_topk'] = top_k(scores, capacity, ids).astype('int8')
        path = EXT_MODELS/'recommended_rules.joblib'; joblib.dump(rule_artifacts['chosen_model'], path)
        saved_models['recommended_rules'] = path.relative_to(ROOT).as_posix()
        for split in prediction_frames:
            name = f'rules_{rule_summary["chosen_leaves"]}'
            for suffix in ['', '__alert_frozen', '__alert_topk']:
                prediction_frames[split]['recommended_rules'+suffix] = prediction_frames[split][name+suffix]
        del rule_artifacts
        del ztr, zv, zt

        # ---------------- B/C. LightGBM refits without selected fields
        log('B/C. LightGBM refits (frozen hyper-parameters, validation early stopping and threshold)')
        ablation = []; variant_scores = {}; variant_thresholds = {}
        for variant, drop in VARIANTS.items():
            start = time.perf_counter()
            model = fit_lgbm(xtr.drop(columns=drop), ytr, xv.drop(columns=drop), yv, params['num_leaves'], params['reg_lambda'])
            sv = model.predict_proba(xv.drop(columns=drop))[:, 1]; st = model.predict_proba(xt.drop(columns=drop))[:, 1]
            thr = threshold_at_fpr(yv, sv); mt = metrics(yt, st, thr); mv = metrics(yv, sv, thr)
            path = EXT_MODELS/f'{variant}.joblib'; joblib.dump(model, path)
            saved_models[variant] = path.relative_to(ROOT).as_posix()
            for split, scores, ids, capacity in [('validation', sv, ids_v, kv), ('test', st, ids_t, kt)]:
                scores = np.asarray(scores, dtype=float)
                prediction_frames[split][variant] = scores
                prediction_frames[split][variant+'__alert_frozen'] = scores >= thr
                prediction_frames[split][variant+'__alert_topk'] = top_k(scores, capacity, ids).astype('int8')
            flags, same_t = same_budget(st, kt, ids_t, yt)
            row = dict(variant=variant, label=VARIANT_NAMES[variant], dropped=','.join(drop) or '—',
                       best_iteration=int(model.best_iteration_), threshold=thr,
                       val_ap=mv['pr_auc_ap'], val_recall=mv['recall'], test_ap=mt['pr_auc_ap'], test_roc_auc=mt['roc_auc'],
                       test_recall=mt['recall'], test_fpr=mt['fpr'], test_alert_rate=mt['alert_rate'],
                       test_fpr_shift_from_target=mt['fpr']-.05,
                       test_same_budget_recall=same_t['recall'], test_same_budget_tp=same_t['tp'])
            if variant != 'reference_refit':
                row['same_budget_recall_diff_vs_main'], row['same_budget_diff_ci95'] = paired_recall_boot(flags, main_flags_t, yt)
                row['ap_diff_vs_main'], row['ap_diff_ci95'] = paired_ap_boot(st, st_main, yt)
            else:
                row['max_abs_val_score_diff_vs_frozen'] = float(np.max(np.abs(sv - sv_main)))
                row['max_abs_test_score_diff_vs_frozen'] = float(np.max(np.abs(st - st_main)))
                row['frozen_best_iteration'] = int(params['best_iteration'])
            ablation.append(row); variant_scores[variant] = (sv, st); variant_thresholds[variant] = thr
            log(f'{variant}: best_iter={row["best_iteration"]}, test AP={mt["pr_auc_ap"]:.4f}, recall={mt["recall"]:.4f}, '
                f'FPR={mt["fpr"]:.4f}, same-K recall={same_t["recall"]:.4f}, {time.perf_counter()-start:.0f}s')
        ref = ablation[0]
        out['reproduction'] = dict(best_iteration=ref['best_iteration'], frozen_best_iteration=ref['frozen_best_iteration'],
            max_abs_val_score_diff=ref['max_abs_val_score_diff_vs_frozen'], max_abs_test_score_diff=ref['max_abs_test_score_diff_vs_frozen'],
            reproduced=bool(ref['best_iteration'] == ref['frozen_best_iteration'] and ref['max_abs_test_score_diff_vs_frozen'] < 1e-9))
        out['ablation'] = ablation

        # ---------------- B. fairness context
        log('B. Fairness context, no-age control, group thresholds, age-proxy check')
        context = (group_metrics(yv, sv_main >= thr_main, age_v, 'validation', 'global_threshold')
                   + group_metrics(yt, st_main >= thr_main, age_t, 'test', 'global_threshold'))
        sv_na, st_na = variant_scores['no_age']; thr_na = variant_thresholds['no_age']
        context += group_metrics(yv, sv_na >= thr_na, age_v, 'validation', 'no_age_model')
        context += group_metrics(yt, st_na >= thr_na, age_t, 'test', 'no_age_model')
        group_thr = {}
        for name, mask in [('age<50', age_v < 50), ('age>=50', age_v >= 50)]:
            group_thr[name] = threshold_at_fpr(yv[mask], sv_main[mask])
        def by_group(scores, age): return np.where(age < 50, scores >= group_thr['age<50'], scores >= group_thr['age>=50'])
        context += group_metrics(yv, by_group(sv_main, age_v), age_v, 'validation', 'group_thresholds')
        context += group_metrics(yt, by_group(st_main, age_t), age_t, 'test', 'group_thresholds')
        # Same review volume K for both models: group FPRs no longer differ just because one model alerts less.
        context += group_metrics(yv, top_k(sv_main, kv, ids_v), age_v, 'validation', 'main_same_k')
        context += group_metrics(yt, top_k(st_main, kt, ids_t), age_t, 'test', 'main_same_k')
        context += group_metrics(yv, top_k(sv_na, kv, ids_v), age_v, 'validation', 'no_age_same_k')
        context += group_metrics(yt, top_k(st_na, kt, ids_t), age_t, 'test', 'no_age_same_k')
        context = pd.DataFrame(context)
        overall = {}
        for label, pred in [('global_threshold', st_main >= thr_main), ('no_age_model', st_na >= thr_na),
                            ('group_thresholds', by_group(st_main, age_t)),
                            ('main_same_k', top_k(st_main, kt, ids_t)), ('no_age_same_k', top_k(st_na, kt, ids_t))]:
            m = binary_metrics(yt, pred)
            overall[label] = dict(m, **fpr_ratio(context[(context.split == 'test') & (context.setting == label)].to_dict('records')))
        # How much age is still recoverable from the other fields?
        sample = np.random.default_rng(SEED).choice(len(xtr), size=min(300_000, len(xtr)), replace=False)
        proxy = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=31, min_child_samples=150,
                                   n_jobs=4, deterministic=True, force_col_wise=True, random_state=SEED, verbosity=-1)
        xtr_na = xtr.drop(columns=['customer_age'])
        proxy.fit(xtr_na.iloc[sample], (df.loc[tr, 'customer_age'].to_numpy()[sample] >= 50).astype(int))
        path = EXT_MODELS/'age_proxy.joblib'; joblib.dump(proxy, path)
        saved_models['age_proxy'] = path.relative_to(ROOT).as_posix()
        gain = pd.Series(proxy.booster_.feature_importance('gain'), index=xtr_na.columns).sort_values(ascending=False)
        proxy_auc = {s: float(roc_auc_score(a >= 50, proxy.predict_proba(x.drop(columns=['customer_age']))[:, 1]))
                     for s, x, a in [('validation', xv, age_v), ('test', xt, age_t)]}
        for split, x in [('validation', xv), ('test', xt)]:
            prediction_frames[split]['age_proxy'] = proxy.predict_proba(x.drop(columns=['customer_age']))[:, 1]
        out['fairness'] = dict(group_thresholds_from_validation=group_thr, test_overall=overall,
            age_proxy=dict(target='age>=50', features='all model inputs except customer_age', train_rows=len(sample),
                           roc_auc=proxy_auc, top_gain_features=gain.head(5).round(1).to_dict()),
            notes=['FPR denominators are each group\'s normal applications.',
                   'Group thresholds use age at decision time (disparate treatment); shown as a trade-off reference, not a recommendation.'])
        labels = {'global_threshold': 'Main model\nfrozen threshold', 'no_age_model': 'No-age refit\nfrozen threshold',
                  'main_same_k': 'Main model\nsame K alerts', 'no_age_same_k': 'No-age refit\nsame K alerts',
                  'group_thresholds': 'Main model\nper-group thresholds'}
        compare = context[context.split == 'test'].copy(); compare['setting_label'] = compare.setting.map(labels)
        compare['order'] = compare.setting.map({k: i for i, k in enumerate(labels)}); compare = compare.sort_values(['order', 'group'])

        # ---------------- C. velocity drift table
        velocity, out['velocity'] = drift_tables(df)

    # ---------------- save
    curve.to_csv(TABLES/'ext_rule_complexity.csv', index=False, encoding='utf-8-sig')
    rules.to_csv(TABLES/'ext_rules_recommended.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in ablation]).to_csv(
        TABLES/'ext_feature_ablation.csv', index=False, encoding='utf-8-sig')
    context.to_csv(TABLES/'ext_fairness_context.csv', index=False, encoding='utf-8-sig')
    velocity.to_csv(TABLES/'ext_velocity_monthly.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(out['velocity']['test_month_psi_all_fields']).to_csv(TABLES/'ext_psi_all_fields_test.csv', index=False, encoding='utf-8-sig')
    prediction_paths = {}
    for split, scores in prediction_frames.items():
        path = OUT/f'predictions/ext_{split}_scores.parquet'; scores.to_parquet(path, index=False)
        prediction_paths[split] = path.relative_to(ROOT).as_posix()
    artifact_paths = list(saved_models.values()) + list(prediction_paths.values())
    extension_manifest = dict(
        artifact_version=2, seed=SEED, source_sha256=manifest['raw_sha256'],
        frozen_source_hashes=before, models=saved_models, predictions=prediction_paths,
        artifact_sha256={p: sha256(ROOT/p) for p in artifact_paths},
        source_code_sha256={p: sha256(ROOT/p) for p in ['src/common.py', 'src/eda.py', 'src/extensions.py']},
        variants={k: dict(dropped_columns=VARIANTS[k], threshold=float(variant_thresholds[k]),
                          num_leaves=params['num_leaves'], reg_lambda=params['reg_lambda']) for k in VARIANTS},
        rule_selection=rule_summary, rule_curve_thresholds={f'rules_{int(row.leaves)}': float(row.threshold) for row in curve.itertuples()},
        bootstrap=dict(replicates=BOOT, seed=SEED, conditioning='fixed predictions and K; positive-class paired recall bootstrap; class-stratified paired AP bootstrap; no retraining or re-ranking'),
        environment=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__, lightgbm=lgb.__version__))
    save_json(strict_json(extension_manifest), OUT/'extension_manifest.json')
    out['artifacts'] = dict(manifest='outputs/extension_manifest.json', manifest_sha256=sha256(OUT/'extension_manifest.json'),
                            models=saved_models, predictions=prediction_paths)
    out['rule_curve'] = curve.to_dict(orient='records'); out['recommended_rules'] = rules.to_dict(orient='records')
    out['fairness_context'] = context.to_dict(orient='records'); out['velocity_monthly'] = velocity.to_dict(orient='records')
    style()
    plot_rules(curve, rule_summary)
    plot_fairness(context, compare)
    plot_drift(velocity, pd.DataFrame(ablation))
    after = frozen_hashes(); out['frozen_hashes_after'] = after
    out['frozen_artifacts_unchanged'] = before == after
    assert before == after, 'A frozen artifact changed during the supplementary stage'
    save_json(strict_json(out), OUT/'extensions.json')
    log(f'Done. Reproduced frozen LightGBM: {out["reproduction"]["reproduced"]}; recommended rules: '
        f'{rule_summary["chosen_leaves"]} leaves, test recall {rule_summary["chosen_test"]["recall"]:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-descriptive', action='store_true', help='refresh PSI and target/fallback metadata without any fitting')
    arguments = parser.parse_args()
    refresh_descriptive() if arguments.refresh_descriptive else run()
