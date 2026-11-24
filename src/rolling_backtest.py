"""Retrospective rolling-origin robustness checks; never replace frozen results.

For test months 4..7, fit on 0..test-2 and validate on test-1. Each window
selects its own baseline from the original three configurations. Ablations
inherit that window's configuration and independently early-stop on its past
validation month. All four variants use a common, predeclared 5% review budget.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(key, '4')
import json
import time
import platform
from pathlib import Path
from datetime import datetime, timezone, timedelta
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from .common import (ROOT, SEED, load_data, clean_features, metrics, binary_metrics,
                     threshold_at_fpr, top_k, wilson, sha256, save_json)
from .extensions import fit_lgbm, frozen_hashes, strict_json

OUT = ROOT/'outputs'
WORK = OUT/'rolling'
CONFIGS = [(15, 5.), (31, 10.), (63, 20.)]
VARIANTS = {
    'baseline': [],
    'no_age': ['customer_age'],
    'no_velocity_short': ['velocity_6h', 'velocity_24h'],
    'no_velocity_all': ['velocity_6h', 'velocity_24h', 'velocity_4w'],
}
REVIEW_RATE = .05
BOOT = 500


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def intervals(row):
    row = dict(row)
    row['recall_ci_low'], row['recall_ci_high'] = wilson(row['tp'], row['fraud'])
    row['fpr_ci_low'], row['fpr_ci_high'] = wilson(row['fp'], row['fp']+row['tn'])
    return row


def fairness_row(y, flags, age):
    young = age < 50
    a = binary_metrics(y[young], flags[young])
    b = binary_metrics(y[~young], flags[~young])
    alo, ahi = wilson(a['fp'], a['fp']+a['tn'])
    blo, bhi = wilson(b['fp'], b['fp']+b['tn'])
    return dict(fpr_younger=a['fpr'], fpr_older=b['fpr'],
                fpr_gap=b['fpr']-a['fpr'],
                fpr_ratio=b['fpr']/a['fpr'] if a['fpr'] else None,
                young_n=a['n'], older_n=b['n'], young_fp=a['fp'], older_fp=b['fp'],
                young_normal=a['fp']+a['tn'], older_normal=b['fp']+b['tn'],
                young_recall=a['recall'], older_recall=b['recall'],
                older_fpr_ci_low=blo, older_fpr_ci_high=bhi,
                young_fpr_ci_low=alo, young_fpr_ci_high=ahi)


def weighted_ap_setup(y, scores):
    """Sort once; bootstrap only changes multiplicities, including score ties."""
    order = np.argsort(-np.asarray(scores), kind='stable')
    sorted_scores = np.asarray(scores)[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(order)-1]
    positive = np.asarray(y)[order] == 1

    def calculate(weights):
        w = np.asarray(weights)[order]
        tp = np.cumsum(w*positive)[ends]
        total = np.cumsum(w)[ends]
        precision = np.divide(tp, total, out=np.zeros(len(tp), dtype=float), where=total > 0)
        return float(np.sum(np.diff(np.r_[0., tp])*precision)/tp[-1])
    return calculate


def paired_differences(y, scores, frozen_flags, top_flags, age, seed):
    """Fixed-prediction, paired stratified bootstrap; alerts are NOT reselected.

    Four strata (age group x label) preserve each denominator and pair every
    variant with baseline on identical sampled applications. Thus intervals
    condition on observed group sizes/base rates and exclude model refitting
    and threshold estimation. AP's tie-aware weighted formula is checked
    against sklearn on an independently generated nonuniform weight vector.
    """
    names = list(VARIANTS)
    strata = [np.flatnonzero((age >= 50) == old) for old in [False, True]]
    strata = [idx[y[idx] == label] for idx in strata for label in [0, 1]]
    young_neg, young_pos, old_neg, old_pos = strata
    positive = y == 1
    rng = np.random.default_rng(seed)
    ap_functions = {v: weighted_ap_setup(y, scores[v]) for v in names}
    sample_weights = np.random.default_rng(seed+1).integers(0, 4, len(y))
    for v in names:
        expected = average_precision_score(y, scores[v], sample_weight=sample_weights)
        if not np.isclose(ap_functions[v](sample_weights), expected, atol=1e-12):
            raise AssertionError('Weighted AP bootstrap disagrees with sklearn')
    flags_f = np.column_stack([frozen_flags[v] for v in names]).astype(float)
    flags_k = np.column_stack([top_flags[v] for v in names]).astype(float)
    boots = {v: {key: [] for key in ['ap', 'recall_frozen', 'recall_topk', 'fpr_gap_topk']}
             for v in names[1:]}
    npos = int(positive.sum())
    for _ in range(BOOT):
        sampled = np.concatenate([rng.choice(idx, len(idx), replace=True) for idx in strata])
        weights = np.bincount(sampled, minlength=len(y))
        aps = np.array([ap_functions[v](weights) for v in names])
        rf = weights[positive] @ flags_f[positive] / npos
        rk = weights[positive] @ flags_k[positive] / npos
        gap = (weights[old_neg] @ flags_k[old_neg]/len(old_neg)
               - weights[young_neg] @ flags_k[young_neg]/len(young_neg))
        for j, v in enumerate(names[1:], 1):
            for key, values in [('ap', aps), ('recall_frozen', rf), ('recall_topk', rk),
                                ('fpr_gap_topk', gap)]:
                boots[v][key].append(float(values[j]-values[0]))
    result = []
    ap0 = average_precision_score(y, scores['baseline'])
    for v in names[1:]:
        row = {'variant': v, 'ap_diff': float(average_precision_score(y, scores[v])-ap0)}
        for key, flags in [('recall_frozen', frozen_flags), ('recall_topk', top_flags)]:
            row[key+'_diff'] = float((flags[v][positive].astype(float)-flags['baseline'][positive]).mean())
        delta = top_flags[v].astype(float)-top_flags['baseline'].astype(float)
        row['fpr_gap_topk_diff'] = float(delta[old_neg].mean()-delta[young_neg].mean())
        for key, values in boots[v].items():
            row[key+'_ci_low'], row[key+'_ci_high'] = np.quantile(values, [.025, .975]).tolist()
        result.append(row)
    return result


def draw_figure(rows, fairness, differences):
    frame = pd.DataFrame(rows)
    f = pd.DataFrame(fairness)
    d = pd.DataFrame(differences)
    colors = {'baseline': '#23678d', 'no_age': '#bd7021',
              'no_velocity_short': '#619758', 'no_velocity_all': '#9259a2'}
    labels = {'baseline': 'Full-feature baseline', 'no_age': 'Without age',
              'no_velocity_short': 'Without velocity 6h/24h', 'no_velocity_all': 'Without all velocity'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for variant in VARIANTS:
        common = (frame.variant == variant) & (frame.split == 'test')
        a = frame[common & (frame.policy == 'fixed_review_5pct')]
        b = frame[common & (frame.policy == 'frozen_threshold')]
        g = f[(f.variant == variant) & (f.policy == 'fixed_review_5pct')]
        axes[0, 0].plot(a.test_month+1, 100*a.recall, 'o-', color=colors[variant], label=labels[variant])
        axes[0, 1].plot(b.test_month+1, 100*b.fpr, 'o-', color=colors[variant])
        axes[1, 0].plot(g.test_month+1, 100*g.fpr_gap, 'o-', color=colors[variant])
    for offset, variant in enumerate(list(VARIANTS)[1:]):
        s = d[d.variant == variant]
        y = 100*s.recall_topk_diff
        axes[1, 1].errorbar(s.test_month+1+(offset-1)*.09, y,
            yerr=np.vstack([y-100*s.recall_topk_ci_low, 100*s.recall_topk_ci_high-y]),
            fmt='o', capsize=3, color=colors[variant], label=labels[variant])
    axes[0, 0].set(title='Recall at a fixed 5% review rate', ylabel='Recall (%)')
    axes[0, 1].set(title='FPR after transferring the validation threshold', ylabel='FPR (%)')
    axes[0, 1].axhline(5, color='#999999', ls='--', lw=1)
    axes[1, 0].set(title='Age FPR gap at the same 5% review rate', ylabel='Older minus younger (percentage points)')
    axes[1, 1].set(title='Recall change vs baseline (paired 95% interval)', ylabel='Change (percentage points)')
    axes[1, 1].axhline(0, color='#999999', ls='--', lw=1)
    for ax in axes.ravel():
        ax.set(xticks=[5, 6, 7, 8], xlabel='Test month (1-based)')
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Four retrospective rolling windows | independent historical fitting in every window', fontsize=12)
    fig.savefig(OUT/'figures/rolling_backtest.png', dpi=160)
    plt.close(fig)


def run():
    started = time.perf_counter()
    for p in [WORK/'models', WORK/'predictions', OUT/'tables', OUT/'figures']:
        p.mkdir(parents=True, exist_ok=True)
    before = frozen_hashes()
    protocol = dict(schema_version=1, created_at_singapore=timestamp(), seed=SEED,
        design='Post-hoc retrospective rolling-origin robustness check; not fresh holdouts',
        windows=[dict(train_months=list(range(m-1)), validation_month=m-1, test_month=m) for m in range(4, 8)],
        baseline_candidates=[dict(num_leaves=a, reg_lambda=b) for a, b in CONFIGS],
        baseline_selection='Highest validation recall at FPR<=5%; validation AP breaks ties, candidate order breaks exact ties',
        ablation_selection='Use each window baseline configuration; independently early-stop on that validation month; never select variants on test',
        variants=VARIANTS, review_rate=REVIEW_RATE, validation_max_fpr=.05,
        equal_budget='floor(0.05*N) per test month for ALL variants; differs from main experiment 0.056412 budget',
        bootstrap_reps=BOOT,
        uncertainty='Paired age-by-label-stratified bootstrap of fixed predictions; observed strata sizes, fitted models, thresholds and top-K flags held fixed; excludes retraining, threshold estimation, and future-month uncertainty',
        caveats=['All months have already been inspected: this is exploratory robustness evidence.',
                 'Training sets overlap, so the four windows are not four independent replicates.',
                 'Min/max across months is an observed range, not a confidence interval.',
                 'Labels are treated as available by the next month; the dataset has no reporting-delay information.'],
        source_sha256=sha256(Path(__file__)), raw_sha256=sha256(ROOT/'data/raw/Base.csv'),
        environment=dict(python=platform.python_version()))
    save_json(strict_json(protocol), OUT/'rolling_protocol.json')
    protocol_hash = sha256(OUT/'rolling_protocol.json')
    log('Rolling protocol written before any new window fitting; loading data')
    df = load_data()
    windows, metric_rows, fairness, differences, manifests = [], [], [], [], []
    with threadpool_limits(limits=4):
        for test_month in range(4, 8):
            wid = f'window_{test_month+1}'
            folder = WORK/'models'/wid
            folder.mkdir(parents=True, exist_ok=True)
            tr = df.month <= test_month-2
            va = df.month == test_month-1
            te = df.month == test_month
            log(f'{wid}: train months 0..{test_month-2}; validate {test_month-1}; test {test_month}')
            xtr, schema = clean_features(df.loc[tr])
            schema['training_months'] = list(range(test_month-1))
            xv, _ = clean_features(df.loc[va], schema)
            ytr = df.loc[tr, 'fraud_bool'].to_numpy()
            yv = df.loc[va, 'fraud_bool'].to_numpy()
            save_json(schema, folder/'schema.json')
            models, selections, validation_scores, candidates = {}, {}, {}, []
            val = pd.DataFrame({'row_id': df.index[va], 'month': test_month-1,
                                'fraud_bool': yv, 'customer_age': df.loc[va, 'customer_age'].to_numpy()})
            best_key = None
            for leaves, reg in CONFIGS:
                model = fit_lgbm(xtr, ytr, xv, yv, leaves, reg)
                scores = model.predict_proba(xv)[:, 1]
                threshold = threshold_at_fpr(yv, scores)
                met = metrics(yv, scores, threshold)
                spec = dict(num_leaves=leaves, reg_lambda=reg,
                            best_iteration=int(model.best_iteration_), threshold=threshold, validation=met)
                candidates.append(spec)
                val[f'candidate_leaves{leaves}'] = scores
                key = (met['recall'], met['pr_auc_ap'])
                if best_key is None or key > best_key:
                    best_key = key
                    models['baseline'], selections['baseline'], validation_scores['baseline'] = model, spec, scores
                log(f'{wid} candidate {leaves}: validation AP={met["pr_auc_ap"]:.4f}, recall={met["recall"]:.4f}')
            chosen = selections['baseline']
            for variant, drops in list(VARIANTS.items())[1:]:
                model = fit_lgbm(xtr.drop(columns=drops), ytr, xv.drop(columns=drops), yv,
                                 chosen['num_leaves'], chosen['reg_lambda'])
                scores = model.predict_proba(xv.drop(columns=drops))[:, 1]
                threshold = threshold_at_fpr(yv, scores)
                models[variant], validation_scores[variant] = model, scores
                selections[variant] = dict(num_leaves=chosen['num_leaves'], reg_lambda=chosen['reg_lambda'],
                    best_iteration=int(model.best_iteration_), threshold=threshold,
                    validation=metrics(yv, scores, threshold))
            for variant in VARIANTS:
                joblib.dump(models[variant], folder/f'{variant}.joblib')
                val[variant] = validation_scores[variant]
                val[variant+'__alert_frozen'] = validation_scores[variant] >= selections[variant]['threshold']
                val[variant+'__alert_topk'] = top_k(validation_scores[variant], int(REVIEW_RATE*len(yv)), val.row_id)
            val_path = WORK/'predictions'/f'{wid}_validation.parquet'
            val.to_parquet(val_path, index=False)
            manifest = dict(window_id=wid, train_months=list(range(test_month-1)),
                validation_month=test_month-1, test_month=test_month,
                candidates=candidates, variants=selections, dropped_features=VARIANTS,
                review_rate=REVIEW_RATE, protocol_sha256=protocol_hash,
                frozen_before_test_predictions=True, frozen_at_singapore=timestamp(),
                artifact_hashes={str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p)
                                 for p in list(folder.glob('*.joblib'))+[folder/'schema.json', val_path]})
            save_json(manifest, folder/'manifest.json')
            manifests.append(str((folder/'manifest.json').relative_to(ROOT)).replace('\\', '/'))
            # No target-month prediction is made until this window's choices are saved.
            xt, _ = clean_features(df.loc[te], schema)
            yt = df.loc[te, 'fraud_bool'].to_numpy()
            age = df.loc[te, 'customer_age'].to_numpy()
            test = pd.DataFrame({'row_id': df.index[te], 'month': test_month, 'fraud_bool': yt, 'customer_age': age})
            k = int(REVIEW_RATE*len(yt))
            scores, flags_f, flags_k = {}, {}, {}
            for variant, drops in VARIANTS.items():
                scores[variant] = models[variant].predict_proba(xt.drop(columns=drops))[:, 1]
                threshold = selections[variant]['threshold']
                flags_f[variant] = scores[variant] >= threshold
                flags_k[variant] = top_k(scores[variant], k, test.row_id)
                test[variant] = scores[variant]
                test[variant+'__alert_frozen'] = flags_f[variant]
                test[variant+'__alert_topk'] = flags_k[variant]
                for split, y, s, ids in [('validation', yv, validation_scores[variant], val.row_id),
                                          ('test', yt, scores[variant], test.row_id)]:
                    met = metrics(y, s, threshold)
                    for policy, bm in [('frozen_threshold', met),
                                       ('fixed_review_5pct', binary_metrics(y, top_k(s, int(REVIEW_RATE*len(y)), ids)))]:
                        entry = dict(window_id=wid, test_month=test_month, variant=variant, split=split,
                                     policy=policy, threshold=threshold, pr_auc_ap=met['pr_auc_ap'], roc_auc=met['roc_auc'])
                        entry.update(intervals(bm))
                        metric_rows.append(entry)
                for policy, flags in [('frozen_threshold', flags_f[variant]), ('fixed_review_5pct', flags_k[variant])]:
                    fairness.append(dict(window_id=wid, test_month=test_month, variant=variant,
                                         policy=policy, **fairness_row(yt, flags, age)))
            test.to_parquet(WORK/'predictions'/f'{wid}_test.parquet', index=False)
            log(f'{wid}: paired bootstrap of four variants (500 replicates)')
            diff = paired_differences(yt, scores, flags_f, flags_k, age, SEED+test_month)
            differences.extend([dict(window_id=wid, test_month=test_month, **d) for d in diff])
            windows.append(dict(window_id=wid, train_months=list(range(test_month-1)),
                validation_month=test_month-1, test_month=test_month,
                train_n=int(tr.sum()), validation_n=int(va.sum()), test_n=len(yt), test_fraud=int(yt.sum()),
                selected_num_leaves=chosen['num_leaves'], selected_reg_lambda=chosen['reg_lambda'],
                selected_best_iteration=chosen['best_iteration'], k=k, validation_k=int(REVIEW_RATE*len(yv))))
            log(f'{wid} complete: baseline recall={binary_metrics(yt,flags_f["baseline"])["recall"]:.4f}; same 5% review recall={binary_metrics(yt,flags_k["baseline"])["recall"]:.4f}')
            del xtr, xv, xt, models
    after = frozen_hashes()
    assert before == after, 'Frozen main artifacts changed during rolling backtest'
    frame = pd.DataFrame(metric_rows)
    baseline = frame[(frame.variant == 'baseline') & (frame.split == 'test')]
    summary = {'observed_ranges_not_confidence_intervals': True, 'baseline': {}, 'ablations': {}}
    for policy in ['frozen_threshold', 'fixed_review_5pct']:
        subset = baseline[baseline.policy == policy]
        summary['baseline'][policy] = {key: [float(subset[key].min()), float(subset[key].max())]
                                      for key in ['pr_auc_ap', 'recall', 'fpr', 'alert_rate']}
    dd = pd.DataFrame(differences)
    ff = pd.DataFrame(fairness)
    fixed_fairness = ff[ff.policy == 'fixed_review_5pct'].set_index(['window_id', 'variant'])
    for v in list(VARIANTS)[1:]:
        subset = dd[dd.variant == v]
        summary['ablations'][v] = dict(positive_ap_months=int((subset.ap_diff > 0).sum()),
            positive_topk_recall_months=int((subset.recall_topk_diff > 0).sum()),
            topk_recall_ci_excludes_zero_months=int(((subset.recall_topk_ci_low > 0) | (subset.recall_topk_ci_high < 0)).sum()),
            smaller_age_fpr_gap_months=sum(
                abs(fixed_fairness.loc[(w['window_id'], v), 'fpr_gap'])
                < abs(fixed_fairness.loc[(w['window_id'], 'baseline'), 'fpr_gap'])
                for w in windows))
    artifact_paths = [p for p in WORK.rglob('*') if p.is_file()]
    result = dict(schema_version=1, completed_at_singapore=timestamp(), seconds=time.perf_counter()-started,
        protocol=protocol, protocol_sha256=protocol_hash, windows=windows, metrics=metric_rows,
        fairness=fairness, differences=differences, summary=summary, window_manifests=manifests,
        frozen_hashes_before=before, frozen_artifacts_unchanged=before == after,
        artifact_hashes={str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in artifact_paths})
    for name, rows in [('windows', windows), ('metrics', metric_rows), ('fairness', fairness), ('differences', differences)]:
        pd.DataFrame(rows).to_csv(OUT/'tables'/f'rolling_{name}.csv', index=False, encoding='utf-8-sig')
    draw_figure(metric_rows, fairness, differences)
    save_json(strict_json(result), OUT/'rolling_backtest.json')
    log(f'Completed all four rolling windows in {result["seconds"]:.0f}s; frozen main artifacts unchanged')


if __name__ == '__main__':
    run()
