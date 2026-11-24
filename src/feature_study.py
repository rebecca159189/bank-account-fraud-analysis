"""Feature analysis and feature engineering for BAF (post-hoc; frozen results stay read only).

Four parts (each can be run alone with --parts; finished parts are cached in outputs/feature_study.json):
  iv          WOE / IV of every input field on training months 0-5, plus PSI of the same bins
              (months 0-5 versus validation month 6)                       -> which fields carry signal, which drift
  selection   drop fields with IV < 0.02 or PSI > 0.25 (rule fixed in advance, uses months 0-6 only),
              retrain the frozen LightGBM recipe and compare
  derived     three groups of derived features (velocity ratios, identity/consistency, category encodings),
              each group and all groups added to the frozen LightGBM recipe; month 6 chooses, month 7 reports;
              then baseline versus "all derived" in four rolling windows with paired bootstrap intervals
  importance  permutation importance of the frozen LightGBM on month 6, compared with TreeSHAP and IV rankings

Discipline (same as src/extensions.py and src/coursework.py)
  * Frozen artifacts are only READ; their SHA-256 are recorded before and after.
  * Bins, encodings, selection rules and early stopping use months 0-6 only; month 7 is reported once.
  * Designed after month-7 results were seen: exploratory. Nothing here replaces the frozen model.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(key, '4')
import argparse
import json
import time
from datetime import datetime, timezone, timedelta
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from .common import *
from .coursework import fit_lgbm_cfg, eval_scores, strict_json, frozen_hashes, log

OUT = ROOT/'outputs'; MODELS = OUT/'models'; FIG = OUT/'figures'; TABLES = OUT/'tables'
RESULT = OUT/'feature_study.json'
PARTS = ['iv', 'selection', 'derived', 'importance']
N_BINS = 10              # quantile bins for continuous fields
DISCRETE_MAX = 20        # fields with at most this many distinct values keep one bin per value
IV_MIN = 0.02            # IV below this: "no predictive power" (common credit-scoring rule of thumb)
PSI_MAX = 0.25           # PSI above this: "significant shift"
TE_SMOOTHING = 100       # target-encoding smoothing strength (pseudo-count)
TE_FOLDS = 5
BOOT = 200
PERM_REPEATS = 3
IV_LEVELS = [(0.02, '无预测力'), (0.1, '弱'), (0.3, '中'), (0.5, '强'), (np.inf, '很强（应核查）')]


def iv_level(v):
    return next(label for limit, label in IV_LEVELS if v < limit)


# ======================================================================= WOE / IV
def fit_bins(s, n_bins=N_BINS, discrete_max=DISCRETE_MAX):
    """Bin specification learned on TRAINING values only. Missing values always get their own bin."""
    if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object:
        return dict(kind='categories', levels=sorted(s.dropna().astype(str).unique().tolist()))
    v = s.dropna().to_numpy(float)
    uniq = np.unique(v)
    if len(uniq) <= discrete_max:
        return dict(kind='values', values=[float(u) for u in uniq])
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins+1)[1:-1]))
    return dict(kind='edges', edges=edges.tolist())


def bin_labels(spec):
    """Bin labels in their natural order (missing and unseen last)."""
    if spec['kind'] == 'categories': lab = list(spec['levels'])
    elif spec['kind'] == 'values': lab = [f'{v:g}' for v in spec['values']]
    else:
        e = spec['edges']
        lab = [f'({"-inf" if i == 0 else f"{e[i-1]:.4g}"}, {"inf" if i == len(e) else f"{e[i]:.4g}"}]' for i in range(len(e)+1)]
    return lab + ['其他', '缺失']


def assign_bins(s, spec):
    """Bin label for every row; unseen levels -> '其他', missing -> '缺失'."""
    out = np.full(len(s), '缺失', dtype=object)
    ok = s.notna().to_numpy()
    if spec['kind'] == 'categories':
        v = s[ok].astype(str).to_numpy(object)
        out[ok] = np.where(np.isin(v, spec['levels']), v, '其他')
    elif spec['kind'] == 'values':
        m = {float(u): f'{float(u):g}' for u in spec['values']}
        out[ok] = s[ok].astype(float).map(m).fillna('其他').to_numpy(object)
    else:
        edges = np.asarray(spec['edges'], float)
        idx = np.searchsorted(edges, s[ok].to_numpy(float), side='right')
        out[ok] = np.asarray(bin_labels(spec)[:len(edges)+1], dtype=object)[idx]
    return out


def woe_table(bins, y, order=None):
    """WOE = ln(bad share / good share): positive means riskier than average. 0.5 smoothing per bin."""
    d = pd.DataFrame({'bin': bins, 'y': np.asarray(y)})
    g = d.groupby('bin', sort=False).y.agg(n='size', bad='sum')
    if order is not None: g = g.reindex([b for b in order if b in g.index])
    g = g.reset_index()
    g['good'] = g.n - g.bad
    nb = len(g)
    g['bad_share'] = (g.bad + .5)/(g.bad.sum() + .5*nb)
    g['good_share'] = (g.good + .5)/(g.good.sum() + .5*nb)
    g['woe'] = np.log(g.bad_share/g.good_share)
    g['iv_part'] = (g.bad_share - g.good_share)*g.woe
    g['fraud_rate'] = g.bad/g.n
    return g, float(g.iv_part.sum())


def psi_from_bins(ref_bins, cur_bins, eps=1e-4):
    levels = sorted(set(ref_bins) | set(cur_bins))
    a = pd.Series(ref_bins).value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy() + eps
    b = pd.Series(cur_bins).value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy() + eps
    return float(np.sum((b - a)*np.log(b/a)))


def base_field(col):
    return col[:-len('__missing')] if col.endswith('__missing') else col


# ======================================================================= derived features
VELOCITY = ['velocity_6h', 'velocity_24h', 'velocity_4w']
GROUPS = {
    'ratios': ['v_ratio_6h_24h', 'v_ratio_24h_4w'],
    'consistency': ['missing_count', 'valid_phones', 'free_email_low_similarity', 'limit_per_income',
                    'bank_months_per_age', 'address_months_total', 'dob_emails_per_device_email'],
    'encodings': [f'freq_{c}' for c in CATEGORICAL] + [f'te_{c}' for c in CATEGORICAL],
}
GROUP_LABELS = {'ratios': '速度比率', 'consistency': '身份与一致性', 'encodings': '类别频率与目标编码'}


def fit_encoders(x, y):
    """Frequency maps and smoothed target-encoding maps learned on the training rows only."""
    prior = float(np.mean(y)); enc = dict(prior=prior, freq={}, te={})
    for c in CATEGORICAL:
        s = x[c].astype(str)
        enc['freq'][c] = s.value_counts(normalize=True).to_dict()
        g = pd.DataFrame({'k': s.to_numpy(), 'y': y}).groupby('k').y.agg(['sum', 'count'])
        enc['te'][c] = ((g['sum'] + TE_SMOOTHING*prior)/(g['count'] + TE_SMOOTHING)).to_dict()
    return enc


def oof_target_encoding(x, y, seed=SEED):
    """Out-of-fold target encoding for the TRAINING rows (a row never sees its own label)."""
    out = pd.DataFrame(index=x.index, columns=[f'te_{c}' for c in CATEGORICAL], dtype='float32')
    for fit_idx, apply_idx in KFold(TE_FOLDS, shuffle=True, random_state=seed).split(x):
        enc = fit_encoders(x.iloc[fit_idx], y[fit_idx])
        for c in CATEGORICAL:
            out.iloc[apply_idx, out.columns.get_loc(f'te_{c}')] = (
                x[c].iloc[apply_idx].astype(str).map(enc['te'][c]).fillna(enc['prior']).to_numpy('float32'))
    return out


def add_derived(x, enc, groups=('ratios', 'consistency', 'encodings'), te_override=None):
    """Return a copy of x with the requested derived groups appended (row-wise formulas + training encodings)."""
    x = x.copy(); f32 = 'float32'
    if 'ratios' in groups:
        x['v_ratio_6h_24h'] = (x.velocity_6h/(x.velocity_24h.abs() + 1)).astype(f32)
        x['v_ratio_24h_4w'] = (x.velocity_24h/(x.velocity_4w.abs() + 1)).astype(f32)
    if 'consistency' in groups:
        flags = [c for c in x if c.endswith('__missing')]
        x['missing_count'] = x[flags].sum(axis=1).astype(f32)
        x['valid_phones'] = (x.phone_home_valid + x.phone_mobile_valid).astype(f32)
        x['free_email_low_similarity'] = (x.email_is_free*(1 - x.name_email_similarity)).astype(f32)
        x['limit_per_income'] = (x.proposed_credit_limit/x.income.clip(lower=.01)).astype(f32)
        x['bank_months_per_age'] = (x.bank_months_count/x.customer_age.clip(lower=1)).astype(f32)
        x['address_months_total'] = (x.prev_address_months_count.fillna(0) + x.current_address_months_count).astype(f32)
        x['dob_emails_per_device_email'] = (x.date_of_birth_distinct_emails_4w/(x.device_distinct_emails_8w.fillna(0) + 1)).astype(f32)
    if 'encodings' in groups:
        for c in CATEGORICAL:
            s = x[c].astype(str)
            x[f'freq_{c}'] = s.map(enc['freq'][c]).fillna(0).astype(f32)
            x[f'te_{c}'] = s.map(enc['te'][c]).fillna(enc['prior']).astype(f32)
        if te_override is not None:
            for col in te_override:
                x[col] = te_override[col].to_numpy(f32)
    return x


# ======================================================================= context
class Ctx:
    """Data, splits and frozen artifacts (read only), loaded lazily."""
    def __init__(self):
        self.manifest = json.loads((MODELS/'frozen_manifest.json').read_text(encoding='utf-8'))
        self.schema = json.loads((MODELS/'schema.json').read_text(encoding='utf-8'))
        log('Loading Base (read only)')
        df = load_data(); self.df = df
        self.masks = {m: (df.month == m).to_numpy() for m in range(8)}
        tr = (df.month <= 5).to_numpy(); va = self.masks[6]; te = self.masks[7]
        self.xtr, _ = clean_features(df.loc[tr], self.schema); self.xv, _ = clean_features(df.loc[va], self.schema)
        self.xt, _ = clean_features(df.loc[te], self.schema)
        self.ytr, self.yv, self.yt = (df.loc[m, 'fraud_bool'].to_numpy() for m in (tr, va, te))
        self.ids_v, self.ids_t = df.index[va].to_numpy(), df.index[te].to_numpy()
        self.kt = int(np.floor(self.manifest['review_budget_rate']*len(self.yt)))
        self.lgb_params = self.manifest['models']['lightgbm']['params']
        self._pred = None

    def frozen_scores(self, model='lightgbm'):
        if self._pred is None:
            self._pred = (pd.read_parquet(OUT/'predictions/validation.parquet'), pd.read_parquet(OUT/'predictions/test.parquet'))
            assert np.array_equal(self._pred[0].row_id.to_numpy(), self.ids_v)
            assert np.array_equal(self._pred[1].row_id.to_numpy(), self.ids_t)
        return self._pred[0][model].to_numpy(float), self._pred[1][model].to_numpy(float)

    def row(self, label, sv, st, **extra):
        return dict(label=label, **eval_scores(self.yv, sv, self.yt, st, self.ids_t, self.kt), **extra)

    def fit(self, xtr, xv):
        return fit_lgbm_cfg(xtr, self.ytr, xv, self.yv, self.lgb_params)


def save_table(df, name): df.to_csv(TABLES/name, index=False, encoding='utf-8-sig')


def fig_save(fig, name): fig.tight_layout(); fig.savefig(FIG/name, dpi=150); plt.close(fig)


# ======================================================================= part 1: WOE / IV
def part_iv(c, res):
    fields = [col for col in c.xtr if not col.endswith('__missing')]
    rows, bins_all = [], []
    for f in fields:
        spec = fit_bins(c.xtr[f])
        btr = assign_bins(c.xtr[f], spec); bv = assign_bins(c.xv[f], spec)
        tab, iv = woe_table(btr, c.ytr, bin_labels(spec))
        tab.insert(0, 'field', f); bins_all.append(tab)
        top = tab.loc[tab.woe.idxmax()]
        rows.append(dict(field=f, iv=iv, level=iv_level(iv), psi_train_vs_month6=psi_from_bins(btr, bv),
                         n_bins=len(tab), riskiest_bin=str(top['bin']), riskiest_bin_fraud_rate=float(top.fraud_rate),
                         riskiest_bin_woe=float(top.woe)))
    t = pd.DataFrame(rows).sort_values('iv', ascending=False).reset_index(drop=True)
    t['keep_by_rule'] = (t.iv >= IV_MIN) & (t.psi_train_vs_month6 <= PSI_MAX)
    save_table(t, 'fs_iv.csv'); save_table(pd.concat(bins_all), 'fs_woe_bins.csv')
    log(f'iv: {len(t)} fields; top: ' + ', '.join(f'{r.field}={r.iv:.3f}' for r in t.head(5).itertuples()))
    # figures
    fig, ax = plt.subplots(1, 2, figsize=(13, 7.5), gridspec_kw=dict(width_ratios=[1.4, 1]))
    tt = t.iloc[::-1]
    colors = ['#c0504d' if not k else '#147d92' for k in tt.keep_by_rule]
    ax[0].barh(tt.field, tt.iv, color=colors); ax[0].axvline(IV_MIN, color='k', ls=':', lw=1)
    ax[0].set(title=f'IV on training months (red = dropped by rule IV<{IV_MIN} or PSI>{PSI_MAX})', xlabel='Information Value')
    ax[0].tick_params(axis='y', labelsize=8)
    ax[1].scatter(t.iv, t.psi_train_vs_month6, color=['#147d92' if k else '#c0504d' for k in t.keep_by_rule])
    for r in t.itertuples():
        if r.iv > .1 or r.psi_train_vs_month6 > .1: ax[1].annotate(r.field, (r.iv, r.psi_train_vs_month6), fontsize=7)
    ax[1].axvline(IV_MIN, color='k', ls=':', lw=1); ax[1].axhline(PSI_MAX, color='k', ls=':', lw=1)
    ax[1].set(xscale='symlog', yscale='symlog', xlabel='IV (training)', ylabel='PSI (months 0-5 vs month 6)', title='Signal versus stability')
    fig_save(fig, 'fs_iv_psi.png')
    top4 = t.field.head(4).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    allb = pd.concat(bins_all)
    for ax, f in zip(axes.ravel(), top4):
        b = allb[allb.field == f]
        ax.bar(range(len(b)), b.woe, color=['#c0504d' if w > 0 else '#147d92' for w in b.woe])
        ax.set_xticks(range(len(b))); ax.set_xticklabels([{'缺失': 'missing', '其他': 'other'}.get(str(v), str(v))[:14] for v in b.bin], rotation=60, ha='right', fontsize=7)
        ax.axhline(0, color='k', lw=.8); ax.set_title(f'{f}  (IV={t.set_index("field").iv[f]:.3f})'); ax.set_ylabel('WOE = ln(bad share / good share)')
    fig_save(fig, 'fs_woe_top.png')
    return dict(n_fields=len(t), iv_min=IV_MIN, psi_max=PSI_MAX, n_bins=N_BINS, table=t.to_dict(orient='records'),
                dropped_by_rule=t.loc[~t.keep_by_rule, 'field'].tolist(),
                level_counts=t.level.value_counts().to_dict())


# ======================================================================= part 2: rule-based selection
def part_selection(c, res):
    iv = res['parts'].get('iv') or part_iv(c, res)
    drop = list(iv['dropped_by_rule'])
    drop_cols = [col for col in c.xtr if base_field(col) in drop]
    rows = [c.row('LightGBM（冻结参考）', *c.frozen_scores(), kind='frozen')]
    t0 = time.perf_counter(); m = c.fit(c.xtr, c.xv)
    sv, st = m.predict_proba(c.xv)[:, 1], m.predict_proba(c.xt)[:, 1]
    fsv, fst = c.frozen_scores()
    reproduce = float(max(np.abs(sv - fsv).max(), np.abs(st - fst).max()))
    rows.append(c.row('同配置重训（全部字段）', sv, st, kind='reference_refit', n_features=c.xtr.shape[1],
                      best_iteration=int(m.best_iteration_), fit_seconds=time.perf_counter()-t0))
    log(f'selection: reference refit max |score - frozen| = {reproduce:.2e}')
    if drop_cols:
        t0 = time.perf_counter(); m = c.fit(c.xtr.drop(columns=drop_cols), c.xv.drop(columns=drop_cols))
        rows.append(c.row(f'按规则删去 {len(drop)} 个字段', m.predict_proba(c.xv.drop(columns=drop_cols))[:, 1],
                          m.predict_proba(c.xt.drop(columns=drop_cols))[:, 1], kind='rule_selected',
                          n_features=c.xtr.shape[1]-len(drop_cols), best_iteration=int(m.best_iteration_),
                          fit_seconds=time.perf_counter()-t0))
        log(f'  dropped {drop}: test AP={rows[-1]["test_ap"]:.4f}')
    save_table(pd.DataFrame(rows), 'fs_selection.csv')
    return dict(dropped_fields=drop, dropped_columns=drop_cols, reference_reproduces_frozen_max_abs=reproduce, rows=rows)


# ======================================================================= part 3: derived features
def derived_frames(xtr, ytr, others, groups):
    """Encoders on training rows; training rows get out-of-fold target encoding."""
    enc = fit_encoders(xtr, ytr)
    te = oof_target_encoding(xtr, ytr) if 'encodings' in groups else None
    out = [add_derived(xtr, enc, groups, te_override=te)]
    out += [add_derived(x, enc, groups) for x in others]
    return out


def boot_diff(y, a, b, k, ids, reps=BOOT, seed=SEED):
    """Paired label-stratified bootstrap of (b - a) for AP and same-capacity recall; predictions held fixed."""
    rng = np.random.default_rng(seed); y = np.asarray(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    fa, fb = top_k(a, k, ids), top_k(b, k, ids)
    ap, rc = [], []
    for _ in range(reps):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        ap.append(average_precision_score(y[i], b[i]) - average_precision_score(y[i], a[i]))
        rc.append(fb[i][y[i] == 1].mean() - fa[i][y[i] == 1].mean())
    q = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    return dict(ap_diff=float(average_precision_score(y, b) - average_precision_score(y, a)), ap_diff_ci95=q(ap),
                same_budget_recall_diff=float(fb[y == 1].mean() - fa[y == 1].mean()), same_budget_recall_diff_ci95=q(rc))


def part_derived(c, res):
    variants = [('reference', ()), ('ratios', ('ratios',)), ('consistency', ('consistency',)),
                ('encodings', ('encodings',)), ('all', ('ratios', 'consistency', 'encodings'))]
    labels = {'reference': '同配置重训（无衍生特征）', 'all': '加入全部三组'}
    rows, scores = [], {}
    for name, groups in variants:
        t0 = time.perf_counter()
        xtr, xv, xt = derived_frames(c.xtr, c.ytr, [c.xv, c.xt], groups) if groups else (c.xtr, c.xv, c.xt)
        m = c.fit(xtr, xv); sv, st = m.predict_proba(xv)[:, 1], m.predict_proba(xt)[:, 1]
        scores[name] = (sv, st)
        label = labels.get(name, f'加入「{GROUP_LABELS.get(name, name)}」')
        rows.append(c.row(label, sv, st, variant=name, n_features=xtr.shape[1], n_new=xtr.shape[1]-c.xtr.shape[1],
                          best_iteration=int(m.best_iteration_), fit_seconds=time.perf_counter()-t0))
        if name == 'all':
            gain = pd.Series(m.booster_.feature_importance('gain'), index=xtr.columns)
            new_gain = (gain[[col for col in xtr if col not in c.xtr]]/gain.sum()).sort_values(ascending=False)
        log(f'derived {name}: val AP={rows[-1]["val_ap"]:.4f} recall={rows[-1]["val_recall"]:.4f} | test AP={rows[-1]["test_ap"]:.4f}')
    # the project rule picks on month 6 only: validation recall at FPR<=5%, AP breaks ties
    chosen = max(rows, key=lambda r: (r['val_recall'], r['val_ap']))['variant']
    ref = scores['reference']
    for r in rows:
        r['chosen_on_month6'] = r['variant'] == chosen
        if r['variant'] != 'reference':
            r.update({f'vs_reference_{k}': v for k, v in boot_diff(c.yt, ref[1], scores[r['variant']][1], c.kt, c.ids_t).items()})
    save_table(pd.DataFrame(rows), 'fs_derived.csv')
    # rolling windows: reference versus "all" (pre-declared, not the month-6 choice, so no window borrows later months)
    windows = []
    df = c.df
    for test_month in range(4, 8):
        trm = (df.month <= test_month-2).to_numpy(); vam = c.masks[test_month-1]; tem = c.masks[test_month]
        xtr, schema = clean_features(df.loc[trm]); xv, _ = clean_features(df.loc[vam], schema); xt, _ = clean_features(df.loc[tem], schema)
        ytr, yv, yt = (df.loc[m, 'fraud_bool'].to_numpy() for m in (trm, vam, tem))
        ids_t = df.index[tem].to_numpy(); k = int(np.floor(.05*len(yt)))
        out = {}
        for name, groups in [('reference', ()), ('all', ('ratios', 'consistency', 'encodings'))]:
            a, b, t3 = derived_frames(xtr, ytr, [xv, xt], groups) if groups else (xtr, xv, xt)
            m = fit_lgbm_cfg(a, ytr, b, yv, c.lgb_params)
            sv, st = m.predict_proba(b)[:, 1], m.predict_proba(t3)[:, 1]
            thr = threshold_at_fpr(yv, sv); mt = metrics(yt, st, thr)
            out[name] = dict(scores=st, test_ap=mt['pr_auc_ap'], test_recall=mt['recall'], test_fpr=mt['fpr'],
                             same_budget_recall=float(binary_metrics(yt, top_k(st, k, ids_t))['recall']),
                             best_iteration=int(m.best_iteration_))
        d = boot_diff(yt, out['reference']['scores'], out['all']['scores'], k, ids_t)
        windows.append(dict(window=f'window_{test_month+1}', train_months=list(range(test_month-1)), validation_month=test_month-1,
                            test_month=test_month, k=k, **{f'{n}_{key}': v for n in out for key, v in out[n].items() if key != 'scores'}, **d))
        log(f'derived rolling window {test_month+1}: AP {out["reference"]["test_ap"]:.4f} -> {out["all"]["test_ap"]:.4f}')
    save_table(pd.DataFrame(windows), 'fs_derived_rolling.csv')
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    labs = [r['variant'] for r in rows]
    x = np.arange(len(rows)); w = .38
    ax[0].bar(x-w/2, [r['val_ap'] for r in rows], w, label='month 6 (choice)', color='#9bbb59')
    ax[0].bar(x+w/2, [r['test_ap'] for r in rows], w, label='month 7 (report)', color='#147d92')
    ax[0].set_xticks(x); ax[0].set_xticklabels(labs); ax[0].set_ylabel('AP'); ax[0].legend(); ax[0].set_title('Derived feature groups, frozen LightGBM recipe')
    wx = np.arange(len(windows))
    ax[1].errorbar(wx, [wd['ap_diff'] for wd in windows],
                   yerr=np.array([[wd['ap_diff']-wd['ap_diff_ci95'][0], wd['ap_diff_ci95'][1]-wd['ap_diff']] for wd in windows]).T,
                   fmt='o', color='#c0504d', capsize=4)
    ax[1].axhline(0, color='k', lw=.8); ax[1].set_xticks(wx); ax[1].set_xticklabels([f'test month {wd["test_month"]}' for wd in windows])
    ax[1].set_ylabel('AP(all derived) - AP(reference)'); ax[1].set_title('Rolling windows, paired bootstrap 95% interval')
    fig_save(fig, 'fs_derived.png')
    return dict(groups={g: GROUPS[g] for g in GROUPS}, group_labels=GROUP_LABELS, rows=rows, chosen_on_month6=chosen,
                new_feature_gain_share=new_gain.to_dict(), rolling=windows, te_smoothing=TE_SMOOTHING, te_folds=TE_FOLDS, bootstrap_reps=BOOT,
                note='Rolling windows compare the pre-declared "all groups" set, fixed LightGBM configuration (num_leaves/reg_lambda of the frozen model), early stopping on each window validation month.')


# ======================================================================= part 4: importance cross-check
def part_importance(c, res):
    iv = res['parts'].get('iv') or part_iv(c, res)
    model = joblib.load(MODELS/'lightgbm.joblib')
    sv = model.predict_proba(c.xv)[:, 1]
    fsv, _ = c.frozen_scores()
    assert np.allclose(sv, fsv, rtol=0, atol=1e-12), 'frozen LightGBM does not reproduce stored validation scores'
    base = average_precision_score(c.yv, sv)
    fields = sorted({base_field(col) for col in c.xv})
    rng = np.random.default_rng(SEED); rows = []
    for f in fields:
        cols = [col for col in c.xv if base_field(col) == f]
        drops = []
        for _ in range(PERM_REPEATS):
            xp = c.xv.copy(); perm = rng.permutation(len(xp))
            for col in cols: xp[col] = xp[col].to_numpy()[perm] if not isinstance(xp[col].dtype, pd.CategoricalDtype) else pd.Categorical.from_codes(xp[col].cat.codes.to_numpy()[perm], categories=xp[col].cat.categories)
            drops.append(base - average_precision_score(c.yv, model.predict_proba(xp)[:, 1]))
        rows.append(dict(field=f, perm_ap_drop=float(np.mean(drops)), perm_ap_drop_sd=float(np.std(drops))))
    perm = pd.DataFrame(rows)
    shp = np.load(OUT/'predictions/validation_shap.npz', allow_pickle=False)
    feats = [str(s) for s in shp['features']]; vals = shp['shap_values']
    shap_field = pd.Series({f: float(np.abs(sum(vals[:, i] for i, col in enumerate(feats) if base_field(col) == f)).mean()) for f in fields})
    t = perm.set_index('field')
    t['shap_mean_abs'] = shap_field
    t['iv'] = pd.DataFrame(iv['table']).set_index('field').iv
    for col in ['perm_ap_drop', 'shap_mean_abs', 'iv']:
        t[col.replace('perm_ap_drop', 'perm').replace('shap_mean_abs', 'shap') + '_rank'] = t[col].rank(ascending=False, method='min').astype(int)
    t = t.sort_values('perm_ap_drop', ascending=False).reset_index()
    save_table(t, 'fs_importance.csv')
    rho = {f'{a}_vs_{b}': float(spearmanr(t[a], t[b]).statistic) for a, b in
           [('perm_ap_drop', 'shap_mean_abs'), ('perm_ap_drop', 'iv'), ('shap_mean_abs', 'iv')]}
    top10 = {k: t.sort_values(k, ascending=False).field.head(10).tolist() for k in ['perm_ap_drop', 'shap_mean_abs', 'iv']}
    agree = sorted(set(top10['perm_ap_drop']) & set(top10['shap_mean_abs']) & set(top10['iv']))
    log(f'importance: Spearman {rho}; top-10 in all three: {agree}')
    fig, ax = plt.subplots(1, 3, figsize=(15, 6.5), sharey=True)
    order = t.sort_values('perm_ap_drop').field
    for a, (col, title, color) in zip(ax, [('perm_ap_drop', 'Permutation: AP drop (month 6)', '#147d92'),
                                           ('shap_mean_abs', 'TreeSHAP: mean |contribution|', '#9bbb59'),
                                           ('iv', 'IV (training months)', '#c0504d')]):
        a.barh(order, t.set_index('field').loc[order, col], color=color); a.set_title(title); a.tick_params(axis='y', labelsize=8)
    fig_save(fig, 'fs_importance.png')
    return dict(base_val_ap=float(base), repeats=PERM_REPEATS, spearman=rho, top10=top10, top10_all_three=agree,
                table=t.to_dict(orient='records'))


RUNNERS = dict(iv=part_iv, selection=part_selection, derived=part_derived, importance=part_importance)


def run(parts=None, force=False):
    parts = parts or PARTS
    before = frozen_hashes()
    FIG.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True)
    res = json.loads(RESULT.read_text(encoding='utf-8')) if RESULT.exists() else {}
    res.setdefault('parts', {})
    res.update(stage='feature analysis and feature engineering (post-hoc)',
               created_singapore=datetime.now(timezone(timedelta(hours=8))).isoformat(),
               discipline=['frozen artifacts read only; hashes checked before/after',
                           'bins, encodings, selection rule and early stopping use months 0-6 only; month 7 reported once',
                           'designed after month-7 results were seen: exploratory, does not replace the frozen model'],
               frozen_hashes_before=before)
    todo = [p for p in parts if force or p not in res['parts']]
    if todo:
        ctx = Ctx()
        with threadpool_limits(limits=4):
            for p in PARTS:
                if p not in todo: continue
                log(f'=== part: {p}'); t0 = time.perf_counter()
                res['parts'][p] = strict_json(RUNNERS[p](ctx, res)); res['parts'][p]['seconds'] = round(time.perf_counter()-t0, 1)
                save_json(strict_json(res), RESULT)
    else:
        log('All requested parts already done (use --force to redo).')
    after = frozen_hashes()
    res['frozen_hashes_after'] = after; res['frozen_artifacts_unchanged'] = bool(before == after)
    save_json(strict_json(res), RESULT)
    if before != after: raise ValueError('A frozen artifact changed during the feature study.')
    log('Done. Frozen artifacts unchanged.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--parts', nargs='*', choices=PARTS, help='run only these parts (default: all missing)')
    ap.add_argument('--force', action='store_true', help='redo parts that already have results')
    a = ap.parse_args(); run(a.parts, a.force)
