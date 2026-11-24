"""Course-method supplement: data-preparation and evaluation ideas taught in NTU AAI (CA6000/6001/6002/6003),
applied to the BAF data AFTER the frozen month-7 evaluation was seen.

Six parts (each can be run alone with --parts):
  skew_scale     skewness / kurtosis, Box-Cox-family and log transforms, three scalers      (CA6003 W2, CA6000 T11)
  outliers       IQR / Z-score outliers: fraud rate inside them, clipping, dropping         (CA6003 W1, W4)
  distributions  fraud-vs-normal distributions, correlation, multicollinearity (VIF)         (CA6002 2.1, CA6003 W4)
  split          random split / K-fold versus the time split used by the project            (CA6001 Ch1, CA6003 W4)
(The earlier MLP learning-curve, ROC/confusion-matrix and Shapiro normality parts were removed on 2026-10-21:
 they repeated the main report or carried no information at this sample size.)
  smote          SMOTE versus class weights versus nothing                                 (CA6003 W2-W3)
  docs           model card, feature dictionary and data-contract check                    (CA6003 W5)

Discipline (same as src/extensions.py)
  * Frozen artifacts are only READ; their SHA-256 are recorded before and after.
  * Every new fit uses months 0-5 only (the random-split study pools months 0-6 on purpose and says so),
    every new choice (thresholds, early stopping) uses month 6 only, month 7 is reported.
  * Questions were formed after month-7 results were seen: exploratory, nothing is re-selected.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(key, '4')
import argparse
import json
import time
import warnings
from datetime import datetime, timezone, timedelta
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder, PowerTransformer,
                                   RobustScaler, StandardScaler)
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from .common import *          # ROOT, SEED, CATEGORICAL, MISSING_FIELDS, load_data, clean_features, make_preprocessor, ...
from .evaluate import style

OUT = ROOT/'outputs'; MODELS = OUT/'models'; FIG = OUT/'figures'; TABLES = OUT/'tables'; DOCS = ROOT/'docs'
RESULT = OUT/'coursework.json'
PARTS = ['skew_scale', 'outliers', 'distributions', 'split', 'smote', 'docs']
FROZEN_FILES = ['outputs/models/frozen_manifest.json', 'outputs/models/lightgbm.joblib',
                'outputs/models/distilled_tree.joblib', 'outputs/models/preprocessor.joblib',
                'outputs/models/schema.json', 'outputs/models/logistic_regression.joblib',
                'outputs/models/mlp.joblib', 'outputs/models/isolation_forest.joblib',
                'outputs/evaluation.json', 'outputs/explanation.json',
                'outputs/predictions/validation.parquet', 'outputs/predictions/test.parquet']
SKEW_LIMIT = 1.0        # |skewness| above this is called "highly skewed" (a common rule of thumb)
CONT_MIN_UNIQUE = 20    # a numeric field with more distinct values than this is treated as continuous
CORR_SAMPLE = 200_000


def log(s): print(time.strftime('%H:%M:%S'), s, flush=True)


def strict_json(x):
    """NaN/inf are not valid JSON; store them as null."""
    if isinstance(x, dict): return {k: strict_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [strict_json(v) for v in x]
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.bool_): return bool(x)
    return x


def frozen_hashes():
    return {p: sha256(ROOT/p) for p in FROZEN_FILES if (ROOT/p).exists()}


# ======================================================================= small reusable pieces
def continuous_columns(x, min_unique=CONT_MIN_UNIQUE):
    """Numeric, non-flag columns with more than `min_unique` distinct training values."""
    cols = [c for c in x if c not in CATEGORICAL and not c.endswith('__missing')]
    return [c for c in cols if x[c].nunique(dropna=True) > min_unique]


def skew_kurtosis_table(x, cols, skew_limit=SKEW_LIMIT):
    """Per-field shape statistics (pandas: unbiased skewness, EXCESS kurtosis; NaN ignored)."""
    rows = []
    for c in cols:
        s = x[c].astype('float64')
        v = s.dropna()
        rows.append(dict(field=c, n_valid=int(len(v)), missing_share=float(s.isna().mean()),
                         skewness=float(s.skew()), excess_kurtosis=float(s.kurt()),
                         minimum=float(v.min()), median=float(v.median()), p99=float(v.quantile(.99)), maximum=float(v.max()),
                         highly_skewed=bool(abs(s.skew()) > skew_limit), log1p_ok=bool(v.min() >= 0)))
    return pd.DataFrame(rows).sort_values('skewness', key=lambda t: -t.abs()).reset_index(drop=True)


class FenceClipper(BaseEstimator, TransformerMixin):
    """Winsorise to Tukey fences q1 - k*IQR, q3 + k*IQR learned on the training rows only. NaN passes through."""
    def __init__(self, k=1.5): self.k = k
    def fit(self, X, y=None):
        X = np.asarray(X, dtype='float64')
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        self.lo_, self.hi_ = q1 - self.k*(q3-q1), q3 + self.k*(q3-q1)
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, dtype='float64'), self.lo_, self.hi_)


def _log1p(a): return np.log1p(np.clip(np.asarray(a, dtype='float64'), 0, None))


def build_prep(x, kind, cont=(), log_cols=()):
    """Same layout as common.make_preprocessor (median impute -> scale numeric, one-hot categorical);
    only the numeric treatment changes. kinds: minmax, robust, yeo, log1p, clip."""
    numeric = [c for c in x if c not in CATEGORICAL]
    impute = lambda: SimpleImputer(strategy='median', keep_empty_features=True)
    std = lambda: StandardScaler()
    if kind == 'minmax':
        blocks = [('num', Pipeline([('i', impute()), ('s', MinMaxScaler())]), numeric)]
    elif kind == 'robust':
        blocks = [('num', Pipeline([('i', impute()), ('s', RobustScaler())]), numeric)]
    else:
        special = [c for c in (log_cols if kind == 'log1p' else cont) if c in numeric]
        rest = [c for c in numeric if c not in special]
        if kind == 'yeo':
            tf = Pipeline([('i', impute()), ('p', PowerTransformer(method='yeo-johnson', standardize=True))])
        elif kind == 'log1p':
            tf = Pipeline([('i', impute()), ('l', FunctionTransformer(_log1p, validate=False)), ('s', std())])
        elif kind == 'clip':
            tf = Pipeline([('c', FenceClipper(1.5)), ('i', impute()), ('s', std())])
        else:
            raise ValueError(kind)
        blocks = [('special', tf, special), ('rest', Pipeline([('i', impute()), ('s', std())]), rest)]
    blocks.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32), CATEGORICAL))
    return ColumnTransformer(blocks, sparse_threshold=0, verbose_feature_names_out=False)


def outlier_table(x, y, cols):
    """For each field: how many rows fall outside the 1.5*IQR / 3*IQR fences and |z|>3 (training stats),
    and the fraud rate inside versus outside."""
    rows, any3 = [], np.zeros(len(x), dtype=bool)
    y = np.asarray(y)
    for c in cols:
        v = x[c].astype('float64').to_numpy()
        ok = ~np.isnan(v)
        q1, q3 = np.nanpercentile(v, [25, 75]); iqr = q3-q1
        mu, sd = np.nanmean(v), np.nanstd(v)
        rules = {'iqr1.5': (v < q1-1.5*iqr) | (v > q3+1.5*iqr), 'iqr3': (v < q1-3*iqr) | (v > q3+3*iqr),
                 'z3': np.abs(v-mu) > 3*sd if sd > 0 else np.zeros(len(v), bool)}
        for name, flag in rules.items():
            flag = flag & ok
            n_in = int(flag.sum()); n_out = int((ok & ~flag).sum())
            r_in = float(y[flag].mean()) if n_in else None; r_out = float(y[ok & ~flag].mean()) if n_out else None
            rows.append(dict(field=c, rule=name, flagged_rows=n_in, flagged_share=n_in/len(v),
                             fraud_rate_flagged=r_in, fraud_rate_not_flagged=r_out,
                             lift=(r_in/r_out) if (r_in is not None and r_out) else None))
        any3 |= rules['iqr3'] & ok
    return pd.DataFrame(rows), any3


def fence_mask(x, cols, k=3.0, ref=None):
    """Rows outside q1-k*IQR..q3+k*IQR in ANY of `cols` (fences from `ref`, default x itself)."""
    ref = x if ref is None else ref
    flag = np.zeros(len(x), dtype=bool)
    for c in cols:
        q1, q3 = np.nanpercentile(ref[c].astype('float64'), [25, 75])
        v = x[c].astype('float64').to_numpy()
        flag |= (v < q1-k*(q3-q1)) | (v > q3+k*(q3-q1))   # NaN compares False
    return flag


def vif_table(x, cols):
    """Variance inflation factors = diagonal of the inverse correlation matrix (median-imputed fields)."""
    m = x[cols].astype('float64')
    m = m.fillna(m.median())
    keep = [c for c in cols if m[c].std() > 0]
    corr = np.corrcoef(m[keep].to_numpy().T)
    vif = np.diag(np.linalg.pinv(corr))
    return pd.DataFrame(dict(field=keep, vif=vif)).sort_values('vif', ascending=False).reset_index(drop=True)


def smote(z, y, ratio=1.0, k=5, seed=SEED):
    """Plain SMOTE (Chawla et al. 2002) on the minority class only: new = a + u*(neighbour - a), u~U(0,1).
    ratio = desired minority/majority count after oversampling. Returns the AUGMENTED arrays (originals first).
    Limitation: interpolates one-hot and flag columns too, giving fractional values (SMOTE-NC would avoid this)."""
    y = np.asarray(y); rng = np.random.default_rng(seed)
    pos = z[y == 1]; n_neg = int((y == 0).sum()); n_new = int(ratio*n_neg) - len(pos)
    if n_new <= 0: return z, y
    nn = NearestNeighbors(n_neighbors=min(k, len(pos)-1)+1).fit(pos)
    neigh = nn.kneighbors(pos, return_distance=False)[:, 1:]
    base = rng.integers(0, len(pos), n_new)
    other = neigh[base, rng.integers(0, neigh.shape[1], n_new)]
    gap = rng.random((n_new, 1), dtype='float32')
    synth = (pos[base] + gap*(pos[other]-pos[base])).astype(z.dtype)
    return np.vstack([z, synth]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])


def fit_lr(z, y, C, class_weight='balanced'):
    return LogisticRegression(C=C, class_weight=class_weight, solver='lbfgs', max_iter=500, tol=1e-5,
                              random_state=SEED).fit(z, y)


def fit_lgbm_cfg(xtr, ytr, xv, yv, params, spw=None, n_estimators=900, early=True):
    """Frozen LightGBM recipe from src/train.py; `spw` overrides scale_pos_weight, early=False fixes the tree count."""
    import lightgbm as lgb
    ratio = float((ytr == 0).sum()/(ytr == 1).sum()) if spw is None else spw
    model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=.04, num_leaves=params['num_leaves'],
        min_child_samples=150, reg_lambda=params['reg_lambda'], colsample_bytree=.9, subsample=.9, subsample_freq=1,
        scale_pos_weight=ratio, n_jobs=4, deterministic=True, force_col_wise=True, random_state=SEED,
        verbosity=-1, metric='average_precision')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if early:
            model.fit(xtr, ytr, eval_set=[(xv, yv)], eval_metric='average_precision',
                      callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False)])
        else:
            model.fit(xtr, ytr)
    return model


def eval_scores(yv, sv, yt, st, ids_t=None, kt=None):
    """Threshold from validation FPR<=5%; everything else on month 7 (frozen-threshold semantics),
    plus same-capacity recall and the naive 0.5 cut-off (to expose the accuracy paradox)."""
    sv = np.asarray(sv, dtype=float); st = np.asarray(st, dtype=float)
    thr = threshold_at_fpr(yv, sv); mv = metrics(yv, sv, thr); mt = metrics(yt, st, thr)
    p5 = binary_metrics(yt, st >= 0.5)
    out = dict(threshold=thr, val_ap=mv['pr_auc_ap'], val_recall=mv['recall'], val_fpr=mv['fpr'],
               test_ap=mt['pr_auc_ap'], test_roc_auc=mt['roc_auc'], test_recall=mt['recall'], test_fpr=mt['fpr'],
               test_precision=mt['precision'], test_alert_rate=mt['alert_rate'],
               test_mean_score=float(st.mean()), test_base_rate=float(np.mean(yt)),
               test_accuracy_at_0_5=float((p5['tp']+p5['tn'])/p5['n']), test_recall_at_0_5=p5['recall'], test_fpr_at_0_5=p5['fpr'])
    if ids_t is not None and kt is not None:
        out['test_same_budget_recall'] = binary_metrics(yt, top_k(st, kt, ids_t))['recall']
    return out


# ======================================================================= shared context
class Ctx:
    """Loads data and frozen artifacts once (read only)."""
    def __init__(self):
        self.manifest = json.loads((MODELS/'frozen_manifest.json').read_text(encoding='utf-8'))
        self.schema = json.loads((MODELS/'schema.json').read_text(encoding='utf-8'))
        assert sha256(ROOT/'data/raw/Base.csv') == self.manifest['raw_sha256'], 'raw data changed'
        log('Loading Base and frozen artifacts (read only)')
        df = load_data(); self.df = df
        tr = (df.month <= 5).to_numpy(); va = (df.month == 6).to_numpy(); te = (df.month == 7).to_numpy()
        self.xtr, _ = clean_features(df.loc[tr], self.schema); self.xv, _ = clean_features(df.loc[va], self.schema)
        self.xt, _ = clean_features(df.loc[te], self.schema)
        self.ytr, self.yv, self.yt = (df.loc[m, 'fraud_bool'].to_numpy() for m in (tr, va, te))
        self.ids_v, self.ids_t = df.index[va].to_numpy(), df.index[te].to_numpy()
        self.val_pred = pd.read_parquet(OUT/'predictions/validation.parquet')
        self.test_pred = pd.read_parquet(OUT/'predictions/test.parquet')
        assert np.array_equal(self.val_pred.row_id.to_numpy(), self.ids_v)
        assert np.array_equal(self.test_pred.row_id.to_numpy(), self.ids_t)
        self.kt = int(np.floor(self.manifest['review_budget_rate']*len(self.yt)))
        self.cont = continuous_columns(self.xtr)
        self.lgb_params = self.manifest['models']['lightgbm']['params']
        self.lr_C = self.manifest['models']['logistic_regression']['params']['C']

    def frozen_scores(self, model):
        return self.val_pred[model].to_numpy(float), self.test_pred[model].to_numpy(float)

    def row(self, label, sv, st, **extra):
        return dict(label=label, **eval_scores(self.yv, sv, self.yt, st, self.ids_t, self.kt), **extra)


def save_table(df, name):
    df.to_csv(TABLES/name, index=False, encoding='utf-8-sig')


def fig_save(fig, name):
    fig.tight_layout(); fig.savefig(FIG/name, dpi=150); plt.close(fig)


# ======================================================================= part 1
def part_skew_scale(c):
    cont = c.cont
    shape = skew_kurtosis_table(c.xtr, cont)
    save_table(shape, 'cw_skew_kurtosis.csv')
    log_cols = shape[shape.highly_skewed & shape.log1p_ok].field.tolist()
    log(f'skew: {int(shape.highly_skewed.sum())} of {len(cont)} continuous fields have |skew|>{SKEW_LIMIT}; log1p candidates: {log_cols}')
    rows = [c.row('LR 标准化（冻结参考）', *c.frozen_scores('logistic_regression'), family='LR', kind='reference')]
    for kind, label in [('minmax', 'LR Min-Max 缩放'), ('robust', 'LR Robust 缩放'),
                        ('yeo', 'LR Yeo-Johnson 幂变换'), ('log1p', 'LR 对偏斜字段取 log1p')]:
        t0 = time.perf_counter()
        prep = build_prep(c.xtr, kind, cont, log_cols)
        ztr = prep.fit_transform(c.xtr).astype('float32'); zv = prep.transform(c.xv).astype('float32'); zt = prep.transform(c.xt).astype('float32')
        lr = fit_lr(ztr, c.ytr, c.lr_C)
        rows.append(c.row(label, lr.predict_proba(zv)[:, 1], lr.predict_proba(zt)[:, 1], family='LR', kind=kind,
                          fit_seconds=time.perf_counter()-t0, n_iter=int(lr.n_iter_[0])))
        log(f'  {label}: test AP={rows[-1]["test_ap"]:.4f} recall={rows[-1]["test_recall"]:.4f}')
        del ztr, zv, zt
    rows.append(c.row('LightGBM（冻结参考）', *c.frozen_scores('lightgbm'), family='LightGBM', kind='reference'))
    # trees are (almost) invariant to monotone transforms: refit LightGBM with log1p-ed fields
    if log_cols:
        def tr(x):
            x = x.copy()
            for col in log_cols: x[col] = _log1p(x[col]).astype('float32')
            return x
        t0 = time.perf_counter()
        model = fit_lgbm_cfg(tr(c.xtr), c.ytr, tr(c.xv), c.yv, c.lgb_params)
        sv, st = model.predict_proba(tr(c.xv))[:, 1], model.predict_proba(tr(c.xt))[:, 1]
        ft = c.frozen_scores('lightgbm')[1]
        rows.append(c.row('LightGBM 对偏斜字段取 log1p', sv, st, family='LightGBM', kind='log1p', fit_seconds=time.perf_counter()-t0,
                          spearman_with_frozen_test=float(stats.spearmanr(st, ft).statistic), best_iteration=int(model.best_iteration_)))
    table = pd.DataFrame(rows); save_table(table, 'cw_scaling_comparison.csv')
    # figure: skewness bars + histogram before/after for the most skewed field
    style()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    top = shape.head(12).iloc[::-1]
    ax[0].barh(top.field, top.skewness, color=['#c0504d' if h else '#4f81bd' for h in top.highly_skewed])
    ax[0].axvline(SKEW_LIMIT, color='k', ls=':'); ax[0].axvline(-SKEW_LIMIT, color='k', ls=':')
    ax[0].set(title='Skewness of continuous fields (training months)', xlabel='skewness')
    if log_cols:
        f = shape[shape.field.isin(log_cols)].field.iloc[0]
        v = c.xtr[f].astype('float64').dropna()
        ax[1].hist(v.clip(upper=v.quantile(.995)), bins=60, color='#4f81bd'); ax[1].set(title=f'{f}: raw (clipped at p99.5)', xlabel=f)
        ax[2].hist(_log1p(v), bins=60, color='#9bbb59'); ax[2].set(title=f'{f}: after log1p', xlabel=f'log1p({f})')
    fig_save(fig, 'cw_skewness.png')
    return dict(n_continuous=len(cont), continuous_fields=cont, highly_skewed=shape[shape.highly_skewed].field.tolist(),
                log1p_fields=log_cols, skew_limit=SKEW_LIMIT, table='outputs/tables/cw_scaling_comparison.csv',
                rows=rows, shape_top=shape.head(8).to_dict(orient='records'))


# ======================================================================= part 2
def part_outliers(c):
    tab, any3 = outlier_table(c.xtr, c.ytr, c.cont)
    save_table(tab, 'cw_outliers.csv')
    row_level = dict(share_rows_any_iqr3=float(any3.mean()), fraud_rate_any_iqr3=float(c.ytr[any3].mean()) if any3.any() else None,
                     fraud_rate_none=float(c.ytr[~any3].mean()), base_rate=float(c.ytr.mean()))
    row_level['lift'] = row_level['fraud_rate_any_iqr3']/row_level['fraud_rate_none'] if row_level['fraud_rate_any_iqr3'] else None
    log(f'outliers: {row_level["share_rows_any_iqr3"]:.1%} rows are 3xIQR outliers in some field; fraud rate {row_level["fraud_rate_any_iqr3"]} vs {row_level["fraud_rate_none"]:.4f}')
    rows = [c.row('LR 标准化（冻结参考）', *c.frozen_scores('logistic_regression'), kind='reference')]
    # (a) winsorise at 1.5 IQR fences (learned on training)
    prep = build_prep(c.xtr, 'clip', c.cont)
    ztr = prep.fit_transform(c.xtr).astype('float32'); zv = prep.transform(c.xv).astype('float32'); zt = prep.transform(c.xt).astype('float32')
    lr = fit_lr(ztr, c.ytr, c.lr_C)
    rows.append(c.row('LR 1.5×IQR 截尾（winsorise）', lr.predict_proba(zv)[:, 1], lr.predict_proba(zt)[:, 1], kind='clip'))
    del ztr, zv, zt
    # (b) drop training rows that are 3xIQR outliers (evaluation rows are never dropped)
    keep = ~any3
    xk, yk = c.xtr.loc[keep], c.ytr[keep]
    prep = make_preprocessor(xk)
    zk = prep.fit_transform(xk).astype('float32'); zv = prep.transform(c.xv).astype('float32'); zt = prep.transform(c.xt).astype('float32')
    lr = fit_lr(zk, yk, c.lr_C)
    rows.append(c.row('LR 训练时删除 3×IQR 离群行', lr.predict_proba(zv)[:, 1], lr.predict_proba(zt)[:, 1], kind='drop',
                      train_rows_dropped=int(any3.sum()), train_frauds_dropped=int(c.ytr[any3].sum()),
                      share_frauds_dropped=float(c.ytr[any3].sum()/c.ytr.sum())))
    del zk, zv, zt
    table = pd.DataFrame(rows); save_table(table, 'cw_outlier_experiments.csv')
    style()
    iqr3 = tab[tab.rule == 'iqr3'].dropna(subset=['lift']).sort_values('lift').tail(12)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.barh(iqr3.field, iqr3.lift, color='#4f81bd'); ax.axvline(1, color='k', ls=':')
    ax.set(title='Fraud-rate lift inside 3×IQR outliers (training months)', xlabel='fraud rate inside / outside')
    fig_save(fig, 'cw_outlier_lift.png')
    return dict(row_level=row_level, rows=rows, table='outputs/tables/cw_outlier_experiments.csv',
                top_lift=iqr3.sort_values('lift', ascending=False).head(5).to_dict(orient='records'))


# ======================================================================= part 3
def part_distributions(c):
    x, y = c.xtr, c.ytr
    rows = []
    for col in c.cont:
        a = x[col].astype('float64').to_numpy()
        f, n = a[(y == 1) & ~np.isnan(a)], a[(y == 0) & ~np.isnan(a)]
        rng = np.random.default_rng(SEED)
        ks = stats.ks_2samp(f, rng.choice(n, min(len(n), 100_000), replace=False)).statistic
        pooled = np.sqrt(((len(f)-1)*f.var(ddof=1)+(len(n)-1)*n.var(ddof=1))/(len(f)+len(n)-2))
        rows.append(dict(field=col, median_fraud=float(np.median(f)), median_normal=float(np.median(n)),
                         standardized_mean_diff=float((f.mean()-n.mean())/pooled) if pooled > 0 else None, ks_statistic=float(ks)))
    dist = pd.DataFrame(rows).sort_values('ks_statistic', ascending=False).reset_index(drop=True)
    save_table(dist, 'cw_class_distribution.csv')
    # correlation (Spearman is robust to the skew found in part 1) on a fixed random sample
    samp = x[c.cont].astype('float64').sample(min(CORR_SAMPLE, len(x)), random_state=SEED)
    spear = samp.corr(method='spearman'); pear = samp.corr(method='pearson')
    pairs = []
    for i, a in enumerate(c.cont):
        for b in c.cont[i+1:]:
            pairs.append(dict(field_a=a, field_b=b, spearman=float(spear.loc[a, b]), pearson=float(pear.loc[a, b])))
    pairs = pd.DataFrame(pairs).sort_values('spearman', key=lambda t: -t.abs()).reset_index(drop=True)
    save_table(pairs, 'cw_correlation_pairs.csv')
    numeric = [col for col in x if col not in CATEGORICAL and not col.endswith('__missing')]
    vif = vif_table(x.sample(min(CORR_SAMPLE, len(x)), random_state=SEED), numeric)
    save_table(vif, 'cw_vif.csv')
    style()
    top = dist.head(6).field.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    nrng = np.random.default_rng(SEED)
    for ax, col in zip(axes.ravel(), top):
        a = x[col].astype('float64').to_numpy(); ok = ~np.isnan(a)
        lo, hi = np.nanpercentile(a, [1, 99])
        g0 = a[(y == 0) & ok]; g1 = a[(y == 1) & ok]
        g0 = np.clip(nrng.choice(g0, min(20000, len(g0)), replace=False), lo, hi); g1 = np.clip(g1, lo, hi)
        parts = ax.violinplot([g0, g1], showmedians=True)
        for body, colr in zip(parts['bodies'], ['#4f81bd', '#c0504d']): body.set_facecolor(colr); body.set_alpha(.6)
        ax.set_xticks([1, 2], ['normal', 'fraud']); ax.set_title(col)
    fig.suptitle('Distribution by label, top-6 fields by KS statistic (training months, clipped at p1–p99)')
    fig_save(fig, 'cw_distribution_by_label.png')
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(spear.to_numpy(), vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(c.cont)), c.cont, rotation=90, fontsize=7); ax.set_yticks(range(len(c.cont)), c.cont, fontsize=7)
    fig.colorbar(im, ax=ax, label='Spearman'); ax.set_title('Spearman correlation, continuous fields (200k-row training sample)')
    fig_save(fig, 'cw_correlation_heatmap.png')
    return dict(top_ks=dist.head(5).to_dict(orient='records'), strong_pairs=pairs[pairs.spearman.abs() >= .7].to_dict(orient='records'),
                n_pairs=len(pairs), vif_over_5=vif[vif.vif > 5].to_dict(orient='records'), vif_max=float(vif.vif.max()))


# ======================================================================= part 4
def part_split(c):
    """Random hold-out and 5-fold CV on months 0-6 pooled, versus the project's time split."""
    pool_x = pd.concat([c.xtr, c.xv]); pool_y = np.concatenate([c.ytr, c.yv])
    idx = np.arange(len(pool_y))
    tr_i, va_i = train_test_split(idx, test_size=len(c.yv)/len(pool_y), stratify=pool_y, random_state=SEED)
    t0 = time.perf_counter()
    model = fit_lgbm_cfg(pool_x.iloc[tr_i], pool_y[tr_i], pool_x.iloc[va_i], pool_y[va_i], c.lgb_params)
    sv = model.predict_proba(pool_x.iloc[va_i])[:, 1]; st = model.predict_proba(c.xt)[:, 1]
    thr = threshold_at_fpr(pool_y[va_i], sv)
    hold = metrics(pool_y[va_i], sv, thr); fut = metrics(c.yt, st, thr)
    main = c.manifest['models']['lightgbm']['validation']
    fv, ft = c.frozen_scores('lightgbm'); main_thr = c.manifest['models']['lightgbm']['threshold']
    frozen_test = metrics(c.yt, ft, main_thr)
    cmp_rows = [
        dict(setting='随机切分（月 0–6 打乱，1/7 留出）', estimate_source='随机留出集', est_ap=hold['pr_auc_ap'], est_recall=hold['recall'],
             est_fpr=hold['fpr'], month7_ap=fut['pr_auc_ap'], month7_recall=fut['recall'], month7_fpr=fut['fpr'],
             best_iteration=int(model.best_iteration_), fit_seconds=time.perf_counter()-t0),
        dict(setting='时间切分（项目采用：月 0–5 训练，月 6 选阈值）', estimate_source='月 6 验证集', est_ap=main['pr_auc_ap'], est_recall=main['recall'],
             est_fpr=main['fpr'], month7_ap=frozen_test['pr_auc_ap'], month7_recall=frozen_test['recall'], month7_fpr=frozen_test['fpr'],
             best_iteration=int(c.lgb_params['best_iteration']), fit_seconds=None)]
    for r in cmp_rows:
        r['ap_gap_estimate_minus_month7'] = r['est_ap']-r['month7_ap']; r['recall_gap_estimate_minus_month7'] = r['est_recall']-r['month7_recall']
    log(f'split: random hold-out AP {hold["pr_auc_ap"]:.4f} vs month-7 AP {fut["pr_auc_ap"]:.4f}; time split {main["pr_auc_ap"]:.4f} vs {frozen_test["pr_auc_ap"]:.4f}')
    # 5-fold stratified CV, fixed tree count (no early stopping inside folds)
    folds = []
    n_trees = int(model.best_iteration_)
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(idx, pool_y), 1):
        m = fit_lgbm_cfg(pool_x.iloc[a], pool_y[a], None, None, c.lgb_params, n_estimators=n_trees, early=False)
        s = m.predict_proba(pool_x.iloc[b])[:, 1]; t = threshold_at_fpr(pool_y[b], s); mm = metrics(pool_y[b], s, t)
        folds.append(dict(fold=k, ap=mm['pr_auc_ap'], recall=mm['recall'], roc_auc=mm['roc_auc']))
        log(f'  fold {k}: AP={mm["pr_auc_ap"]:.4f} recall={mm["recall"]:.4f}')
    cv = pd.DataFrame(folds)
    cv_summary = dict(ap_mean=float(cv.ap.mean()), ap_std=float(cv.ap.std(ddof=1)), recall_mean=float(cv.recall.mean()),
                      recall_std=float(cv.recall.std(ddof=1)), trees=n_trees)
    # forward-in-time reference from the rolling back-test, if it has been run
    rolling = None
    rpath = TABLES/'rolling_metrics.csv'
    if rpath.exists():
        r = pd.read_csv(rpath, encoding='utf-8-sig')
        r = r[(r.variant == 'baseline') & (r.split == 'test') & (r.policy == 'frozen_threshold')]
        if len(r): rolling = dict(windows=int(len(r)), ap_mean=float(r.pr_auc_ap.mean()), ap_min=float(r.pr_auc_ap.min()), ap_max=float(r.pr_auc_ap.max()))
    save_table(pd.DataFrame(cmp_rows), 'cw_split_comparison.csv'); save_table(cv, 'cw_random_cv_folds.csv')
    style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(2); w = .35
    ax.bar(xs-w/2, [r['est_ap'] for r in cmp_rows], w, label='estimate on hold-out', color='#4f81bd')
    ax.bar(xs+w/2, [r['month7_ap'] for r in cmp_rows], w, label='true month-7 result', color='#c0504d')
    ax.set_xticks(xs, ['random split', 'time split']); ax.set_ylabel('AP'); ax.legend(); ax.set_title('Estimated vs real performance (LightGBM, same recipe)')
    fig_save(fig, 'cw_split_comparison.png')
    return dict(rows=cmp_rows, cv=cv_summary, folds=folds, rolling_reference=rolling,
                note='The random split pools months 0-6 (train+validation months) by design; month 7 is still used only for reporting.')


# ======================================================================= part 5
def part_smote(c):
    prep = joblib.load(MODELS/'preprocessor.joblib')
    ztr = prep.transform(c.xtr).astype('float32'); zv = prep.transform(c.xv).astype('float32'); zt = prep.transform(c.xt).astype('float32')
    rows = [c.row('LR 类别权重（冻结参考）', *c.frozen_scores('logistic_regression'), family='LR', kind='class_weight', label_en='LR + class weight (frozen)'),
            c.row('LightGBM scale_pos_weight（冻结参考）', *c.frozen_scores('lightgbm'), family='LightGBM', kind='class_weight', label_en='LightGBM + scale_pos_weight (frozen)')]
    t0 = time.perf_counter()
    lr = fit_lr(ztr, c.ytr, c.lr_C, class_weight=None)
    rows.append(c.row('LR 不处理不平衡', lr.predict_proba(zv)[:, 1], lr.predict_proba(zt)[:, 1], family='LR', kind='none', fit_seconds=time.perf_counter()-t0, label_en='LR, no imbalance handling'))
    log('  LR unweighted done'); t0 = time.perf_counter()
    zs, ys = smote(ztr, c.ytr, ratio=1.0)
    lr = fit_lr(zs, ys, c.lr_C, class_weight=None)
    rows.append(c.row('LR + SMOTE（少数类补到 1:1）', lr.predict_proba(zv)[:, 1], lr.predict_proba(zt)[:, 1], family='LR', kind='smote',
                      fit_seconds=time.perf_counter()-t0, synthetic_rows=int(len(ys)-len(c.ytr)), label_en='LR + SMOTE (1:1)'))
    del zs, ys, ztr, zv, zt
    log('  LR + SMOTE done'); t0 = time.perf_counter()
    model = fit_lgbm_cfg(c.xtr, c.ytr, c.xv, c.yv, c.lgb_params, spw=1.0)
    rows.append(c.row('LightGBM 不处理不平衡（scale_pos_weight=1）', model.predict_proba(c.xv)[:, 1], model.predict_proba(c.xt)[:, 1],
                      family='LightGBM', kind='none', fit_seconds=time.perf_counter()-t0, best_iteration=int(model.best_iteration_), label_en='LightGBM, scale_pos_weight=1'))
    table = pd.DataFrame(rows); save_table(table, 'cw_imbalance_methods.csv')
    style(); fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    lab = [r['label_en'] for r in rows]
    ax[0].barh(lab[::-1], [r['test_ap'] for r in rows][::-1], color='#4f81bd'); ax[0].set(title='Test AP', xlabel='AP')
    ax[1].barh(lab[::-1], [r['test_mean_score'] for r in rows][::-1], color='#c0504d'); ax[1].axvline(rows[0]['test_base_rate'], color='k', ls=':', label='true base rate')
    ax[1].set(title='Mean score on month 7 (dotted: true base rate)', xlabel='mean score'); ax[1].legend(); ax[1].set_yticklabels([])
    fig_save(fig, 'cw_imbalance_methods.png')
    return dict(rows=rows, table='outputs/tables/cw_imbalance_methods.csv',
                note='SMOTE interpolates over the one-hot and missing-flag columns as well (plain SMOTE, not SMOTE-NC).')


# ======================================================================= part 6
FIELD_DOC = {  # Chinese description, datasheet range text, role
    'income': ('申请人年收入（十分位）', '[0.1, 0.9]'), 'name_email_similarity': ('邮箱与姓名的相似度，越高越相似', '[0, 1]'),
    'prev_address_months_count': ('前住址居住月数', '[-1, 380]，-1 为缺失'), 'current_address_months_count': ('现住址居住月数', '[-1, 429]，-1 为缺失'),
    'customer_age': ('申请人年龄（取整到十岁）', '[10, 90]'), 'days_since_request': ('申请提交后已过去的天数', '[0, 79]'),
    'intended_balcon_amount': ('开户初始转入金额', '[-16, 114]，负数为缺失'), 'payment_type': ('信贷付款计划类型（已匿名）', '5 个取值'),
    'zip_count_4w': ('近 4 周同邮编的申请数', '[1, 6830]'), 'velocity_6h': ('近 6 小时平均每小时申请数', '[-175, 16818]'),
    'velocity_24h': ('近 24 小时平均每小时申请数', '[1297, 9586]'), 'velocity_4w': ('近 4 周平均每小时申请数', '[2825, 7020]'),
    'bank_branch_count_8w': ('所选网点近 8 周申请总数', '[0, 2404]'), 'date_of_birth_distinct_emails_4w': ('近 4 周同生日申请人使用的不同邮箱数', '[0, 39]'),
    'employment_status': ('就业状态（已匿名）', '7 个取值'), 'credit_risk_score': ('内部申请风险评分', '[-191, 389]'),
    'email_is_free': ('邮箱是否为免费域名', '0/1'), 'housing_status': ('居住状态（已匿名）', '7 个取值'),
    'phone_home_valid': ('家庭电话是否有效', '0/1'), 'phone_mobile_valid': ('手机号是否有效', '0/1'),
    'bank_months_count': ('旧账户账龄月数', '[-1, 32]，-1 为缺失'), 'has_other_cards': ('是否持有同银行其他卡', '0/1'),
    'proposed_credit_limit': ('申请的信用额度', '[200, 2000]'), 'foreign_request': ('请求来源国是否与银行所在国不同', '0/1'),
    'source': ('申请渠道：浏览器或 App', '2 个取值'), 'session_length_in_minutes': ('会话时长（分钟）', '[-1, 107]，-1 为缺失'),
    'device_os': ('设备操作系统', '5 个取值'), 'keep_alive_session': ('退出时是否选择保持登录', '0/1'),
    'device_distinct_emails_8w': ('该设备近 8 周出现的不同邮箱数', '[-1, 2]，-1 为缺失'), 'device_fraud_count': ('该设备上的欺诈申请数', '[0, 1]（本数据恒为 0，建模时丢弃）'),
    'month': ('申请所在月份 0–7', '[0, 7]（只用于切分，不作为特征）'), 'fraud_bool': ('是否欺诈（目标）', '0/1'),
}
RANGES = {'income': (.1, .9), 'name_email_similarity': (0, 1), 'prev_address_months_count': (-1, 380), 'current_address_months_count': (-1, 429),
          'customer_age': (10, 90), 'days_since_request': (0, 79), 'intended_balcon_amount': (-16, 114), 'zip_count_4w': (1, 6830),
          'velocity_6h': (-175, 16818), 'velocity_24h': (1297, 9586), 'velocity_4w': (2825, 7020), 'bank_branch_count_8w': (0, 2404),
          'date_of_birth_distinct_emails_4w': (0, 39), 'credit_risk_score': (-191, 389), 'bank_months_count': (-1, 32),
          'proposed_credit_limit': (200, 2000), 'session_length_in_minutes': (-1, 107), 'device_distinct_emails_8w': (-1, 2),
          'device_fraud_count': (0, 1), 'month': (0, 7)}
BINARY = ['email_is_free', 'phone_home_valid', 'phone_mobile_valid', 'has_other_cards', 'foreign_request', 'keep_alive_session', 'fraud_bool']


def contract_check(df):
    """Compare every field with the ranges / value sets stated in the official datasheet."""
    rows = []
    for col, (lo, hi) in RANGES.items():
        v = df[col].astype('float64')
        rows.append(dict(field=col, rule=f'datasheet range [{lo}, {hi}]', observed_min=float(v.min()), observed_max=float(v.max()),
                         violations=int(((v < lo) | (v > hi)).sum())))
    for col in BINARY:
        v = df[col].astype('float64')
        rows.append(dict(field=col, rule='value in {0,1}', observed_min=float(v.min()), observed_max=float(v.max()), violations=int((~v.isin([0, 1])).sum())))
    return pd.DataFrame(rows)


def _pct(v, d=2): return '—' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{100*v:.{d}f}%'


def part_docs(c):
    DOCS.mkdir(exist_ok=True)
    df = c.df; manifest = c.manifest; ev = json.loads((OUT/'evaluation.json').read_text(encoding='utf-8'))
    schema_csv = pd.read_csv(TABLES/'eda_schema.csv', encoding='utf-8-sig').set_index('column')
    contract = contract_check(df); save_table(contract, 'cw_data_contract.csv')
    dropped = c.schema['drop_constants']; categories = c.schema['categories']
    sens = {'customer_age': '敏感属性（用于公平性分析；模型默认使用它）'}
    # ---- feature dictionary
    lines = ['# 特征说明与数据契约（字段字典）', '',
             '> 由 `src/coursework.py` 的 docs 部分根据官方数据说明 `references/datasheet.txt` 与 `outputs/tables/eda_schema.csv` 自动生成，请勿手工改数字。',
             '> 对应 CA6003 第 5 周的"特征文档""数据契约"。数据为 Feedzai 发布的合成数据，不是真实客户资料。', '',
             '## 字段表', '', '| 字段 | 类型 | 说明 | 数据说明给出的取值 | 本数据观测范围 | 缺失处理 | 建模中的角色 | 备注 |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for col in schema_csv.index:
        r = schema_csv.loc[col]; desc, rng = FIELD_DOC.get(col, ('', ''))
        kind = '类别' if col in CATEGORICAL else ('目标' if col == 'fraud_bool' else ('二值' if col in BINARY else '数值'))
        role = '目标' if col == 'fraud_bool' else ('仅用于时间切分' if col == 'month' else ('常量，已丢弃' if col in dropped else '特征'))
        miss = str(r['handling']).replace('|', '/')
        if col in MISSING_FIELDS: miss += f'；缺失占比 {_pct(r["effective_missing_fraction"])}；另加 `{col}__missing` 标记列'
        note = sens.get(col, '')
        if col in ('velocity_6h', 'velocity_24h', 'velocity_4w'): note = '逐月单调下降，相当于隐藏的时间指标（见教程第 14 章）'
        lines.append(f'| `{col}` | {kind} | {desc} | {rng} | {r["raw_min"]:g} ~ {r["raw_max"]:g} | {miss} | {role} | {note} |')
    lines += ['', '## 类别字段的取值（训练月）', '']
    for col, levels in categories.items(): lines.append(f'- `{col}`：{", ".join(levels)}')
    lines += ['', '## 数据契约核对', '', '逐字段核对全部 100 万行是否落在官方数据说明给出的范围内（说明里的范围是四舍五入后的，轻微越界应当先看原因，再决定是否放宽契约）。', '',
              '| 字段 | 规则 | 观测最小 | 观测最大 | 违反行数 |', '| --- | --- | --- | --- | --- |']
    for _, r in contract.iterrows(): lines.append(f'| `{r.field}` | {r.rule} | {r.observed_min:g} | {r.observed_max:g} | {int(r.violations):,} |')
    total = int(contract.violations.sum())
    lines += ['', f'合计违反 {total:,} 个字段取值；缺失用 -1 或负数表示的字段见上表"缺失处理"，这些不算违反（它们是官方规定的缺失编码）。']
    (DOCS/'特征说明_字段字典.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # ---- model card
    lg = manifest['models']['lightgbm']; test = next(r for r in ev['test_results'] if r['model'] == 'lightgbm')
    same = next((r for r in ev['same_budget_results'] if r['model'] == 'lightgbm'), None)
    fair = [r for r in ev['fairness'] if r['model'] == 'lightgbm' and r['split'] == 'test']
    gs = {r['group']: r for r in fair}
    hashes = frozen_hashes()
    L = ['# 模型卡：LightGBM 开户欺诈评分模型（BAF 学习项目）', '',
         '> 由 `src/coursework.py` 自动生成，数字全部来自冻结产物，请勿手工改。对应 CA6003 第 5 周的"模型卡""血缘与溯源"。', '',
         '## 1 基本信息', '', f'- 用途：个人学习项目，用于练习欺诈识别建模、阈值选择、解释、公平性与漂移检查。',
         f'- 模型：LightGBM 梯度提升树，候选 `{lg["candidate"]}`，超参数 {json.dumps({k: v for k, v in lg["params"].items()}, ensure_ascii=False)}。',
         f'- 随机种子 {manifest["seed"]}；训练月 {manifest["split"]["train"]}，验证月 {manifest["split"]["validation"]}，测试月 {manifest["split"]["test"]}。',
         f'- 选择规则：{manifest["selection"]}。冻结阈值 {lg["threshold"]:.4f}（在验证月按 FPR ≤ 5% 确定）。',
         '- **不适用于**：真实的开户审批、任何会影响真实申请人的决定。数据是合成的，模型没有经过任何生产环境验证。', '',
         '## 2 数据', '', f'- BAF Base（Feedzai 发布的合成开户申请数据）：{len(df):,} 行，欺诈 {int(df.fraud_bool.sum()):,} 笔（{_pct(df.fraud_bool.mean())}）。',
         f'- 原始文件 SHA-256：`{manifest["raw_sha256"]}`。', '- 缺失值：5 个字段用 -1、1 个字段用负数表示缺失（官方说明），建模前还原为缺失并加标记列。', '- 字段说明与契约核对见 [特征说明_字段字典.md](特征说明_字段字典.md)。', '',
         '## 3 测试月表现（冻结阈值，月 7）', '', '| 指标 | 数值 |', '| --- | --- |',
         f'| 召回率 | {_pct(test["recall"])}（95% 区间 {_pct(test["recall_ci"][0])}–{_pct(test["recall_ci"][1])}） |',
         f'| 误报率 FPR | {_pct(test["fpr"])}（验证月目标 ≤ 5%） |', f'| 精确率 | {_pct(test["precision"])} |',
         f'| AP | {test["pr_auc_ap"]:.3f}（随机基线约 {test["fraud"]/test["n"]:.3f}） |', f'| ROC-AUC | {test["roc_auc"]:.3f} |', f'| 告警占比 | {_pct(test["alert_rate"])} |']
    if same: L.append(f'| 相同审核量（{same["k"]:,} 笔）下召回率 | {_pct(same["recall"])} |')
    L += ['', '注意：冻结阈值口径与"相同审核量"口径不要混在一句话里比较。', '', '## 4 公平性（按年龄分组，测试月）', '']
    if 'age<50' in gs and 'age>=50' in gs:
        L += ['| 组 | 样本数 | 欺诈数 | 误报率 | 召回率 |', '| --- | --- | --- | --- | --- |']
        for g in ['age<50', 'age>=50']: L.append(f'| {g} | {gs[g]["n"]:,} | {gs[g]["fraud"]:,} | {_pct(gs[g]["fpr"])} | {_pct(gs[g]["recall"])} |')
        L += ['']
        ext_path = OUT/'extensions.json'
        fr = json.loads(ext_path.read_text(encoding='utf-8')).get('fairness', {}).get('test_overall', {}) if ext_path.exists() else {}
        if fr:
            names = [('main_same_k', '主模型'), ('no_age_same_k', '去掉年龄特征的重训模型')]
            L += ['相同审核量下，年长组与年轻组的误报率之比（审核量对齐后再比，避免把审核量变化混进来）：', '', '| 口径 | 年长组 FPR | 年轻组 FPR | 比值 |', '| --- | --- | --- | --- |']
            for key, label in names:
                if key in fr and fr[key].get('fpr_ratio'):
                    L.append(f'| {label} | {_pct(fr[key]["fpr_older"])} | {_pct(fr[key]["fpr_younger"])} | {fr[key]["fpr_ratio"]:.2f} 倍 |')
            L += ['', '去掉年龄特征后差距缩小但没有消失，因为其余特征仍可推断年龄（见教程第 13 章）。', '']
        else:
            L += [f'冻结阈值下年长组误报率约为年轻组的 {gs["age>=50"]["fpr"]/gs["age<50"]["fpr"]:.1f} 倍（补充分析阶段尚未运行，缺少去掉年龄特征的对照）。', '']
    L += ['## 5 已知局限', '', '- 数据为合成数据，结论不能直接外推到真实银行。', '- 结果主要来自一个测试月；4 个窗口的滚动回测是事后补充的检查，不是预先设定的评估方案。',
          '- AP 只有约 0.2：欺诈本身占比很低，模型是帮助排序审核队列，不是自动拒绝。', '- 窗口计数类特征（velocity_*）逐月漂移，携带时间信息，未来月的阈值可能迁移。',
          '- 规则化（决策树蒸馏）会损失效果：相同审核量下规则的召回率低于原模型。', '', '## 6 血缘与溯源', '',
          '- 代码流水线：`run_analysis.py` 依次运行 eda → train → evaluate → explain → extensions → rolling_backtest → coursework → feature_study → verify → report。',
          '- 冻结产物的 SHA-256（生成本卡时重新计算）：', '', '| 文件 | SHA-256 |', '| --- | --- |']
    for p, h in hashes.items(): L.append(f'| `{p}` | `{h[:16]}…` |')
    L += ['', f'- 运行环境：{manifest["environment"]["platform"]}。', '']
    (DOCS/'模型卡_LightGBM.md').write_text('\n'.join(L), encoding='utf-8')
    return dict(contract_total_violations=total, contract_fields_with_violations=contract[contract.violations > 0].field.tolist(),
                files=['docs/特征说明_字段字典.md', 'docs/模型卡_LightGBM.md'])


RUNNERS = dict(skew_scale=part_skew_scale, outliers=part_outliers, distributions=part_distributions, split=part_split,
               smote=part_smote, docs=part_docs)


def run(parts=None, force=False):
    parts = parts or PARTS
    before = frozen_hashes()
    FIG.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True)
    res = json.loads(RESULT.read_text(encoding='utf-8')) if RESULT.exists() else {}
    res.setdefault('parts', {})
    for k in [k for k in res['parts'] if k not in PARTS]: del res['parts'][k]          # parts removed on 2026-10-21
    for k in [k for k in res['parts'].get('distributions', {}) if k.startswith('shapiro')]: del res['parts']['distributions'][k]
    res.update(stage='course-method supplement (post-hoc)', created_singapore=datetime.now(timezone(timedelta(hours=8))).isoformat(),
               discipline=['frozen artifacts read only; hashes checked before/after',
                           'fits on months 0-5 (random-split study pools months 0-6 on purpose), choices on month 6, month 7 reported only',
                           'designed after month-7 results were seen: exploratory, does not re-select the frozen winner'],
               frozen_hashes_before=before)
    todo = [p for p in parts if force or p not in res['parts']]
    if todo:
        ctx = Ctx()
        with threadpool_limits(limits=4):
            for p in todo:
                log(f'=== part: {p}'); t0 = time.perf_counter()
                res['parts'][p] = strict_json(RUNNERS[p](ctx)); res['parts'][p]['seconds'] = round(time.perf_counter()-t0, 1)
                save_json(strict_json(res), RESULT)           # keep finished parts if a later one fails
    else:
        log('All requested parts already done (use --force to redo).')
    after = frozen_hashes()
    res['frozen_hashes_after'] = after; res['frozen_artifacts_unchanged'] = bool(before == after)
    save_json(strict_json(res), RESULT)
    if before != after: raise ValueError('A frozen artifact changed during the course-method supplement.')
    log('Done. Frozen artifacts unchanged.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--parts', nargs='*', choices=PARTS, help='run only these parts (default: all missing)')
    ap.add_argument('--force', action='store_true', help='redo parts that already have results')
    a = ap.parse_args(); run(a.parts, a.force)
