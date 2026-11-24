"""Shared data handling and evaluation. No fitted object may see months 6 or 7."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CATEGORICAL = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
SENTINELS = ['prev_address_months_count', 'current_address_months_count', 'bank_months_count',
             'session_length_in_minutes', 'device_distinct_emails_8w']
MISSING_FIELDS = SENTINELS + ['intended_balcon_amount']

def save_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def default(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default), encoding='utf-8')

def load_data():
    header = pd.read_csv(ROOT/'data/raw/Base.csv', nrows=0).columns
    dtypes = {c: ('category' if c in CATEGORICAL else 'float32') for c in header}
    dtypes.update(fraud_bool='int8', month='int8', customer_age='int16')
    df = pd.read_csv(ROOT/'data/raw/Base.csv', dtype=dtypes)
    assert df.shape == (1_000_000, 32), f'Unexpected Base shape: {df.shape}'
    assert set(df.month.unique()) == set(range(8))
    assert set(df.fraud_bool.unique()) == {0, 1}
    df.index.name = 'row_id'
    return df

def clean_features(df, schema=None):
    x = df.drop(columns=['fraud_bool', 'month']).copy()
    for c in SENTINELS: x.loc[x[c] == -1, c] = np.nan
    x.loc[x.intended_balcon_amount < 0, 'intended_balcon_amount'] = np.nan
    for c in MISSING_FIELDS: x[c+'__missing'] = x[c].isna().astype('float32')
    if schema is None:
        constants = [c for c in x if x[c].nunique(dropna=False) <= 1]
        categories = {c: sorted(x[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL}
        schema = {'drop_constants': constants, 'categories': categories,
                  'training_months': list(range(6)), 'excluded': ['fraud_bool', 'month']}
    x = x.drop(columns=schema['drop_constants'])
    for c, levels in schema['categories'].items():
        x[c] = pd.Categorical(x[c], categories=levels)
    return x, schema

def make_preprocessor(x):
    numeric = [c for c in x if c not in CATEGORICAL]
    return ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                          ('scale', StandardScaler())]), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32), CATEGORICAL)
    ], sparse_threshold=0, verbose_feature_names_out=False)

def threshold_at_fpr(y, scores, max_fpr=.05):
    """Lowest feasible threshold, with whole score ties kept together; predict score>=t."""
    neg = np.sort(np.asarray(scores, dtype=float)[np.asarray(y) == 0])
    k = int(np.floor(max_fpr * len(neg)))
    return float(np.nextafter(neg[-(k+1)], np.inf))

def binary_metrics(y, pred):
    y = np.asarray(y); pred = np.asarray(pred, dtype=bool)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=len(y), fraud=int(y.sum()), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                recall=float(tp/(tp+fn)) if tp+fn else None,
                fpr=float(fp/(fp+tn)) if fp+tn else None,
                precision=float(tp/(tp+fp)) if tp+fp else 0.,
                alerts=int(tp+fp), alert_rate=float((tp+fp)/len(y)))

def metrics(y, scores, threshold):
    s = np.asarray(scores, dtype=float)
    out = binary_metrics(y, s >= threshold)
    p, r, _ = precision_recall_curve(y, s)
    out.update(threshold=float(threshold), pr_auc_ap=float(average_precision_score(y, s)),
               pr_auc_trapezoid=float(auc(r, p)), roc_auc=float(roc_auc_score(y, s)))
    return out

def top_k(scores, k, row_ids):
    """Exact fixed-capacity ranking. Ties use seeded row hashes, never labels/time ordering."""
    ids = np.asarray(row_ids, dtype=np.uint64)
    tie = (ids ^ np.uint64(SEED)) * np.uint64(11400714819323198485)
    order = np.lexsort((tie, -np.asarray(scores, dtype=float)))
    result = np.zeros(len(scores), dtype=bool); result[order[:k]] = True
    return result

def wilson(successes, total, z=1.95996398454):
    if not total: return [None, None]
    p=successes/total; den=1+z*z/total
    mid=(p+z*z/(2*total))/den
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total**2))/den
    return [float(mid-half), float(mid+half)]

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
