"""Independent consistency checks for outputs/feature_study.json (optional stage; skipped if absent)."""
import json
import math
import numpy as np
import pandas as pd
from .common import ROOT, sha256, load_data, clean_features

PARTS = ['iv', 'selection', 'derived', 'importance']


def _finite(x):
    if isinstance(x, dict): return all(_finite(v) for v in x.values())
    if isinstance(x, list): return all(_finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)


def validate_features(out, check, df=None):
    path = out / 'feature_study.json'
    if not path.exists():
        return
    fs = json.loads(path.read_text(encoding='utf-8'))
    parts = fs.get('parts', {})
    check('features: all four parts present', all(p in parts for p in PARTS))
    check('features: every number finite (no NaN/Inf)', _finite(fs))
    before, after = fs.get('frozen_hashes_before', {}), fs.get('frozen_hashes_after', {})
    check('features: frozen artifact hashes identical before and after',
          bool(before) and before == after and fs.get('frozen_artifacts_unchanged') is True)
    check('features: recorded hashes match the files on disk now',
          all((ROOT / p).exists() and sha256(ROOT / p) == h for p, h in after.items()))
    iv = parts.get('iv', {})
    t = pd.DataFrame(iv.get('table', []))
    if len(t):
        check('features: IV non-negative for every field', bool((t.iv >= 0).all()))
        rule = (t.iv >= iv['iv_min']) & (t.psi_train_vs_month6 <= iv['psi_max'])
        check('features: keep/drop flags follow the stated IV/PSI rule', bool((rule == t.keep_by_rule).all()))
        check('features: dropped list equals rule result', sorted(iv['dropped_by_rule']) == sorted(t.loc[~rule, 'field']))
        # recompute IV of one categorical field with an independent, plain groupby
        if df is None: df = load_data()
        schema = json.loads((out / 'models/schema.json').read_text(encoding='utf-8'))
        tr = df.loc[df.month <= 5]
        x, _ = clean_features(tr, schema)
        f = 'housing_status'
        g = pd.DataFrame({'k': x[f].astype(str).fillna('缺失'), 'y': tr.fraud_bool.to_numpy()}).groupby('k').y.agg(['sum', 'count'])
        bad = (g['sum'] + .5)/(g['sum'].sum() + .5*len(g)); good = (g['count']-g['sum'] + .5)/((g['count']-g['sum']).sum() + .5*len(g))
        iv_check = float(((bad-good)*np.log(bad/good)).sum())
        check('features: IV of housing_status reproduces independently', abs(iv_check - float(t.set_index('field').iv[f])) < 1e-9)
    sel = parts.get('selection', {})
    if sel:
        check('features: same-recipe refit reproduces frozen LightGBM scores', sel['reference_reproduces_frozen_max_abs'] < 1e-9)
        ev = json.loads((out / 'evaluation.json').read_text(encoding='utf-8'))
        lg = next(r for r in ev['test_results'] if r['model'] == 'lightgbm')
        fr = next(r for r in sel['rows'] if r['kind'] == 'frozen')
        check('features: frozen reference row equals evaluation.json',
              abs(fr['test_ap'] - lg['pr_auc_ap']) < 1e-12 and abs(fr['test_recall'] - lg['recall']) < 1e-12)
    der = parts.get('derived', {})
    if der:
        rows = der['rows']
        best = max(rows, key=lambda r: (r['val_recall'], r['val_ap']))
        check('features: month-6 choice follows recall-then-AP rule', der['chosen_on_month6'] == best['variant']
              and sum(bool(r['chosen_on_month6']) for r in rows) == 1)
        ref = next(r for r in rows if r['variant'] == 'reference')
        sref = next((r for r in sel.get('rows', []) if r['kind'] == 'reference_refit'), None)
        if sref:
            check('features: derived-study reference equals selection-study reference',
                  abs(ref['test_ap'] - sref['test_ap']) < 1e-12 and abs(ref['val_ap'] - sref['val_ap']) < 1e-12)
        check('features: four rolling windows with ordered intervals', len(der['rolling']) == 4 and all(
            w['ap_diff_ci95'][0] <= w['ap_diff_ci95'][1] and w['same_budget_recall_diff_ci95'][0] <= w['same_budget_recall_diff_ci95'][1]
            for w in der['rolling']))
        check('features: rolling windows never train on their own test month',
              all(max(w['train_months']) < w['validation_month'] < w['test_month'] for w in der['rolling']))
    imp = parts.get('importance', {})
    if imp:
        n = len(imp['table'])
        check('features: importance ranks within 1..n', all(1 <= r[k] <= n for r in imp['table'] for k in ('perm_rank', 'shap_rank', 'iv_rank')))
    for f in ['fs_iv_psi', 'fs_woe_top', 'fs_derived', 'fs_importance']:
        check(f'features: figure {f}.png exists', (out / 'figures' / f'{f}.png').exists())
