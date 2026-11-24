"""Independent consistency checks for outputs/coursework.json (optional stage; skipped if absent)."""
import json
import math
from .common import ROOT, sha256

PARTS = ['skew_scale', 'outliers', 'distributions', 'split', 'smote', 'docs']


def _finite(x):
    if isinstance(x, dict): return all(_finite(v) for v in x.values())
    if isinstance(x, list): return all(_finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)


def validate_coursework(out, check):
    path = out / 'coursework.json'
    if not path.exists():
        return
    cw = json.loads(path.read_text(encoding='utf-8'))
    parts = cw.get('parts', {})
    check('coursework: all six parts present', all(p in parts for p in PARTS))
    check('coursework: every number finite (no NaN/Inf)', _finite(cw))
    before, after = cw.get('frozen_hashes_before', {}), cw.get('frozen_hashes_after', {})
    check('coursework: frozen artifact hashes identical before and after', bool(before) and before == after
          and cw.get('frozen_artifacts_unchanged') is True)
    check('coursework: recorded hashes match the files on disk now',
          all((ROOT / p).exists() and sha256(ROOT / p) == h for p, h in after.items()))
    ev = json.loads((out / 'evaluation.json').read_text(encoding='utf-8'))
    test = {r['model']: r for r in ev['test_results']}
    for name, label_part, prefix in [('logistic_regression', 'skew_scale', 'LR 标准化（冻结参考）'),
                                     ('lightgbm', 'skew_scale', 'LightGBM（冻结参考）')]:
        rows = parts.get(label_part, {}).get('rows', [])
        ref = next((r for r in rows if r['label'] == prefix), None)
        if ref is None:
            check(f'coursework: {name} reference row present', False); continue
        t = test[name]
        check(f'coursework: {name} reference row reproduces frozen test metrics',
              abs(ref['test_recall'] - t['recall']) < 1e-9 and abs(ref['test_fpr'] - t['fpr']) < 1e-9
              and abs(ref['test_ap'] - t['pr_auc_ap']) < 1e-9 and abs(ref['threshold'] - t['threshold']) < 1e-9)
    ss = parts.get('smote', {}).get('rows', [])
    for r in ss:
        check(f"coursework: smote row '{r['label']}' rates within [0,1]",
              all(0 <= r[k] <= 1 for k in ('test_recall', 'test_fpr', 'test_accuracy_at_0_5', 'test_base_rate')))
    sp = parts.get('split', {})
    check('coursework: split study has random and time rows with CV folds', len(sp.get('rows', [])) >= 2 and len(sp.get('folds', [])) == 5)
    for f in ['cw_skewness', 'cw_outlier_lift', 'cw_distribution_by_label', 'cw_correlation_heatmap', 'cw_split_comparison',
              'cw_imbalance_methods']:
        check(f'coursework: figure {f}.png exists', (out / 'figures' / f'{f}.png').exists())
