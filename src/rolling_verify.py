"""Independent numerical checks for saved rolling-origin experiments.

The arithmetic below deliberately does not call common.metrics,
common.threshold_at_fpr, or sklearn's AP implementation.
"""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
MINUS_ONE = ['prev_address_months_count', 'current_address_months_count', 'bank_months_count',
             'session_length_in_minutes', 'device_distinct_emails_8w']
MISSING_FIELDS = MINUS_ONE + ['intended_balcon_amount']


def independent_average_precision(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(y) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid score vector for average precision.')
    positives = int(np.sum(y == 1))
    if not positives:
        return 0.0
    order = np.argsort(-scores, kind='stable')
    ranked_y, ranked_scores = y[order], scores[order]
    ends = np.flatnonzero(np.r_[ranked_scores[1:] != ranked_scores[:-1], True])
    tp = np.cumsum(ranked_y == 1)[ends]
    gained_tp = np.diff(np.r_[0, tp])
    precision = tp / (ends + 1)
    return float(np.sum(gained_tp * precision) / positives)


def independent_fpr_threshold(y, scores, max_fpr=.05):
    """Find the excluded negative-score boundary by cumulative whole-tie counts."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    negative_scores = scores[y == 0]
    if not len(negative_scores) or not np.isfinite(scores).all() or not 0 <= max_fpr < 1:
        raise ValueError('Invalid data for a validation FPR threshold.')
    distinct, counts = np.unique(negative_scores, return_counts=True)
    must_exclude = len(negative_scores) - int(np.floor(max_fpr * len(negative_scores)))
    boundary_index = np.searchsorted(np.cumsum(counts), must_exclude, side='left')
    return float(np.nextafter(distinct[boundary_index], np.inf))


def independent_top_k(scores, k, row_ids, seed=42):
    scores = np.asarray(scores, dtype=np.float64)
    row_ids = np.asarray(row_ids)
    if len(scores) != len(row_ids) or not 0 <= k <= len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid inputs to the capacity ranking check.')
    mask64 = (1 << 64) - 1
    ties = np.fromiter((((int(row_id) ^ seed) * 11400714819323198485) & mask64
                        for row_id in row_ids), dtype=np.uint64, count=len(row_ids))
    records = np.rec.fromarrays([-scores, ties], names=['descending_score', 'tie_hash'])
    order = np.argsort(records, order=['descending_score', 'tie_hash'])
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:k]] = True
    return selected


def independent_binary_metrics(y, flags):
    y = np.asarray(y)
    flags = np.asarray(flags, dtype=bool)
    positive = y == 1
    negative = y == 0
    tp = int(np.count_nonzero(flags & positive))
    fp = int(np.count_nonzero(flags & negative))
    fn = int(np.count_nonzero(~flags & positive))
    tn = int(np.count_nonzero(~flags & negative))
    return dict(n=len(y), fraud=int(positive.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                alerts=tp+fp, recall=tp/(tp+fn) if tp+fn else None,
                fpr=fp/(fp+tn) if fp+tn else None,
                precision=tp/(tp+fp) if tp+fp else 0., alert_rate=(tp+fp)/len(y))


def independent_paired_points(y, scores_a, scores_b, flags_a, flags_b):
    positive = np.asarray(y) == 1
    difference = np.asarray(flags_a, dtype=float)[positive] - np.asarray(flags_b, dtype=float)[positive]
    return dict(recall_difference=float(difference.mean()),
                ap_difference=independent_average_precision(y, scores_a)-independent_average_precision(y, scores_b))


def independent_roc_auc(y, scores):
    y = np.asarray(y) == 1
    scores = np.asarray(scores, dtype=np.float64)
    positives, negatives = int(y.sum()), int((~y).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind='stable')
    sorted_scores, sorted_y = scores[order], y[order]
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    starts = np.r_[0, ends[:-1]+1]
    group_positives = np.diff(np.r_[0, np.cumsum(sorted_y)[ends]])
    average_ranks = (starts+ends+2)/2
    return float((np.sum(group_positives*average_ranks)-positives*(positives+1)/2)/(positives*negatives))


def _wilson(successes, total):
    z = 1.959963984540054
    p = successes/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    radius = z/(1+z*z/total)*np.sqrt(p*(1-p)/total+z*z/(4*total*total))
    return center-radius, center+radius


def _hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _strict_load(path):
    def reject(value):
        raise ValueError(f'Non-standard JSON constant: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def _equal(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, (str, bool)) or isinstance(b, (str, bool)):
        return a == b
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a-b) < 1e-12)


def _raw_features(frame):
    """Reconstruct input cleaning independently of common.clean_features."""
    result = frame.drop(columns=['fraud_bool', 'month']).copy()
    for column in MINUS_ONE:
        result[column] = result[column].mask(result[column] == -1)
    result['intended_balcon_amount'] = result.intended_balcon_amount.mask(result.intended_balcon_amount < 0)
    for column in MISSING_FIELDS:
        result[column+'__missing'] = result[column].isna().astype('float32')
    return result


def _apply_schema(frame, schema):
    result = _raw_features(frame).drop(columns=schema['drop_constants'])
    for column, levels in schema['categories'].items():
        result[column] = pd.Categorical(result[column], categories=levels)
    return result


def _fairness(y, flags, age):
    young = np.asarray(age) < 50
    a = independent_binary_metrics(np.asarray(y)[young], np.asarray(flags)[young])
    b = independent_binary_metrics(np.asarray(y)[~young], np.asarray(flags)[~young])
    alo, ahi = _wilson(a['fp'], a['fp']+a['tn'])
    blo, bhi = _wilson(b['fp'], b['fp']+b['tn'])
    return dict(fpr_younger=a['fpr'], fpr_older=b['fpr'], fpr_gap=b['fpr']-a['fpr'],
                fpr_ratio=b['fpr']/a['fpr'] if a['fpr'] else None,
                young_n=a['n'], older_n=b['n'], young_fp=a['fp'], older_fp=b['fp'],
                young_normal=a['fp']+a['tn'], older_normal=b['fp']+b['tn'],
                young_recall=a['recall'], older_recall=b['recall'],
                older_fpr_ci_low=blo, older_fpr_ci_high=bhi, young_fpr_ci_low=alo, young_fpr_ci_high=ahi)


def validate_rolling(out, check):
    """Validate the persisted rolling stage once its manifest is available."""
    out = Path(out)
    path = out/'rolling_backtest.json'
    if not path.exists():
        return
    result = _strict_load(path)
    protocol = _strict_load(out/'rolling_protocol.json')
    check('rolling: supported strict JSON schema', result['schema_version'] == 1 and protocol['schema_version'] == 1)
    check('rolling: embedded protocol and hash match', result['protocol'] == protocol
          and result['protocol_sha256'] == _hash(out/'rolling_protocol.json'))
    check('rolling: current producer source matches protocol revision', protocol['source_sha256'] == _hash(ROOT/'src/rolling_backtest.py'))
    check('rolling: original raw source hash matches', protocol['raw_sha256'] == _hash(ROOT/'data/raw/Base.csv'))
    check('rolling: common review rate is exactly 5%', protocol['review_rate'] == .05 and protocol['validation_max_fpr'] == .05)
    check('rolling: frozen main artifacts unchanged', result['frozen_artifacts_unchanged']
          and all(_hash(ROOT/p) == digest for p, digest in result['frozen_hashes_before'].items())
          and 'outputs/models/schema.json' in result['frozen_hashes_before'])
    for relative, digest in result['artifact_hashes'].items():
        check(f'rolling artifact hash: {relative}', (ROOT/relative).exists() and _hash(ROOT/relative) == digest)

    header = pd.read_csv(ROOT/'data/raw/Base.csv', nrows=0).columns
    dtypes = {c: ('category' if c in CATEGORIES else 'float32') for c in header}
    dtypes.update(fraud_bool='int8', month='int8', customer_age='int16')
    raw = pd.read_csv(ROOT/'data/raw/Base.csv', dtype=dtypes)
    variants = protocol['variants']
    metric_lookup = {(r['window_id'], r['variant'], r['split'], r['policy']): r for r in result['metrics']}
    fairness_lookup = {(r['window_id'], r['variant'], r['policy']): r for r in result['fairness']}
    difference_lookup = {(r['window_id'], r['variant']): r for r in result['differences']}
    check('rolling: every metric row has a unique window/variant/split/policy',
          len(metric_lookup) == len(result['metrics']) == len(result['windows'])*len(variants)*4)
    check('rolling: fairness and difference rows are complete and unique',
          len(fairness_lookup) == len(result['fairness']) == len(result['windows'])*len(variants)*2
          and len(difference_lookup) == len(result['differences']) == len(result['windows'])*(len(variants)-1))
    expected_months = [r['test_month'] for r in protocol['windows']]
    check('rolling: declared windows are all present once', sorted(r['test_month'] for r in result['windows']) == sorted(expected_months)
          and len(set(r['window_id'] for r in result['windows'])) == len(result['windows']))
    manifest_lookup = {_strict_load(ROOT/p)['window_id']: (ROOT/p, _strict_load(ROOT/p)) for p in result['window_manifests']}
    check('rolling: every window has one frozen manifest', set(manifest_lookup) == {r['window_id'] for r in result['windows']})

    def fields_match(name, saved, expected):
        check(name, all(key in saved and _equal(saved[key], value) for key, value in expected.items()))

    recomputed_differences = []
    for window in result['windows']:
        wid = window['window_id']; manifest_path, manifest = manifest_lookup[wid]
        test_month = int(window['test_month']); validation_month = int(window['validation_month'])
        train_months = window['train_months']
        declared = next(r for r in protocol['windows'] if r['test_month'] == test_month)
        check(f'{wid}: strict chronological train/validation/test isolation',
              train_months == declared['train_months'] == manifest['train_months']
              and validation_month == declared['validation_month'] == manifest['validation_month']
              and test_month == manifest['test_month'] and max(train_months) < validation_month < test_month)
        check(f'{wid}: choices were frozen against the declared protocol', manifest['frozen_before_test_predictions'] is True
              and manifest['protocol_sha256'] == result['protocol_sha256'] and manifest['review_rate'] == .05)
        for relative, digest in manifest['artifact_hashes'].items():
            check(f'{wid}: pre-test artifact hash {Path(relative).name}', _hash(ROOT/relative) == digest)
        folder = manifest_path.parent
        schema = _strict_load(folder/'schema.json')
        raw_train = raw.loc[raw.month.isin(train_months)]
        training_features = _raw_features(raw_train)
        constants = [c for c in training_features if training_features[c].nunique(dropna=False) <= 1]
        categories = {c: sorted(training_features[c].dropna().astype(str).unique().tolist()) for c in CATEGORIES}
        check(f'{wid}: schema constants/categories come only from training rows',
              schema['training_months'] == train_months and schema['drop_constants'] == constants
              and schema['categories'] == categories and set(schema['excluded']) == {'fraud_bool', 'month'})
        check(f'{wid}: training row count matches raw month membership', window['train_n'] == len(raw_train))
        del training_features, raw_train

        frames = {split: pd.read_parquet(out/f'rolling/predictions/{wid}_{split}.parquet') for split in ['validation', 'test']}
        for split, month in [('validation', validation_month), ('test', test_month)]:
            frame = frames[split]; source = raw.loc[raw.month == month]
            check(f'{wid}/{split}: exact raw IDs, month, labels and age match',
                  np.array_equal(frame.row_id.to_numpy(), source.index.to_numpy()) and np.all(frame.month.to_numpy() == month)
                  and np.array_equal(frame.fraud_bool.to_numpy(), source.fraud_bool.to_numpy())
                  and np.array_equal(frame.customer_age.to_numpy(), source.customer_age.to_numpy()))
            check(f'{wid}/{split}: saved sample size and exact 5% capacity match',
                  len(frame) == window[split+'_n'] and window['validation_k' if split == 'validation' else 'k'] == int(np.floor(.05*len(frame))))
        val, test = frames['validation'], frames['test']
        check(f'{wid}: held-out fraud count matches raw labels', window['test_fraud'] == int(test.fraud_bool.sum()))

        candidate_keys = []
        declared_configs = [(r['num_leaves'], r['reg_lambda']) for r in protocol['baseline_candidates']]
        check(f'{wid}: baseline candidate configurations match the protocol',
              [(r['num_leaves'], r['reg_lambda']) for r in manifest['candidates']] == declared_configs)
        for candidate in manifest['candidates']:
            scores = val[f'candidate_leaves{candidate["num_leaves"]}'].to_numpy(dtype=np.float64)
            threshold = independent_fpr_threshold(val.fraud_bool, scores)
            bm = independent_binary_metrics(val.fraud_bool, scores >= threshold)
            ap = independent_average_precision(val.fraud_bool, scores)
            fields_match(f'{wid}/candidate{candidate["num_leaves"]}: validation arithmetic', candidate['validation'], dict(bm, threshold=threshold, pr_auc_ap=ap))
            check(f'{wid}/candidate{candidate["num_leaves"]}: threshold derives only from validation scores', candidate['threshold'] == threshold)
            candidate_keys.append((bm['recall'], ap))
        winner = max(range(len(candidate_keys)), key=lambda i: candidate_keys[i])
        chosen = manifest['candidates'][winner]
        baseline = manifest['variants']['baseline']
        check(f'{wid}: baseline is selected by validation recall/AP with stable candidate ties',
              all(baseline[k] == chosen[k] for k in ['num_leaves', 'reg_lambda', 'best_iteration', 'threshold'])
              and np.array_equal(val.baseline.to_numpy(), val[f'candidate_leaves{chosen["num_leaves"]}'].to_numpy())
              and window['selected_num_leaves'] == chosen['num_leaves'] and window['selected_reg_lambda'] == chosen['reg_lambda']
              and window['selected_best_iteration'] == chosen['best_iteration'])
        check(f'{wid}: variant drop definitions match protocol', manifest['dropped_features'] == variants and set(manifest['variants']) == set(variants))

        models = {v: joblib.load(folder/f'{v}.joblib') for v in variants}
        test_frozen, test_top = {}, {}
        for variant in variants:
            spec = manifest['variants'][variant]
            check(f'{wid}/{variant}: ablation inherits baseline hyperparameters',
                  spec['num_leaves'] == baseline['num_leaves'] and spec['reg_lambda'] == baseline['reg_lambda']
                  and models[variant].best_iteration_ == spec['best_iteration'])
            threshold = independent_fpr_threshold(val.fraud_bool, val[variant])
            check(f'{wid}/{variant}: frozen threshold rederived from validation only', threshold == spec['threshold'])
            for split, frame in frames.items():
                y = frame.fraud_bool.to_numpy(); scores = frame[variant].to_numpy(dtype=np.float64)
                flags_f = scores >= threshold
                flags_k = independent_top_k(scores, int(np.floor(.05*len(frame))), frame.row_id, seed=protocol['seed'])
                check(f'{wid}/{variant}/{split}: frozen and exact-K saved flags match independent ranking',
                      np.array_equal(flags_f, frame[variant+'__alert_frozen'].to_numpy(dtype=bool))
                      and np.array_equal(flags_k, frame[variant+'__alert_topk'].to_numpy(dtype=bool))
                      and int(flags_k.sum()) == int(np.floor(.05*len(frame))))
                ap = independent_average_precision(y, scores)
                auc = independent_roc_auc(y, scores)
                if split == 'validation':
                    normal = y == 0
                    check(f'{wid}/{variant}: whole-tie validation FPR<=5%', int(np.sum(flags_f & normal)) <= int(np.floor(.05*normal.sum())))
                    fields_match(f'{wid}/{variant}: manifest validation metrics reconcile', spec['validation'],
                                 dict(independent_binary_metrics(y, flags_f), threshold=threshold, pr_auc_ap=ap, roc_auc=auc))
                else:
                    test_frozen[variant], test_top[variant] = flags_f, flags_k
                for policy, flags in [('frozen_threshold', flags_f), ('fixed_review_5pct', flags_k)]:
                    saved = metric_lookup[(wid, variant, split, policy)]
                    bm = independent_binary_metrics(y, flags)
                    lo, hi = _wilson(bm['tp'], bm['fraud']); flo, fhi = _wilson(bm['fp'], bm['fp']+bm['tn'])
                    fields_match(f'{wid}/{variant}/{split}/{policy}: AP, ROC, confusion matrix and intervals', saved,
                                 dict(bm, threshold=threshold, pr_auc_ap=ap, roc_auc=auc, recall_ci_low=lo, recall_ci_high=hi, fpr_ci_low=flo, fpr_ci_high=fhi))
                    if split == 'test':
                        fields_match(f'{wid}/{variant}/{policy}: age fairness denominators and rates',
                                     fairness_lookup[(wid, variant, policy)], _fairness(y, flags, frame.customer_age))

        rng = np.random.default_rng(protocol['seed']+test_month)
        with threadpool_limits(limits=4):
            for split, frame in frames.items():
                positions = np.sort(rng.choice(len(frame), size=min(512, len(frame)), replace=False))
                source = raw.iloc[frame.row_id.to_numpy(dtype=int)[positions]]
                reconstructed_features = _apply_schema(source, schema)
                for variant, model in models.items():
                    features = reconstructed_features.drop(columns=variants[variant])
                    check(f'{wid}/{variant}/{split}: model input schema excludes label/month and matches saved columns',
                          model.feature_name_ == list(features.columns) and not {'fraud_bool', 'month'}.intersection(features.columns))
                    reconstructed = model.predict_proba(features)[:, 1]
                    check(f'{wid}/{variant}/{split}: independently reconstructed inputs reproduce scores',
                          np.allclose(reconstructed, frame[variant].to_numpy()[positions], rtol=0, atol=1e-12))

        y = test.fraud_bool.to_numpy(); old_neg = (test.customer_age.to_numpy() >= 50) & (y == 0)
        young_neg = (test.customer_age.to_numpy() < 50) & (y == 0)
        for variant in list(variants)[1:]:
            paired = independent_paired_points(y, test[variant], test.baseline, test_top[variant], test_top['baseline'])
            frozen_difference = float((test_frozen[variant].astype(float)[y == 1]-test_frozen['baseline'][y == 1]).mean())
            delta = test_top[variant].astype(float)-test_top['baseline'].astype(float)
            expected = dict(ap_diff=paired['ap_difference'], recall_frozen_diff=frozen_difference,
                            recall_topk_diff=paired['recall_difference'], fpr_gap_topk_diff=float(delta[old_neg].mean()-delta[young_neg].mean()))
            saved = difference_lookup[(wid, variant)]
            fields_match(f'{wid}/{variant}: paired variant-minus-baseline point estimates', saved, expected)
            check(f'{wid}/{variant}: all paired interval endpoints are finite and ordered', all(
                np.isfinite(saved[key+'_ci_low']) and np.isfinite(saved[key+'_ci_high']) and saved[key+'_ci_low'] <= saved[key+'_ci_high']
                for key in ['ap', 'recall_frozen', 'recall_topk', 'fpr_gap_topk']))
            baseline_gap=float(test_top['baseline'][old_neg].mean()-test_top['baseline'][young_neg].mean())
            variant_gap=float(test_top[variant][old_neg].mean()-test_top[variant][young_neg].mean())
            recomputed_differences.append(dict(window_id=wid, variant=variant,
                                               absolute_age_gap_reduced=abs(variant_gap)<abs(baseline_gap), **expected))

        if test_month == 7:
            for split, frame in frames.items():
                primary=pd.read_parquet(out/f'predictions/{split}.parquet')
                supplementary=pd.read_parquet(out/f'predictions/ext_{split}_scores.parquet')
                check(f'{wid}/{split}: full score arrays align with frozen/supplementary rows',
                      np.array_equal(frame.row_id,primary.row_id) and np.array_equal(frame.row_id,supplementary.row_id))
                check(f'{wid}/{split}: full baseline scores exactly reproduce frozen LightGBM',
                      float(np.max(np.abs(frame.baseline.to_numpy()-primary.lightgbm.to_numpy())))<1e-12)
                for variant in variants:
                    reference='reference_refit' if variant=='baseline' else variant
                    check(f'{wid}/{variant}/{split}: full scores reproduce same-configuration extension refit',
                          float(np.max(np.abs(frame[variant].to_numpy()-supplementary[reference].to_numpy())))<1e-12)

    summary = result['summary']
    check('rolling: observed month ranges are explicitly not confidence intervals', summary['observed_ranges_not_confidence_intervals'] is True)
    metric_frame = pd.DataFrame(result['metrics'])
    for policy, ranges in summary['baseline'].items():
        selected = metric_frame[(metric_frame.variant == 'baseline') & (metric_frame.split == 'test') & (metric_frame.policy == policy)]
        check(f'rolling: baseline observed ranges reconcile for {policy}', all(
            _equal(values[0], float(selected[key].min())) and _equal(values[1], float(selected[key].max())) for key, values in ranges.items()))
    difference_frame = pd.DataFrame(recomputed_differences)
    saved_differences = pd.DataFrame(result['differences'])
    for variant, counts in summary['ablations'].items():
        values = difference_frame[difference_frame.variant == variant]
        intervals = saved_differences[saved_differences.variant == variant]
        fields_match(f'rolling: descriptive win counts reconcile for {variant}', counts,
                     dict(positive_ap_months=int((values.ap_diff > 0).sum()), positive_topk_recall_months=int((values.recall_topk_diff > 0).sum()),
                          smaller_age_fpr_gap_months=int(values.absolute_age_gap_reduced.sum()),
                          topk_recall_ci_excludes_zero_months=int(((intervals.recall_topk_ci_low > 0) | (intervals.recall_topk_ci_high < 0)).sum())))
