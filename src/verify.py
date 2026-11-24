"""Independent arithmetic checks on the saved, final prediction artifacts."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from .common import ROOT, SEED, save_json, threshold_at_fpr, sha256, top_k, load_data, clean_features
from .rolling_verify import validate_rolling
from .coursework_verify import validate_coursework
from .feature_verify import validate_features

def run():
    out=ROOT/'outputs'; result=json.loads((out/'evaluation.json').read_text(encoding='utf-8'))
    explanation=json.loads((out/'explanation.json').read_text(encoding='utf-8'))
    val=pd.read_parquet(out/'predictions/validation.parquet')
    test=pd.read_parquet(out/'predictions/test.parquet'); y=test.fraud_bool.to_numpy()
    checks=[]
    def check(name,condition):
        assert condition,name
        checks.append({'check':name,'passed':True})
    check('validation/test row IDs disjoint', not set(val.row_id).intersection(test.row_id))
    check('full validation/test sizes',len(val)==108168 and len(test)==96843)
    check('full validation/test fraud counts',int(val.fraud_bool.sum())==1450 and int(y.sum())==1428)
    for m in result['test_results']:
        f=m['model']; threshold=result['manifest']['models'][f]['threshold']
        flag=test[f].to_numpy(dtype=np.float64)>=threshold
        tp=int(np.sum(flag & (y==1))); fp=int(np.sum(flag & (y==0)))
        check(f'{f}: saved TP/FP independently reconcile',tp==m['tp'] and fp==m['fp'])
        check(f'{f}: saved alert flag reconciles',np.array_equal(flag,test[f+'__alert_frozen'].to_numpy()))
        vflag=val[f].to_numpy(dtype=np.float64)>=threshold
        vnormal=val.fraud_bool.to_numpy()==0
        check(f'{f}: validation FPR<=5%',np.sum(vflag & vnormal)<=np.floor(.05*vnormal.sum()))
        check(f'{f}: exact same-budget alert count',int(test[f+'__alert_topk'].sum())==result['rules_comparison']['test_k'])
    for row in result['same_budget_results']:
        flag=test[row['model']+'__alert_topk'].to_numpy(dtype=bool)
        check(row['model']+': same-budget catches reconcile',int(np.sum(flag & (y==1)))==row['tp'])
    for f in result['manifest']['models']:
        g=[r for r in result['fairness'] if r['model']==f and r['split']=='test']
        total=next(r for r in result['test_results'] if r['model']==f)
        check(f'{f}: age groups partition test confusion matrix',all(sum(r[k] for r in g)==total[k] for k in ['n','tp','fp','fn','tn']))
    check('five mutually exclusive rules cover test',len(explanation['rules'])==5 and sum(r['test_rows'] for r in explanation['rules'])==len(test))
    check('rule fraud counts reconcile',sum(r['test_fraud'] for r in explanation['rules'])==int(y.sum()))
    check('SHAP raw-score additivity',explanation['shap_additivity_max_error']<1e-6)
    # A tied score must not be partly admitted by a whole-threshold rule.
    example_y=np.array([0]*20+[1,1]); example_s=np.array([.9,.9]+[.1]*18+[.95,.85],dtype='float32')
    t=threshold_at_fpr(example_y,example_s,.05)
    flags=example_s.astype('float64')>=t
    check('handcrafted threshold tie case: no normal and one fraud',int(flags[:20].sum())==0 and int(flags[20:].sum())==1)
    ext_path=out/'extensions.json'
    if ext_path.exists():
        def reject(token): raise ValueError(f'non-standard JSON constant {token}')
        ext=json.loads(ext_path.read_text(encoding='utf-8'),parse_constant=reject)
        check('extensions.json is strict JSON (no NaN/Infinity)',True)
        psi_check=ext['velocity'].get('psi_consistency_with_eda',{})
        # Tolerance 1e-4: this stage reads values as float32 (common.load_data), the EDA used float64; decile edges differ at ~1e-6.
        check('PSI recomputation matches EDA table',psi_check.get('rows_compared',0)>0 and psi_check['max_abs_diff']<1e-4)
        check('supplementary stage left frozen artifacts unchanged',ext['frozen_artifacts_unchanged']
              and all(sha256(ROOT/p)==h for p,h in ext['frozen_hashes_before'].items()))
        check('supplementary refit reproduces frozen LightGBM',ext['reproduction']['reproduced'])
        check('frozen schema is included in supplementary integrity checks',
              'outputs/models/schema.json' in ext['frozen_hashes_before'])
        curve=pd.DataFrame(ext['rule_curve']); rule_summary=ext['rules']
        target=rule_summary['target_fraction_of_teacher']*rule_summary['teacher_validation_recall']
        eligible=curve[curve.val_whole_recall>=target]
        expected_choice=int(eligible.leaves.min()) if len(eligible) else int(curve.sort_values(
            ['val_whole_recall','leaves'],ascending=[False,True]).iloc[0].leaves)
        check('rule target/fallback status is explicit and correct',
              rule_summary['target_met']==bool(len(eligible)) and rule_summary['fallback_used']==(len(eligible)==0)
              and rule_summary['chosen_leaves']==expected_choice
              and abs(rule_summary['target_validation_recall']-target)<1e-12
              and (not rule_summary['fallback_used'] or bool(rule_summary['fallback_reason'])))
        check('rule achieved target fraction reconciles',abs(rule_summary['achieved_fraction_of_teacher']-
              rule_summary['chosen_validation']['recall']/rule_summary['teacher_validation_recall'])<1e-12)
        scores=pd.read_parquet(out/'predictions/ext_test_scores.parquet')
        check('supplementary scores align with frozen test rows',np.array_equal(scores.row_id.to_numpy(),test.row_id.to_numpy()))
        for v in [c for c in scores if c.endswith('__alert_topk')]:
            check(f'{v}: exact same-budget K',int(scores[v].sum())==ext['test_k'])
            score_column=v.removesuffix('__alert_topk')
            check(f'{v}: flags match deterministic score ranking',np.array_equal(
                top_k(scores[score_column],ext['test_k'],scores.row_id),scores[v].to_numpy(dtype=bool)))
        check('supplementary artifacts have a persisted manifest','artifacts' in ext)
        artifacts=ext['artifacts']; manifest_path=ROOT/artifacts['manifest']
        check('supplementary artifact manifest hash matches',sha256(manifest_path)==artifacts['manifest_sha256'])
        artifact_manifest=json.loads(manifest_path.read_text(encoding='utf-8'),parse_constant=reject)
        for path,digest in artifact_manifest['artifact_sha256'].items():
            check(f'persisted supplementary artifact hash: {path}',(ROOT/path).exists() and sha256(ROOT/path)==digest)
        for path,digest in artifact_manifest['source_code_sha256'].items():
            check(f'supplementary source revision: {path}',sha256(ROOT/path)==digest)
        ext_val=pd.read_parquet(ROOT/artifact_manifest['predictions']['validation'])
        check('supplementary validation predictions align',np.array_equal(ext_val.row_id,val.row_id)
              and np.array_equal(ext_val.fraud_bool,val.fraud_bool))
        for row in ext['ablation']:
            name=row['variant']; values=scores[name].to_numpy(dtype=float)
            flags=values>=row['threshold']; same=scores[name+'__alert_topk'].to_numpy(dtype=bool)
            tp=int(np.sum(flags&(y==1))); fp=int(np.sum(flags&(y==0)))
            check(f'{name}: held-out AP, frozen recall/FPR and exact-K TP reconcile',
                  abs(average_precision_score(y,values)-row['test_ap'])<1e-12
                  and abs(tp/int(y.sum())-row['test_recall'])<1e-12
                  and abs(fp/int((y==0).sum())-row['test_fpr'])<1e-12
                  and int(np.sum(same&(y==1)))==row['test_same_budget_tp'])
            validation_values=ext_val[name].to_numpy(dtype=float)
            validation_threshold=threshold_at_fpr(ext_val.fraud_bool,validation_values)
            check(f'{name}: saved threshold is selected only from saved validation predictions',
                  validation_threshold==row['threshold'])
        for row in curve.itertuples():
            name=f'rules_{int(row.leaves)}'; flags=scores[name].to_numpy(dtype=float)>=row.threshold
            check(f'{name}: whole-leaf held-out counts and AP reconcile',
                  int(flags.sum())==row.test_whole_alerts
                  and abs(np.sum(flags&(y==1))/int(y.sum())-row.test_whole_recall)<1e-12
                  and abs(average_precision_score(y,scores[name])-row.test_ap)<1e-12
                  and threshold_at_fpr(ext_val.fraud_bool,ext_val[name])==row.threshold)
        rules=pd.DataFrame(ext['recommended_rules'])
        check('recommended rules partition the test month',int(rules.test_rows.sum())==len(test) and int(rules.test_fraud.sum())==int(y.sum()))
        r=ext['rules']['chosen_test']
        alerted=rules[rules.alert]
        check('recommended whole-leaf alerts reconcile',int(alerted.test_rows.sum())==r['alerts'] and int(alerted.test_fraud.sum())==r['tp'])
        ctx=pd.DataFrame(ext['fairness_context'])
        for (split,setting),g in ctx.groupby(['split','setting']):
            check(f'{setting}/{split}: age groups partition rows',int(g.n.sum())==(len(val) if split=='validation' else len(test)))
        gv=ctx[(ctx.split=='validation')&(ctx.setting=='group_thresholds')]
        check('per-group thresholds keep each validation group FPR<=5%',all(np.floor(.05*(row.fp+row.tn))>=row.fp for row in gv.itertuples()))
        for row in ctx[ctx.split=='test'].itertuples():
            check(f'{row.setting}/{row.group}: base rate = fraud/n',abs(row.base_fraud_rate-row.fraud/row.n)<1e-12)
        raw=load_data()
        psi_rows={r['feature']:r for r in ext['velocity']['test_month_psi_all_fields']}
        for column in ['email_is_free','phone_home_valid','phone_mobile_valid','has_other_cards','foreign_request','keep_alive_session']:
            reference=raw.loc[raw.month<=5,column]; current=raw.loc[raw.month==7,column]
            expected=np.array([(reference==0).sum(),(reference==1).sum(),0,0],dtype=float)
            actual=np.array([(current==0).sum(),(current==1).sum(),0,0],dtype=float)
            p=(expected+.5)/(len(reference)+2); q=(actual+.5)/(len(current)+2)
            independent_psi=float(np.sum((q-p)*np.log(q/p)))
            check(f'{column}: binary PSI preserves both value bins',psi_rows[column]['binning']=='discrete'
                  and psi_rows[column]['bins_including_missing']==4 and abs(psi_rows[column]['psi']-independent_psi)<1e-12)
        schema=json.loads((out/'models/schema.json').read_text(encoding='utf-8'))
        preprocessor=joblib.load(out/'models/preprocessor.joblib')
        rng=np.random.default_rng(SEED)
        for split,predictions,month in [('validation',ext_val,6),('test',scores,7)]:
            raw_split=raw.loc[raw.month==month]
            positions=np.sort(rng.choice(len(raw_split),size=min(512,len(raw_split)),replace=False))
            features,_=clean_features(raw_split.iloc[positions],schema)
            transformed=preprocessor.transform(features).astype('float32')
            for name,path in artifact_manifest['models'].items():
                model=joblib.load(ROOT/path)
                if name in artifact_manifest['variants']:
                    dropped=artifact_manifest['variants'][name]['dropped_columns']
                    reconstructed=model.predict_proba(features.drop(columns=dropped))[:,1]
                elif name=='age_proxy':
                    reconstructed=model.predict_proba(features.drop(columns=['customer_age']))[:,1]
                else:
                    reconstructed=model.predict(transformed)
                check(f'{name}/{split}: saved model reproduces stored sample scores',
                      np.allclose(reconstructed,predictions[name].to_numpy()[positions],rtol=0,atol=1e-12))
            check(f'age proxy/{split}: AUC reconciles',abs(roc_auc_score(
                predictions.customer_age>=50,predictions.age_proxy)-ext['fairness']['age_proxy']['roc_auc'][split])<1e-12)
    validate_rolling(out,check)
    validate_coursework(out,check)
    validate_features(out,check)
    save_json({'passed':True,'count':len(checks),'checks':checks},out/'verification.json')
    print(f'{len(checks)} independent consistency checks passed.')

if __name__=='__main__': run()
