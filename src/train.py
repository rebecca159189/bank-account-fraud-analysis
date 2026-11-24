"""Train/choose all models on months 0..5 / 6, then write a frozen manifest.

This stage deliberately does not score month 7. Run src.evaluate only after this stage.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(key, '4')
import time, copy, warnings, platform, sys
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from .common import *

OUT=ROOT/'outputs'; MODELS=OUT/'models'

def log(s): print(time.strftime('%H:%M:%S'),s,flush=True)

def run():
    MODELS.mkdir(parents=True,exist_ok=True)
    log('Loading Base; fitting transforms only on months 0..5')
    df=load_data(); tr=df.month<=5; va=df.month==6
    xtr,schema=clean_features(df.loc[tr]); xv,_=clean_features(df.loc[va],schema)
    ytr=df.loc[tr,'fraud_bool'].to_numpy(); yv=df.loc[va,'fraud_bool'].to_numpy()
    prep=make_preprocessor(xtr)
    ztr=prep.fit_transform(xtr).astype('float32'); zv=prep.transform(xv).astype('float32')
    joblib.dump(prep,MODELS/'preprocessor.joblib'); save_json(schema,MODELS/'schema.json')
    records=[]; chosen={}; val=pd.DataFrame({'row_id':df.index[va], 'fraud_bool':yv})
    def register(family,name,model,scores,params,elapsed):
        t=threshold_at_fpr(yv,scores); m=metrics(yv,scores,t)
        row=dict(family=family,candidate=name,params=params,fit_seconds=elapsed,**m)
        records.append(row)
        log(f'{name}: val AP={m["pr_auc_ap"]:.4f}, recall={m["recall"]:.4f}, FPR={m["fpr"]:.4f}')
        key=(m['recall'],m['pr_auc_ap'])
        if family not in chosen or key>chosen[family]['key']:
            chosen[family]={'key':key,'candidate':name,'params':params,'threshold':t,'validation':m}
            joblib.dump(model,MODELS/f'{family}.joblib')
            val[family]=scores
    with threadpool_limits(limits=4):
        for c in [.1,1.]:
            start=time.perf_counter(); model=LogisticRegression(C=c,class_weight='balanced',
                    solver='lbfgs',max_iter=500,tol=1e-5,random_state=SEED)
            model.fit(ztr,ytr)
            register('logistic_regression',f'LR_C{c}',model,model.predict_proba(zv)[:,1],
                     {'C':c,'class_weight':'balanced','max_iter':500,'n_iter':int(model.n_iter_[0])},time.perf_counter()-start)
        ratio=float((ytr==0).sum()/(ytr==1).sum())
        for leaves,reg in [(15,5.),(31,10.),(63,20.)]:
            start=time.perf_counter(); model=lgb.LGBMClassifier(n_estimators=900,learning_rate=.04,
                num_leaves=leaves,min_child_samples=150,reg_lambda=reg,colsample_bytree=.9,
                subsample=.9,subsample_freq=1,scale_pos_weight=ratio,n_jobs=4,
                deterministic=True,force_col_wise=True,random_state=SEED,verbosity=-1,metric='average_precision')
            model.fit(xtr,ytr,eval_set=[(xv,yv)],eval_metric='average_precision',
                callbacks=[lgb.early_stopping(60,first_metric_only=True,verbose=False)])
            register('lightgbm',f'LGBM_leaves{leaves}',model,model.predict_proba(xv)[:,1],
                {'num_leaves':leaves,'reg_lambda':reg,'best_iteration':model.best_iteration_,
                 'scale_pos_weight':ratio},time.perf_counter()-start)
        weights=compute_sample_weight('balanced',ytr).astype('float32')
        for lr in [.001,.0004]:
            start=time.perf_counter(); model=MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',
                solver='adam',batch_size=2048,learning_rate_init=lr,alpha=.01,
                random_state=SEED,max_iter=1,shuffle=True)
            best=None; bestkey=(-1,-1); stale=0; history=[]
            for epoch in range(1,31):
                model.partial_fit(ztr,ytr,classes=np.array([0,1]),sample_weight=weights)
                s=model.predict_proba(zv)[:,1]; t=threshold_at_fpr(yv,s); m=metrics(yv,s,t)
                key=(m['recall'],m['pr_auc_ap']); history.append(dict(epoch=epoch,**m))
                if key>bestkey:
                    best=copy.deepcopy(model); bestkey=key; bestepoch=epoch; stale=0
                else: stale+=1
                log(f'MLP lr={lr} epoch={epoch} val AP={m["pr_auc_ap"]:.4f} recall={m["recall"]:.4f}')
                if epoch>=10 and stale>=6: break
            save_json(history,OUT/f'mlp_history_{lr}.json')
            register('mlp',f'MLP_lr{lr}',best,best.predict_proba(zv)[:,1],
                     {'hidden_layer_sizes':[64,32],'learning_rate_init':lr,'best_epoch':bestepoch,
                      'class_weight':'balanced via sample_weight','alpha':.01},time.perf_counter()-start)
        # Exactly one predeclared unsupervised configuration: labels never enter IF fitting or parameter choice.
        start=time.perf_counter(); model=IsolationForest(n_estimators=250,max_samples=4096,
                    contamination='auto',n_jobs=4,random_state=SEED)
        model.fit(ztr)
        register('isolation_forest','IF_250_4096',model,-model.score_samples(zv),
                 {'n_estimators':250,'max_samples':4096,'labels_used_for_fit':False,
                  'labels_used_for_threshold':True},time.perf_counter()-start)
        log('Distilling LightGBM log-odds into 3-4 level trees with <=5 leaves')
        teacher=joblib.load(MODELS/'lightgbm.joblib')
        teacher_train=teacher.predict(xtr,raw_score=True)
        teacher_val=teacher.predict(xv,raw_score=True)
        budget=chosen['lightgbm']['validation']['alert_rate']
        k=int(np.floor(budget*len(yv)))
        student_best=None; student_key=(-1,-1); student_candidates=[]
        for depth in [3,4]:
            student=DecisionTreeRegressor(max_depth=depth,max_leaf_nodes=5,min_samples_leaf=.005,
                                          random_state=SEED)
            student.fit(ztr,teacher_train)
            s=student.predict(zv); m=binary_metrics(yv,top_k(s,k,df.index[va]))
            fidelity=float(spearmanr(teacher_val,s).statistic)
            student_candidates.append(dict(depth=depth,leaves=student.get_n_leaves(),
                validation_same_budget=m,spearman=fidelity,r2=float(r2_score(teacher_val,s))))
            key=(m['recall'],fidelity)
            if key>student_key: student_best=student; student_key=key; studentdepth=depth
        joblib.dump(student_best,MODELS/'distilled_tree.joblib')
        s=student_best.predict(zv); t=threshold_at_fpr(yv,s)
        chosen['distilled_tree']={'candidate':f'Distilled_depth{studentdepth}',
              'params':{'max_depth':studentdepth,'max_leaf_nodes':5,'min_samples_leaf':.005},
              'threshold':t,'validation':metrics(yv,s,t),'fidelity_candidates':student_candidates}
        val['distilled_tree']=s
        # In-sample monthly estimates are kept separate from future-month evaluation.
        monthly=[]
        for month in range(7):
            mask=df.month==month
            xm,_=clean_features(df.loc[mask],schema)
            s=teacher.predict_proba(xm)[:,1]
            monthly.append(dict(month=month,split='train_in_sample' if month<6 else 'validation',
                                **metrics(df.loc[mask,'fraud_bool'],s,chosen['lightgbm']['threshold'])))
        save_json(monthly,OUT/'monthly_development.json')
    supervised=['logistic_regression','lightgbm','mlp']
    winner=max(supervised,key=lambda f: chosen[f]['key'])
    for spec in chosen.values(): spec.pop('key',None)
    pd.DataFrame([{k:(json.dumps(v) if isinstance(v,dict) else v) for k,v in row.items()}
                  for row in records]).to_csv(OUT/'tables/validation_candidates.csv',index=False)
    val.to_parquet(OUT/'predictions/validation.parquet',index=False)
    manifest={'seed':SEED,'raw_sha256':sha256(ROOT/'data/raw/Base.csv'),
       'split':{'train':list(range(6)),'validation':[6],'test':[7]},
       'selection':'maximize validation recall under FPR<=0.05; AP breaks ties',
       'pr_auc_definition':'average_precision_score (AP); trapezoidal PR AUC is saved separately',
       'supervised_winner':winner,'review_budget_rate':budget,
       'test_labels_used_for_selection':False,'models':chosen,
       'environment':{'python':sys.version,'platform':platform.platform()}}
    save_json(manifest,MODELS/'frozen_manifest.json')
    log(f'FROZEN. Supervised validation winner: {winner}. Test month remains unscored.')

if __name__=='__main__': run()
