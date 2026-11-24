"""Exact native TreeSHAP on validation, plus a readable five-leaf student tree."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ.setdefault(key,'4')
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
from threadpoolctl import threadpool_limits
from .common import *
from .evaluate import style

OUT=ROOT/'outputs'; MODELS=OUT/'models'; FIG=OUT/'figures'

def run():
    manifest=json.loads((MODELS/'frozen_manifest.json').read_text(encoding='utf-8'))
    schema=json.loads((MODELS/'schema.json').read_text(encoding='utf-8'))
    df=load_data(); val=df.month==6; tr=df.month<=5; te=df.month==7
    xtr,_=clean_features(df.loc[tr],schema); xv,_=clean_features(df.loc[val],schema); xt,_=clean_features(df.loc[te],schema)
    prep=joblib.load(MODELS/'preprocessor.joblib'); teacher=joblib.load(MODELS/'lightgbm.joblib'); student=joblib.load(MODELS/'distilled_tree.joblib')
    rng=np.random.default_rng(SEED); sample=xv.iloc[np.sort(rng.choice(len(xv),size=5000,replace=False))]
    with threadpool_limits(limits=4):
        contributions=teacher.predict(sample,pred_contrib=True)
        raw=teacher.predict(sample,raw_score=True)
    shap_values=contributions[:,:-1]
    error=float(np.max(np.abs(contributions.sum(axis=1)-raw)))
    assert error<1e-6,'TreeSHAP additivity failed'
    importance=pd.DataFrame({'feature':sample.columns,'mean_abs_shap':np.abs(shap_values).mean(axis=0)}).sort_values('mean_abs_shap',ascending=False)
    importance.to_csv(OUT/'tables/shap_importance.csv',index=False)
    np.savez_compressed(OUT/'predictions/validation_shap.npz',row_id=sample.index.to_numpy(),shap_values=shap_values,base_value=contributions[:,-1],features=np.asarray(sample.columns,dtype=str))
    style(); top=importance.head(15).iloc[::-1]
    fig,ax=plt.subplots(figsize=(9,6)); ax.barh(top.feature,top.mean_abs_shap,color='#147d92')
    ax.set(xlabel='Mean absolute SHAP value (fraud log-odds)',title='LightGBM: exact TreeSHAP on 5,000 validation applications')
    ax.grid(axis='x',alpha=.18); ax.set_axisbelow(True); fig.tight_layout(); fig.savefig(FIG/'shap_importance.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,7))
    for j,f in enumerate(importance.head(15).feature.iloc[::-1]):
        index=sample.columns.get_loc(f); values=shap_values[:,index]
        rawfeature=sample[f]
        if isinstance(rawfeature.dtype,pd.CategoricalDtype):
            colors=np.full(len(sample),.5) # unordered categories do not have a meaningful low/high order
        else:
            v=rawfeature.to_numpy(dtype=float); lo,hi=np.nanquantile(v,[.05,.95])
            colors=np.nan_to_num(np.clip((v-lo)/(hi-lo+1e-12),0,1),nan=.5)
        ax.scatter(values,j+rng.normal(0,.08,len(values)),c=colors,s=3,alpha=.45,cmap='coolwarm',vmin=0,vmax=1,rasterized=True)
    ax.axvline(0,color='#999999',lw=1); ax.set_yticks(range(len(top)),importance.head(15).feature.iloc[::-1])
    ax.set(xlabel='SHAP value (positive increases fraud log-odds)',title='Validation SHAP distribution (blue: low; red: high; categories: neutral)')
    fig.tight_layout(); fig.savefig(FIG/'shap_beeswarm.png',dpi=170); plt.close(fig)
    # Translate standardized split thresholds back into original feature units.
    features=list(prep.get_feature_names_out()); numeric=list(prep.transformers_[0][2])
    scaler=prep.named_transformers_['num'].named_steps['scale']
    medians=prep.named_transformers_['num'].named_steps['impute'].statistics_
    def condition(feature_id,threshold,left):
        name=features[feature_id]; op='<=' if left else '>'
        if feature_id<len(numeric):
            t=threshold*scaler.scale_[feature_id]+scaler.mean_[feature_id]
            if name.endswith('__missing'):
                return f'{name.removesuffix("__missing")} '+('非缺失' if left else '缺失')
            return f'{name} {op} {t:.6g}'
        # One-hot splits separate a single category from all others.
        for c in CATEGORICAL:
            prefix=c+'_'
            if name.startswith(prefix): return f'{c} '+('不等于 ' if left else '等于 ')+name[len(prefix):]
        return f'{name} {op} {threshold:.6g}'
    tree=student.tree_; paths={}
    def walk(node,parts):
        if tree.children_left[node]==tree.children_right[node]: paths[node]=parts; return
        fi=tree.feature[node]; t=tree.threshold[node]
        walk(tree.children_left[node],parts+[condition(fi,t,True)])
        walk(tree.children_right[node],parts+[condition(fi,t,False)])
    walk(0,[])
    ztr=prep.transform(xtr).astype('float32'); zv=prep.transform(xv).astype('float32'); zt=prep.transform(xt).astype('float32')
    train_leaves=student.apply(ztr); val_leaves=student.apply(zv); test_leaves=student.apply(zt)
    rules=[]
    for rank,node in enumerate(sorted(paths,key=lambda n:tree.value[n,0,0],reverse=True),1):
        mask=test_leaves==node
        rules.append({'rule_id':f'R{rank}','node_id':int(node),'path':' 且 '.join(paths[node]),
          'teacher_log_odds':float(tree.value[node,0,0]),'train_rows':int((train_leaves==node).sum()),
          'validation_rows':int((val_leaves==node).sum()),'test_rows':int(mask.sum()),
          'test_fraud':int(df.loc[te,'fraud_bool'].to_numpy()[mask].sum()),
          'test_fraud_rate':float(df.loc[te,'fraud_bool'].to_numpy()[mask].mean()) if mask.any() else None,
          'frozen_threshold_alert':bool(tree.value[node,0,0]>=manifest['models']['distilled_tree']['threshold'])})
    fidelity={}
    for split,x,z in [('validation',xv,zv),('test',xt,zt)]:
        teacher_logit=teacher.predict(x,raw_score=True); student_logit=student.predict(z)
        fidelity[split+'_spearman']=float(spearmanr(teacher_logit,student_logit).statistic)
        fidelity[split+'_r2']=float(r2_score(teacher_logit,student_logit))
        fidelity[split+'_logit_mae']=float(mean_absolute_error(teacher_logit,student_logit))
        teacher_flag=teacher.predict_proba(x)[:,1].astype(float)>=manifest['models']['lightgbm']['threshold']
        student_flag=student_logit>=manifest['models']['distilled_tree']['threshold']
        fidelity[split+'_frozen_decision_agreement']=float(np.mean(teacher_flag==student_flag))
    # Use an unscaled display-only clone so diagram thresholds are readable.
    import copy
    display_tree=copy.deepcopy(student)
    for node in range(tree.node_count):
        fi=tree.feature[node]
        if 0<=fi<len(numeric):
            display_tree.tree_.threshold[node]=tree.threshold[node]*scaler.scale_[fi]+scaler.mean_[fi]
    fig,ax=plt.subplots(figsize=(18,8))
    plot_tree(display_tree,feature_names=features,filled=True,rounded=True,precision=2,
              impurity=False,proportion=True,fontsize=9,ax=ax)
    ax.set_title('Distilled risk tiers: depth <=4, exactly five leaves (values are teacher log-odds)',fontsize=14)
    fig.tight_layout(); fig.savefig(FIG/'distilled_tree.png',dpi=170); plt.close(fig)
    output={'shap_importance':importance.to_dict(orient='records'),'shap_sample_size':len(sample),
       'shap_additivity_max_error':error,'shap_output':'raw fraud log-odds; native exact TreeSHAP',
       'shap_split':'validation month6; never used to retune after test',
       'rules':rules,'student_fidelity':fidelity,'numeric_imputation_medians':dict(zip(numeric,medians)),
       'rule_notes':['Five mutually exclusive risk tiers, ordered by student score; all applications belong to one tier.',
                     'Missing numeric values are replaced by training medians before these splits; displayed thresholds are rounded, exact execution is in the saved model.',
                     'Same-capacity comparison partially reviews tied boundary leaves using a seeded row hash; a whole-leaf OR rule has a coarser capacity.']}
    save_json(output,OUT/'explanation.json'); pd.DataFrame(rules).to_csv(OUT/'tables/distilled_rules.csv',index=False,encoding='utf-8-sig')
    print('SHAP verified:',error,'Rules:',len(rules),'Fidelity:',fidelity)

if __name__=='__main__': run()
