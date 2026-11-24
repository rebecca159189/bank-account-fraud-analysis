"""One final held-out evaluation of frozen development choices. No fitting here."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'4')
import json
from datetime import datetime, timezone, timedelta
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from threadpoolctl import threadpool_limits
from .common import *

OUT=ROOT/'outputs'; MODELS=OUT/'models'; FIG=OUT/'figures'
DISPLAY={'logistic_regression':'Logistic regression','lightgbm':'LightGBM','mlp':'MLP',
         'isolation_forest':'Isolation Forest','distilled_tree':'Distilled rules'}
COLORS=['#577590','#147d92','#c8862d','#9b7b9b','#c4584a']

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
       'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})

def rates_with_ci(m):
    r=dict(m)
    r['recall_ci']=wilson(r['tp'],r['tp']+r['fn'])
    r['fpr_ci']=wilson(r['fp'],r['fp']+r['tn'])
    return r

def run():
    manifest=json.loads((MODELS/'frozen_manifest.json').read_text(encoding='utf-8'))
    assert manifest['test_labels_used_for_selection'] is False
    assert sha256(ROOT/'data/raw/Base.csv')==manifest['raw_sha256']
    df=load_data(); test=df.month==7
    schema=json.loads((MODELS/'schema.json').read_text(encoding='utf-8'))
    xt,_=clean_features(df.loc[test],schema); prep=joblib.load(MODELS/'preprocessor.joblib')
    zt=prep.transform(xt).astype('float32'); yt=df.loc[test,'fraud_bool'].to_numpy()
    validation=pd.read_parquet(OUT/'predictions/validation.parquet')
    test_scores=pd.DataFrame({'row_id':df.index[test],'fraud_bool':yt,'customer_age':df.loc[test,'customer_age'].to_numpy()})
    families=list(DISPLAY); test_results=[]; validation_results=[]
    with threadpool_limits(limits=4):
        for f in families:
            model=joblib.load(MODELS/f'{f}.joblib')
            if f=='lightgbm': s=model.predict_proba(xt)[:,1]
            elif f=='isolation_forest': s=-model.score_samples(zt)
            elif f=='distilled_tree': s=model.predict(zt)
            else: s=model.predict_proba(zt)[:,1]
            test_scores[f]=s
            t=manifest['models'][f]['threshold']
            test_results.append(dict(model=f,**rates_with_ci(metrics(yt,s,t))))
            validation_results.append(dict(model=f,**rates_with_ci(manifest['models'][f]['validation'])))
    # Same fixed review capacity for every model. K uses no test label or score quantile target.
    budget=manifest['review_budget_rate']; k=int(np.floor(budget*len(yt)))
    same=[]; fixed_flags={}
    for f in families:
        flags=top_k(test_scores[f],k,test_scores.row_id)
        fixed_flags[f]=flags
        m=binary_metrics(yt,flags)
        same.append(dict(model=f,k=k,**m))
        test_scores[f+'__alert_frozen']=np.asarray(test_scores[f],dtype=float)>=manifest['models'][f]['threshold']
        test_scores[f+'__alert_topk']=flags.astype('int8')
    a=fixed_flags['lightgbm']; b=fixed_flags['distilled_tree']
    positive=np.where(yt==1)[0]; rng=np.random.default_rng(SEED)
    differences=(a[positive].astype(float)-b[positive].astype(float))
    boot=np.mean(differences[rng.integers(0,len(positive),size=(500,len(positive)))],axis=1)
    comparison={'budget_rate':budget,'test_k':k,'teacher_tp':int((a&(yt==1)).sum()),
       'rules_tp':int((b&(yt==1)).sum()),'caught_difference':int(((a.astype(int)-b.astype(int))*(yt==1)).sum()),
       'recall_gap':float(differences.mean()),'bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
       'bootstrap_replicates':500,'bootstrap_conditioning':'paired positive-class bootstrap, fixed predictions and K; no retraining or re-ranking',
       'tie_break':'seed42 row hash, independent of labels and month ordering',
       'rules_cutoff_tie_count':int((test_scores.distilled_tree==np.min(test_scores.loc[b,'distilled_tree'])).sum()),
       'rules_selected_from_cutoff_tie':int((b & (test_scores.distilled_tree==np.min(test_scores.loc[b,'distilled_tree']))).sum())}
    # Fairness: one global frozen threshold for both age groups.
    fairness=[]; gaps=[]; sensitivity=[]
    for split,raw,sc in [('validation',df.loc[df.month==6],validation),('test',df.loc[test],test_scores)]:
        age=raw.customer_age.to_numpy(); y=raw.fraud_bool.to_numpy()
        for f in families:
            pred=sc[f].to_numpy(dtype=float)>=manifest['models'][f]['threshold']
            group_results={}
            for name,mask in [('age<50',age<50),('age>=50',age>=50)]:
                m=binary_metrics(y[mask],pred[mask]); lo,hi=wilson(m['fp'],m['fp']+m['tn'])
                fairness.append(dict(model=f,split=split,group=name,**m,fpr_ci_low=lo,fpr_ci_high=hi))
                group_results[name]=m
            old=group_results['age>=50']['fpr']; young=group_results['age<50']['fpr']
            gaps.append(dict(model=f,split=split,fpr_gap_older_minus_younger=old-young,
                             fpr_ratio_older_to_younger=old/young if young else None))
            for name,mask in [('age<=50',age<=50),('age>50',age>50)]:
                sensitivity.append(dict(model=f,split=split,group=name,**binary_metrics(y[mask],pred[mask])))
    monthly=json.loads((OUT/'monthly_development.json').read_text(encoding='utf-8'))
    monthly.append(dict(month=7,split='test',**next(m for m in test_results if m['model']=='lightgbm')))
    for m in monthly:
        m['model']='lightgbm'; m['recall_ci_low'],m['recall_ci_high']=wilson(m['tp'],m['tp']+m['fn'])
    output={'manifest':manifest,'test_results':test_results,'validation_results':validation_results,
       'same_budget_results':same,'rules_comparison':comparison,'fairness':fairness,'fairness_gaps':gaps,
       'age_boundary_sensitivity':sensitivity,'monthly':monthly,
       'evaluation_time_singapore':datetime.now(timezone(timedelta(hours=8))).isoformat(),
       'test_prediction_artifact':'outputs/predictions/test.parquet',
       'final_test_policy':'All choices frozen before this stage; this stage only scores and summarizes.'}
    test_scores.to_parquet(OUT/'predictions/test.parquet',index=False)
    save_json(output,OUT/'evaluation.json')
    for name,rows in [('test_metrics',test_results),('validation_metrics',validation_results),
             ('same_budget',same),('fairness',fairness),('fairness_gaps',gaps),
             ('age_boundary_sensitivity',sensitivity),('monthly_recall',monthly)]:
        pd.DataFrame(rows).to_csv(OUT/f'tables/{name}.csv',index=False)
    style(); make_plots(output,test_scores)
    print(pd.DataFrame(test_results)[['model','pr_auc_ap','recall','fpr','alerts']].to_string(index=False))
    print('SAME BUDGET:',comparison)

def make_plots(output,scores):
    rows=output['test_results']; names=[DISPLAY[r['model']] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,metric,title in zip(axes,['pr_auc_ap','recall'],['Test PR-AUC (average precision)','Test recall at validation-frozen threshold']):
        values=[r[metric] for r in rows]; bars=ax.barh(names,values,color=COLORS)
        ax.invert_yaxis(); ax.set_title(title,fontsize=11); ax.set_xlim(0,max(values)*1.25+.005)
        ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
        for bar,value in zip(bars,values): ax.text(value+.004,bar.get_y()+bar.get_height()/2,f'{value:.3f}',va='center')
    fig.tight_layout(); fig.savefig(FIG/'benchmark.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5.2))
    y=scores.fraud_bool.to_numpy()
    for row,color in zip(rows,COLORS):
        f=row['model']; p,r,_=precision_recall_curve(y,scores[f])
        ax.step(r,p,where='post',label=f'{DISPLAY[f]} (AP={row["pr_auc_ap"]:.3f})',color=color,lw=1.7)
    ax.axhline(y.mean(),color='#999999',ls='--',label=f'Prevalence={y.mean():.3%}')
    ax.set(xlabel='Recall',ylabel='Precision',title='Held-out month 7: precision-recall curves',xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=9); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIG/'pr_curves.png',dpi=170); plt.close(fig)
    month=pd.DataFrame(output['monthly'])
    fig,axes=plt.subplots(2,1,figsize=(10,6.4),sharex=True)
    for ax in axes: ax.axvspan(-.3,5.5,color='#eeeeee'); ax.grid(alpha=.18)
    axes[0].plot(month.month,month.recall,'o-',color='#147d92')
    axes[0].fill_between(month.month,month.recall_ci_low,month.recall_ci_high,color='#147d92',alpha=.15)
    axes[0].set(ylabel='Recall',title='LightGBM at one threshold frozen on month 6')
    axes[0].text(.02,.08,'Months 0-5: in-sample (optimistic)',transform=axes[0].transAxes,color='#777777')
    axes[1].plot(month.month,month.fpr,'o-',color='#c4584a'); axes[1].axhline(.05,color='#777777',ls='--',lw=1)
    axes[1].set(ylabel='False positive rate',xlabel='Relative month index (0-7)',xticks=range(8))
    axes[0].annotate('Validation',xy=(6,month.loc[month.month==6,'recall'].iloc[0]),xytext=(.74,.85),textcoords='axes fraction',arrowprops={'arrowstyle':'->'},fontsize=9)
    axes[0].annotate('Final test',xy=(7,month.loc[month.month==7,'recall'].iloc[0]),xytext=(.89,.65),textcoords='axes fraction',arrowprops={'arrowstyle':'->'},fontsize=9)
    fig.tight_layout(); fig.savefig(FIG/'monthly_recall.png',dpi=170); plt.close(fig)
    fair=pd.DataFrame(output['fairness']); fair=fair[fair.split=='test']
    fig,ax=plt.subplots(figsize=(10,5)); pos=np.arange(len(rows)); width=.35
    for offset,group,color in [(-width/2,'age<50','#577590'),(width/2,'age>=50','#c8862d')]:
        g=fair[fair.group==group].set_index('model').loc[[r['model'] for r in rows]]
        v=g.fpr.to_numpy(); err=np.vstack([v-g.fpr_ci_low.to_numpy(),g.fpr_ci_high.to_numpy()-v])
        ax.bar(pos+offset,v,width,label=group,color=color,yerr=err,capsize=3)
    ax.axhline(.05,color='#777777',ls='--',lw=1); ax.set_xticks(pos,names,rotation=10)
    ax.set(ylabel='False positive rate',title='Age fairness on month 7 (global frozen thresholds, Wilson 95% CI)')
    ax.legend(); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True); fig.tight_layout(); fig.savefig(FIG/'fairness.png',dpi=170); plt.close(fig)
    same=output['same_budget_results']; fig,ax=plt.subplots(figsize=(9,4.7))
    bars=ax.barh([DISPLAY[r['model']] for r in same],[r['tp'] for r in same],color=COLORS)
    ax.invert_yaxis(); ax.set(xlabel='Frauds caught',title=f'Equal review capacity on test: exactly {same[0]["k"]:,} alerts per method')
    for bar,r in zip(bars,same): ax.text(r['tp']+8,bar.get_y()+bar.get_height()/2,f'{r["tp"]:,} ({r["recall"]:.1%})',va='center')
    ax.set_xlim(0,max(r['tp'] for r in same)*1.3); ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG/'equal_budget.png',dpi=170); plt.close(fig)

if __name__=='__main__': run()
