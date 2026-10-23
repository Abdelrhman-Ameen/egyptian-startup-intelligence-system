"""Add calibration, survey coverage and uncertainty diagnostics to the completed EDA."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score
import analyze_spain_ventures as eda

def main():
    df,raw,meta,cleaning=eda.load_data()
    pred=pd.read_csv(eda.TABLE/'out_of_fold_predictions.csv')
    rows=[]
    for model,group in pred.loc[pred['repeat']==0].groupby('model'):
        group=group.copy()
        group['bin']=pd.qcut(group.probability,4,duplicates='drop')
        for key,g in group.groupby('bin',observed=True):
            lo,hi=eda.wilson(int(g.non_survival_2014.sum()),len(g))
            rows.append({'model':model,'bin':str(key),'n':len(g),'mean_probability':g.probability.mean(),
                         'observed_non_survival':g.non_survival_2014.mean(),'wilson_low':lo,'wilson_high':hi})
    calibration=eda.save_table('calibration_first_partition',pd.DataFrame(rows))
    fig,ax=plt.subplots(figsize=(7.5,5))
    ax.plot([0,1],[0,1],'--',color='grey',label='Perfect calibration')
    for name in ['Logistic regression','Random forest','Equal-weight ensemble']:
        g=calibration.loc[calibration.model==name]
        ax.plot(g.mean_probability,g.observed_non_survival,'o-',label=name)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Average predicted non-survival score',ylabel='Observed non-survival proportion',
           title='Uncalibrated class-weighted scores: first OOF partition')
    ax.legend(fontsize=8);eda.save_fig('12_calibration',fig)
    # Conditional intervals resample companies' fixed first-partition predictions, not fitted models.
    ci=[];rng=np.random.default_rng(eda.SEED)
    for name in ['Logistic regression','Random forest','Gradient boosting','Equal-weight ensemble']:
        g=pred.loc[(pred['repeat']==0)&(pred.model==name)]
        y=g.non_survival_2014.to_numpy();p=g.probability.to_numpy();zero=np.flatnonzero(y==0);one=np.flatnonzero(y==1)
        scores=[]
        for _ in range(2000):
            idx=np.r_[rng.choice(zero,len(zero),replace=True),rng.choice(one,len(one),replace=True)]
            scores.append([roc_auc_score(y[idx],p[idx]),average_precision_score(y[idx],p[idx])])
        bounds=np.quantile(scores,[.025,.975],axis=0)
        ci.append({'model':name,'roc_auc_first_partition':roc_auc_score(y,p),'roc_auc_low':bounds[0,0],
                   'roc_auc_high':bounds[1,0],'ap_first_partition':average_precision_score(y,p),'ap_low':bounds[0,1],
                   'ap_high':bounds[1,1],'method':'2000 stratified row bootstraps of fixed OOF predictions; ignores refitting uncertainty and fold dependence.'})
    eda.save_table('conditional_bootstrap_intervals',pd.DataFrame(ci))
    # Full score distributions and explicit category semantics are left available for team review.
    counts=[]
    for c in [x for x in df if x.startswith(('Sources_knowledge','Training_experience','Partners','Innovation','Strategy_performance'))]:
        for v,n in df[c].value_counts(dropna=False).items():counts.append({'feature':c,'value':v,'count':n,'percent_all':n/len(df)*100})
    eda.save_table('survey_score_distributions',pd.DataFrame(counts))
    financial=[]
    for i in range(1,5):
        a,b=f'Financial_performance{i}_2008',f'Financial_performance{i}_2014'
        for label,mask in [('All',pd.Series(True,index=df.index)),('Survived',df[eda.TARGET]==1),('Not survived',df[eda.TARGET]==0),('Unknown',df[eda.TARGET].isna())]:
            paired=df.loc[mask,[a,b]].dropna()
            financial.append({'financial_metric':i,'outcome':label,'both_years_available':len(paired)})
    eda.save_table('paired_financial_coverage',pd.DataFrame(financial))
    manifest={'source_sha256':hashlib.sha256((eda.BASE/'raw/technology_entrepreneurs.sav').read_bytes()).hexdigest(),
              'analysis_script_sha256':hashlib.sha256(Path(eda.__file__).read_bytes()).hexdigest(),
              'row_count':len(df),'analysis_date':'2026-10-23','repeats':10,'folds':5,
              'positive_class':'non_survival_2014 = 1 - Survival_2014 (known labels only)',
              'fixed_features':eda.CORE,'hyperparameter_search':False,'independent_test_set':False,
              'cv_interpretation':'Internal exploratory validation; repeated partitions are dependent, not independent samples.',
              'financial_currency_and_units':'Unverified; values retained in source units.',
              'age_firm_unit':'Unverified; no conversion applied.',
              'publication_year':2024,'survey_year':2009,'survey_reference_period':'inception / through 2008',
              'outcome_year':2014,'registry_extraction_year':2016}
    (eda.OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Supplementary diagnostics complete.')

if __name__=='__main__': main()
