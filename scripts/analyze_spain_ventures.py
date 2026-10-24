"""Reproducible EDA and a limited predictive-signal check for Spain ventures.

No Egyptian failure probabilities or causal scenario effects are estimated.
Run from any directory: python scripts/analyze_spain_ventures.py
"""
from pathlib import Path
import json
import re
import unicodedata
import hashlib
import importlib.metadata
import warnings
import numpy as np
import pandas as pd
import pyreadstat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/2026-10-23/spain_technology_ventures'
PROCESSED = BASE / 'processed'
OUT = ROOT / 'reports/spain_technology_ventures'
FIG = OUT / 'figures'
TABLE = OUT / 'tables'
TARGET = 'Survival_2014'
SEED = 42
# Chosen on subject meaning, before inspecting univariate outcome associations.
CORE = ['Founders_team2', 'Employees', 'Technology_strategy', 'Partners1', 'Partners2',
        'Training_experience5', 'Training_experience6', 'Sources_knowledge6', 'Innovation2']
SHORT = {
 'Founders_team2': 'Number of founders', 'Employees': 'Full-time employees',
 'Technology_strategy': 'Technology strategy', 'Partners1': 'Investment partner access',
 'Partners2': 'Technology partner access', 'Training_experience5': 'Same-sector experience score',
 'Training_experience6': 'Management experience score', 'Sources_knowledge6': 'Customer dialogue score',
 'Innovation2': 'R&D emphasis score', 'Age_firm': 'Firm age (unit unresolved)',
 'Age_entrepreneur': 'Entrepreneur age', 'Gender_entrepreneur': 'Gender',
 'Founders_team1': 'Founding team indicator',
}

def save_table(name, df):
    df.to_csv(TABLE / f'{name}.csv', index=False, encoding='utf-8-sig')
    return df

def save_fig(name, fig):
    fig.savefig(FIG / f'{name}.png', dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def wilson(k, n):
    if not n:
        return np.nan, np.nan
    z = 1.959963984540054
    ph = k/n
    den = 1+z*z/n
    center = (ph+z*z/(2*n))/den
    radius = z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    return center-radius, center+radius

def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(adjusted, 1)
    return result

def canonical_sector(value):
    if pd.isna(value):
        return np.nan
    value = ''.join(c for c in unicodedata.normalize('NFKD', str(value)) if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', value.strip().upper())

def load_data():
    path = BASE / 'raw/technology_entrepreneurs.sav'
    raw, meta = pyreadstat.read_sav(path, user_missing=True)
    df = raw.copy()
    changes = []
    for col, ranges in meta.missing_ranges.items():
        count = 0
        for item in ranges:
            mask = df[col].between(item['lo'], item['hi'])
            count += int(mask.sum())
            df.loc[mask, col] = np.nan
        changes.append({'feature': col, 'declared_missing_replaced': count,
                        'source_missing_codes': json.dumps(ranges)})
    for col in ['Sector']:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x,str) else x).replace('',np.nan)
    df.index = pd.Index(np.arange(1,len(df)+1), name='source_row')
    # The row key is lineage only. It is not a genuine company identifier or a feature.
    return df, raw, meta, pd.DataFrame(changes)

def feature_role(col):
    if col == TARGET:
        return 'Outcome only', 'Survival at 2014; unknown labels excluded from supervised work.'
    if col.endswith('_2014'):
        return 'Exclude: future outcome', 'Observed at outcome year, collected in 2016; target leakage.'
    if col.endswith('_2008'):
        return 'Sensitivity only', '2008 values extracted in 2016; availability at prediction time and financial units unverified.'
    if col == 'Sector':
        return 'EDA only initially', 'Free text, high cardinality; require reviewed sector taxonomy.'
    if col == 'Age_firm':
        return 'Exclude until clarified', 'Age unit absent; 4-350 conflicts with an unqualified years interpretation.'
    if col in ['Gender_entrepreneur','Age_entrepreneur']:
        return 'Description only', 'Small subgroups; omitted from this business-feature baseline.'
    if col == 'Founders_team1':
        return 'Redundant', 'Duplicates the single-versus-multiple-founder distinction in founder count.'
    return 'Candidate survey predictor', '2009 survey referring to inception or performance through 2008; score anchors need questionnaire.'

def metrics(y,p):
    pred = np.asarray(p)>=.5
    return {'roc_auc':roc_auc_score(y,p), 'average_precision':average_precision_score(y,p),
            'balanced_accuracy':balanced_accuracy_score(y,pred),'accuracy':accuracy_score(y,pred),
            'non_survival_precision':precision_score(y,pred,zero_division=0),
            'non_survival_recall':recall_score(y,pred,zero_division=0),
            'non_survival_f1':f1_score(y,pred,zero_division=0),'brier':brier_score_loss(y,p)}

def prep(df):
    x = df.copy().astype(float)
    for c in x:
        if c == 'Employees' or c.startswith('Financial_performance'):
            x[c] = np.sign(x[c])*np.log1p(np.abs(x[c]))
    return x

def make_models():
    return {
      'Logistic regression': make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                  LogisticRegression(C=.3, class_weight='balanced',max_iter=2000,random_state=SEED)),
      'Random forest': make_pipeline(SimpleImputer(strategy='median'),
                  RandomForestClassifier(n_estimators=150,max_depth=3,min_samples_leaf=8,
                                         class_weight='balanced',random_state=SEED,n_jobs=1)),
      'Gradient boosting': make_pipeline(SimpleImputer(strategy='median'),
                  HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=5,min_samples_leaf=12,
                        l2_regularization=5,class_weight='balanced',early_stopping=False,random_state=SEED)),
    }

def run_eda():
    for path in [PROCESSED, OUT, FIG, TABLE]: path.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','axes.titlesize':13})
    df, raw, meta, cleaning = load_data()
    cleaning.to_csv(PROCESSED/'missing_code_audit.csv',index=False)
    df.to_csv(PROCESSED/'spain_ventures_clean.csv',encoding='utf-8-sig')
    known = df[TARGET].notna()
    labeled = df.loc[known].copy()
    y = (1-labeled[TARGET]).astype(int)
    labeled['non_survival_2014'] = y
    survey = [c for c in df if not c.endswith(('_2008','_2014'))]
    full = [c for c in survey if c not in ['Sector','Age_firm','Gender_entrepreneur','Age_entrepreneur','Founders_team1']]
    future = [c for c in df if c.endswith('_2014') and c != TARGET]
    finance = [c for c in df if c.endswith('_2008')]
    labeled[CORE+['non_survival_2014']].to_csv(PROCESSED/'core_model_input.csv',encoding='utf-8-sig')
    df.loc[~known].to_csv(PROCESSED/'unknown_outcome_rows.csv',encoding='utf-8-sig')
    roles = []
    for c in df:
        role, reason = feature_role(c)
        roles.append({'feature':c,'description':meta.column_names_to_labels[c], 'role':role,'reason':reason,
                      'missing_all':int(df[c].isna().sum()),'missing_labeled':int(labeled[c].isna().sum()),
                      'unique_nonmissing':int(df[c].nunique()),'min':df[c].min() if c!='Sector' else None,
                      'max':df[c].max() if c!='Sector' else None,'source_value_labels':json.dumps(meta.variable_value_labels.get(c,{})),
                      'in_core_baseline':c in CORE})
    dictionary = save_table('feature_dictionary_and_decisions',pd.DataFrame(roles))
    save_table('numeric_summary',df.select_dtypes('number').describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T.reset_index(names='feature'))
    target_table = pd.DataFrame({'status':['Survived in 2014','Did not survive in 2014','Outcome unknown'],
         'count':[int((df[TARGET]==1).sum()),int((df[TARGET]==0).sum()),int(df[TARGET].isna().sum())]})
    target_table['percent_all'] = target_table['count']/len(df)*100
    save_table('target_distribution',target_table)
    fig, ax = plt.subplots(figsize=(8,4))
    bars=ax.bar(target_table.status,target_table['count'],color=['#466477','#a3483a','#9d9d9d'])
    ax.bar_label(bars,labels=[f'{n} ({p:.1f}% of all rows)' for n,p in zip(target_table['count'],target_table.percent_all)],padding=5)
    ax.set(ylim=(0,130),ylabel='Companies',title='150 labeled companies; 25 outcomes remain unknown')
    save_fig('01_target',fig)

    missing = pd.DataFrame({'feature':df.columns,'missing_count':df.isna().sum().values,
        'missing_percent':df.isna().mean().values*100}).sort_values('missing_count',ascending=False)
    save_table('missingness_all',missing)
    status = df[TARGET].map({0:'Not survived',1:'Survived'}).fillna('Unknown')
    by_status = df.isna().groupby(status).mean().T*100
    save_table('missingness_by_outcome',by_status.reset_index(names='feature'))
    fig,axes=plt.subplots(1,2,figsize=(13,7),gridspec_kw={'width_ratios':[1.2,1]})
    top=missing.head(16).iloc[::-1]
    axes[0].barh(top.feature,top.missing_percent,color='#466477');axes[0].set(xlabel='Missing (%)',title='Most incomplete fields',xlim=(0,100))
    show=finance+future+['Employees','Age_firm']
    im=axes[1].imshow(by_status.loc[show,['Survived','Not survived','Unknown']],vmin=0,vmax=100,cmap='Greys',aspect='auto')
    axes[1].set(yticks=range(len(show)),yticklabels=show,xticks=range(3),xticklabels=['Survived','Not survived','Unknown'],title='Missingness depends on outcome')
    for i in range(len(show)):
        for j in range(3):
            val=by_status.loc[show[i],['Survived','Not survived','Unknown'][j]]
            axes[1].text(j,i,f'{val:.0f}%',ha='center',va='center',color='white' if val>55 else 'black',fontsize=9)
    fig.colorbar(im,ax=axes[1],label='Missing (%)',shrink=.7);fig.tight_layout()
    save_fig('02_missingness',fig)

    # Audit recording/availability effects rather than treating them as valid predictors.
    audits=[]
    for c in finance+future:
        for outcome in ['Survived','Not survived','Unknown']:
            rows=df.loc[status==outcome,c]
            audits.append({'feature':c,'outcome':outcome,'rows':len(rows),'available':int(rows.notna().sum()),'missing':int(rows.isna().sum())})
    save_table('financial_availability_by_outcome',pd.DataFrame(audits))
    missing_tests=[]
    for c in df.columns.drop(TARGET):
        tab=pd.crosstab(known,df[c].isna()).reindex(index=[False,True],columns=[False,True],fill_value=0)
        if tab.values.sum() and (tab.sum(axis=0)>0).all():
            odds,p=stats.fisher_exact(tab)
            missing_tests.append({'feature':c,'unknown_outcome_missing_percent':df.loc[~known,c].isna().mean()*100,
                                 'known_outcome_missing_percent':df.loc[known,c].isna().mean()*100,'p_value':p})
    mt=pd.DataFrame(missing_tests); mt['q_bh']=bh(mt.p_value);save_table('label_missingness_tests',mt.sort_values('p_value'))
    retention=[]
    for name,cols in [('All 47 columns',list(df)),('Survey fields plus label',survey+[TARGET]),
                      ('Core 9 plus label',CORE+[TARGET]),('2008 financial fields plus label',finance+[TARGET])]:
        subset=df.dropna(subset=cols)
        retention.append({'selection':name,'remaining_rows':len(subset),'non_survived':int((subset[TARGET]==0).sum()),
                          'survived':int((subset[TARGET]==1).sum())})
    save_table('complete_case_retention',pd.DataFrame(retention))

    # Quality flags: never silently remove or winsorize original records.
    issues=[]
    def flag(mask,field,issue):
        for idx in df.index[mask]:
            issues.append({'source_row':idx,'feature':field,'value':df.loc[idx,field],'issue':issue,'survival_2014':df.loc[idx,TARGET]})
    flag(df.Employees>=250,'Employees','Conflicts with described sample selection (<250 employees); measurement timing needs clarification.')
    flag(df.Age_firm>96,'Age_firm','If recorded in months, exceeds described 8-year limit; unit remains unverified.')
    flag((df.Founders_team1==0)&(df.Founders_team2>1) | (df.Founders_team1==1)&(df.Founders_team2==1), 'Founders_team2','Team flag conflicts with founder count.')
    for c in df.select_dtypes('number'):
        if c.startswith(('Sources_knowledge','Partners','Innovation','Strategy_performance')):
            flag(df[c].notna()&(~df[c].between(1,7) | (df[c]%1!=0)),c,'Outside expected observed integer score range 1-7.')
        elif c.startswith('Training_experience'):
            flag(df[c].notna()&(~df[c].between(1,5) | (df[c]%1!=0)),c,'Outside expected observed integer score range 1-5.')
    save_table('quality_flags',pd.DataFrame(issues))
    outliers=[]
    for c in ['Age_firm','Employees','Founders_team2','Age_entrepreneur']+finance+future[1:]:
        if c not in df or c=='Internationalization_2014': continue
        s=df[c].dropna(); q1,q3=s.quantile([.25,.75]); lo,hi=q1-1.5*(q3-q1),q3+1.5*(q3-q1)
        outliers.append({'feature':c,'n':len(s),'min':s.min(),'median':s.median(),'max':s.max(),
                         'lower_fence':lo,'upper_fence':hi,'flagged_iqr':int(((s<lo)|(s>hi)).sum()),'zeros':int((s==0).sum()),'negative':int((s<0).sum())})
    save_table('outlier_audit',pd.DataFrame(outliers))
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,c in zip(axes.flat,['Employees','Founders_team2','Age_firm','Age_entrepreneur','Financial_performance1_2008','Financial_performance2_2008']):
        combined=labeled[c].dropna()
        transform=c=='Employees' or c.startswith('Financial')
        if transform: combined=np.sign(combined)*np.log1p(abs(combined))
        bins=np.histogram_bin_edges(combined,bins=12)
        for value,label,color in [(1,'Survived','#466477'),(0,'Not survived','#a3483a')]:
            values=df.loc[df[TARGET]==value,c].dropna()
            transform=c=='Employees' or c.startswith('Financial')
            if transform: values=np.sign(values)*np.log1p(abs(values))
            ax.hist(values,bins=bins,weights=np.ones(len(values))/len(values),alpha=.55,label=f'{label} (n={len(values)})',color=color)
        ax.set_title(SHORT.get(c,c),fontsize=11)
        ax.set_xlabel('Signed log(1 + value)' if transform else ('Recorded units' if c=='Age_firm' else 'Value'))
        ax.set_ylabel('Proportion within class');ax.legend(fontsize=7)
    fig.tight_layout();save_fig('03_distributions',fig)

    normalized=df.Sector.map(canonical_sector)
    sector=pd.DataFrame({'sector_original':df.Sector,'sector_normalized':normalized}).drop_duplicates()
    save_table('sector_normalization_mapping',sector)
    counts=normalized.value_counts()
    save_table('sector_counts',counts.reset_index(name='count').rename(columns={'index':'sector'}))
    fig,ax=plt.subplots(figsize=(9,4))
    counts.head(12).iloc[::-1].plot.barh(ax=ax,color='#466477');ax.set(xlabel='Companies',ylabel='',title='Sector text after case, accent and whitespace normalization')
    save_fig('04_sectors',fig)

    # Pairwise tests are exploratory. BH controls discovery across the tested numeric features.
    assoc=[]
    for c in df.select_dtypes('number').columns.drop(TARGET):
        a=labeled.loc[y==1,c].dropna();b=labeled.loc[y==0,c].dropna()
        if min(len(a),len(b))>=3 and pd.concat([a,b]).nunique()>1:
            u,p=stats.mannwhitneyu(a,b,alternative='two-sided')
            assoc.append({'feature':c,'role':feature_role(c)[0],'n_not_survived':len(a),'n_survived':len(b),
                          'median_not_survived':a.median(),'median_survived':b.median(),
                          'rank_biserial_not_survived_minus_survived':2*u/(len(a)*len(b))-1,'p_value':p})
    association=pd.DataFrame(assoc);association['q_bh_all_numeric']=bh(association.p_value)
    eligible=association.role.isin(['Candidate survey predictor','Description only','Redundant','Exclude until clarified'])
    association.loc[eligible,'q_bh_survey']=bh(association.loc[eligible,'p_value'])
    save_table('univariate_associations',association.sort_values('p_value'))
    categorical=[]
    for col in ['Technology_strategy','Founders_team1','Gender_entrepreneur']:
        tab=pd.crosstab(labeled[col],y)
        odds,p=stats.fisher_exact(tab) if tab.shape==(2,2) else (np.nan,np.nan)
        for cat,rows in labeled.groupby(col):
            k=int((rows[TARGET]==0).sum()); n=len(rows);low,high=wilson(k,n)
            categorical.append({'feature':col,'category':cat,'label':meta.variable_value_labels.get(col,{}).get(cat,str(cat)),
                                'n':n,'not_survived':k,'rate':k/n,'wilson_low':low,'wilson_high':high,'fisher_p':p})
    cat=pd.DataFrame(categorical)
    cat_q=dict(zip(cat.drop_duplicates('feature').feature,bh(cat.drop_duplicates('feature').fisher_p)))
    cat['fisher_q_bh_3_tests']=cat.feature.map(cat_q)
    save_table('categorical_outcome_rates',cat)
    fig,ax=plt.subplots(figsize=(8,4))
    names=[f'{SHORT[r.feature]}: {r.label} (n={r.n})' for r in cat.itertuples()]
    ax.errorbar(cat.rate,np.arange(len(cat)),xerr=np.array([cat.rate-cat.wilson_low,cat.wilson_high-cat.rate]),fmt='o',color='#466477',capsize=3)
    ax.set(yticks=np.arange(len(cat)),yticklabels=names,xlabel='Observed non-survival proportion',xlim=(0,.75),title='Subgroup rates with 95% Wilson intervals')
    ax.axvline(y.mean(),ls='--',color='#a3483a',label='Labeled-sample rate');ax.legend(fontsize=8)
    save_fig('05_subgroup_rates',fig)
    corr=labeled[full].corr(method='spearman',min_periods=30)
    pairs=[]
    for i,c in enumerate(corr):
        for j in range(i+1,len(corr)):
            c2=corr.columns[j];pairs.append({'feature_1':c,'feature_2':c2,'spearman_rho':corr.iloc[i,j],
                                            'pairwise_n':int(labeled[[c,c2]].dropna().shape[0])})
    pairdf=pd.DataFrame(pairs);pairdf['absolute_rho']=pairdf.spearman_rho.abs()
    save_table('correlation_pairs',pairdf.sort_values('absolute_rho',ascending=False))
    fig,ax=plt.subplots(figsize=(13,11))
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');ax.set(xticks=range(len(corr)),yticks=range(len(corr)),
        xticklabels=corr.columns,yticklabels=corr.columns,title='Survey-feature Spearman correlations (pairwise complete records)')
    ax.tick_params(axis='x',rotation=90,labelsize=7);ax.tick_params(axis='y',labelsize=7);fig.colorbar(im,ax=ax,shrink=.65)
    save_fig('06_correlations',fig)
    survey_assoc=association.loc[association.feature.isin(full)].assign(abs_effect=lambda x:abs(x.rank_biserial_not_survived_minus_survived)).nlargest(10,'abs_effect').iloc[::-1]
    fig,ax=plt.subplots(figsize=(9,5))
    ax.barh(survey_assoc.feature,survey_assoc.rank_biserial_not_survived_minus_survived,color='#466477')
    ax.axvline(0,color='black',lw=.8);ax.set(xlabel='Rank-biserial effect (positive = higher scores among non-survivors)',title='Largest exploratory survey associations; check multiple-testing results')
    save_fig('07_associations',fig)

    numeric=df.select_dtypes('number')
    summary={'source': 'https://zenodo.org/records/10607031', 'retrieved':'2026-10-23',
      'rows':len(df),'columns':len(df.columns),'labeled_rows':len(labeled),'survived':int((y==0).sum()),
      'not_survived':int(y.sum()),'unknown_outcome':int((~known).sum()),'non_survival_rate':float(y.mean()),
      'non_survival_wilson_95':wilson(int(y.sum()),len(y)),
      'unknown_label_extreme_bounds':[float(y.sum()/len(df)),float((y.sum()+(~known).sum())/len(df))],
      'class_ratio':float((y==0).sum()/y.sum()),'majority_accuracy':float((y==0).mean()),
      'duplicate_rows':int(df.duplicated().sum()),'duplicate_survey_rows':int(df[survey].duplicated().sum()),
      'constant_columns':df.columns[df.nunique(dropna=True)<=1].tolist(),
      'missing_cells':int(df.isna().sum().sum()),'missing_percent':float(df.isna().mean().mean()*100),
      'declared_missing_cells_replaced':int(cleaning.declared_missing_replaced.sum()),
      'original_sectors':int(df.Sector.nunique()),'normalized_sectors':int(normalized.nunique()),
      'singleton_normalized_sectors':int((counts==1).sum()),'sector_missing':int(df.Sector.isna().sum()),
      'sector_encoding_replacement_characters':int(df.Sector.str.contains(chr(65533),na=False).sum()),
      'employee_selection_conflicts':int((df.Employees>=250).sum()),'age_over_96_if_months':int((df.Age_firm>96).sum()),
      'zero_employees':int((df.Employees==0).sum()),'core_features':CORE,'full_survey_features':full,
      'future_features_excluded':future,'financial_sensitivity_features':finance,
      'survey_associations_q_below_05':association.loc[association.q_bh_survey<.05,'feature'].tolist(),
      'financial_2008_missing_labeled':int(labeled[finance[0]].isna().sum()),
      'financial_2008_missing_unknown':int(df.loc[~known,finance[0]].isna().sum()),
      'paired_financial_rows':int(df[[finance[0],'Financial_performance1_2014']].dropna().shape[0]),
      'versions':{p:importlib.metadata.version(p) for p in ['pandas','numpy','scipy','scikit-learn','pyreadstat','matplotlib']}}
    (OUT/'eda_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return df, labeled, y, summary

def run_models(df,labeled,y,summary,repeats=10):
    print('Running fixed, exploratory baselines; no hyperparameter selection.',flush=True)
    core_x=prep(labeled[CORE]);models=make_models(); metrics_rows=[]; predictions=[];folds=[]
    # Same folds for all models, all transforms fit exclusively on training rows.
    for repeat in range(repeats):
        cv=StratifiedKFold(5,shuffle=True,random_state=SEED+repeat)
        pp={name:np.zeros(len(y)) for name in models}; pp['Prevalence baseline']=np.zeros(len(y))
        for fold,(tr,te) in enumerate(cv.split(core_x,y),1):
            for pos in te: folds.append({'repeat':repeat,'fold':fold,'source_row':int(y.index[pos])})
            pp['Prevalence baseline'][te]=y.iloc[tr].mean()
            for name,model in models.items():
                fitted=clone(model).fit(core_x.iloc[tr],y.iloc[tr]); pp[name][te]=fitted.predict_proba(core_x.iloc[te])[:,1]
        pp['Equal-weight ensemble']=np.mean([pp[n] for n in models],axis=0)
        for name,p in pp.items():
            metrics_rows.append({'repeat':repeat,'model':name,**metrics(y,p)})
            for idx,actual,prob in zip(y.index,y,p):
                predictions.append({'repeat':repeat,'model':name,'source_row':int(idx),'non_survival_2014':int(actual),'probability':float(prob)})
        print(f'Finished repeat {repeat+1}/{repeats}',flush=True)
    metricdf=save_table('baseline_metrics_by_repeat',pd.DataFrame(metrics_rows))
    pred=save_table('out_of_fold_predictions',pd.DataFrame(predictions))
    save_table('fold_assignments',pd.DataFrame(folds))
    measures=[c for c in metricdf if c not in ['repeat','model']]
    statsdf=metricdf.groupby('model')[measures].agg(['mean','std','min','max'])
    statsdf.columns=['_'.join(c) for c in statsdf.columns];statsdf=statsdf.reset_index()
    save_table('baseline_summary',statsdf)
    confusion=[]
    for name,rows in pred.loc[pred['repeat']==0].groupby('model'):
        tn,fp,fn,tp=confusion_matrix(rows.non_survival_2014,rows.probability>=.5).ravel()
        confusion.append({'model':name,'true_survived':tn,'false_non_survival':fp,'missed_non_survival':fn,'correct_non_survival':tp})
    save_table('first_repeat_confusion',pd.DataFrame(confusion))
    # Sensitivity experiments are diagnostics, never candidates selected using these scores.
    sensitivities=[]
    experiments={
      'Core 9 survey features':core_x,
      'All 32 survey features':prep(labeled[summary['full_survey_features']]),
      'Core + 2008 finance (availability unverified)':prep(labeled[CORE+summary['financial_sensitivity_features']]),
      '2008 finance missingness only (audit)':labeled[summary['financial_sensitivity_features']].isna().astype(int),
      '2014 finance missingness only (INVALID leakage demo)':labeled[[c for c in summary['future_features_excluded'] if c.startswith('Financial')]].isna().astype(int),
      'Core missingness only (audit)':labeled[CORE].isna().astype(int),
    }
    for name,x in experiments.items():
        for repeat in range(5):
            p=np.zeros(len(y));cv=StratifiedKFold(5,shuffle=True,random_state=SEED+repeat)
            for tr,te in cv.split(x,y):
                fit=clone(models['Logistic regression']).fit(x.iloc[tr],y.iloc[tr]);p[te]=fit.predict_proba(x.iloc[te])[:,1]
            sensitivities.append({'experiment':name,'repeat':repeat,**metrics(y,p)})
    # Known employee selection conflicts: mask values, do not discard companies.
    x=core_x.copy();x.loc[labeled.Employees>=250,'Employees']=np.nan
    for repeat in range(5):
        p=np.zeros(len(y));cv=StratifiedKFold(5,shuffle=True,random_state=SEED+repeat)
        for tr,te in cv.split(x,y):
            fit=clone(models['Logistic regression']).fit(x.iloc[tr],y.iloc[tr]);p[te]=fit.predict_proba(x.iloc[te])[:,1]
        sensitivities.append({'experiment':'Core: mask employee counts >=250','repeat':repeat,**metrics(y,p)})
    sens=save_table('sensitivity_metrics_by_repeat',pd.DataFrame(sensitivities))
    ss=sens.groupby('experiment')[measures].mean().reset_index();save_table('sensitivity_summary',ss)
    # One fixed model and one fixed metric, with shuffled-label CV as a sanity check.
    score, null, pvalue=permutation_test_score(models['Logistic regression'],core_x,y,
        scoring='roc_auc',cv=StratifiedKFold(5,shuffle=True,random_state=SEED),n_permutations=199,random_state=SEED,n_jobs=1)
    save_table('permutation_null_auc',pd.DataFrame({'permutation':range(1,len(null)+1),'cv_roc_auc':null}))
    perm={'model':'Core logistic regression','metric':'mean fold ROC-AUC','observed':score,
          'permutations':199,'p_value':pvalue,'null_mean':float(null.mean()),
          'interpretation':'Exploratory test, conditional on this sample and fixed pipeline; not a population or Egyptian validation.'}
    (OUT/'permutation_test.json').write_text(json.dumps(perm,indent=2),encoding='utf-8')
    plot_model_metrics(statsdf,y)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name,rows in pred.loc[pred['repeat']==0].groupby('model'):
        if name=='Prevalence baseline':continue
        fpr,tpr,_=roc_curve(rows.non_survival_2014,rows.probability);axes[0].plot(fpr,tpr,label=name)
        pr,rc,_=precision_recall_curve(rows.non_survival_2014,rows.probability);axes[1].plot(rc,pr,label=name)
    axes[0].plot([0,1],[0,1],'--',color='grey');axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='ROC curves: first OOF partition')
    axes[1].axhline(y.mean(),ls='--',color='grey');axes[1].set(xlabel='Non-survival recall',ylabel='Non-survival precision',title='Precision-recall: first OOF partition');axes[1].legend(fontsize=8)
    fig.tight_layout();save_fig('09_roc_pr',fig)
    fig,ax=plt.subplots(figsize=(9,4.8))
    ax.barh(ss.experiment,ss.roc_auc,color=['#466477' if 'INVALID' not in x else '#a3483a' for x in ss.experiment]);ax.axvline(.5,ls='--',color='black')
    ax.set(xlim=(0,1),xlabel='Mean pooled OOF ROC-AUC over 5 partitions',title='Availability can look predictive even when it is not usable at decision time')
    save_fig('10_sensitivity_and_leakage',fig)
    fig,ax=plt.subplots(figsize=(7,3.8));ax.hist(null,bins=18,color='#b8c5cc');ax.axvline(score,color='#a3483a',label=f'Observed {score:.3f}; permutation p={pvalue:.3f}')
    ax.set(xlabel='Mean fold ROC-AUC with permuted labels',ylabel='Permutations',title='Does the fixed logistic baseline beat shuffled labels?');ax.legend(fontsize=9)
    save_fig('11_permutation_test',fig)
    # Coefficients across training folds show directional instability, not causation.
    coefficients=[]
    cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
    for fold,(tr,te) in enumerate(cv.split(core_x,y),1):
        fitted=clone(models['Logistic regression']).fit(core_x.iloc[tr],y.iloc[tr])
        for c,coef in zip(CORE,fitted[-1].coef_[0]):coefficients.append({'fold':fold,'feature':c,'standardized_coefficient':coef})
    save_table('logistic_coefficients_by_fold',pd.DataFrame(coefficients))
    print(statsdf[['model','roc_auc_mean','average_precision_mean','balanced_accuracy_mean','non_survival_recall_mean']].to_string(index=False),flush=True)
    print('Permutation:',perm,flush=True)
    return statsdf

def plot_model_metrics(statsdf,y):
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    names=statsdf.model.tolist();positions=np.arange(len(names))
    for ax,measure,baseline,title in zip(axes,['roc_auc','average_precision','balanced_accuracy'],[.5,y.mean(),.5],['ROC-AUC','Non-survival AP','Balanced accuracy at 0.5']):
        ax.errorbar(statsdf[measure+'_mean'],positions,xerr=statsdf[measure+'_std'],fmt='o',capsize=3,color='#466477')
        ax.axvline(baseline,ls='--',color='#a3483a');ax.set(yticks=positions,yticklabels=names if ax is axes[0] else [],xlim=(0,1),title=title)
    fig.supxlabel('Mean +/- SD across 10 partitions; these bars are not confidence intervals',fontsize=11)
    fig.tight_layout();save_fig('08_baseline_metrics',fig)

def main():
    warnings.filterwarnings('error',category=RuntimeWarning)
    df,labeled,y,summary=run_eda()
    run_models(df,labeled,y,summary)

if __name__=='__main__':
    main()
