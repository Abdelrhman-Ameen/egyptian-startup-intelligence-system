"""Check data lineage, cleaning, evaluation coverage and the delivered notebook."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import pyreadstat
import nbformat
import analyze_spain_ventures as eda

def main():
    df,raw,meta,cleaning=eda.load_data()
    original=eda.BASE/'raw/technology_entrepreneurs.sav'
    assert hashlib.md5(original.read_bytes()).hexdigest()=='b4b6cfee378eb1694e86cd98c18fc9ea'
    reference,_=pyreadstat.read_sav(original,user_missing=False)
    reference.index=df.index
    pd.testing.assert_frame_equal(df,reference)
    assert df.shape==(175,47)
    assert df.Survival_2014.value_counts().to_dict()=={1.0:110,0.0:40}
    assert df.Survival_2014.isna().sum()==25
    assert cleaning.declared_missing_replaced.sum()==242
    assert df.dropna().Survival_2014.eq(1).all() and len(df.dropna())==34
    assert df.duplicated().sum()==0
    summary=json.loads((eda.OUT/'eda_summary.json').read_text())
    assert set(eda.CORE).isdisjoint(summary['future_features_excluded']+['Survival_2014','source_row'])
    predictions=pd.read_csv(eda.TABLE/'out_of_fold_predictions.csv')
    folds=pd.read_csv(eda.TABLE/'fold_assignments.csv')
    measured=pd.read_csv(eda.TABLE/'baseline_metrics_by_repeat.csv')
    assert predictions.shape[0]==150*10*5
    assert not predictions.duplicated(['repeat','model','source_row']).any()
    assert predictions.probability.between(0,1).all()
    assert folds.shape[0]==150*10
    assert not folds.duplicated(['repeat','source_row']).any()
    known=set(df.index[df.Survival_2014.notna()])
    for (repeat,model),group in predictions.groupby(['repeat','model']):
        assert set(group.source_row)==known
        assert group.non_survival_2014.sum()==40
        independent=eda.metrics(group.non_survival_2014,group.probability)
        row=measured.loc[(measured['repeat']==repeat)&(measured.model==model)].iloc[0]
        for key,value in independent.items(): assert np.isclose(value,row[key],atol=1e-12), (repeat,model,key)
    for (repeat,fold),group in folds.groupby(['repeat','fold']):
        outcomes=df.loc[group.source_row,'Survival_2014']
        assert len(group)==30 and outcomes.eq(0).sum()==8
    notebook=eda.ROOT/'notebooks/01_spain_technology_ventures_eda.ipynb'
    nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not [o for c in code for o in c.outputs if o.output_type=='error']
    report=(eda.OUT/'spain_ventures_eda_report.html').read_text(encoding='utf-8')
    assert report.count('data:image/png;base64,')==12
    results={'checks_passed':True,'source_md5_verified':True,'cleaning_matches_spss_reader':True,
             'raw_rows':175,'labeled_rows':150,'unknown_rows':25,'cross_validated_predictions':len(predictions),
             'cv_fold_assignments':len(folds),'models_including_baseline_and_ensemble':5,
             'notebook_cells':len(nb.cells),'executed_code_cells':len(code),'notebook_errors':0,
             'embedded_report_charts':12,'metric_recalculation_matches':True}
    (eda.OUT/'validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
