"""Build a standalone HTML report and an executed, readable EDA notebook."""
from pathlib import Path
import base64
import html
import json
import sys
import hashlib
import textwrap
import inspect
import pandas as pd
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import analyze_spain_ventures as eda

ROOT=eda.ROOT
OUT=eda.OUT
TABLE=eda.TABLE
FIG=eda.FIG
NOTEBOOK=ROOT/'notebooks/01_spain_technology_ventures_eda.ipynb'

def read(name): return pd.read_csv(TABLE/f'{name}.csv')
def table(df): return '<div class="table">'+df.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}',na_rep='')+'</div>'
def figure(name,caption):
    data=base64.b64encode((FIG/f'{name}.png').read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
def details(title,body):return f'<details><summary>{title}</summary>{body}</details>'

def build_report():
    s=json.loads((OUT/'eda_summary.json').read_text())
    b=read('baseline_summary').set_index('model')
    perm=json.loads((OUT/'permutation_test.json').read_text())
    model_table=pd.DataFrame([{
       'Model':name,'ROC-AUC':b.loc[name,'roc_auc_mean'],'AP (non-survival)':b.loc[name,'average_precision_mean'],
       'Balanced accuracy':b.loc[name,'balanced_accuracy_mean'],'Precision':b.loc[name,'non_survival_precision_mean'],
       'Recall':b.loc[name,'non_survival_recall_mean'],'F1':b.loc[name,'non_survival_f1_mean'],
       'Brier (lower is better)':b.loc[name,'brier_mean']}
       for name in ['Prevalence baseline','Logistic regression','Random forest','Gradient boosting','Equal-weight ensemble']])
    useful_questions=pd.DataFrame([
      ['Which early firm and founder characteristics are associated with survival by 2014?', 'Yes, exploratory', '150 labeled Spanish ventures. Associations are uncertain and not causal.'],
      ['Can early survey responses distinguish survivors from non-survivors in this cohort?', 'Small ML experiment', 'Evaluate a historical binary classifier using pre-outcome fields; current baseline signal is weak.'],
      ['How much does the choice of model or ensemble help?', 'Yes, methodological', 'Same target, same folds and feature set. The equal-weight ensemble did not improve on logistic regression.'],
      ['What is an Egyptian startup\'s failure probability in the next 12 or 24 months?', 'No', 'Different country, historical cohort, no such horizon labels and no local validation.'],
      ['When will a startup close? What is its one-, two- or five-year survival curve?', 'No', 'No closure dates, event durations or individual last-observation dates.'],
      ['What happens to cash runway if expenses fall by 20%?', 'No direct support', 'No cash balance, monthly receipts/payments, burn or runway. Use founder inputs and cash-flow calculations.'],
      ['How would inflation or an exchange-rate shock affect this company?', 'No direct support', 'No exposure data or varied country-year panel from which to learn the relationship.'],
      ['Would adding a co-founder or spending more on R&D cause survival to improve?', 'No causal answer', 'Observational survey associations do not identify interventions; no measured R&D spending is provided.'],
      ['Can we study financial changes between 2008 and 2014?', 'Limited descriptive use', '71 paired operating-income observations; selected subset, unknown units, six-year interval. Not monthly or 12-month growth.'],
      ['Can the dataset or related papers support RAG?', 'Yes, with a narrow scope', 'Use the source description and research as historical Spanish evidence, with dates and limitations; not Egyptian market facts.'],
    ],columns=['Project question','Use','Reason'])
    useful_questions.to_csv(TABLE/'project_question_fit.csv',index=False,encoding='utf-8-sig')
    source='<a href="https://zenodo.org/records/10607031">Zenodo record and original download</a>'
    parts=[f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Spain Technology Ventures: EDA and project fit</title><style>
    body{{font:16px/1.6 Arial,sans-serif;color:#202020;background:white;margin:0}}main{{max-width:1120px;margin:40px auto;padding:0 28px}}
    h1{{font-size:30px;line-height:1.2;margin-bottom:8px}}h2{{font-size:23px;margin-top:42px;border-bottom:1px solid #bbb;padding-bottom:7px}}h3{{font-size:18px;margin-top:28px}}
    p,li{{max-width:1000px}}a{{color:#a12f2f}}.meta,figcaption{{color:#555;font-size:14px}}.lead{{font-size:18px;border-left:3px solid #555;padding:4px 18px}}
    table{{border-collapse:collapse;width:100%;font-size:14px;line-height:1.45}}th,td{{text-align:left;padding:10px 12px;border:1px solid #ccc;vertical-align:top}}th{{background:#eee}}
    .table{{overflow:auto;margin:18px 0}}figure{{margin:28px 0}}img{{max-width:100%;height:auto}}figcaption{{margin-top:8px}}
    code,pre{{font-family:Consolas,monospace;font-size:13px}}pre{{background:#f3f3f3;padding:15px;overflow:auto}}details{{margin:20px 0}}summary{{cursor:pointer;font-weight:bold}}
    @media print{{main{{margin:0}}h2{{break-after:avoid}}figure,tr{{break-inside:avoid}}details{{display:block}}}}
    </style></head><body><main><h1>Spain Technology Ventures</h1><p class="meta">EDA and project fit | Egyptian Startup Intelligence System | 23 October 2026</p>
    <p class="lead"><b>Keep this dataset for a small, reproducible ML experiment and data-quality training.</b> It does not support a dependable Egyptian startup risk score or our cash-flow simulation. The usable label is survival in 2014 among Spanish ventures that had already reached the 2009 survey.</p>
    <p>The original SAV file was downloaded again and checked against Zenodo's published MD5 checksum. This report uses the actual records and their SPSS metadata, not the earlier shortlist counts alone. Source: {source}.</p>
    <p><a href="#source">Source</a> · <a href="#target">Target</a> · <a href="#quality">Quality</a> · <a href="#insights">Insights</a> · <a href="#models">ML check</a> · <a href="#fit">Project questions</a> · <a href="#next">Team actions</a></p>
    <h2 id="source">1. What was collected</h2>
    <p>There are <b>175 rows and 47 original fields</b>, with one reported venture per row. The public file has no company name or stable company identifier. A generated <code>source_row</code> preserves row lineage only and is never a model input. Exact duplicate rows: <b>0</b>; exact duplicate survey profiles: <b>0</b>; constant fields: <b>0</b>. Without identifiers, this does not independently prove that all entities are distinct.</p>
    <table><tr><th>Date</th><th>Meaning</th></tr>
    <tr><td>2009</td><td>Founder-manager questionnaire, referring to resources at or near inception and performance through 2008.</td></tr>
    <tr><td>2008 and 2014</td><td>Years attached to the financial measurements; the binary survival outcome refers to 2014.</td></tr>
    <tr><td>2016</td><td>Researchers retrieved financial and survival information from SABI.</td></tr>
    <tr><td>2024</td><td>Dataset publication year. It is not the observation year.</td></tr></table>
    <p>The source describes independent Spanish technology ventures, fewer than 250 employees and up to eight years old at recruitment. The file contains observations that need reconciliation with those criteria. The <a href="https://accedacris.ulpgc.es/jspui/handle/10553/165201">authors' institutional repository</a> confirms the study design and the dataset's CC BY 4.0 license. Attribution must accompany reuse.</p>
    <p>Field groups: knowledge-source scores (7), partner access (2), training and experience (6), innovation (4), strategy and perceived performance (10), firm/founder descriptors (8), survival (1), internationalization (1), and financial measures for two years (8). Score values are mostly 1-7, with training/experience at 1-5. They are ordinal responses, not years of experience, actual customer counts, growth percentages or investment amounts. The exact response anchors are not included in the SAV labels and remain to be obtained.</p>
    <h2 id="target">2. Target and imbalance</h2>
    <p><code>Survival_2014 = 1</code> means survived; <code>0</code> means did not survive. For evaluation, <code>non_survival_2014 = 1 - Survival_2014</code>, so the minority event is the positive class. The dataset does not provide a documented closure date, reason or bankruptcy event. The 25 unknown outcomes are kept for descriptive auditing and excluded from supervised fitting; they are not assigned invented labels.</p>''',
    table(read('target_distribution')),figure('01_target','Class balance across all 175 rows. The labeled sample contains 110 survivors and 40 non-survivors.'),
    f'''<p>Among the 150 known outcomes, non-survival is <b>26.7%</b>, survival is <b>73.3%</b>, and the majority/minority ratio is <b>2.75:1</b>. This is manageable class imbalance; the more serious constraint is only 40 minority examples. Five-fold validation leaves <b>8 non-survivors per validation fold</b>. One missed event therefore changes fold recall by 12.5 percentage points.</p>
    <p>A model that always predicts survival gets 73.3% accuracy while detecting no non-survivors. Report average precision (AP), minority recall/precision/F1, balanced accuracy and ROC-AUC. AP's prevalence baseline is 0.267. The Wilson 95% interval for the observed minority proportion is 20.2%-34.3%; it does not establish a national failure rate. If all 25 missing outcomes were survivors or non-survivors, the full-file non-survival proportion would range from 22.9% to 37.1%. These are extreme-case bounds, not predictions.</p>
    <p>Class weights were used inside training. No SMOTE or artificial rows were added. If resampling is tried later, it belongs inside training folds only. Standard SMOTE also needs care with ordinal and categorical fields. Neither weighting nor oversampling adds new real companies.</p>
    <h2 id="quality">3. Cleaning, missingness and leakage</h2>
    <p><b>9999 is a missing-value code.</b> The SPSS file explicitly declares it, including the text version for Sector. The cleaning step replaced {s['declared_missing_cells_replaced']} actual coded cells with missing values using metadata, then preserved system-missing values. The result has {s['missing_cells']:,} missing cells ({s['missing_percent']:.2f}% of the 8,225 original cells). The raw SAV and raw-value export are retained unchanged.</p>''',
    figure('02_missingness','Missingness percentages use each outcome group as denominator: 110 survived, 40 not survived and 25 unknown.'),
    '''<p>Each 2008 financial field is missing in 64/175 rows (36.6%). Within known labels, those fields are missing in <b>20/40 non-survivors (50%)</b> and <b>19/110 survivors (17.3%)</b>. All 25 unknown-outcome rows lack all four 2008 financial fields. This pattern is inconsistent with a simple assumption that missing records are unrelated to outcomes. It does not identify the exact missing-data mechanism.</p>
    <p><b>Blind complete-case deletion destroys the target.</b> Keeping only rows complete across every column leaves 34 records, all survivors. Keeping the nine core predictors and a known label retains 142 complete records; fold-local imputation lets the exploratory model retain all 150 labeled records.</p>''',
    table(read('complete_case_retention')),
    '''<p><b>Exclude all five non-target 2014 fields.</b> They are future information for an earlier prediction. Their missingness indicators are also future information. The 2008 financial values need a separate decision: the year is earlier, but the records were extracted in 2016, and their availability strongly follows eventual outcomes. We therefore keep them out of the primary survey baseline and use them only for an explicitly limited sensitivity check. Imputation cannot establish that a field was available at the original decision time.</p>
    <h3>Other quality issues</h3>
    <ul><li><b>Firm age:</b> values range from 4 to 350; the unit is not specified. A months interpretation is plausible but unverified. Under that assumption, 42 rows exceed 96 months. Do not silently divide by 12 or discard those rows. Firm age is excluded from the baseline pending clarification.</li>
    <li><b>Employees:</b> median 4, range 0-1,100 among 165 observed values. Two values, 300 and 1,100, conflict with the stated fewer-than-250 selection rule unless timing or another explanation resolves it. There are 19 zero-employee companies; these are not automatically invalid. Outliers are flagged, retained in raw data and checked in a sensitivity experiment.</li>
    <li><b>Sector:</b> 97 original strings become 88 after case, accent and whitespace normalization. Of those 88 categories, 69 occur once. Ten values are missing. A reviewed taxonomy is needed before sector-level inference or categorical modeling. UTF-8 source accents are intact; normalization is not a claim of semantic equivalence.</li>
    <li><b>Financial values:</b> profits may legitimately be negative; extreme profitability values need review. Currency, scale and precise accounting definitions are not resolved by the labels. No currency conversion or revenue-growth label was invented.</li>
    <li><b>Redundant fields:</b> where both are observed, the founding-team flag agrees with whether founder count exceeds one. The baseline keeps count alone.</li>
    <li><b>Coverage:</b> one country and one historical recruitment cohort. Founders who did not reach the 2009 survey are not represented. The file has 146 male and 20 female respondents, with 9 missing; such small, unbalanced subgroups cannot justify demographic conclusions. Gender and entrepreneur age are descriptive only in this experiment.</li>
    <li><b>Internationalization:</b> source categories 1 and 3 both describe internationalization; 69.7% of values are missing. Resolve the ambiguous coding before using it as a separate target.</li></ul>''',
    figure('03_distributions','Common histogram bins and within-class proportions allow comparisons despite unequal class sizes. Financial axes retain source-unit meaning after a signed-log display transformation.'),
    figure('04_sectors','Normalization only removes formatting differences. It does not infer a sector taxonomy.'),
    details('Complete feature dictionary and decisions',table(read('feature_dictionary_and_decisions'))),
    details('Outliers and source inconsistencies',table(read('outlier_audit'))+table(read('quality_flags'))),
    details('Missingness, score distributions and paired-year coverage',table(read('missingness_all'))+table(read('paired_financial_coverage'))+table(read('survey_score_distributions'))),
    '''<h2 id="insights">4. What the exploratory comparisons show</h2>
    <p>Survivors have a median of <b>5 employees</b> versus <b>2</b> among non-survivors. Same-sector experience scores have medians <b>3 versus 2</b>. These are useful hypotheses to test with larger data; they do not show that hiring or experience causes survival. The employee comparison has an unadjusted p-value of 0.033 but a survey-family BH-adjusted q-value of 0.371. <b>No individual survey feature passes q &lt; 0.05</b> in the exploratory numeric screening.</p>
    <p>Pioneers have an observed non-survival proportion of 32.6% (14/43), versus 24.5% (26/106) for followers. Fisher's exact p-value is 0.317 and the intervals overlap widely. This is not sufficient evidence to recommend one technology strategy over the other. Subgroup comparisons are unadjusted for confounding.</p>
    <p>Foreign-sales and foreign-market-expansion scores have Spearman correlation 0.769; local-sales-growth and market-share scores have correlation 0.692. Some questionnaire items carry overlapping information. These are self-reported scores, not measured growth rates. Use regularization and avoid treating 32 correlated inputs as 32 independent pieces of evidence.</p>''',
    figure('05_subgroup_rates','95% Wilson intervals express sampling uncertainty within the observed cohort. Demographic differences are descriptive, not deployment rules.'),
    figure('07_associations','Effects and tests use observed values only, with sample sizes recorded in the table. A high position is not a validated predictive feature.'),
    details('All outcome associations and adjusted p-values',table(read('univariate_associations'))+table(read('categorical_outcome_rates'))),
    details('Full correlation map and strongest pairs',figure('06_correlations','32 candidate survey predictors; pairwise sample sizes are retained in the correlation table.')+table(read('correlation_pairs').head(20))),
    '''<h2 id="models">5. A small predictive-signal check</h2>
    <p>We tested three fixed model families plus a simple equal-weight probability ensemble and a prevalence baseline. All use the same nine business features, picked for their meaning rather than their univariate ranking: founder count, employees, technology strategy, access to investment and technology partners, sector and management experience scores, customer dialogue, and R&D emphasis.</p>
    <p><b>Protocol:</b> 5 stratified folds, repeated with 10 seeds (42-51). Each company is predicted once per repeat by a model that did not train on it. Median imputation and logistic-regression scaling fit only on the training fold. Employee counts receive a fixed log(1+x) transform. Survey scores are treated as ordered numeric inputs, an approximation that assumes useful numeric spacing. No feature selection, hyperparameter search, threshold tuning or synthetic data was used. The classification threshold stays at 0.5. The ensemble averages the three held-out predictions. This is internal exploratory cross-validation, not an independent final test or an Egyptian validation.</p>
    <p>Logistic regression uses C=0.3 and balanced class weights. Random forest uses 150 trees, maximum depth 3 and at least 8 rows per leaf. Histogram gradient boosting uses 80 iterations, at most 5 leaves, at least 12 rows per leaf and L2 regularization 5. These settings are reproducible trial choices, not tuned optima.</p>''',
    table(model_table),figure('08_baseline_metrics','Mean scores across 10 complete OOF partitions. Standard deviations show partition variability; repeated partitions share companies and are not independent samples.'),
    f'''<p>The core logistic baseline is the strongest of these trials, with mean ROC-AUC <b>{b.loc['Logistic regression','roc_auc_mean']:.3f}</b> and AP <b>{b.loc['Logistic regression','average_precision_mean']:.3f}</b>. Its minority recall is 57.3%, but precision is only 34.3% at the fixed threshold. The ensemble's ROC-AUC is <b>{b.loc['Equal-weight ensemble','roc_auc_mean']:.3f}</b>; averaging these models did not improve the logistic baseline. This does not rule out every other model, but it gives no basis for a reliable deployed risk score.</p>
    <p>For the first OOF partition, logistic regression detects 21/40 non-survivors, misses 19 and falsely flags 42 survivors. A fixed logistic-regression label-permutation check with 199 permutations gives <b>p={perm['p_value']:.3f}</b> for mean fold ROC-AUC. It does not clear a conventional 0.05 threshold. This is a limited signal check, not proof that no relationship exists.</p>''',
    figure('09_roc_pr','Curves from the first 5-fold OOF partition. They are not resubstitution/training curves.'),
    figure('11_permutation_test','The permutation statistic averages fold AUCs; it is different from AUC pooled across a repeat, so the displayed point estimates need not match exactly.'),
    '''<p><b>Scores are not calibrated risk probabilities.</b> Core-model Brier scores are worse than the constant prevalence baseline (0.196). Class weighting shifts the effective class distribution. Calibration needs its own validation and more observations; this report does not convert these scores into personal risk advice.</p>''',
    figure('12_calibration','Four equal-frequency bins from the first OOF partition. Small bin sizes make the observed proportions noisy; interval counts are in the calibration table.'),
    details('Confusion matrices, interval limits and calibration data',table(read('first_repeat_confusion'))+table(read('conditional_bootstrap_intervals'))+table(read('calibration_first_partition'))),
    '''<h3>Sensitivity checks and the misleading shortcut</h3>
    <p>Using all 32 survey features gives logistic regression an average ROC-AUC of 0.635 over five partitions, still modest and not a separately confirmed result. Removing the influence of the employee values that conflict with recruitment criteria barely changes the core result. Adding 2008 financial values does not improve this trial.</p>
    <p>More tellingly, using <b>only the missingness of the 2014 financial fields</b> gives an apparent ROC-AUC of <b>0.776</b>. This is an intentionally invalid leakage demonstration: it uses information collected at the outcome year. It must never be presented as successful forecasting. Even 2008 financial missingness alone gives about 0.617; this reinforces the need to audit the 2016 retrieval process and the availability of fields at decision time.</p>''',
    figure('10_sensitivity_and_leakage','Sensitivity trials all use logistic regression and five repeated partitions. The red result is invalid for forecasting and is shown only to demonstrate leakage.'),
    table(read('sensitivity_summary')[['experiment','roc_auc','average_precision','balanced_accuracy','non_survival_recall']]),
    '''<h2 id="fit">6. Which project questions can this source answer?</h2>''',table(useful_questions),
    '''<p>The defensible modeling question is: <b>“Among ventures represented in this Spanish survey, do early recorded business characteristics help distinguish survival from non-survival by 2014?”</b> The outcome is a fixed calendar-year status, about five years after recruitment, not a rolling five-year prediction from each company's birth and not a 24-month failure target.</p>
    <p>For the product, keep founder-entered cash, receipts, expenses and scenario assumptions in a separate cash-flow calculator. No field in this source gives a direct coefficient for “cut expenses 20%” or “increase R&D spending.” A model explanation describes associations learned from these records; it cannot estimate the effect of taking such an action.</p>
    <h2 id="next">7. What the team should do next</h2>
    <ol><li><b>Tarek:</b> review the metadata-based missing-code handling and source dictionary; document unresolved age, accounting units, selection inconsistencies and questionnaire anchors. Keep original values and row lineage. Do not merge this cohort with bankruptcy tables as if their outcomes meant the same thing.</li>
    <li><b>Youssef:</b> use the saved folds, positive-class definition and fixed baselines as the starting experiment. Compare future changes on the same protocol, report all attempted models and require an untouched evaluation set or nested validation if tuning is introduced. Treat the current result as an exploratory experiment.</li>
    <li><b>Karim:</b> review financial definitions and selection effects, then keep the cash-flow simulator based on founder inputs. Review the tentative employee/experience findings and the correlated survey fields without turning them into causal advice.</li>
    <li><b>Abdelrahman Ameen:</b> keep live Egyptian prediction conditional on suitable training and validation data. The architecture can still deliver sourced RAG answers and deterministic scenarios while this ML track remains an offline experiment.</li>
    <li><b>Abdelrahman Mohamed:</b> check the documented examples, terminology and outcome labels, and record questions the report does not answer. Software implementation remains shared across the team.</li></ol>
    <p><b>Decision:</b> retain this source as a benchmark and a documented EDA case. Do not make it the sole training foundation for the application's Egyptian risk engine. The next data search should prioritize compatible, larger cohorts with pre-outcome observations, dated outcomes, a stated event definition and documented collection procedures. More rows help only if those definitions match.</p>
    <h2>Reproducibility and sources</h2>
    <p>The notebook, download and analysis scripts, original SAV, cleaning audit, model inputs, per-fold assignments, per-repeat predictions, feature decisions and numeric tables are saved separately. The HTML report embeds its figures and can be shared as one file. The notebook can be rerun from the project root or its own folder.</p>
    <pre>python scripts/download_spain_ventures.py
python scripts/inspect_spain_source.py
python scripts/analyze_spain_ventures.py
python scripts/supplement_spain_analysis.py
python scripts/build_spain_eda_deliverables.py
python scripts/verify_spain_eda.py</pre>
    <p>Install the pinned analysis dependencies from <code>scripts/requirements-spain-eda.txt</code> into a virtual environment first. Downloading uses public source URLs and no account credentials.</p>
    <ul><li>Garcia-Cabrera, A. M., and Garcia-Soto, M. G. (2024). <a href="https://doi.org/10.5281/zenodo.10607031">Technology-based entrepreneurs database</a>. Zenodo. CC BY 4.0. Dataset attribution; analysis, formatting normalization and derived labels are our transformations.</li>
    <li><a href="https://accedacris.ulpgc.es/jspui/handle/10553/165201">Institutional metadata record</a>: sampling and collection timeline.</li>
    <li>Garcia-Cabrera, Garcia-Soto and Nieves (2021). <a href="https://doi.org/10.1007/s11365-020-00656-z">Knowledge, innovation and NTBF short- and long-term performance</a>. Related research, not a replication claim.</li>
    <li>Garcia-Cabrera, Garcia-Soto and Olivares-Mesa (2019). <a href="https://doi.org/10.1111/jsbm.12411">Entrepreneurs' resources, technology strategy, and new technology-based firms' performance</a>. Related research.</li>
    <li><a href="https://scikit-learn.org/stable/common_pitfalls.html">Scikit-learn: leakage and preprocessing</a>; <a href="https://scikit-learn.org/stable/modules/model_evaluation.html">classification metrics</a>.</li></ul>
    <p class="meta">The linked papers' downloadable full texts could not be retrieved during this run. Unresolved questionnaire anchors and measurement units were therefore left unresolved rather than guessed. Source checksum, dates and software versions are recorded with the results.</p>
    </main></body></html>''']
    destination=OUT/'spain_ventures_eda_report.html'
    destination.write_text('\n'.join(parts),encoding='utf-8')
    return destination

def build_notebook():
    cells=[]
    def md(s):cells.append(nbformat.v4.new_markdown_cell(inspect.cleandoc(s)))
    def code(s):cells.append(nbformat.v4.new_code_cell(inspect.cleandoc(s)))
    md('''# Spain Technology Ventures: EDA and project fit
    Egyptian Startup Intelligence System · 23 October 2026

    **Decision:** useful as a small historical ML experiment. The present data and results do not support a dependable Egyptian startup risk score or cash-runway simulation.

    This notebook inspects the original source, audits cleaning and missingness, examines outcome imbalance and associations, and reviews a fixed internal cross-validation experiment. It uses the scripts in `scripts/` so the detailed calculations have one maintained implementation.

    Source: [Technology-based entrepreneurs database, Zenodo](https://doi.org/10.5281/zenodo.10607031), Garcia-Cabrera and Garcia-Soto (2024), CC BY 4.0. Observations relate to 2008/2014, not 2024. See the [institutional source description](https://accedacris.ulpgc.es/jspui/handle/10553/165201).
    ''')
    code('''from pathlib import Path
    import sys, json, hashlib
    import numpy as np
    import pandas as pd
    from IPython.display import display, Image
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'scripts/analyze_spain_ventures.py').exists())
    sys.path.insert(0, str(ROOT / 'scripts'))
    import analyze_spain_ventures as eda
    BASE, OUT = eda.BASE, eda.OUT
    TABLE, FIG = eda.TABLE, eda.FIG
    def read_table(name): return pd.read_csv(TABLE / f'{name}.csv')
    def show_figure(name): display(Image(filename=str(FIG / f'{name}.png')))
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.max_colwidth', 90)
    df, raw, metadata, cleaning_audit = eda.load_data()
    summary = json.loads((OUT / 'eda_summary.json').read_text())
    manifest = json.loads((OUT / 'analysis_manifest.json').read_text())
    assert hashlib.sha256((BASE / 'raw/technology_entrepreneurs.sav').read_bytes()).hexdigest() == manifest['source_sha256']
    assert hashlib.sha256(Path(eda.__file__).read_bytes()).hexdigest() == manifest['analysis_script_sha256']
    print(f'{len(df)} companies, {df.shape[1]} source columns; source checksum verified.')''')
    md('''## 1. Source and unit of observation
    The sample consists of founder-manager respondents from Spanish technology ventures operating at recruitment in 2009. Survey questions refer to inception or performance through 2008. Registry financial and survival information was retrieved in 2016 and refers to 2008/2014. A row is a reported company, not a monthly observation. There is no company ID or country-year panel. `source_row` is lineage only.

    The recruitment description says fewer than 250 employees and up to eight years of age. We check the file against those claims, without inventing fixes.''')
    code('''display(df.head())
    display(pd.DataFrame({'dtype': df.dtypes.astype(str), 'nonmissing': df.notna().sum(), 'unique': df.nunique()}))
    print('Exact duplicate rows:', df.duplicated().sum())
    print('Constant columns:', df.columns[df.nunique() <= 1].tolist())''')
    md('''## 2. Missing-value codes and data dictionary
    SPSS declares `9999` as missing (including text `9999` in Sector). The loader applies the file's own missing ranges, preserves genuine zero values and stores the unmodified SAV separately. No statistical imputation is performed for EDA. Fold-local imputation is used only during modeling.''')
    code('''display(cleaning_audit.loc[cleaning_audit.declared_missing_replaced > 0])
    print('Coded missing cells replaced:', cleaning_audit.declared_missing_replaced.sum())
    print('Clean missing cells:', df.isna().sum().sum())
    display(read_table('feature_dictionary_and_decisions')[['feature','description','role','missing_all','in_core_baseline']])''')
    md('''## 3. Outcome and imbalance
    Original target: `Survival_2014`, where 1=survived and 0=did not survive. It does not identify bankruptcy, a failure cause or a dated shutdown.

    Use `non_survival_2014 = 1 - Survival_2014` as the evaluation target, keeping unknown outcomes unknown. A 2009-survey-to-2014 status is not a rolling two-year failure label or survival time from founding.''')
    code('''known = df.Survival_2014.notna()
    labeled = df.loc[known].copy()
    y = (1 - labeled.Survival_2014).astype(int)
    display(read_table('target_distribution'))
    print(f'Class ratio: {(y == 0).sum() / (y == 1).sum():.2f}:1')
    print(f'Always-survive accuracy: {(y == 0).mean():.1%}; non-survival recall: 0%')
    print(f'Average precision prevalence baseline: {y.mean():.3f}')
    print('95% Wilson interval for observed non-survival proportion:', eda.wilson(int(y.sum()), len(y)))
    print('Unknown-outcome extreme bounds:', summary['unknown_label_extreme_bounds'])
    show_figure('01_target')''')
    md('''## 4. Missingness and selection
    All 25 unknown-outcome rows lack the 2008 financial fields. Among labeled records, financial missingness is 50% for non-survivors and 17.3% for survivors. Such patterns can reflect selection or retrieval procedures; the mechanism cannot be established from this file alone.''')
    code('''display(read_table('missingness_all'))
    show_figure('02_missingness')
    display(read_table('financial_availability_by_outcome'))
    display(read_table('label_missingness_tests').head(10))''')
    md('''### Why not just use `dropna()`?
    Deleting every incomplete row leaves 34 records, all survivors. That would eliminate the minority class. For model training, exclude unknown targets and impute only selected feature values inside each training fold.''')
    code('''display(read_table('complete_case_retention'))
    display(df.dropna().Survival_2014.value_counts())''')
    md('''## 5. Ranges, outliers and source inconsistencies
    Age is stored as 4-350 in an undocumented unit. Do not call these years or convert to months without evidence. If months were intended, 42 records would exceed the recruitment description's 8-year limit.

    Employee values 300 and 1,100 need reconciliation with the <250 selection rule. Zero employees may be legitimate. Negative profits and extreme financial ratios are flagged, not silently deleted. Currency and accounting units remain unverified.''')
    code('''display(read_table('numeric_summary'))
    display(read_table('outlier_audit'))
    display(read_table('quality_flags').head(15))
    show_figure('03_distributions')''')
    md('''## 6. Sector text and score distributions
    Case, whitespace and accent normalization reduces 97 source labels to 88; 69 normalized categories have only one record. This is not an industry taxonomy. Sector is excluded from the first baseline. Survey values are ordinal scores, not financial amounts or annual growth.''')
    code('''display(read_table('sector_normalization_mapping').head(15))
    display(read_table('sector_counts').head(15))
    print('Original / normalized / singleton sectors:', summary['original_sectors'], summary['normalized_sectors'], summary['singleton_normalized_sectors'])
    show_figure('04_sectors')
    display(read_table('survey_score_distributions').head(21))''')
    md('''## 7. Outcome associations
    These are exploratory univariate associations. Mann-Whitney tests use observed values, and rank-biserial effects indicate direction and size. BH q-values adjust for screening multiple numeric features. No individual survey feature meets q<0.05.

    Survivors have median employee count 5 versus 2 and sector-experience score 3 versus 2. These are hypotheses for follow-up, not causal recommendations. Fisher tests and Wilson intervals describe categorical groups. Gender and entrepreneur age are omitted from the business baseline; small, imbalanced demographic samples cannot establish reliable subgroup effects.''')
    code('''associations = read_table('univariate_associations')
    display(associations.loc[associations.feature.isin(summary['full_survey_features'])].head(15))
    display(read_table('categorical_outcome_rates'))
    show_figure('05_subgroup_rates')
    show_figure('07_associations')''')
    md('''## 8. Redundancy and correlations
    Foreign-sales and foreign-market-expansion scores correlate at about 0.77; local-sales and market-share scores at about 0.69. Treating every item as independent increases overfitting risk. Correlations are descriptive, pairwise complete, and do not imply causal links. The founders-team flag duplicates a distinction already represented by founder count.''')
    code('''display(read_table('correlation_pairs').head(20))
    display(pd.crosstab(df.Founders_team1, df.Founders_team2))
    show_figure('06_correlations')''')
    md('''## 9. Feature availability and leakage
    Exclude all five non-target 2014 fields, including their missing indicators. Keep 2008 financial fields out of the primary trial because they were retrieved in 2016 and availability is strongly outcome-associated. Exclude unresolved firm age, high-cardinality sector, demographic descriptors and the redundant team flag.

    The primary nine features are selected by business meaning, not by ranking their outcome correlations. The 32-feature trial is a sensitivity analysis.''')
    code('''print('Primary features:', eda.CORE)
    print('Excluded future fields:', summary['future_features_excluded'])
    display(read_table('feature_dictionary_and_decisions')[['feature','role','reason']])''')
    md('''## 10. Fixed ML baselines and an equal-weight ensemble
    Logistic regression, random forest and histogram gradient boosting use identical features and folds. Five stratified folds repeated ten times leave 8 non-survivors in each validation fold. Median imputation and scaling are trained within each fold; class weights do not touch validation labels. The 0.5 threshold, feature set and model settings are fixed. No hyperparameter search or SMOTE is used.

    The saved results below come from a completed run. Set `RETRAIN=True` to regenerate them using the same protocol. The default reads verified saved results to avoid needlessly repeating 50 fits per model and 199 label permutations.''')
    code('''models = eda.make_models()
    for name, model in models.items():
        print(name, '\\n', model, '\\n')
    RETRAIN = False
    if RETRAIN:
        eda.run_models(df, labeled, y, summary, repeats=10)
    baseline = read_table('baseline_summary')
    display(baseline[['model','roc_auc_mean','roc_auc_std','average_precision_mean','balanced_accuracy_mean',
                      'accuracy_mean','non_survival_precision_mean','non_survival_recall_mean','non_survival_f1_mean','brier_mean']])
    show_figure('08_baseline_metrics')''')
    md('''The core logistic trial has mean ROC-AUC 0.602 and AP 0.377. Random forest and boosting are weaker here. The equal-weight ensemble reaches 0.565 ROC-AUC and does not improve on logistic regression. Its existence does not compensate for a small, weakly informative dataset.

    Standard deviations across repeated partitions describe split variability, not independent-sample confidence intervals. These results are internal exploratory validation; no untouched final test set or local Egyptian validation exists.''')
    code('''display(read_table('first_repeat_confusion'))
    show_figure('09_roc_pr')
    display(read_table('conditional_bootstrap_intervals'))''')
    md('''The bootstrap intervals condition on fixed first-partition OOF predictions and resample rows within classes. They ignore model-refitting uncertainty and fold dependence; they are supplementary uncertainty checks, not final deployment intervals.''')
    md('''## 11. Permutation test and calibration
    The fixed logistic pipeline's 199-permutation ROC-AUC test gives p=0.09, insufficient to clear a conventional 0.05 threshold. The permutation statistic averages fold AUCs, unlike the pooled-repeat AUC in the main table.

    Class-weighted predicted scores are not calibrated probabilities. Their Brier scores are worse than the constant prevalence baseline. Do not present these scores as an Egyptian startup's personal probability of failure.''')
    code('''display(json.loads((OUT/'permutation_test.json').read_text()))
    show_figure('11_permutation_test')
    display(read_table('calibration_first_partition'))
    show_figure('12_calibration')''')
    md('''## 12. Sensitivity and deliberate leakage demonstration
    The all-survey logistic trial reaches around 0.635 ROC-AUC over five partitions. Adding 2008 financial values does not help the chosen trial. Masking the conflicting employee sizes has little effect.

    **The 0.776 ROC-AUC from 2014 financial missingness is invalid for forecasting.** It is shown only to demonstrate how future information can appear useful. The 2008 missingness-only result also warrants an availability audit.''')
    code('''display(read_table('sensitivity_summary'))
    show_figure('10_sensitivity_and_leakage')''')
    md('''## 13. Questions our project can answer with this source
    The defensible question is whether early recorded characteristics distinguish survival by 2014 **within this historical Spanish cohort**. No rolling 12/24-month target, dated shutdown, cash balance, expenses, runway, FX exposure or economic panel is provided.

    We cannot interpret changing a questionnaire score or founder count as an estimated real-world intervention. Related papers and the source description can support narrowly attributed historical RAG explanations.''')
    code('''display(read_table('project_question_fit'))
    display(read_table('paired_financial_coverage'))''')
    md('''## 14. Recommended use and team handoff
    Keep the dataset for EDA, a reproducible small-data benchmark and illustrating leakage. Do not use it as the sole basis of a live Egyptian risk score. Keep RAG and founder-input cash scenarios moving while the team looks for compatible regional or international cohorts.

    - Tarek: missing codes, dictionary, lineage and unresolved units/selection rules.
    - Youssef: fixed folds, baselines, transparent comparisons and validation design.
    - Karim: feature definitions, finance availability, explainability limits and cash simulation.
    - Abdelrahman Ameen: architecture, RAG and conditional integration of validated ML.
    - Abdelrahman Mohamed: example checks, terminology and the manual feedback log.

    Software work is shared. Before a broader experiment, obtain the questionnaire anchors, clarify firm-age units and the two employee-size inconsistencies, and verify the survival event definition and the historical availability of financial fields. The linked full-text article downloads were unavailable in this run; these details have not been guessed.

    Raw and cleaned files are separate. Scripts are in `scripts/`; this notebook is in `notebooks/`; report tables and figures are in `reports/spain_technology_ventures/`.
    ''')
    code('''print('Software versions:', summary['versions'])
    print('Source SHA-256:', manifest['source_sha256'])
    print('Independent test set:', manifest['independent_test_set'])
    assert set(y.unique()) == {0, 1}
    assert len(y) == 150 and int(y.sum()) == 40
    assert not set(eda.CORE) & set(summary['future_features_excluded'])
    assert 'source_row' not in eda.CORE
    print('Basic lineage, target and feature checks passed.')''')
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python','version':sys.version.split()[0]}})
    NOTEBOOK.parent.mkdir(parents=True,exist_ok=True)
    # Temporary kernel is scoped to this build and invokes the active project environment.
    kernel_root=ROOT/'tmp/spain_eda_kernels'
    kernel=(kernel_root/'spain-eda');kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
       'display_name':'Spain EDA','language':'python','env':{'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}}),encoding='utf-8')
    manager=KernelManager(kernel_name='spain-eda',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
    client=NotebookClient(nb,km=manager,timeout=300,resources={'metadata':{'path':str(ROOT)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.validate(nb)
    nbformat.write(nb,NOTEBOOK)
    print(f'Executed notebook: {NOTEBOOK}, {len(cells)} cells')
    return NOTEBOOK

def main():
    report=build_report()
    print('Report:',report,flush=True)
    build_notebook()

if __name__=='__main__': main()
