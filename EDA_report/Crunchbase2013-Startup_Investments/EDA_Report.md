# Startup Investment Ecosystem

## Exploratory Data Analysis Report

**Analysis source:** Existing `Startup_Investments.ipynb` notebook and its saved outputs


> **Evidence note.** This report documents saved notebook results and visualizations. It does not rerun the EDA, change source data, or modify the notebook. Interpretations based only on field names are labeled as inferred. No cleaning, imputation, outlier treatment, or target selection is presented as completed work.

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dataset Overview](#2-dataset-overview)
3. [Understanding the Problem](#3-understanding-the-problem)
4. [Data Quality Assessment](#4-data-quality-assessment)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Visualization Analysis](#6-visualization-analysis)
7. [Outlier Analysis](#7-outlier-analysis)
8. [Feature Relationships](#8-feature-relationships)
9. [Machine Learning Readiness](#9-machine-learning-readiness)
10. [Key Findings and Insights](#10-key-findings-and-insights)
11. [Limitations](#11-limitations)
12. [Recommendations and Next Steps](#12-recommendations-and-next-steps)
13. [Final Conclusion](#13-final-conclusion)
14. [Appendix A: Field Guide](#appendix-a-field-guide)
15. [Appendix B: Numerical Summary](#appendix-b-numerical-summary)
16. [Appendix C: Complete Outlier Screening Results](#appendix-c-complete-outlier-screening-results)

## 1. Executive Summary

The notebook examines a startup and investment ecosystem represented by 11 related CSV tables. Together, the tables contain **1,500,237 rows and 154 columns**. The largest table, `objects.csv`, contains 462,651 records and 40 fields. The table names and fields indicate coverage of companies and other entities, people, funding rounds, investments, offices, degrees, funds, milestones, acquisitions, IPOs, and relationships. This subject-matter description is inferred from the schema; the dataset's publisher, collection period, and formal data dictionary are not documented.

The EDA found **9,260,295 missing cells** across the tables, with substantial variation by table and field. No exact duplicate rows were reported in any table. The notebook's inferred candidate primary-key checks found no missing or duplicate candidate keys. Date-like strings were sample-parseable at 100% for the non-missing sampled values in the checked fields, but were not converted in the source tables. Potential outliers were flagged by IQR and Z-score rules; they were **not removed, capped, or transformed**.

In the 196,553 records filtered as companies, the candidate `status` label is dominated by `operating` (183,441; 93.33%). The remaining observed categories are acquired (4.78%), closed (1.31%), and IPO (0.58%). This is a highly imbalanced outcome if treated as a classification target, and it is not a confirmed prediction objective.

The most consequential statistical pattern is a long right tail in financial and count variables. For example, funding-round `raised_amount_usd` has a median of $1.6 million and a maximum of $3.835 billion; its skewness is 50.22. This gap between typical and extreme values means means and standard deviations can be strongly influenced by a small number of very large observations.

**Overall ML readiness: exploratory, not training-ready.** The dataset is rich and has checked candidate keys, but it is a set of linked tables rather than a ready-made company-level modeling matrix. A specific prediction question, target, prediction date, correct relational joins, domain review of missing values/zeros/outliers, and leakage-safe preprocessing remain necessary.

## 2. Dataset Overview

### 2.1 Scope and dimensions

The notebook loaded 11 CSV files from the project's local `data/` folder. The table rows should not be added together and treated as unique companies: the tables have different record grains (for example, one record per office, funding round, or relationship), and entities can appear in multiple tables.

| Table                  |                Rows |       Columns |       Missing cells | Missing cells (%) | Exact duplicate rows |
| ---------------------- | ------------------: | ------------: | ------------------: | ----------------: | -------------------: |
| `objects.csv`        |             462,651 |            40 |           7,917,475 |             42.78 |                    0 |
| `relationships.csv`  |             402,878 |            11 |             511,067 |             11.53 |                    0 |
| `people.csv`         |             226,709 |             6 |             198,663 |             14.60 |                    0 |
| `offices.csv`        |             112,718 |            15 |             431,467 |             25.52 |                    0 |
| `degrees.csv`        |             109,610 |             8 |              91,144 |             10.39 |                    0 |
| `investments.csv`    |              80,902 |             6 |                   0 |              0.00 |                    0 |
| `funding_rounds.csv` |              52,928 |            23 |              78,511 |              6.45 |                    0 |
| `milestones.csv`     |              39,456 |             9 |              18,243 |              5.14 |                    0 |
| `acquisitions.csv`   |               9,562 |            12 |               9,658 |              8.42 |                    0 |
| `funds.csv`          |               1,564 |            11 |                 753 |              4.38 |                    0 |
| `ipos.csv`           |               1,259 |            13 |               3,314 |             20.25 |                    0 |
| **Total**        | **1,500,237** | **154** | **9,260,295** |                — |          **0** |

The missing-cell percentage is reported per table as in the notebook. A single overall percentage is not reported here because tables have different row grains, and combining all cells would not by itself describe the usable coverage of any modeling population.

### 2.2 Tables and their roles

| Table                  | Plain-language description inferred from table and field names                                                   |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `objects.csv`        | Registry of entities such as companies, people, and products, with names, locations, status, and summary counts. |
| `people.csv`         | Person records and basic identity/affiliation information.                                                       |
| `degrees.csv`        | Education or degree records linked to objects/people.                                                            |
| `offices.csv`        | Organization office descriptions, locations, and coordinates.                                                    |
| `funding_rounds.csv` | Funding event details, round types, amounts, valuations, and participant counts.                                 |
| `investments.csv`    | Links between funding rounds, funded entities, and investor entities.                                            |
| `funds.csv`          | Fund records, including reported name, funding date, and amount.                                                 |
| `milestones.csv`     | Dated milestone records associated with entities.                                                                |
| `acquisitions.csv`   | Acquisition-event parties, date, terms, and price fields.                                                        |
| `ipos.csv`           | IPO/public-listing event, valuation and raised amount, currency, and ticker fields.                              |
| `relationships.csv`  | Person-to-entity relationship records, including role/title and date fields.                                     |

The notebook does not provide an external data dictionary. These descriptions are best-effort interpretations of the field names and should be checked against the original dataset documentation before making domain-sensitive decisions.

### 2.3 Feature types and possible target

The notebook's numerical summary includes **22 numeric features** across the tables, while its categorical frequency summary covers **19 categorical fields**. Other columns are inferred as identifiers, date-like fields, free text/high-cardinality strings, or binary indicators. These counts reflect the notebook's role-classification and reporting logic; they are not a claim that every other field is a useful model feature.

`objects.csv` contains 40 columns. The notebook identified company records using `entity_type == "Company"` and reported **196,553 company rows**. `status` could be investigated as a candidate classification label, but the notebook explicitly did not select it as a target. A prediction task needs a defined outcome and the date at which a prediction is meant to be made.

### 2.4 Potential real-world questions

Subject to representative coverage, correct joins, and a clear time boundary, the data could help investigate:

- How funding-round amounts differ by round type, time, or company attributes.
- How recorded company lifecycle statuses vary across selected categories.
- How investment and relationship networks connect organizations and investors.
- How acquisitions and IPO records are distributed among entities.
- Whether company-level attributes available at a historical cutoff are associated with later outcomes.

These are possible analyses, not findings of causation or a confirmed project objective.

## 3. Understanding the Problem

The notebook is an exploratory analysis of a multi-table business dataset, not a completed prediction study. It can describe observed records and associations, but the available notebook does not establish who was eligible to be included, how records were sampled, when the dataset was collected, or what "success" means.

### Questions it can support

- What entity and event tables are available, and how much information is missing?
- What are typical and extreme reported funding, acquisition, fund, and IPO amounts?
- How are numeric measurements distributed, and which observations trigger conventional outlier screens?
- How do the recorded statuses vary across the ten most common company categories shown in the notebook?
- Which numeric variables have the strongest within-table linear correlations?

### Task types and inference boundaries

- **Classification:** Possible only after a target is approved. Company `status` is a candidate, not a confirmed label.
- **Regression:** Possible for an amount or valuation field only after defining the prediction population, time point, currency convention, and handling of zero/missing values.
- **Clustering or network analysis:** Plausible with carefully selected entity attributes or relationship/investment edges, but not performed in the notebook.
- **Descriptive analysis:** Supported by the saved tables and plots.

The notebook does not prove that funding causes a company to succeed, that a recorded status is a future outcome, or that the observed records represent all startups. Correlation and grouped percentages are descriptive associations.

## 4. Data Quality Assessment

### 4.1 Missingness

The notebook found missing values in **68 of 154 columns**. Missingness ranges from no missing cells in `investments.csv` to 42.78% of all cells in `objects.csv`. The most incomplete fields are:

| Table / field                         | Missing count | Missing (%) | Potential impact and suggested review                                                                                        |
| ------------------------------------- | ------------: | ----------: | ---------------------------------------------------------------------------------------------------------------------------- |
| `offices.csv / created_at`          |       112,718 |      100.00 | No usable timestamps in this field; exclude from features unless the source can restore it.                                  |
| `offices.csv / updated_at`          |       112,718 |      100.00 | Same; retaining an all-missing field adds no information.                                                                    |
| `objects.csv / closed_at`           |       459,842 |       99.39 | Likely absent for many non-closed records, but semantics must be confirmed; missingness may itself reflect lifecycle status. |
| `objects.csv / short_description`   |       455,034 |       98.35 | Text coverage is very sparse; most records cannot use this field without an explicit missing-text strategy.                  |
| `objects.csv / first_investment_at` |       445,695 |       96.34 | Likely structural for non-investors; do not treat absence as a date or ordinary random missingness.                          |
| `objects.csv / last_investment_at`  |       445,695 |       96.34 | Same; consider a separate indicator only when justified by task timing.                                                      |
| `objects.csv / first_funding_at`    |       431,144 |       93.19 | May be structurally absent for unfunded entities; verify before imputing.                                                    |
| `objects.csv / last_funding_at`     |       431,144 |       93.19 | Same; potential leakage if later than a prediction cutoff.                                                                   |
| `objects.csv / state_code`          |       407,891 |       88.16 | Location detail is incomplete; consider broader geography or an explicit unknown category if appropriate.                    |
| `people.csv / birthplace`           |       167,872 |       74.04 | Sparse personal data; likely exclude unless clearly relevant and ethically appropriate.                                      |
| `ipos.csv / source_description`     |         1,086 |       86.26 | Source notes are largely absent and are not ordinarily a predictive measure.                                                 |

The missingness chart also shows high missingness in `objects.csv / parent_id` (about 94%), `people.csv / birthplace`, IPO source fields, `objects.csv / city`, and `acquisitions.csv / term_code`. The exact count for `parent_id` is not displayed in the saved missingness table excerpt, so it is not stated here. The notebook did not impute or remove missing values. Structural absence (for example, no IPO for a company) should not automatically be replaced with a statistical average.

### 4.2 Duplicates and key checks

- **Exact duplicate rows:** 0 reported in each of the 11 tables.
- **Candidate key checks:** The notebook's inferred candidate keys had zero missing key values and zero duplicate key rows. The fields checked were `acquisition_id`, `id` in degrees/investments/milestones/objects/people, `funding_round_id`, `fund_id`, `ipo_id`, `office_id`, and `relationship_id`.
- These are reassuring checks, but the notebook notes that candidate keys were inferred from field names. They do not validate all cross-table foreign-key relationships or prove that a proposed join has the correct grain.

### 4.3 Types, categories, and low-variance fields

- Date-like fields are stored as strings in the source data. The notebook's sampled non-missing date values were 100% parseable in checked fields. This is a parseability check, not a type conversion; the source columns remain unchanged.
- In `objects.csv / region`, the notebook found **16 case/space-normalization collision groups** and zero rows with leading or trailing spaces. This suggests values that may become duplicates after case/spacing normalization; it does not establish that the labels are semantically identical. Review before standardizing.
- Constant or near-constant fields include `milestones.csv / milestone_code` (one value, `other`, across 39,456 rows), `funding_rounds.csv / pre_money_currency_code` (one observed value, with many missing records), and the all-missing office timestamps. `pre_money_valuation` and its USD counterpart have 27 unique values but a 99.94% most-common value. These fields can add little predictive information or encode missingness rather than useful variation.
- No domain dictionary was available to validate all category labels. Category validity and inconsistent codes therefore remain unresolved.
- IDs, URLs, long text, and data-entry/audit timestamps may be unsuitable as direct predictors. Their utility depends on the task and leakage review.

### 4.4 Detected versus fixed

The checks detected missingness, potential category-format collisions, low-variance fields, and numerical outlier flags. **The notebook did not fix these issues**: no records were deleted, no missing data were imputed, no category normalization was applied, and no outliers were removed or capped.

## 5. Exploratory Data Analysis

### 5.1 Descriptive statistics

The notebook excludes identifier fields from numerical summaries because numeric-looking IDs are keys rather than measurements. It reports count, mean, median, standard deviation, minimum, quartiles, maximum, skewness, and kurtosis for the 22 numeric features. Selected measures are below; the complete displayed numeric summary is in Appendix B.

| Table / feature                            |        Mean |      Median |                  Q1–Q3 |     Minimum–maximum | Skewness | Interpretation                                                                                                          |
| ------------------------------------------ | ----------: | ----------: | ----------------------: | -------------------: | -------: | ----------------------------------------------------------------------------------------------------------------------- |
| `funding_rounds.csv / raised_amount_usd` |   7,946,092 |   1,600,000 |      246,330–6,700,000 |     0–3,835,050,000 |    50.22 | Mean is much higher than median, consistent with a strong upper tail.                                                   |
| `funding_rounds.csv / participants`      |        1.53 |           1 |                    0–2 |                0–36 |     2.95 | Most rounds have few participants; high counts are unusual under the quartiles.                                         |
| `acquisitions.csv / price_amount`        | 388,619,100 |           0 |              0–500,000 | 0–2,600,000,000,000 |    97.61 | Extreme values dominate the mean; zero values need domain interpretation.                                               |
| `funds.csv / raised_amount`              | 399,481,300 | 100,000,000 | 16,300,000–275,000,000 |    0–89,000,000,000 |    30.16 | Fund amounts are highly right-skewed with a wide range.                                                                 |
| `objects.csv / funding_total_usd`        |     892,672 |           0 |                    0–0 |     0–5,700,000,000 |   147.41 | At least 75% of values are zero under the reported quartiles; verify whether zeros mean no funding or unavailable data. |
| `objects.csv / investment_rounds`        |       0.174 |           0 |                    0–0 |               0–529 |    73.99 | Many zero counts and a small tail of very large values.                                                                 |
| `objects.csv / relationships`            |        1.79 |           1 |                    0–2 |             0–1,189 |    84.71 | Most entities have few relationships in this field, with extreme maximums.                                              |
| `offices.csv / latitude`                 |       10.90 |           0 |                0–30.84 |       −43.77–69.65 |     1.05 | The zero median and quartile warrant checking for placeholder coordinates.                                              |
| `offices.csv / longitude`                |     −16.33 |           0 |                    0–0 |     −159.48–176.92 |   −0.88 | The zero quartiles warrant checking for placeholder coordinates; not proof that zero is invalid.                        |

For each summary, **Q1** is the 25th percentile and **Q3** the 75th percentile; half of the recorded values lie between them. The standard deviation measures spread around the mean and can be inflated by extreme values. Skewness near zero indicates approximate symmetry; large positive skewness indicates a long right tail. Several financial fields have very high kurtosis, meaning unusually heavy tails relative to a normal distribution. Many reported zeros may be true zeros, placeholders, or missingness conventions; the notebook does not resolve which.

### 5.2 Categorical frequencies and company status

Company status was counted for **196,553 company records**:

| Recorded status |             Count |        Percentage |
| --------------- | ----------------: | ----------------: |
| Operating       |           183,441 |            93.33% |
| Acquired        |             9,394 |             4.78% |
| Closed          |             2,584 |             1.31% |
| IPO             |             1,134 |             0.58% |
| **Total** | **196,553** | **100.00%** |

The breakdown is highly imbalanced: operating records outnumber each other class. If status were approved as a classification target, a model could achieve a misleadingly high accuracy simply by favoring the majority class. Class-aware metrics and stratified evaluation would be important. Status is not automatically a measure of success; `operating` could reflect the dataset's snapshot and recording rules.

For the top observed company categories, the notebook reports status percentages within each category. Examples: biotech records show 10.27% acquired, 2.64% closed, and 5.58% IPO; consulting shows 2.72% acquired, 0.62% closed, and 0.42% IPO; web shows 6.42% acquired, 4.04% closed, and 0.31% IPO. These are descriptive percentages for the selected categories, not adjusted risk estimates. Category coverage and missing category codes may affect comparisons.

## 6. Visualization Analysis

The figures below are existing notebook charts embedded from saved outputs; they were not recreated for this report.

### Missingness

![Columns with the highest missing-value percentages](eda_report_assets/missingness.png)

**Figure 1. “Columns with the highest missing-value percentages.”** This bar chart ranks the 20 most incomplete fields. It is useful for prioritizing review because the highest bars are two fully empty office audit timestamps, followed by extremely incomplete company lifecycle and descriptive fields. The chart displays percentages, not the reason for missingness; it cannot distinguish “not applicable” from “not collected.”

### Candidate label distribution

![Company status distribution (candidate label only)](eda_report_assets/company_status.png)

**Figure 2. “Company status distribution (candidate label only).”** The bar heights make the majority-class imbalance visible: operating records dominate the three less frequent outcomes. The plot is a count chart, so it does not adjust for time, sector, or company attributes and does not validate status as a target.

### Numerical associations

![Numerical feature correlations: objects.csv](eda_report_assets/object_correlations.png)

**Figure 3. “Numerical feature correlations: objects.csv.”** This heatmap shows pairwise Pearson correlations among numeric fields in `objects.csv`. The strongest visible association is `investment_rounds` with `invested_companies` (r = 0.98); `funding_rounds` with `funding_total_usd` is weaker (r = 0.30). The heatmap summarizes linear association, not causation, and can be affected by zero-heavy distributions and snapshot aggregation.

![funding_rounds.csv post-money valuation scatterplot](eda_report_assets/post_money_scatter.png)

**Figure 4. “funding_rounds.csv: post_money_valuation_usd vs post_money_valuation (r=1.00).”** The scatterplot compares the reported valuation and its USD counterpart. The near-perfect line indicates the two columns are almost duplicates in this table; verify currency and conversion semantics before retaining both as features. Extreme values compress the central cloud, so the plot does not show fine-grained variation among smaller rounds.

### Outlier visual

![Potential outliers in numerical objects fields](eda_report_assets/object_outliers.png)

**Figure 5. “objects.csv: potential outliers.”** These boxplots show candidate IQR outliers for selected object-table counts and logo dimensions. Several count features have many zero observations, so the IQR rule's upper bound can be zero and label every positive count as an outlier. The plots flag unusual values, not errors; the logo dimensions may also include image-format or metadata artifacts.

The notebook additionally generated per-table numerical histograms/boxplots, low-cardinality categorical frequency plots, correlation heatmaps, and scatterplots for strongest numeric pairs. Large-table scatterplots and distribution plots use a seeded sample of up to 5,000 or 10,000 values, respectively, for legibility. No causal conclusions follow from any plot.

## 7. Outlier Analysis

The notebook applied two screening rules to the 22 numeric features:

- **IQR rule:** values below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR are flagged.
- **Z-score rule:** values more than 3 population standard deviations from the mean are flagged.

The bounds and counts below are calculated from non-missing observations. The IQR and Z-score counts need not match because they use different definitions. A large IQR count can occur for discrete or zero-inflated fields: for example, when Q1 = Q3 = 0, every positive value exceeds the IQR upper bound of zero. This does not mean every flagged observation is a data error.

| Table / feature                                   |                   IQR bounds | Observations |    IQR flags (%) | Z > 3 flags (%) |           Maximum |
| ------------------------------------------------- | ---------------------------: | -----------: | ---------------: | --------------: | ----------------: |
| `objects.csv / milestones`                      |                       0 to 0 |      462,651 | 100,358 (21.69%) |  16,769 (3.62%) |                 9 |
| `objects.csv / logo_height`                     |             −208.5 to 347.5 |      462,651 |   42,484 (9.18%) |   7,636 (1.65%) |             9,600 |
| `objects.csv / logo_width`                      |                 −324 to 540 |      462,651 |   38,436 (8.31%) |   9,175 (1.98%) |            18,200 |
| `relationships.csv / sequence`                  |                     −2 to 6 |      402,878 |   37,654 (9.35%) |   4,709 (1.17%) |           399,946 |
| `offices.csv / longitude`                       |                       0 to 0 |      112,718 |  32,368 (28.72%) |     910 (0.81%) |            176.92 |
| `objects.csv / funding_rounds`                  |                       0 to 0 |      462,651 |   31,707 (6.85%) |  11,173 (2.42%) |                15 |
| `objects.csv / relationships`                   |                     −3 to 5 |      462,651 |   30,199 (6.53%) |   2,568 (0.56%) |             1,189 |
| `objects.csv / funding_total_usd`               |                       0 to 0 |      462,651 |   27,874 (6.02%) |   1,629 (0.35%) |     5,700,000,000 |
| `objects.csv / investment_rounds`               |                       0 to 0 |      462,651 |   16,986 (3.67%) |   1,208 (0.26%) |               529 |
| `objects.csv / invested_companies`              |                       0 to 0 |      462,651 |   16,986 (3.67%) |   1,490 (0.32%) |               459 |
| `funding_rounds.csv / raised_amount_usd`        |    −9,434,175 to 16,380,500 |       52,928 |   5,400 (10.20%) |     287 (0.54%) |     3,835,050,000 |
| `funding_rounds.csv / raised_amount`            |    −9,275,000 to 16,125,000 |       52,928 |   5,395 (10.19%) |     280 (0.53%) |     3,835,050,000 |
| `funding_rounds.csv / participants`             |                     −3 to 5 |       52,928 |    2,487 (4.70%) |     981 (1.85%) |                36 |
| `acquisitions.csv / price_amount`               |       −750,000 to 1,250,000 |        9,562 |   2,345 (24.52%) |       1 (0.01%) | 2,600,000,000,000 |
| `funding_rounds.csv / post_money_valuation`     |                       0 to 0 |       52,928 |    1,279 (2.42%) |      34 (0.06%) |    24,324,230,000 |
| `funding_rounds.csv / post_money_valuation_usd` |                       0 to 0 |       52,928 |    1,194 (2.26%) |      31 (0.06%) |    24,324,230,000 |
| `funds.csv / raised_amount`                     | −371,750,000 to 663,050,000 |        1,564 |      152 (9.72%) |      10 (0.64%) |    89,000,000,000 |
| `ipos.csv / raised_amount`                      |                       0 to 0 |        1,259 |     138 (10.96%) |       2 (0.16%) |    24,400,000,000 |
| `ipos.csv / valuation_amount`                   |                       0 to 0 |        1,259 |      108 (8.58%) |       4 (0.32%) |   108,960,000,000 |
| `funding_rounds.csv / pre_money_valuation_usd`  |                       0 to 0 |       52,928 |       31 (0.06%) |       6 (0.01%) |    15,000,000,000 |
| `funding_rounds.csv / pre_money_valuation`      |                       0 to 0 |       52,928 |       31 (0.06%) |       6 (0.01%) |    15,000,000,000 |
| `offices.csv / latitude`                        |         −46.2566 to 77.0943 |      112,718 |        0 (0.00%) |      3 (0.003%) |           69.6502 |

Notable range and distribution patterns:

- `acquisitions.csv / price_amount` has a maximum of 2.6 trillion and 24.52% IQR flags, but only one Z-score flag. This illustrates how a highly skewed distribution can cause the two rules to disagree. Validate units, currency, and extreme records against the source before interpreting the maximum.
- `funding_rounds.csv / raised_amount_usd` has 5,400 IQR flags but 287 Z-score flags. Extreme but valid funding rounds may be economically important; removing them could erase precisely the events the analysis aims to study.
- `objects.csv / milestones`, `funding_total_usd`, and several count measures have an IQR upper bound of zero because the middle half of values are zero. Positive counts are not automatically erroneous.
- `offices.csv / longitude` has 32,368 IQR flags due to Q1 = Q3 = 0; review coordinate completeness and possible zero placeholders, without assuming all flagged longitudes are invalid.
- Logo dimensions may reflect image assets rather than company characteristics and may be inappropriate predictors.

The notebook sampled up to five IQR-flagged row examples per feature for inspection. It did not verify them against an external source or apply any treatment. Recommended next step: confirm units and data semantics, inspect the most extreme valid records, then choose feature-specific handling (retain, log-transform, robust-scale, winsorize, or exclude) only if the project objective justifies it.

## 8. Feature Relationships

The notebook calculated correlations among non-identifier numeric fields within each table; tables were not automatically joined. The strongest reported pairs are:

| Table                  | Feature pair                                              | Pearson r | Interpretation                                                                                                                           |
| ---------------------- | --------------------------------------------------------- | --------: | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `funding_rounds.csv` | `pre_money_valuation_usd` and `pre_money_valuation`   |    1.0000 | Near-duplicate representations; inspect currency semantics and avoid redundant features.                                                 |
| `funding_rounds.csv` | `post_money_valuation_usd` and `post_money_valuation` |    0.9995 | Nearly redundant; same caution.                                                                                                          |
| `objects.csv`        | `investment_rounds` and `invested_companies`          |    0.9816 | Very strong association between two investment-related counts; field definitions may overlap.                                            |
| `funding_rounds.csv` | `raised_amount_usd` and `raised_amount`               |    0.9381 | Strong association; likely related amount representations, but not necessarily identical.                                                |
| `objects.csv`        | `logo_width` and `logo_height`                        |    0.8290 | Larger image widths tend to accompany larger heights; likely an asset-size relationship, not a business outcome.                         |
| `offices.csv`        | `latitude` and `longitude`                            |  −0.6041 | Geographic coordinates show a moderate negative linear association in the observed sample; this is not a geographic causal relationship. |
| `ipos.csv`           | `valuation_amount` and `raised_amount`                |    0.3402 | Moderate positive association in the recorded IPO fields.                                                                                |
| `objects.csv`        | `funding_rounds` and `milestones`                     |    0.3304 | Moderate association between event/summary counts.                                                                                       |
| `objects.csv`        | `funding_rounds` and `funding_total_usd`              |    0.3049 | Modest positive association between funding round count and total funding.                                                               |

Correlation does not establish cause and effect. Near-duplicate amount fields could create multicollinearity or redundant information in a model. Some fields are summary counts of events recorded in the other tables, so a cross-table model must define a single entity-level row and a time cutoff. The notebook generated scatterplots for the strongest pairs, with samples used for legibility when tables were large.

The notebook also compared company status percentages across the top ten `category_code` groups. Biotech has the highest displayed IPO share (5.58%) among those categories; web has 4.04% closed; advertising has 5.12% acquired. These are raw within-category percentages and can reflect differences in category size, observation date, missing category coverage, or other confounders.

## 9. Machine Learning Readiness

**No target was selected, no feature matrix was built, and no model was trained or evaluated.** The notebook explicitly describes status as a candidate label only.

| Readiness area         | Current evidence                                                                 | Work needed before training                                                                                                                         |
| ---------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Target and objective   | Company `status` counts are available, but no objective or target is approved. | Define the business question, outcome classes or numeric target, and prediction horizon.                                                            |
| Unit of analysis       | 11 related tables have different grains.                                         | Define one row per prediction unit; join or aggregate with validated keys and prevent duplicate entity leakage.                                     |
| Missing values         | High and likely structural missingness exists in some fields.                    | Investigate why values are missing; impute within training folds only; retain missingness indicators only when justified.                           |
| Dates                  | Date-like strings were sample-parseable but remain strings.                      | Convert and derive only date features known at the prediction cutoff.                                                                               |
| Categories             | Currency, status, geography, round type, and other categories are present.       | Standardize labels with domain rules, then encode within a train-only pipeline; handle rare/unknown classes.                                        |
| Scaling                | Financial values have widely varying magnitudes and skew.                        | Consider log/robust transformations for scale-sensitive models after resolving zeros and units; tree models may need less scaling.                  |
| Class balance          | Candidate company status is 93.33% operating.                                    | If approved as target, use stratified splits, class-aware metrics, and possibly class weighting/resampling within training data.                    |
| Leakage                | Status and later event/date/aggregate fields can reveal outcomes.                | Exclude the target itself and features observed after the prediction time; time-bound funding, milestones, IPO/acquisition, and snapshot summaries. |
| Correlation/redundancy | Some numeric pairs have r ≈ 1.                                                  | Review redundant currency representations and highly correlated features for the intended model.                                                    |
| Outliers               | Many IQR flags occur in zero-heavy and heavy-tailed fields.                      | Verify extreme records and use domain-aware handling rather than automatic deletion.                                                                |
| Validation             | The notebook contains no model validation.                                       | Use temporal and/or company-group splits if the goal is future generalization; avoid the same entity appearing across train and test after joins.   |

If the intended task is company-status prediction, recorded status may be retrospective rather than a future label. A valid study would need a historical snapshot: features must be known before a fixed prediction date, while the target must describe an outcome after that date. Without this design, reported model performance could be inflated by leakage.

## 10. Key Findings and Insights

### Data quality

- The dataset has 11 tables, 1,500,237 total rows, and 154 columns; missingness is uneven and reaches 42.78% of cells in `objects.csv`.
- The notebook found 68 columns with at least some missingness. Two office audit timestamp columns are entirely empty.
- No exact duplicate rows were found, and all 11 inferred candidate keys passed their missing/duplicate checks.
- Date-like fields were strings, although checked non-missing samples were fully parseable. Source fields were not converted.
- `objects.csv / region` has 16 normalization-collision groups. Category values generally lack an external domain dictionary for validation.

### Statistical and distribution findings

- Amount and count fields are frequently right-skewed. `raised_amount_usd` in funding rounds has median $1.6 million, mean about $7.95 million, maximum $3.835 billion, and skewness 50.22.
- Company `funding_total_usd` has median and Q3 equal to zero, while its maximum is $5.7 billion. This makes the interpretation of zero values especially important.
- Company status among company records is heavily concentrated in operating (93.33%).

### Outliers and relationships

- The IQR rule flags 100% of `objects.csv / milestones` because its upper bound is zero; this is a consequence of a zero-heavy distribution, not proof of invalid records.
- `acquisitions.csv / price_amount` reaches 2.6 trillion and has a 24.52% IQR flag rate, but only 0.01% Z-score flags.
- Near-perfect correlations exist between some likely parallel amount/valuation fields; the strongest association among object counts is `investment_rounds` vs. `invested_companies` (r = 0.9816).
- None of these associations establishes causation or guarantees that an extreme record is erroneous.

### Potential ML opportunity

- A company-level status classification study is possible only as a proposed direction. The label is imbalanced and the prediction date is undefined, so leakage control and a target definition must come first.
- The relational tables could support funding, investor-network, or event-history analyses after their join keys and record grains are validated.

## 11. Limitations

- **Source and context:** The notebook does not identify the dataset publisher, collection date, observation window, sampling method, or formal variable definitions.
- **Representativeness:** Historical business records may overrepresent entities with public funding, prominent outcomes, or better reporting. This is a potential bias, not quantified by the EDA.
- **Missingness:** The analysis reports counts, not a full missingness mechanism study. It does not establish whether absence is random, structurally inapplicable, or a source omission.
- **Outlier rules:** IQR and Z-score thresholds are generic screening rules. They can over-flag discrete zero-heavy data, understate unusual values in skewed fields, and cannot validate source accuracy.
- **Correlation:** Pearson correlation detects linear pairwise association and is sensitive to extreme values and coding conventions. It cannot establish causation and may conceal nonlinear patterns.
- **Sampling in plots:** Some numerical charts use a deterministic sample of up to 10,000 observations and scatterplots up to 5,000. Those charts aid readability but are not a replacement for the full numerical summaries.
- **Cross-table interpretation:** No automatic joins were done, so relationships between event tables and entities are not established by the correlation analysis.
- **Modeling:** No target, preprocessing pipeline, model, or validation metrics exist in the notebook. No predictive performance claim can be made.
- **Unanswerable from this dataset alone:** Why a company succeeded or failed, the causal impact of funding, whether an unrecorded event did not occur, or how a future startup will perform.

## 12. Recommendations and Next Steps

### Priority 1 — Essential data-quality and project-definition work

1. Obtain the source citation, collection/snapshot date, field definitions, currency conventions, and any official data dictionary.
2. Confirm the project question, prediction unit, target definition, and prediction time before selecting features.
3. Validate candidate primary and foreign keys and document the grain of every join. Test that joining event tables does not multiply company-level records unintentionally.
4. Investigate missingness in high-value fields, especially company funding/lifecycle fields and office coordinates. Separate “not applicable” from “not recorded.”
5. Confirm whether zero amounts, counts, and coordinates are true zeros or source-specific placeholders.
6. Review the 2.6-trillion acquisition-price maximum, high funding-round values, extreme relationship sequences, and unusual logo dimensions against source records.
7. Resolve the 16 `region` normalization-collision groups using domain-approved rules; do not normalize all categories blindly.

### Priority 2 — Optional transformations

- Convert validated date strings into datetimes and derive elapsed-time or calendar features only as appropriate.
- Consider log or robust transformations for heavily skewed positive financial variables after determining how zero and negative values should be represented.
- Evaluate removal of constant/all-empty fields and redundant currency-converted features after verifying definitions.
- Retain valid extreme financial events when they are part of the business question; avoid indiscriminate trimming.

### Priority 3 — Feature engineering opportunities

- Derive company age or time-from-founding features only using dates available at a defined prediction cutoff.
- Aggregate funding and relationship histories to a company-level snapshot, using only events before the prediction date.
- Create careful indicators for event presence or structurally missing attributes where the domain supports them.
- Consider geography, category, and network summaries only after checking coverage, privacy, and join consistency.

### Priority 4 — ML preprocessing and validation

- Implement imputers, encoders, scalers, and transformations in a pipeline fitted only on training data.
- Use a chronological split and/or group split by company where future or unseen-entity generalization is the aim.
- If classifying status, report per-class precision/recall, macro-F1, confusion matrix, and balanced accuracy alongside accuracy.
- Check for leakage from status, IPO/acquisition/closure fields, future milestones, later funding events, and precomputed snapshot totals.

### Priority 5 — Model training and evaluation

After the target and leakage-safe design are approved, establish a simple baseline, compare appropriate models, tune using training/validation data only, and evaluate once on a held-out test set. Report the split strategy, metrics, uncertainty, and limitations. These steps are recommendations; no model training has occurred.

## 13. Final Conclusion

The existing analysis establishes that this is a substantial, connected startup-investment dataset with useful company, financing, people, geography, and event information. The notebook successfully profiled table sizes, missingness, inferred key quality, descriptive statistics, categorical frequencies, numeric correlations, candidate status distribution, and potential outliers without altering the source rows.

Its strengths are the breadth of related records, the lack of detected exact duplicate rows, and the zero-duplicate results for the inferred candidate keys. Its weaknesses for immediate modeling are substantial and uneven missingness, unresolved zero semantics and extreme values, string-encoded dates, incomplete domain documentation, redundant fields, and an unselected target across tables of different grains.

The dataset is **suitable for further descriptive analysis and may support a carefully scoped ML project**, but it is **not ready for model training as-is**. The next step is not automatic cleaning; it is to confirm provenance and the project objective, define a leakage-safe prediction snapshot, validate relationships and value semantics, then build a documented preprocessing and evaluation pipeline.

## Appendix A: Field Guide

The following meanings are inferred from column names and the notebook's table descriptions, not confirmed by a source dictionary. Names such as `id`, `object_id`, and `*_object_id` should be treated as keys; `*_at` fields are date/time-like strings; `*_code` fields are categorical codes; URL and description fields are provenance or text fields.

### `objects.csv` (40 fields)

| Field                   | Name-based description                             |
| ----------------------- | -------------------------------------------------- |
| `id`                  | Record identifier for the object row.              |
| `entity_type`         | Entity class, such as Company, Person, or Product. |
| `entity_id`           | Identifier associated with the entity type.        |
| `parent_id`           | Possible parent/hierarchy reference.               |
| `name`                | Display name.                                      |
| `normalized_name`     | Normalized or canonical name.                      |
| `permalink`           | URL-friendly record slug.                          |
| `category_code`       | Industry/category code.                            |
| `status`              | Recorded lifecycle status.                         |
| `founded_at`          | Founding date.                                     |
| `closed_at`           | Closure date, where recorded.                      |
| `domain`              | Website domain.                                    |
| `homepage_url`        | Homepage address.                                  |
| `twitter_username`    | Social-media username.                             |
| `logo_url`            | Logo asset location.                               |
| `logo_width`          | Logo image width metadata.                         |
| `logo_height`         | Logo image height metadata.                        |
| `short_description`   | Brief text summary.                                |
| `description`         | Longer text description.                           |
| `overview`            | Overview text.                                     |
| `tag_list`            | Tags or keywords.                                  |
| `country_code`        | Country code.                                      |
| `state_code`          | State/province code.                               |
| `city`                | City name.                                         |
| `region`              | Region label.                                      |
| `first_investment_at` | First recorded investment date.                    |
| `last_investment_at`  | Last recorded investment date.                     |
| `investment_rounds`   | Investment-round summary count.                    |
| `invested_companies`  | Invested-company summary count.                    |
| `first_funding_at`    | First recorded funding date.                       |
| `last_funding_at`     | Last recorded funding date.                        |
| `funding_rounds`      | Funding-round count.                               |
| `funding_total_usd`   | Recorded total funding in USD.                     |
| `first_milestone_at`  | First recorded milestone date.                     |
| `last_milestone_at`   | Last recorded milestone date.                      |
| `milestones`          | Milestone count.                                   |
| `relationships`       | Relationship count.                                |
| `created_by`          | Record creator/source user label.                  |
| `created_at`          | Record creation timestamp.                         |
| `updated_at`          | Record update timestamp.                           |

### Supporting table fields

| Table                  | Fields and name-based description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `acquisitions.csv`   | `id`, `acquisition_id` (record/event IDs); `acquiring_object_id`, `acquired_object_id` (entity links); `term_code` (deal-term category); `price_amount`, `price_currency_code` (reported price and currency); `acquired_at` (event date); `source_url`, `source_description` (provenance); `created_at`, `updated_at` (audit timestamps).                                                                                                                                                                                                                                                                                                                                                              |
| `degrees.csv`        | `id` (record ID); `object_id` (entity link); `degree_type`, `subject`, `institution` (education attributes); `graduated_at` (date); `created_at`, `updated_at` (audit timestamps).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `funding_rounds.csv` | `id`, `funding_round_id` (record/event IDs); `object_id` (entity link); `funded_at` (date); `funding_round_type`, `funding_round_code` (round categories); `raised_amount_usd`, `raised_amount`, `raised_currency_code` (amount and currency representations); `pre_money_valuation_usd`, `pre_money_valuation`, `pre_money_currency_code` (pre-money valuation); `post_money_valuation_usd`, `post_money_valuation`, `post_money_currency_code` (post-money valuation); `participants` (participant count); `is_first_round`, `is_last_round` (round indicators); `source_url`, `source_description`, `created_by` (provenance); `created_at`, `updated_at` (audit timestamps). |
| `funds.csv`          | `id`, `fund_id` (record IDs); `object_id` (entity link); `name` (fund name); `funded_at` (date); `raised_amount`, `raised_currency_code` (amount and currency); `source_url`, `source_description` (provenance); `created_at`, `updated_at` (audit timestamps).                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `investments.csv`    | `id` (record ID); `funding_round_id` (round link); `funded_object_id` (recipient link); `investor_object_id` (investor link); `created_at`, `updated_at` (audit timestamps).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `ipos.csv`           | `id`, `ipo_id` (record/event IDs); `object_id` (entity link); `valuation_amount`, `valuation_currency_code` (valuation); `raised_amount`, `raised_currency_code` (amount raised); `public_at` (listing date); `stock_symbol` (ticker); `source_url`, `source_description` (provenance); `created_at`, `updated_at` (audit timestamps).                                                                                                                                                                                                                                                                                                                                                               |
| `milestones.csv`     | `id` (record ID); `object_id` (entity link); `milestone_at` (date); `milestone_code` (milestone category); `description` (text); `source_url`, `source_description` (provenance); `created_at`, `updated_at` (audit timestamps).                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `offices.csv`        | `id`, `office_id` (record IDs); `object_id` (entity link); `description`, `region`, `address1`, `address2`, `city`, `zip_code`, `state_code`, `country_code` (location attributes); `latitude`, `longitude` (coordinates); `created_at`, `updated_at` (audit timestamps).                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `people.csv`         | `id` (record ID); `object_id` (entity link); `first_name`, `last_name` (name); `birthplace` (place); `affiliation_name` (affiliation).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `relationships.csv`  | `id`, `relationship_id` (record IDs); `person_object_id`, `relationship_object_id` (person/entity links); `start_at`, `end_at` (relationship dates); `is_past` (past/current indicator); `sequence` (ordering/count-like field); `title` (role/title); `created_at`, `updated_at` (audit timestamps).                                                                                                                                                                                                                                                                                                                                                                                                    |

## Appendix B: Numerical Summary

The notebook reports these statistics for every numeric feature it selected. `std` is sample standard deviation. All values are as displayed or rounded to four decimals in the existing notebook output.

| Table / feature                                   |   Count |        Mean |      Median |            Std |        Min |         Q1 |          Q3 |               Max |     Skew |
| ------------------------------------------------- | ------: | ----------: | ----------: | -------------: | ---------: | ---------: | ----------: | ----------------: | -------: |
| `acquisitions.csv / price_amount`               |   9,562 | 388,619,100 |           0 | 26,603,740,000 |          0 |          0 |     500,000 | 2,600,000,000,000 |  97.6088 |
| `funding_rounds.csv / raised_amount_usd`        |  52,928 |   7,946,092 |   1,600,000 |     42,168,200 |          0 |    246,330 |   6,700,000 |     3,835,050,000 |  50.2157 |
| `funding_rounds.csv / raised_amount`            |  52,928 |   8,056,120 | 1,565,056.5 |     44,799,140 |          0 |    250,000 |   6,600,000 |     3,835,050,000 |  48.1763 |
| `funding_rounds.csv / pre_money_valuation_usd`  |  52,928 |   329,452.5 |           0 |     65,318,030 |          0 |          0 |           0 |    15,000,000,000 | 228.8399 |
| `funding_rounds.csv / pre_money_valuation`      |  52,928 |   329,452.5 |           0 |     65,318,030 |          0 |          0 |           0 |    15,000,000,000 | 228.8399 |
| `funding_rounds.csv / post_money_valuation_usd` |  52,928 |   1,824,359 |           0 |    128,706,500 |          0 |          0 |           0 |    24,324,230,000 | 145.7047 |
| `funding_rounds.csv / post_money_valuation`     |  52,928 |   1,862,279 |           0 |    128,768,600 |          0 |          0 |           0 |    24,324,230,000 | 145.4972 |
| `funding_rounds.csv / participants`             |  52,928 |      1.5286 |           1 |         2.0602 |          0 |          0 |           2 |                36 |   2.9531 |
| `funds.csv / raised_amount`                     |   1,564 | 399,481,300 | 100,000,000 |  2,468,491,000 |          0 | 16,300,000 | 275,000,000 |    89,000,000,000 |  30.1553 |
| `ipos.csv / valuation_amount`                   |   1,259 | 360,780,000 |           0 |  5,173,185,000 |          0 |          0 |           0 |   108,960,000,000 |  19.5320 |
| `ipos.csv / raised_amount`                      |   1,259 |  58,879,500 |           0 |    870,828,500 |          0 |          0 |           0 |    24,400,000,000 |  24.8616 |
| `objects.csv / logo_width`                      | 462,651 |    189.1598 |           0 |       432.4035 |          0 |          0 |         216 |            18,200 |   6.5283 |
| `objects.csv / logo_height`                     | 462,651 |    130.4887 |           0 |       337.1916 |          0 |          0 |         139 |             9,600 |   6.9366 |
| `objects.csv / investment_rounds`               | 462,651 |      0.1739 |           0 |         3.6380 |          0 |          0 |           0 |               529 |  73.9858 |
| `objects.csv / invested_companies`              | 462,651 |      0.1381 |           0 |         2.5492 |          0 |          0 |           0 |               459 |  76.2947 |
| `objects.csv / funding_rounds`                  | 462,651 |      0.1137 |           0 |         0.5242 |          0 |          0 |           0 |                15 |   7.2663 |
| `objects.csv / funding_total_usd`               | 462,651 |   892,672.2 |           0 |     17,001,200 |          0 |          0 |           0 |     5,700,000,000 | 147.4090 |
| `objects.csv / milestones`                      | 462,651 |      0.2652 |           0 |         0.5686 |          0 |          0 |           0 |                 9 |   2.7323 |
| `objects.csv / relationships`                   | 462,651 |      1.7943 |           1 |         5.4567 |          0 |          0 |           2 |             1,189 |  84.7086 |
| `offices.csv / latitude`                        | 112,718 |     10.9014 |           0 |        18.9779 |  −43.7676 |          0 |     30.8377 |           69.6502 |   1.0478 |
| `offices.csv / longitude`                       | 112,718 |   −16.3292 |           0 |        43.9792 | −159.4803 |          0 |           0 |          176.9163 | −0.8841 |
| `relationships.csv / sequence`                  | 402,878 |   3,020.999 |           1 |      28,343.06 |          1 |          1 |           3 |           399,946 |  10.2192 |

## Appendix C: Complete Outlier Screening Results

IQR bounds and rates are copied from the saved notebook outlier summary. `Z` is the number of observations with absolute Z-score greater than 3. Values have not been treated.

| Table / feature                                   |        Min |               Max |     IQR lower |   IQR upper | IQR count | IQR (%) | Z count |  Z (%) |
| ------------------------------------------------- | ---------: | ----------------: | ------------: | ----------: | --------: | ------: | ------: | -----: |
| `objects.csv / milestones`                      |          0 |                 9 |             0 |           0 |   100,358 | 21.6919 |  16,769 | 3.6245 |
| `objects.csv / logo_height`                     |          0 |             9,600 |       −208.5 |       347.5 |    42,484 |  9.1827 |   7,636 | 1.6505 |
| `objects.csv / logo_width`                      |          0 |            18,200 |         −324 |         540 |    38,436 |  8.3078 |   9,175 | 1.9831 |
| `relationships.csv / sequence`                  |          1 |           399,946 |           −2 |           6 |    37,654 |  9.3463 |   4,709 | 1.1688 |
| `offices.csv / longitude`                       | −159.4803 |          176.9163 |             0 |           0 |    32,368 | 28.7159 |     910 | 0.8073 |
| `objects.csv / funding_rounds`                  |          0 |                15 |             0 |           0 |    31,707 |  6.8533 |  11,173 | 2.4150 |
| `objects.csv / relationships`                   |          0 |             1,189 |           −3 |           5 |    30,199 |  6.5274 |   2,568 | 0.5551 |
| `objects.csv / funding_total_usd`               |          0 |     5,700,000,000 |             0 |           0 |    27,874 |  6.0248 |   1,629 | 0.3521 |
| `objects.csv / investment_rounds`               |          0 |               529 |             0 |           0 |    16,986 |  3.6714 |   1,208 | 0.2611 |
| `objects.csv / invested_companies`              |          0 |               459 |             0 |           0 |    16,986 |  3.6714 |   1,490 | 0.3221 |
| `funding_rounds.csv / raised_amount_usd`        |          0 |     3,835,050,000 |   −9,434,175 |  16,380,500 |     5,400 | 10.2025 |     287 | 0.5422 |
| `funding_rounds.csv / raised_amount`            |          0 |     3,835,050,000 |   −9,275,000 |  16,125,000 |     5,395 | 10.1931 |     280 | 0.5290 |
| `funding_rounds.csv / participants`             |          0 |                36 |           −3 |           5 |     2,487 |  4.6988 |     981 | 1.8535 |
| `acquisitions.csv / price_amount`               |          0 | 2,600,000,000,000 |     −750,000 |   1,250,000 |     2,345 | 24.5242 |       1 | 0.0105 |
| `funding_rounds.csv / post_money_valuation`     |          0 |    24,324,230,000 |             0 |           0 |     1,279 |  2.4165 |      34 | 0.0642 |
| `funding_rounds.csv / post_money_valuation_usd` |          0 |    24,324,230,000 |             0 |           0 |     1,194 |  2.2559 |      31 | 0.0586 |
| `funds.csv / raised_amount`                     |          0 |    89,000,000,000 | −371,750,000 | 663,050,000 |       152 |  9.7187 |      10 | 0.6394 |
| `ipos.csv / raised_amount`                      |          0 |    24,400,000,000 |             0 |           0 |       138 | 10.9611 |       2 | 0.1589 |
| `ipos.csv / valuation_amount`                   |          0 |   108,960,000,000 |             0 |           0 |       108 |  8.5782 |       4 | 0.3177 |
| `funding_rounds.csv / pre_money_valuation_usd`  |          0 |    15,000,000,000 |             0 |           0 |        31 |  0.0586 |       6 | 0.0113 |
| `funding_rounds.csv / pre_money_valuation`      |          0 |    15,000,000,000 |             0 |           0 |        31 |  0.0586 |       6 | 0.0113 |
| `offices.csv / latitude`                        |  −43.7676 |           69.6502 |     −46.2566 |     77.0943 |         0 |  0.0000 |       3 | 0.0027 |
