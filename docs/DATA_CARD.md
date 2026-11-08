# Data card

**Source:** [UCI Polish Companies Bankruptcy](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data), Sebastian Tomczak (2016), DOI `10.24432/C5F600`. UCI publishes it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Derived demo cases and numerical experiment outputs retain this attribution. This release acquires the source archive rather than redistributing the historical research collection.

**Primary file:** `5year.arff`: 5,910 financial statements, 64 ratios, 410 positive outcomes. UCI describes this fifth file as bankruptcy status after one year. The filename is not a five-year prediction horizon. The five files are not pooled.

**Selection:** bankrupt firms were analyzed in 2000–2012 and operating firms in 2007–2013, according to UCI. The sample is not a contemporary population prevalence estimate.

**Identity and timing:** the released files used here lack reliable company and observation-date keys. Exact feature-vector hashes define duplicate groups; these do not establish independent companies. Neither a temporal nor country-generalization claim is supported.

**Cleaning:** numeric infinities become missing; ratios are not indiscriminately clipped to [0,1]. Negative equity can be legitimate. Missing values are imputed from training folds. Complete-case deletion, target-dependent filtering and automatic accounting repairs are not used. Definition inconsistencies are flagged, not silently corrected.

**Reproducibility:** `python -m ventureguard.data` records the original archive SHA-256, row counts, missingness and review counts in `reports/ventureguard/data_audit.json`.

## Compact feature contract

| Ratio | Required definition |
|---|---|
| Attr1 | Net profit / total assets |
| Attr2 | Total liabilities / total assets |
| Attr3 | Working capital / total assets |
| Attr4 | Current assets / short-term liabilities |
| Attr7 | EBIT / total assets |
| Attr9 | Sales / total assets |
| Attr10 | Equity / total assets |
| Attr23 | Net profit / sales |
| Attr40 | (Current assets − inventory − receivables) / short-term liabilities |
| Attr46 | (Current assets − inventory) / short-term liabilities |
| Attr51 | Short-term liabilities / total assets |
| Attr58 | Total costs / total sales |
| Attr63 | Sales / short-term liabilities |

Sales and total sales must not be silently interchanged. Numerators and denominators must use compatible reporting periods and units. Undefined ratios are missing, not zero. Thirteen ratios are not thirteen independent data-entry tasks: source accounting items are shared. The dataset does not supply validated usability-cost labels or sufficient raw cash balances to reconstruct a planning schedule.

## Separate planning inputs

Cash planning uses explicit opening cash, monthly collections, payments, a cash buffer and scenario assumptions. Those inputs are supplied by the user and are never inferred from the UCI ratios. No profiles, statements or scenario payloads are persisted by the app.
