# VentureGuard experiment results

Generated from frozen nested evaluation. All figures below are internal source-domain estimates, after earlier exploratory analysis.

| Schema | Procedure | Mean AP | AP SD | Mean ROC-AUC | Mean Brier | Mean Recall@10% |
|---|---|---:|---:|---:|---:|---:|
| core13 | logistic | 0.320 | 0.032 | 0.797 | 0.0565 | 0.463 |
| core13 | hgb | 0.353 | 0.027 | 0.827 | 0.0539 | 0.498 |
| core13 | mlp | 0.335 | 0.035 | 0.830 | 0.0547 | 0.493 |
| core13 | nested_selection | 0.353 | 0.027 | 0.827 | 0.0539 | 0.498 |
| core13 | training_prevalence | 0.069 | 0.000 | 0.500 | 0.0646 | 0.101 |
| full64 | logistic | 0.682 | 0.048 | 0.918 | 0.0351 | 0.724 |
| full64 | hgb | 0.799 | 0.026 | 0.955 | 0.0271 | 0.798 |
| full64 | mlp | 0.748 | 0.033 | 0.942 | 0.0342 | 0.741 |
| full64 | nested_selection | 0.799 | 0.026 | 0.955 | 0.0271 | 0.798 |
| full64 | training_prevalence | 0.069 | 0.000 | 0.500 | 0.0646 | 0.101 |
| missingness_only | hgb | 0.459 | 0.066 | 0.764 | 0.0448 | 0.538 |
| missingness_only | training_prevalence | 0.069 | 0.000 | 0.500 | 0.0646 | 0.101 |
| full64_without_explicit_indicators | hgb | 0.791 | 0.030 | 0.955 | 0.0281 | 0.788 |
| full64_without_explicit_indicators | training_prevalence | 0.069 | 0.000 | 0.500 | 0.0646 | 0.101 |

## Interpretation

The full-schema nested procedure achieved mean AP 0.799, versus 0.353 for the compact schema. This is a matched internal comparison, not proof of real data-entry cost or external usefulness.

Missingness-only gradient boosting achieved mean AP 0.459. Missingness patterns therefore contain substantial label-associated information in this source. This does not establish leakage or quantify how much of the full model is causal financial signal.

Calibration and class-weight alternatives were chosen inside outer training. Compare the raw and selected Brier values in fold_metrics.csv. Fold SD is not a confidence interval. The full-data release model has no separate external test result.

## Missing-input stress test

Artificial random masking, evaluated on held-out cases without refitting. This does not reproduce all real missing-data mechanisms.

| Schema | Added masking | Mean AP | Mean Brier |
|---|---:|---:|---:|
| core13 | 0% | 0.353 | 0.0539 |
| core13 | 10% | 0.332 | 0.0548 |
| core13 | 30% | 0.263 | 0.0582 |
| full64 | 0% | 0.799 | 0.0271 |
| full64 | 10% | 0.423 | 0.0952 |
| full64 | 30% | 0.251 | 0.2267 |

## Reproduce

```bash
python -m ventureguard.training
python -m ventureguard.reporting
pytest -q
```

See DATA_CARD.md and MODEL_CARD.md for provenance, mathematical assumptions, selection and limitations. RAG is deferred.
