# Spain Technology Ventures data

Source: García-Cabrera, Antonia Mercedes, and García-Soto, María Gracia (2024), *Technology-based entrepreneurs database*, [Zenodo, DOI 10.5281/zenodo.10607031](https://doi.org/10.5281/zenodo.10607031).

The source dataset is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). That license applies to the source data and the derived data exports in this folder. This analysis is not endorsed by the original authors.

The dataset has 175 records and 47 source columns. Survey responses were collected in 2009; survival is recorded for 2014. The 2024 date is the publication year, not the observation year. Of the 175 records, 110 have a positive survival outcome, 40 a negative outcome, and 25 an unknown outcome.

## Files

- `raw/technology_entrepreneurs.sav`: the unchanged original SPSS file, downloaded on 23 October 2026.
- `raw/zenodo_record.json`: source metadata, authors, and license.
- `raw/download_manifest.json`: download URL and verified checksums.
- `processed/spain_ventures_raw_export.csv`: source values exported to CSV, including the original missing-value codes.
- `processed/spss_metadata.json`: source variable labels, value labels, and missing-value definitions.
- `processed/spain_ventures_clean.csv`: declared missing-value codes replaced with missing values; a `source_row` column preserves row lineage.
- `processed/core_model_input.csv`: the nine baseline inputs and derived non-survival target for the 150 records with known outcomes. `source_row` is an identifier, not a model input.
- `processed/unknown_outcome_rows.csv`: the 25 records excluded from supervised evaluation because their outcomes are unknown.
- `processed/missing_code_audit.csv`: counts of the missing-value codes replaced during cleaning.

No unknown outcomes were converted to failures or survivors. Missing model inputs are imputed within each training fold, not filled in the source data. Financial units, the firm-age unit, and the survey scale anchors remain unverified.

The [executed notebook](../../../notebooks/01_spain_technology_ventures_eda.ipynb), [analysis scripts](../../../scripts/README.md), and [report](../../../reports/spain_technology_ventures/spain_ventures_eda_report.html) document the transformations, comparisons, and limitations.
