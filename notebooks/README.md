# Project notebooks

Start with `01_spain_technology_ventures_eda.ipynb`. It contains executed outputs for the first dataset in the project handout, including data quality, missingness, imbalance, descriptive comparisons and a small ML benchmark.

Use a Python 3.12 environment with the dependencies in `../scripts/requirements-spain-eda.txt`. The notebook locates the project root from its current folder and imports the shared analysis code from `../scripts/`.

The notebook reads saved model results by default. Set `RETRAIN = True` in its model section to repeat the same cross-validation experiment. To regenerate every result and report, follow the commands in `../scripts/README.md`.

The source and processed data are in `../research/2026-10-23/spain_technology_ventures/`. The shareable report is `../reports/spain_technology_ventures/spain_ventures_eda_report.html`.
