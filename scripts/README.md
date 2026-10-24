# Analysis scripts

These scripts download and analyze the Spain Technology Ventures dataset. They resolve paths relative to this project, so they can be run from any working directory. They leave the original SAV file unchanged.

Use Python 3.12 and install the pinned dependencies into a virtual environment:

```powershell
python -m venv .venv-eda
.venv-eda/Scripts/python.exe -m pip install -r scripts/requirements-spain-eda.txt
```

Run from the project root in this order:

```powershell
.venv-eda/Scripts/python.exe scripts/download_spain_ventures.py
.venv-eda/Scripts/python.exe scripts/inspect_spain_source.py
.venv-eda/Scripts/python.exe scripts/analyze_spain_ventures.py
.venv-eda/Scripts/python.exe scripts/supplement_spain_analysis.py
.venv-eda/Scripts/python.exe scripts/build_spain_eda_deliverables.py
.venv-eda/Scripts/python.exe scripts/verify_spain_eda.py
```

- `download_spain_ventures.py`: original public Zenodo record and SAV, checksum verification and download manifest. The snapshot date is fixed at 23 October 2026; change it and the output directory when starting a new dated research snapshot.
- `inspect_spain_source.py`: original coded-value export and SPSS definitions. It also attempts the related public research-paper downloads and reports unavailable sources without treating them as evidence.
- `analyze_spain_ventures.py`: metadata-based cleaning, descriptive tables, figures, fixed cross-validation, sensitivity checks and a label-permutation test. No deployed model is produced.
- `supplement_spain_analysis.py`: calibration, conditional bootstrap diagnostics, score distributions and reproducibility manifest.
- `build_spain_eda_deliverables.py`: standalone HTML report and executed notebook. Run it after the analysis scripts, because it reads their saved results.
- `verify_spain_eda.py`: data lineage, target counts, fold coverage, saved metrics and notebook execution checks.

Outputs are in `reports/spain_technology_ventures/`; the notebook is in `notebooks/`; original and processed data are in `research/2026-10-23/spain_technology_ventures/`. The initial experiment uses 150 known outcomes, with non-survival as the positive class. It is a historical Spanish cohort experiment, not a validated Egyptian risk model.

Dataset attribution: Garcia-Cabrera, A. M., and Garcia-Soto, M. G. (2024), *Technology-based entrepreneurs database*, [Zenodo](https://doi.org/10.5281/zenodo.10607031), CC BY 4.0. Cleaning, derived fields and analysis outputs are documented transformations of that source.
