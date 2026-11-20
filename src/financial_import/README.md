# Financial statement import

Run from the repository root with Python 3.12.

```powershell
python -m pip install -r src/financial_import/requirements.txt
python -m uvicorn financial_import.app:app --app-dir src
```

Open [the local upload screen](http://127.0.0.1:8000/). The screen has a downloadable CSV template. You can also upload your own file if its figures are arranged in a table.

## Import a statement

1. Upload an `.xlsx` or `.csv` file. Use CSV UTF-8 when exporting from Excel. The importer detects comma, semicolon and tab separators, and the screen lets you override the detected separator.
2. Select the worksheet, the locations of the labels and amounts, and the first row or column that contains a line item. Pick the numbers for one reporting period. For example, from a profit and loss statement with columns for 2024 and 2025, pick only the 2025 column.
3. Review the suggested mappings. Labels can be in Arabic or English. Choose the correct field manually if the document uses a different label. Select whether the figures are reported in units, thousands or millions. Add them to the review. You can repeat the process for a balance sheet from another worksheet or file.
4. Review all selected amounts, enter a three-letter currency code and confirm a common reporting period. Use balance sheet balances from the end of that period. The screen displays the ratios it can calculate and shows the missing field or a zero denominator when it cannot calculate one.

The amount reader supports decimal points, English or Arabic-Indic digits, Arabic decimal and thousands separators, and accounting negatives such as `(120.50)`. Use one currency per import. The importer scales reported units and performs no currency conversion. For Excel formulas, the preview marks values cached in the workbook. Recalculate and save the workbook if a required formula has no cached value.

## Backend contract

| Route | Behavior |
| --- | --- |
| `GET /api/financial-statements/fields` | Field identifiers, labels and suggested aliases |
| `GET /api/financial-statements/template.csv` | Empty CSV with one row per supported item |
| `POST /api/financial-statements/preview?format=xlsx` | Accepts the raw file body and returns visible worksheet cells |
| `POST /api/financial-statements/preview?format=csv` | Accepts the raw CSV body and an optional `separator=comma\|semicolon\|tab` query parameter |
| `POST /api/financial-statements/confirm` | Validates the reviewed JSON and returns exact normalized amounts as strings and calculable ratios as numbers |

Each confirmation has `currency`, `period_start`, `period_end` and `items`. An item looks like `{"field":"total_assets","amount":"1250.50","scale":1000}`. The `features` object is keyed by the 13 selected UCI Poland attributes. A ratio with unavailable inputs has a `null` value. The separate `amounts` object holds the source line items for future scenario calculations.

The upload limit is 2 MiB per file, 8 visible sheets and 400 rows by 32 columns per sheet. This version processes files in memory for local use. The importer sends no content to Gemini and does not save files to disk. Before publishing, put the endpoints behind the app's authentication and use HTTPS.

The current screen prepares inputs and ratios. Bankruptcy scoring and scenario simulation are separate integration work.
