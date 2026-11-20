from io import BytesIO
import re
from zipfile import ZipFile

from fastapi.testclient import TestClient
from openpyxl import Workbook

from src.financial_import.app import app
from src.financial_import.readers import MAX_UPLOAD


client = TestClient(app)
preview = "/api/financial-statements/preview"
confirm = "/api/financial-statements/confirm"


def make_workbook():
    workbook = Workbook()
    income = workbook.active
    income.title = "Income"
    income.append(["Item", 2025, "Missing cache"])
    income.append(["Net profit", "=40+2", "=10-5"])
    balance = workbook.create_sheet("Balance Sheet")
    balance.append(["Item", 2025])
    balance.append(["Total assets", 200.5])
    workbook.create_sheet("Notes").sheet_state = "hidden"
    initial = BytesIO()
    workbook.save(initial)
    output = BytesIO()
    with ZipFile(initial) as source, ZipFile(output, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name == "xl/worksheets/sheet1.xml":
                data, replacements = re.subn(
                    rb"<f>40\+2</f>(?:<v\s*/>|<v></v>)",
                    b"<f>40+2</f><v>42</v>",
                    data,
                )
                assert replacements == 1
            target.writestr(name, data)
    return output.getvalue()


def statement(*items):
    return {
        "currency": "EGP",
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
        "items": [{"field": field, "amount": amount, "scale": scale} for field, amount, scale in items],
    }


def test_csv_preview_accepts_arabic_and_preserves_zero():
    payload = "بند;2025;2024\nصافي الربح;(١٬٢٥٠٫٥٠);100\nالمبيعات;0;200\n"
    response = client.post(preview, params={"format": "csv"}, content=payload.encode("utf-8-sig"))
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["sheets"][0]["rows"] == [
        ["بند", "2025", "2024"],
        ["صافي الربح", "(١٬٢٥٠٫٥٠)", "100"],
        ["المبيعات", "0", "200"],
    ]


def test_excel_preview_exposes_visible_sheets_and_marks_formula_cache():
    response = client.post(preview, params={"format": "xlsx"}, content=make_workbook())
    assert response.status_code == 200
    sheets = response.json()["sheets"]
    assert [sheet["name"] for sheet in sheets] == ["Income", "Balance Sheet"]
    assert sheets[0]["rows"][1] == [
        "Net profit",
        {"value": 42, "status": "formula_cached"},
        {"value": None, "status": "formula_missing"},
    ]
    assert sheets[1]["rows"][1] == ["Total assets", "200.5"]


def test_confirm_calculates_supported_ratios_and_tracks_missing_inputs():
    response = client.post(confirm, json=statement(
        ("net_profit", "(١٠٠٫٥٠)", 1000),
        ("total_assets", "1", 1000000),
        ("sales", "2000", 1000),
        ("total_costs", "100000", 1),
        ("total_liabilities", "500000", 1),
        ("equity", "200000", 1),
    ))
    assert response.status_code == 200
    result = response.json()
    assert result["amounts"]["net_profit"] == "-100500.00"
    assert len(result["features"]) == 13
    assert result["features"]["Attr1"]["value"] == -0.1005
    assert result["features"]["Attr9"]["value"] == 2
    assert result["features"]["Attr23"]["value"] == -0.05025
    assert result["features"]["Attr58"]["value"] is None
    assert result["features"]["Attr58"]["missing_fields"] == ["total_sales"]
    assert result["warnings"]


def test_zero_denominator_is_distinct_from_missing():
    result = client.post(confirm, json=statement(("net_profit", "10", 1), ("sales", "0", 1))).json()
    ratio = result["features"]["Attr23"]
    assert ratio["value"] is None
    assert ratio["missing_fields"] == []
    assert ratio["zero_denominator"] == "sales"


def test_rejects_duplicate_items_invalid_spreadsheets_and_oversized_uploads():
    duplicate = statement(("sales", "10", 1), ("sales", "20", 1))
    assert client.post(confirm, json=duplicate).status_code == 422
    assert client.post(confirm, json=statement(("sales", "NaN", 1))).status_code == 422
    assert client.post(preview, params={"format": "xlsx"}, content=b"not a zip").status_code == 422
    assert client.post(preview, params={"format": "csv"}, content=b"x" * (MAX_UPLOAD + 1)).status_code == 413
