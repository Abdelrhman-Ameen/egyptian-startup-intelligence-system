"""Versioned UCI acquisition and explicit data-quality decisions."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import arff

ROOT = Path(os.environ.get("VENTUREGUARD_HOME", Path(__file__).resolve().parents[2]))
SOURCE_URL = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"
SOURCE_SHA256 = "17377929aa0b204bbf957e56462cf827c19fe4e2ce89f27dfbc77f9ea2bb16c9"
CORE = [f"Attr{i}" for i in [1, 2, 3, 4, 7, 9, 10, 23, 40, 46, 51, 58, 63]]
LABELS = {
    "Attr1": "Net profit / total assets", "Attr2": "Total liabilities / total assets",
    "Attr3": "Working capital / total assets", "Attr4": "Current assets / short-term liabilities",
    "Attr7": "EBIT / total assets", "Attr9": "Sales / total assets",
    "Attr10": "Equity / total assets", "Attr23": "Net profit / sales",
    "Attr40": "(Current assets − inventory − receivables) / short-term liabilities",
    "Attr46": "(Current assets − inventory) / short-term liabilities",
    "Attr51": "Short-term liabilities / total assets", "Attr58": "Total costs / total sales",
    "Attr63": "Sales / short-term liabilities",
    "Attr5": "Source-defined liquidity duration (days)",
    "Attr6": "Retained earnings / total assets", "Attr8": "Book equity / total liabilities",
    "Attr11": "(Gross profit + extraordinary items + financial expenses) / total assets",
    "Attr12": "Gross profit / short-term liabilities", "Attr13": "(Gross profit + depreciation) / sales",
    "Attr14": "(Gross profit + interest) / total assets", "Attr15": "Liabilities × 365 / (gross profit + depreciation)",
    "Attr16": "(Gross profit + depreciation) / liabilities", "Attr17": "Total assets / liabilities",
    "Attr18": "Gross profit / total assets", "Attr19": "Gross profit / sales", "Attr20": "Inventory × 365 / sales",
    "Attr21": "Current-year sales / previous-year sales", "Attr22": "Operating profit / total assets",
    "Attr24": "Three-year gross profit / total assets", "Attr25": "(Equity − share capital) / total assets",
    "Attr26": "(Net profit + depreciation) / total liabilities", "Attr27": "Operating profit / financial expenses",
    "Attr28": "Working capital / fixed assets", "Attr29": "Logarithm of total assets",
    "Attr30": "(Total liabilities − cash) / sales", "Attr31": "(Gross profit + interest) / sales",
    "Attr32": "Current liabilities × 365 / cost of products sold", "Attr33": "Operating expenses / short-term liabilities",
    "Attr34": "Operating expenses / total liabilities", "Attr35": "Profit on sales / total assets",
    "Attr36": "Total sales / total assets", "Attr37": "(Current assets − inventory) / long-term liabilities",
    "Attr38": "Constant capital / total assets", "Attr39": "Profit on sales / sales",
    "Attr41": "Source-defined liability / operating cash proxy", "Attr42": "Operating profit / sales",
    "Attr43": "Receivables rotation + inventory turnover (days)", "Attr44": "Receivables × 365 / sales",
    "Attr45": "Net profit / inventory", "Attr47": "Inventory × 365 / cost of products sold",
    "Attr48": "Source-defined EBITDA / total assets", "Attr49": "Source-defined EBITDA / sales",
    "Attr50": "Current assets / total liabilities", "Attr52": "Short-term liabilities × 365 / cost of products sold",
    "Attr53": "Equity / fixed assets", "Attr54": "Constant capital / fixed assets", "Attr55": "Working capital",
    "Attr56": "(Sales − cost of products sold) / sales", "Attr57": "Source-defined working-capital / profit proxy",
    "Attr59": "Long-term liabilities / equity", "Attr60": "Sales / inventory", "Attr61": "Sales / receivables",
    "Attr62": "Short-term liabilities × 365 / sales", "Attr64": "Sales / fixed assets",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquire() -> Path:
    target = ROOT / "data/raw/poland.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        # Reuse the original downloaded archive when present, not a transformed CSV.
        previous = ROOT / "research/2026-10-08/raw/uci_poland_bankruptcy.zip"
        payload = previous.read_bytes() if previous.exists() else urlopen(SOURCE_URL, timeout=90).read()
        if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
            raise ValueError("Downloaded archive checksum changed; no cache file was written.")
        target.write_bytes(payload)
    if digest(target) != SOURCE_SHA256:
        raise ValueError("Source archive checksum changed. Review the source and protocol before training.")
    return target


def load_data(write_audit: bool = True):
    source = acquire()
    with ZipFile(source) as archive:
        member = next(n for n in archive.namelist() if n.lower().endswith("5year.arff"))
        records, _ = arff.loadarff(io.StringIO(archive.read(member).decode("utf-8")))
    frame = pd.DataFrame(records)
    frame.columns = [f"Attr{i}" for i in range(1, 65)] + ["class"]
    y = frame.pop("class").astype(int).to_numpy()
    x = frame.astype(float).replace([np.inf, -np.inf], np.nan)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    audit = {
        "source": SOURCE_URL, "source_sha256": digest(source), "member": member,
        "target": "Bankruptcy status after one year as defined by UCI for 5year.arff",
        "rows": len(x), "positive": int(y.sum()), "prevalence": float(y.mean()),
        "features": len(x.columns), "unique_exact_vectors": int(len(np.unique(groups))),
        "missing_by_feature": x.isna().sum().astype(int).to_dict(),
        "missingness_by_target": {str(k): float(x.loc[y == k].isna().any(axis=1).mean()) for k in [0, 1]},
        "balance_identity_review_count": int(((x.Attr2 + x.Attr10 - 1).abs() > .02).sum()),
        "balance_identity_policy": "Flag for definition review; never auto-repair or drop by target.",
        "conflicting_exact_vector_groups": int(pd.DataFrame({"g": groups, "y": y}).groupby("g").y.nunique().gt(1).sum()),
        "exclusions": "No blanket complete-case deletion or outcome-dependent cleaning.",
        "limitations": ["No reliable company IDs or observation dates", "Historical Polish source domain", "Prior exploratory analysis exists"],
    }
    if write_audit:
        out = ROOT / "reports/ventureguard"
        out.mkdir(parents=True, exist_ok=True)
        (out / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return x, y, groups, audit


if __name__ == "__main__":
    print(json.dumps(load_data()[3], indent=2))
