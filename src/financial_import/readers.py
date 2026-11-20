"""Read bounded spreadsheets in memory, with cached Excel values when available."""

import csv
from datetime import date, datetime
from decimal import Decimal
from io import BytesIO, StringIO
import math
from zipfile import BadZipFile, ZipFile

# Use the hardened standard-library XML reader supplied by defusedxml.
import os

os.environ["OPENPYXL_LXML"] = "False"

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
from defusedxml.common import DefusedXmlException
from xml.etree.ElementTree import ParseError


MAX_UPLOAD = 2 * 1024 * 1024
MAX_UNCOMPRESSED = 12 * 1024 * 1024
MAX_ROWS = 400
MAX_COLUMNS = 32
MAX_SHEETS = 8
MAX_PREVIEW_TEXT = 400_000


class ImportProblem(ValueError):
    pass


def read_csv(payload: bytes, delimiter: str | None = None) -> list[dict]:
    try:
        encoding = "utf-16" if payload.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        content = payload.decode(encoding)
    except UnicodeError as exc:
        raise ImportProblem("احفظ الملف بصيغة CSV UTF-8 من Excel وبعدها ارفعه تاني.") from exc
    if "\x00" in content:
        raise ImportProblem("احفظ الملف بصيغة CSV UTF-8 من Excel وبعدها ارفعه تاني.")
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(content[:8192], delimiters=",;\t").delimiter
        except csv.Error:
            first = next((line for line in content.splitlines() if line.strip()), "")
            delimiter = max((",", ";", "\t"), key=first.count)
    rows = []
    try:
        for row in csv.reader(StringIO(content, newline=""), delimiter=delimiter, strict=True):
            if len(rows) >= MAX_ROWS or len(row) > MAX_COLUMNS:
                raise ImportProblem("كل جدول يقبل لحد 400 صف و32 عمود. اختصر الملف وارفعه تاني.")
            rows.append(row)
    except csv.Error as exc:
        raise ImportProblem("صيغة CSV فيها مشكلة. راجع الفواصل أو اختار الفاصل يدويًا.") from exc
    return [{"name": "CSV", "rows": rows}]


def check_archive(payload: bytes) -> None:
    with ZipFile(BytesIO(payload)) as archive:
        files = archive.infolist()
        if len(files) > 180 or sum(file.file_size for file in files) > MAX_UNCOMPRESSED:
            raise ImportProblem("محتوى ملف Excel كبير. انسخ الجدول المطلوب في ملف جديد وارفعه تاني.")
        if any(file.filename.lower().endswith("vbaproject.bin") for file in files):
            raise ImportProblem("احفظ الملف بصيغة XLSX من غير ماكروز.")


def preview_cell(cell):
    value = cell.value
    if cell.data_type == "e":
        return {"value": str(value), "status": "error"}
    if isinstance(value, (datetime, date)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return {"value": str(value), "status": "error"}
    if isinstance(value, float):
        # Preserve Excel's numeric value in a form the amount parser can read.
        return format(Decimal(str(value)), "f")
    if isinstance(value, int) and abs(value) >= 2**53:
        return str(value)
    return value


def read_xlsx(payload: bytes) -> list[dict]:
    source = None
    cached = None
    try:
        check_archive(payload)
        source = load_workbook(BytesIO(payload), read_only=True, keep_links=False, data_only=False)
        sheets = [sheet for sheet in source.worksheets if sheet.sheet_state == "visible"]
        if len(sheets) > MAX_SHEETS:
            raise ImportProblem("الرفع يقبل لحد 8 أوراق ظاهرة. احتفظ بالأوراق المالية المطلوبة.")
        output = []
        for sheet in sheets:
            if (sheet.max_row or 0) > MAX_ROWS or (sheet.max_column or 0) > MAX_COLUMNS:
                raise ImportProblem("كل ورقة تقبل لحد 400 صف و32 عمود. انسخ النطاق المطلوب في ملف جديد.")
            width = sheet.max_column or MAX_COLUMNS
            rows = []
            for row in sheet.iter_rows(max_row=MAX_ROWS + 1, max_col=width):
                if len(rows) == MAX_ROWS:
                    if any(cell.value is not None for cell in row):
                        raise ImportProblem("الورقة أكتر من 400 صف. اختصر النطاق المطلوب.")
                    break
                rows.append([
                    {"value": None, "status": "formula_missing"} if cell.data_type == "f" else preview_cell(cell)
                    for cell in row
                ])
                if sheet.max_row and len(rows) == sheet.max_row:
                    break

            if any(isinstance(cell, dict) and cell["status"] == "formula_missing" for row in rows for cell in row):
                if cached is None:
                    cached = load_workbook(BytesIO(payload), read_only=True, keep_links=False, data_only=True)
                for index, row in enumerate(cached[sheet.title].iter_rows(max_row=len(rows), max_col=width)):
                    for col, cell in enumerate(row):
                        original = rows[index][col]
                        if isinstance(original, dict) and original["status"] == "formula_missing":
                            value = preview_cell(cell)
                            if value is not None:
                                rows[index][col] = value if isinstance(value, dict) else {"value": value, "status": "formula_cached"}

            output.append({"name": sheet.title, "rows": rows})
        return output
    except (BadZipFile, InvalidFileException, ParseError, DefusedXmlException, KeyError, IndexError, TypeError, OverflowError, RuntimeError, OSError) as exc:
        raise ImportProblem("مش قادرين نقرأ ملف Excel ده. افتحه واحفظ نسخة XLSX جديدة وبعدها ارفعها.") from exc
    finally:
        if source is not None:
            source.close()
        if cached is not None:
            cached.close()


def preview_sheet(payload: bytes, extension: str, delimiter: str | None) -> dict:
    sheets = read_csv(payload, delimiter) if extension == "csv" else read_xlsx(payload)
    usable = []
    for sheet in sheets:
        rows = sheet["rows"]
        while rows and not any(cell is not None and cell != "" for cell in rows[-1]):
            rows.pop()
        if rows:
            usable.append(sheet)
    if not usable:
        raise ImportProblem("الملف فاضي. ارفع جدول فيه أسماء البنود والأرقام.")
    text_size = 0
    for sheet in usable:
        for row in sheet["rows"]:
            for cell in row:
                value = cell["value"] if isinstance(cell, dict) else cell
                if isinstance(value, str):
                    text_size += len(value)
                    if text_size > MAX_PREVIEW_TEXT:
                        raise ImportProblem("النصوص في الجدول كبيرة. ارفع نسخة فيها البنود والأرقام المطلوبة بس.")
    return {"sheets": usable}
