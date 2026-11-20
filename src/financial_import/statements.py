"""Input contract and UCI Poland ratio calculations for reviewed statements."""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, DecimalException, localcontext
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


@dataclass(frozen=True)
class FinancialField:
    label: str
    aliases: tuple[str, ...]


FIELDS = {
    "net_profit": FinancialField("صافي الربح", ("net income", "net profit after tax", "صافي الخسارة", "صافي الربح بعد الضريبة")),
    "sales": FinancialField("المبيعات", ("sales", "sales revenue", "المبيعات")),
    "total_sales": FinancialField("إجمالي المبيعات", ("total sales", "اجمالي المبيعات")),
    "total_assets": FinancialField("إجمالي الأصول", ("total assets", "اجمالي الأصول")),
    "total_liabilities": FinancialField("إجمالي الالتزامات", ("total liabilities", "إجمالي الخصوم")),
    "current_assets": FinancialField("الأصول المتداولة", ("current assets",)),
    "short_term_liabilities": FinancialField("الالتزامات قصيرة الأجل", ("short term liabilities", "current liabilities", "الالتزامات المتداولة", "الخصوم المتداولة")),
    "ebit": FinancialField("الربح قبل الفوائد والضرائب (EBIT)", ("earnings before interest and taxes", "الربح قبل الفوائد والضرائب")),
    "equity": FinancialField("حقوق الملكية", ("total equity", "shareholders equity", "صافي حقوق الملكية", "إجمالي حقوق الملكية")),
    "inventory": FinancialField("المخزون", ("inventory", "inventories", "مخزون")),
    "receivables": FinancialField("المدينون", ("trade receivables", "accounts receivable", "الذمم المدينة")),
    "total_costs": FinancialField("إجمالي التكاليف", ("total costs", "اجمالي التكاليف")),
    "cash_balance": FinancialField("النقد وما يعادله", ("cash and cash equivalents", "cash balance", "النقدية وما في حكمها")),
}

# Keep source definitions explicit and keep any uncomputable ratio visible.
RATIOS = {
    "Attr1": ("صافي الربح ÷ إجمالي الأصول", ("net_profit",), "total_assets"),
    "Attr2": ("إجمالي الالتزامات ÷ إجمالي الأصول", ("total_liabilities",), "total_assets"),
    "Attr3": ("رأس المال العامل ÷ إجمالي الأصول", ("current_assets", "short_term_liabilities"), "total_assets"),
    "Attr4": ("الأصول المتداولة ÷ الالتزامات قصيرة الأجل", ("current_assets",), "short_term_liabilities"),
    "Attr7": ("EBIT ÷ إجمالي الأصول", ("ebit",), "total_assets"),
    "Attr9": ("المبيعات ÷ إجمالي الأصول", ("sales",), "total_assets"),
    "Attr10": ("حقوق الملكية ÷ إجمالي الأصول", ("equity",), "total_assets"),
    "Attr23": ("صافي الربح ÷ المبيعات", ("net_profit",), "sales"),
    "Attr40": ("(الأصول المتداولة − المخزون − المدينون) ÷ الالتزامات قصيرة الأجل", ("current_assets", "inventory", "receivables"), "short_term_liabilities"),
    "Attr46": ("(الأصول المتداولة − المخزون) ÷ الالتزامات قصيرة الأجل", ("current_assets", "inventory"), "short_term_liabilities"),
    "Attr51": ("الالتزامات قصيرة الأجل ÷ إجمالي الأصول", ("short_term_liabilities",), "total_assets"),
    "Attr58": ("إجمالي التكاليف ÷ إجمالي المبيعات", ("total_costs",), "total_sales"),
    "Attr63": ("المبيعات ÷ الالتزامات قصيرة الأجل", ("sales",), "short_term_liabilities"),
}
NUMBER = re.compile(r"^[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?$")
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬−", "01234567890123456789.,-")


def parse_amount(raw: str) -> Decimal:
    value = unicodedata.normalize("NFKC", raw).translate(DIGITS).strip()
    if value.startswith("(") and value.endswith(")"):
        inner = value[1:-1].strip()
        if inner.startswith(("+", "-")):
            raise ValueError("الإشارة السالبة مكتوبة مرتين. راجع الرقم.")
        value = "-" + inner
    if not NUMBER.fullmatch(value):
        raise ValueError("اكتب مبلغًا مثل 1250.50 أو ١٬٢٥٠٫٥٠. الأرقام بين أقواس بتتحسب بالسالب.")
    return Decimal(value.replace(",", ""))


class ReviewedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    amount: Annotated[str, Field(min_length=1, max_length=64)]
    scale: Literal[1, 1000, 1000000]

    @model_validator(mode="after")
    def validate_amount_and_field(self):
        if self.field not in FIELDS:
            raise ValueError("البند المالي مش ضمن قائمة البنود المتاحة.")
        try:
            parse_amount(self.amount)
        except DecimalException as exc:
            raise ValueError("راجع صيغة المبلغ.") from exc
        return self


class ReviewedStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    period_start: date
    period_end: date
    items: Annotated[list[ReviewedItem], Field(min_length=1, max_length=len(FIELDS))]

    @model_validator(mode="after")
    def check_statement(self):
        if self.period_start > self.period_end:
            raise ValueError("بداية الفترة لازم تسبق نهايتها أو تساويها.")
        if len({item.field for item in self.items}) != len(self.items):
            raise ValueError("اختار قيمة واحدة لكل بند مالي وراجع البنود المكررة.")
        return self


def as_decimal_string(number: Decimal) -> str:
    return format(number, "f")


def prepare_statement(review: ReviewedStatement) -> dict:
    # Keep normalized amounts exact through unit conversion and subtraction.
    with localcontext(prec=144):
        return calculate_statement(review)


def calculate_statement(review: ReviewedStatement) -> dict:
    values = {item.field: parse_amount(item.amount) * item.scale for item in review.items}
    features = {}
    for feature, (label, numerators, denominator) in RATIOS.items():
        missing = list(dict.fromkeys(field for field in (*numerators, denominator) if field not in values))
        zero = denominator in values and values[denominator] == 0
        result = None
        if not missing and not zero:
            numerator = values[numerators[0]] - sum(values[field] for field in numerators[1:])
            result = float(numerator / values[denominator])
        features[feature] = {
            "label": label,
            "value": result,
            "missing_fields": missing,
            "zero_denominator": denominator if zero else None,
        }

    warnings = []
    if all(key in values for key in ("total_assets", "total_liabilities", "equity")):
        assets, liabilities, equity = (values[key] for key in ("total_assets", "total_liabilities", "equity"))
        if abs(assets - (liabilities + equity)) > max(Decimal("0.01"), abs(assets) * Decimal("0.02")):
            warnings.append("راجع الميزانية: إجمالي الأصول مختلف عن مجموع الالتزامات وحقوق الملكية بأكتر من هامش التقريب.")
    return {
        "currency": review.currency,
        "period_start": review.period_start.isoformat(),
        "period_end": review.period_end.isoformat(),
        "amounts": {item: as_decimal_string(amount) for item, amount in values.items()},
        "features": features,
        "warnings": warnings,
    }
