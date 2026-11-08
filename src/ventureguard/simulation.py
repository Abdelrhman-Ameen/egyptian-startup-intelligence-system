"""Monthly cash accounting. No estimated causal elasticities or ML predictions."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

Amount = Annotated[FiniteFloat, Field(ge=0, le=1e12)]


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expense_change_pct: Annotated[FiniteFloat, Field(ge=-.95, le=2)] = 0
    receipt_change_pct: Annotated[FiniteFloat, Field(ge=-1, le=2)] = 0
    fx_change_pct: Annotated[FiniteFloat, Field(ge=-.95, le=3)] = 0
    fx_payment_share: Annotated[FiniteFloat, Field(ge=0, le=1)] = 0
    collection_delay_months: int = Field(default=0, ge=0, le=12)
    start_month: int = Field(default=1, ge=1, le=36)
    one_off_cost: Amount = 0
    funding_amount: Amount = 0
    funding_month: int = Field(default=1, ge=1, le=36)


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    currency: str = Field(default="EGP", pattern=r"^[A-Z]{3}$")
    opening_cash: Amount
    cash_floor: Amount = 0
    receipts: list[Amount] = Field(min_length=1, max_length=36)
    payments: list[Amount] = Field(min_length=1, max_length=36)
    scenario: Scenario = Field(default_factory=Scenario)

    @model_validator(mode="after")
    def validate_horizon(self):
        if len(self.receipts) != len(self.payments):
            raise ValueError("Receipts and payments must cover the same months.")
        if self.scenario.start_month > len(self.receipts):
            raise ValueError("Scenario start must be within the planning horizon.")
        if self.scenario.funding_amount and self.scenario.funding_month > len(self.receipts):
            raise ValueError("Funding month must be within the planning horizon.")
        return self


def money(value):
    return Decimal(str(value)).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)


def schedule(request, scenario):
    n = len(request.receipts)
    receipts = [Decimal(0) for _ in range(n)]
    deferred = Decimal(0)
    for index, value in enumerate(request.receipts):
        active = index + 1 >= scenario.start_month
        receipt = money(value) * (1 + Decimal(str(scenario.receipt_change_pct)) if active else 1)
        receipt = money(receipt)
        destination = index + (scenario.collection_delay_months if active else 0)
        if destination < n:
            receipts[destination] += receipt
        else:
            deferred += receipt
    balance = money(request.opening_cash)
    floor = money(request.cash_floor)
    rows = []
    first_floor = 0 if balance < floor else None
    first_zero = 0 if balance <= 0 else None
    for i, original in enumerate(request.payments):
        month = i + 1
        opening = balance
        payment = money(original)
        if month >= scenario.start_month:
            payment *= 1 + Decimal(str(scenario.expense_change_pct))
            payment *= 1 + Decimal(str(scenario.fx_payment_share)) * Decimal(str(scenario.fx_change_pct))
        payment = money(payment)
        one_off = money(scenario.one_off_cost) if month == scenario.start_month else Decimal(0)
        funding = money(scenario.funding_amount) if month == scenario.funding_month else Decimal(0)
        balance = opening + receipts[i] + funding - payment - one_off
        if first_floor is None and balance < floor:
            first_floor = month
        if first_zero is None and balance <= 0:
            first_zero = month
        rows.append({"month": month, "opening_cash": float(opening), "receipts": float(receipts[i]),
                     "payments": float(payment), "one_off_cost": float(one_off), "funding": float(funding),
                     "closing_cash": float(balance)})
    minimum = min([money(request.opening_cash)] + [money(r["closing_cash"]) for r in rows])
    return {"months": rows, "closing_cash": float(balance), "first_below_floor": first_floor,
            "first_nonpositive": first_zero, "minimum_cash": float(minimum),
            "financing_gap_to_zero": float(max(Decimal(0), -minimum)),
            "deferred_receipts_beyond_horizon": float(deferred)}


def simulate(request: SimulationRequest):
    baseline = schedule(request, Scenario())
    alternative = schedule(request, request.scenario)
    return {"currency": request.currency, "cash_floor": float(money(request.cash_floor)), "baseline": baseline, "scenario": alternative,
            "closing_cash_difference": float(money(alternative["closing_cash"]) - money(baseline["closing_cash"])),
            "assumptions": ["Receipts are cash collections, not accrual revenue.",
                            "Changed receipts are an explicit assumption, not a learned demand response.",
                            "FX applies only to the stated payment share; receipt FX exposure is not modeled.",
                            "Funding is available at the start of its month; monthly endpoints cannot identify intramonth shortfalls.",
                            "Delayed collections beyond the horizon are reported separately, not lost.",
                            "Negative balances indicate financing gaps; continued operations are illustrative.",
                            "No bankruptcy probability or causal effect is inferred from this schedule."]}
