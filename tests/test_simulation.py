from decimal import Decimal
import pytest
from pydantic import ValidationError
from ventureguard.simulation import SimulationRequest, simulate


def request(**kwargs):
    return SimulationRequest(opening_cash=1000000, cash_floor=200000,
                             receipts=[200000] * 12, payments=[300000] * 12, **kwargs)


def test_known_baseline_and_cut():
    result = simulate(request(scenario={"expense_change_pct": -.15}))
    assert result["baseline"]["closing_cash"] == -200000
    assert result["baseline"]["first_nonpositive"] == 10
    assert result["baseline"]["first_below_floor"] == 9
    assert result["scenario"]["closing_cash"] == 340000
    assert result["scenario"]["first_below_floor"] is None
    assert result["closing_cash_difference"] == 540000


def test_identity_scenario_equals_baseline():
    result = simulate(request())
    assert result["baseline"] == result["scenario"]


def test_delayed_receipts_preserve_money_beyond_horizon():
    result = simulate(request(scenario={"collection_delay_months": 2}))
    months = result["scenario"]["months"]
    assert months[0]["receipts"] == months[1]["receipts"] == 0
    assert sum(m["receipts"] for m in months) + result["scenario"]["deferred_receipts_beyond_horizon"] == 2400000
    assert result["scenario"]["closing_cash"] == -600000


def test_timing_and_fx_multiply_on_exposed_share():
    result = simulate(request(scenario={"start_month": 3, "expense_change_pct": -.2,
                       "fx_change_pct": .25, "fx_payment_share": .4, "one_off_cost": 10000}))
    months = result["scenario"]["months"]
    assert months[1]["payments"] == 300000
    assert months[2]["payments"] == 264000
    assert months[2]["one_off_cost"] == 10000
    assert months[3]["one_off_cost"] == 0


def test_rollforward_for_each_month_and_total():
    result = simulate(request(scenario={"funding_amount": 500000, "funding_month": 4, "collection_delay_months": 1}))
    rows = result["scenario"]["months"]
    for row in rows:
        assert row["closing_cash"] == row["opening_cash"] + row["receipts"] + row["funding"] - row["payments"] - row["one_off_cost"]
    assert rows[-1]["closing_cash"] == 1000000 + sum(r["receipts"] + r["funding"] - r["payments"] - r["one_off_cost"] for r in rows)


def test_zero_is_valid_and_flagged_at_opening():
    result = simulate(SimulationRequest(opening_cash=0, receipts=[0], payments=[0]))
    assert result["baseline"]["first_nonpositive"] == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_invalid_amounts_rejected(bad):
    with pytest.raises(ValidationError):
        SimulationRequest(opening_cash=bad, receipts=[1], payments=[1])


def test_mismatched_months_rejected():
    with pytest.raises(ValidationError):
        SimulationRequest(opening_cash=100, receipts=[1, 2], payments=[1])


def test_timing_outside_horizon_rejected():
    with pytest.raises(ValidationError):
        request(scenario={"start_month": 13})


def test_no_duplicate_unknown_fields():
    with pytest.raises(ValidationError):
        request(scenario={"unknown_payment": 10})
