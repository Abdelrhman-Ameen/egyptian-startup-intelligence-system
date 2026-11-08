import pytest
from fastapi.testclient import TestClient
from ventureguard.api import app, requests_seen


@pytest.fixture
def client():
    requests_seen.clear()
    with TestClient(app) as c:
        yield c


def test_health_and_research_metadata(client):
    assert client.get("/health").json()["model_loaded"]
    info = client.get("/v1/model-info").json()
    assert info["rag_status"] == "deferred"
    assert len(info["results"]) >= 10


@pytest.mark.parametrize("schema", ["core13", "full64"])
def test_historical_examples_use_outer_exclusion(client, schema):
    examples = client.get("/v1/examples").json()
    for case in examples:
        response = client.post("/v1/benchmark/predict", json={"schema_name": schema, "example_id": case["id"]})
        assert response.status_code == 200
        value = response.json()
        assert 0 <= value["score"] <= 1
        assert value["evaluation_mode"] == "outer_fold_excluded"
        assert value["observed_outcome"] in [0, 1]
        assert "global" in value["explanation_note"].lower()


def test_missing_and_ambiguous_inputs_rejected(client):
    assert client.post("/v1/benchmark/predict", json={}).status_code == 422
    assert client.post("/v1/benchmark/predict", json={"features": {}, "example_id": "x"}).status_code == 422
    assert client.post("/v1/benchmark/predict", json={"features": {"Attr1": 1}}).status_code == 422
    assert client.post("/v1/benchmark/predict", json={"example_id": "missing"}).status_code == 422


def test_simulation_api_and_no_extra_fields(client):
    payload = {"opening_cash": 100, "receipts": [50], "payments": [70]}
    assert client.post("/v1/simulate", json=payload).json()["baseline"]["closing_cash"] == 80
    assert client.post("/v1/simulate", json={**payload, "untracked_debt": 20}).status_code == 422


def test_oversize_stream_rejected(client):
    response = client.post("/v1/simulate", content=b" " * 65537, headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_server_export_matches_saved_oof_predictions(client):
    import pandas as pd
    from ventureguard.data import ROOT
    expected = pd.read_csv(ROOT / "reports/ventureguard/oof_predictions.csv")
    case = client.get("/v1/examples").json()[0]
    actual = client.post("/v1/benchmark/predict", json={"schema_name": "core13", "example_id": case["id"]}).json()
    row = int(case["id"].split("-")[1])
    saved = expected.query("schema == 'core13' and family == 'nested_selection' and row == @row").score.iloc[0]
    assert actual["score"] == pytest.approx(saved, abs=1e-9)
