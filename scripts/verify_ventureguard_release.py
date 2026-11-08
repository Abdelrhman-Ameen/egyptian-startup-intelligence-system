"""Verify a running local or deployed release against frozen research outputs."""
import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", default="reports/ventureguard/validation.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = pd.read_csv(root / "reports/ventureguard/oof_predictions.csv")
    with httpx.Client(base_url=args.url, timeout=90) as client:
        assert client.get("/health").raise_for_status().json()["model_loaded"]
        assert client.get("/").raise_for_status().headers["content-type"].startswith("text/html")
        assert client.get("/docs").status_code == 200
        assert client.get("/v1/model-info").raise_for_status().json()["rag_status"] == "deferred"
        cases = client.get("/v1/examples").raise_for_status().json()
        checked = 0
        for schema in ["core13", "full64"]:
            for case in cases:
                value = client.post("/v1/benchmark/predict", json={"schema_name": schema, "example_id": case["id"]}).raise_for_status().json()
                row = int(case["id"].split("-")[1])
                score = expected[(expected.schema == schema) & (expected.family == "nested_selection") & (expected.row == row)].score.iloc[0]
                assert abs(value["score"] - score) < 1e-9
                assert value["evaluation_mode"] == "outer_fold_excluded"
                checked += 1
        simulation = client.post("/v1/simulate", json={"opening_cash": 1000000, "cash_floor": 200000,
                    "receipts": [200000] * 12, "payments": [300000] * 12,
                    "scenario": {"expense_change_pct": -.15}}).raise_for_status().json()
        assert simulation["scenario"]["closing_cash"] == 340000
        assert simulation["baseline"]["closing_cash"] == -200000
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            client.post("/v1/benchmark/predict", json={"schema_name": "full64", "example_id": cases[0]["id"]}).raise_for_status()
            latencies.append((time.perf_counter() - start) * 1000)
    result = {"passed": True, "url": args.url, "historical_predictions_verified": checked,
              "tolerance": 1e-9, "cash_identity_example_passed": True,
              "warm_request_count": len(latencies), "warm_latency_median_ms": float(np.median(latencies)),
              "warm_latency_p95_ms": float(np.percentile(latencies, 95)),
              "latency_scope": "Sequential warm client HTTP requests, not a load or cold-start benchmark.",
              "public_endpoint": urlparse(args.url).hostname not in ["localhost", "127.0.0.1"]}
    target = root / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
