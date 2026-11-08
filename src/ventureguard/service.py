"""CPU research inference using trusted, versioned local artifacts."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import joblib
from .data import ROOT, LABELS
from .training import predict_bundle


class ResearchService:
    def __init__(self, directory=None):
        self.directory = directory or ROOT / "artifacts"
        self.bundle = joblib.load(self.directory / "research_models.joblib")
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        self.examples = json.loads((self.directory / "examples.json").read_text(encoding="utf-8"))
        importance = pd.read_csv(ROOT / "reports/ventureguard/permutation_importance.csv")
        self.importance = {schema: data.groupby("feature").ap_decrease.mean().sort_values(ascending=False)
                           for schema, data in importance.groupby("schema")}

    def example_list(self):
        return [{k: v for k, v in case.items() if k != "features"} for case in self.examples]

    def predict(self, schema, features=None, example_id=None):
        spec = self.bundle["schemas"][schema]
        example = None
        if example_id:
            example = next((case for case in self.examples if case["id"] == example_id), None)
            if example is None:
                raise ValueError("Unknown historical example.")
            features = example["features"]
            fitted = spec["folds"][example["fold"]]
        else:
            fitted = spec["final"]
            unknown = set(features) - set(spec["columns"])
            if unknown:
                raise ValueError("Unexpected financial fields: " + ", ".join(sorted(unknown)))
        missing = [col for col in spec["columns"] if features.get(col) is None]
        if len(missing) / len(spec["columns"]) > .25:
            raise ValueError("Provide at least 75% of this schema's fields. This is an input policy, not a confidence guarantee.")
        frame = pd.DataFrame([[features.get(col, np.nan) for col in spec["columns"]]], columns=spec["columns"], dtype=float)
        score = float(predict_bundle(fitted, frame)[0])
        outlying = [col for col in frame if pd.notna(frame.iloc[0][col]) and
                    not spec["low"][col] <= frame.iloc[0][col] <= spec["high"][col]]
        # Local gradients of ratios can make invalid accounting states. Instead display
        # held-out global relevance plus the actual case's values, without causal claims.
        drivers = [{"feature": col, "label": LABELS.get(col, col),
                    "value": None if pd.isna(frame.iloc[0][col]) else float(frame.iloc[0][col]),
                    "mean_heldout_ap_decrease": float(value)}
                   for col, value in self.importance[schema].head(5).items()]
        return {"schema": schema, "model_version": self.bundle["version"],
                "model_family": fitted["selection"]["family"], "score": score,
                "calibrated": fitted["selection"]["calibration_used"],
                "scope": "Historical Polish research benchmark; not a validated Egyptian startup probability.",
                "evaluation_mode": "outer_fold_excluded" if example else "research_input_unverified_membership",
                "observed_outcome": example["outcome"] if example else None,
                "missing_fields": missing, "outside_reference_range": outlying,
                "reference_range_note": "Central 99% of source values; a diagnostic flag, not a validity or confidence test.",
                "important_features": drivers,
                "explanation_note": "Global held-out permutation relevance, not local attribution or a causal explanation. Correlated ratios can share relevance.",
                "features": {col: None if pd.isna(frame.iloc[0][col]) else float(frame.iloc[0][col]) for col in frame},
                "warnings": ["Demo cases are outcome-stratified illustrations, not a prevalence sample."] if example else ["Training membership is not verified for submitted research inputs."]}
