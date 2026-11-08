"""Post-fit diagnostics using frozen outer-fold predictions and estimators."""
from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from threadpoolctl import threadpool_limits
from .data import ROOT, load_data, CORE
from .training import metrics, predict_bundle, save_json


def main():
    out = ROOT / "reports/ventureguard"
    x, y, groups, audit = load_data(write_audit=False)
    frozen = joblib.load(ROOT / "artifacts/research_models.joblib")
    folds = pd.read_csv(out / "folds.csv")
    scores = pd.read_csv(out / "oof_predictions.csv")
    rng = np.random.default_rng(20261108)
    stress_rows = []
    with threadpool_limits(limits=2):
        for schema, spec in frozen["schemas"].items():
            for fraction in [0., .1, .3]:
                for fold, fitted in enumerate(spec["folds"]):
                    selected = folds.row[folds.fold == fold].to_numpy()
                    damaged = x[spec["columns"]].iloc[selected].copy()
                    mask = rng.random(damaged.shape) < fraction
                    damaged = damaged.mask(mask)
                    p = predict_bundle(fitted, damaged)
                    stress_rows.append({"schema": schema, "fraction_randomly_masked": fraction, "fold": fold,
                                        **metrics(y[selected], p)})
    pd.DataFrame(stress_rows).to_csv(out / "stress_metrics.csv", index=False)
    subgroup_rows = []
    definitions = {"all64_complete": ~x.isna().any(axis=1), "any64_missing": x.isna().any(axis=1),
                   "Attr21_missing": x.Attr21.isna(), "Attr21_observed": x.Attr21.notna()}
    for schema in frozen["schemas"]:
        base = scores[(scores.schema == schema) & (scores.family == "nested_selection")].sort_values("row")
        for name, mask in definitions.items():
            block = base.loc[mask.to_numpy()]
            both_classes = block.target.nunique() == 2
            subgroup_rows.append({"schema": schema, "subgroup": name, "n": len(block), "positive": int(block.target.sum()),
                                  "prevalence": float(block.target.mean()), "mean_score": float(block.score.mean()),
                                  "average_precision": float(average_precision_score(block.target, block.score)) if both_classes else None,
                                  "roc_auc": float(roc_auc_score(block.target, block.score)) if both_classes else None,
                                  "brier": float(brier_score_loss(block.target, block.score))})
    pd.DataFrame(subgroup_rows).to_csv(out / "subgroup_metrics.csv", index=False)
    diagnostics = {"stress_scope": "Artificial independent masking of held-out inputs; not a model of real missingness or external validation.",
                   "subgroup_scope": "Post-hoc pooled outer-fold predictions; models were not refitted on complete cases. Groups differ in prevalence and sample size.",
                   "stress": stress_rows, "subgroups": subgroup_rows}
    save_json(out / "diagnostics.json", diagnostics)
    summary = json.loads((out / "metrics.json").read_text())["results"]
    report = ["# VentureGuard experiment results", "", "Generated from frozen nested evaluation. All figures below are internal source-domain estimates, after earlier exploratory analysis.", "",
              "| Schema | Procedure | Mean AP | AP SD | Mean ROC-AUC | Mean Brier | Mean Recall@10% |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        report.append(f"| {row['schema']} | {row['family']} | {row['average_precision']:.3f} | {row['average_precision_sd']:.3f} | {row['roc_auc']:.3f} | {row['brier']:.4f} | {row['recall_at_10pct']:.3f} |")
    full = next(r for r in summary if r["schema"] == "full64" and r["family"] == "nested_selection")
    core = next(r for r in summary if r["schema"] == "core13" and r["family"] == "nested_selection")
    mask = next(r for r in summary if r["schema"] == "missingness_only" and r["family"] == "hgb")
    report += ["", "## Interpretation", "",
               f"The full-schema nested procedure achieved mean AP {full['average_precision']:.3f}, versus {core['average_precision']:.3f} for the compact schema. This is a matched internal comparison, not proof of real data-entry cost or external usefulness.", "",
               f"Missingness-only gradient boosting achieved mean AP {mask['average_precision']:.3f}. Missingness patterns therefore contain substantial label-associated information in this source. This does not establish leakage or quantify how much of the full model is causal financial signal.", "",
               "Calibration and class-weight alternatives were chosen inside outer training. Compare the raw and selected Brier values in fold_metrics.csv. Fold SD is not a confidence interval. The full-data release model has no separate external test result.", "",
               "## Missing-input stress test", "", "Artificial random masking, evaluated on held-out cases without refitting. This does not reproduce all real missing-data mechanisms.", "",
               "| Schema | Added masking | Mean AP | Mean Brier |", "|---|---:|---:|---:|"]
    stress = pd.DataFrame(stress_rows)
    stress_summary = []
    for (schema, fraction), block in stress.groupby(["schema", "fraction_randomly_masked"]):
        report.append(f"| {schema} | {fraction:.0%} | {block.average_precision.mean():.3f} | {block.brier.mean():.4f} |")
        stress_summary.append({"schema": schema, "fraction_randomly_masked": float(fraction),
                               "average_precision": float(block.average_precision.mean()), "brier": float(block.brier.mean())})
    metadata_path = ROOT / "artifacts/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["stress_summary"] = stress_summary
    metadata["review_tie_policy"] = "Expected uniform selection within boundary score ties; source row order is not used."
    save_json(metadata_path, metadata)
    report += ["", "## Reproduce", "", "```bash", "python -m ventureguard.training", "python -m ventureguard.reporting", "pytest -q", "```", "",
               "See DATA_CARD.md and MODEL_CARD.md for provenance, mathematical assumptions, selection and limitations. RAG is deferred."]
    (ROOT / "docs/RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Post-fit diagnostics and docs/RESULTS.md generated.")


if __name__ == "__main__":
    main()
