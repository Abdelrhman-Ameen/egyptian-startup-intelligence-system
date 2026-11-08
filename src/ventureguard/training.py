"""Run the frozen, source-domain nested evaluation and export CPU artifacts.

Run: python -m ventureguard.training
An interrupted run resumes completed outer-fold experiments from checkpoints.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

from .data import ROOT, CORE, LABELS, load_data, digest
from .models import make_model, candidate_configs

OUT = ROOT / "reports/ventureguard"
ART = ROOT / "artifacts"
CONFIG = json.loads((ROOT / "configs/experiment.json").read_text())


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def metrics(y, p):
    k = max(1, int(np.ceil(.1 * len(y))))
    p = np.asarray(p)
    cutoff = np.partition(p, -k)[-k]
    above, tied = p > cutoff, p == cutoff
    # Expected uniform selection within boundary ties, independent of source row order.
    # A constant predictor therefore receives the appropriate random-review reference.
    expected_tp = float(y[above].sum()) + (k - int(above.sum())) * float(y[tied].mean())
    return {"average_precision": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p)), "recall_at_10pct": expected_tp / float(y.sum()),
            "precision_at_10pct": expected_tp / k, "mean_score": float(np.mean(p)),
            "prevalence": float(np.mean(y)), "n": int(len(y)), "positive": int(np.sum(y))}


def score_feature(p):
    return logit(np.clip(p, 1e-7, 1 - 1e-7)).reshape(-1, 1)


def fit_calibrator(y, p):
    return LogisticRegression(C=1e6, max_iter=1000).fit(score_feature(p), y)


def calibrate(model, p):
    return p if model is None else model.predict_proba(score_feature(p))[:, 1]


def search(x, y, groups, splits, family, seed, indicators=True, mask_only=False):
    trials, best = [], None
    for cfg in candidate_configs(family):
        predictions = np.full(len(y), np.nan)
        aps, model_epochs = [], []
        for train, valid in splits:
            model = make_model(family, cfg, seed, indicators, mask_only,
                               CONFIG["mlp_max_epochs"], CONFIG["mlp_patience"])
            model.fit(x.iloc[train], y[train])
            predictions[valid] = model.predict_proba(x.iloc[valid])[:, 1]
            aps.append(float(average_precision_score(y[valid], predictions[valid])))
            if family == "mlp":
                model_epochs.append(int(model[-1].epochs_))
        trial = {"family": family, "config": cfg, "inner_mean_ap": float(np.mean(aps)),
                 "inner_fold_ap": aps, "epochs": model_epochs}
        trials.append(trial)
        if best is None or trial["inner_mean_ap"] > best["inner_mean_ap"]:
            best = dict(trial, predictions=predictions)
    # Calibration decisions use cross-fitted mappings entirely inside outer training.
    p = best.pop("predictions")
    calibrated = np.empty(len(y))
    for train, valid in splits:
        mapping = fit_calibrator(y[train], p[train])
        calibrated[valid] = calibrate(mapping, p[valid])
    raw_brier = brier_score_loss(y, p)
    cal_brier = brier_score_loss(y, calibrated)
    use_cal = bool(cal_brier < raw_brier)
    mapping = fit_calibrator(y, p) if use_cal else None
    best.update({"inner_raw_brier": float(raw_brier), "inner_crossfit_calibrated_brier": float(cal_brier),
                 "calibration_used": use_cal})
    fitted = make_model(family, best["config"], seed, indicators, mask_only,
                        CONFIG["mlp_max_epochs"], CONFIG["mlp_patience"])
    fitted.fit(x, y)
    return {"pipeline": fitted, "calibrator": mapping, "selection": best, "trials": trials}


def predict_bundle(bundle, x):
    return calibrate(bundle["calibrator"], bundle["pipeline"].predict_proba(x)[:, 1])


def summarize(rows):
    frame = pd.DataFrame(rows)
    cols = ["average_precision", "roc_auc", "brier", "log_loss", "recall_at_10pct", "precision_at_10pct"]
    result = []
    for (schema, family), block in frame.groupby(["schema", "family"], sort=False):
        row = {"schema": schema, "family": family, "folds": len(block)}
        for col in cols:
            row[col] = float(block[col].mean())
            row[col + "_sd"] = float(block[col].std(ddof=1))
        result.append(row)
    return result


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(exist_ok=True)
    cache = ROOT / ".training-cache"
    cache.mkdir(exist_ok=True)
    x, y, groups, audit = load_data()
    seed = CONFIG["seed"]
    outer = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y, groups))
    fold_ids = np.empty(len(y), dtype=int)
    for fold, (train, valid) in enumerate(outer):
        assert not set(groups[train]) & set(groups[valid])
        fold_ids[valid] = fold
    pd.DataFrame({"row": np.arange(len(y)), "fold": fold_ids, "group": groups, "target": y}).to_csv(OUT / "folds.csv", index=False)
    rows, prediction_rows, selections = [], [], []
    fold_bundles = {"core13": [], "full64": []}
    families = ["logistic", "hgb", "mlp"]
    with threadpool_limits(limits=2):
        for schema in ["core13", "full64", "missingness_only", "full64_without_explicit_indicators"]:
            columns = CORE if schema == "core13" else list(x.columns)
            xx = x[columns]
            for fold, (tr, te) in enumerate(outer):
                inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed + fold + 1).split(xx.iloc[tr], y[tr], groups[tr]))
                candidates = []
                for family in families if schema in fold_bundles else ["hgb"]:
                    key = cache / f"v1-{schema}-{fold}-{family}.joblib"
                    if key.exists():
                        bundle = joblib.load(key)
                    else:
                        bundle = search(xx.iloc[tr], y[tr], groups[tr], inner, family, seed,
                                        schema != "full64_without_explicit_indicators", schema == "missingness_only")
                        joblib.dump(bundle, key, compress=3)
                    p = predict_bundle(bundle, xx.iloc[te])
                    raw = bundle["pipeline"].predict_proba(xx.iloc[te])[:, 1]
                    row = {"schema": schema, "family": family, "fold": fold,
                           **metrics(y[te], p), "raw_brier": float(brier_score_loss(y[te], raw)),
                           "calibration_used": bundle["selection"]["calibration_used"]}
                    rows.append(row)
                    prediction_rows.extend({"schema": schema, "family": family, "row": int(r), "fold": fold,
                                            "target": int(y[r]), "score": float(prob)} for r, prob in zip(te, p))
                    selections.append({"schema": schema, "fold": fold, **bundle["selection"], "trials": bundle["trials"]})
                    candidates.append((bundle["selection"]["inner_mean_ap"], bundle, p))
                    print(f"{schema} fold={fold} {family}: AP={row['average_precision']:.3f} Brier={row['brier']:.4f}", flush=True)
                    save_json(OUT / "progress.json", {"last": f"{schema}/{fold}/{family}", "completed": len(rows), "elapsed_seconds": time.perf_counter() - started})
                if schema in fold_bundles:
                    _, winner, p = max(candidates, key=lambda item: item[0])
                    # A demo case is evaluated by a model whose entire fitting/selection excluded its outer fold.
                    fold_bundles[schema].append(winner)
                    rows.append({"schema": schema, "family": "nested_selection", "fold": fold, **metrics(y[te], p)})
                    prediction_rows.extend({"schema": schema, "family": "nested_selection", "row": int(r), "fold": fold,
                                            "target": int(y[r]), "score": float(prob)} for r, prob in zip(te, p))
                baseline = np.full(len(te), y[tr].mean())
                rows.append({"schema": schema, "family": "training_prevalence", "fold": fold, **metrics(y[te], baseline)})
        # Full-data selection is separate from outer evaluation and is used only for the exported general research estimator.
        release = {}
        all_inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed + 100).split(x, y, groups))
        for schema in fold_bundles:
            cols = CORE if schema == "core13" else list(x.columns)
            options = []
            for family in families:
                key = cache / f"v1-final-{schema}-{family}.joblib"
                if key.exists():
                    bundle = joblib.load(key)
                else:
                    bundle = search(x[cols], y, groups, all_inner, family, seed)
                    joblib.dump(bundle, key, compress=3)
                options.append(bundle)
            final = max(options, key=lambda b: b["selection"]["inner_mean_ap"])
            release[schema] = {
                "columns": cols, "final": final, "folds": fold_bundles[schema],
                "median": x[cols].median().fillna(0).to_dict(),
                "low": x[cols].quantile(.005).fillna(0).to_dict(),
                "high": x[cols].quantile(.995).fillna(0).to_dict(),
            }
            print(f"release {schema}: {final['selection']['family']}", flush=True)
        # Seed sensitivity of a fixed neural configuration. No best-seed selection.
        seed_rows = []
        for neural_seed in CONFIG["mlp_seed_sensitivity"]:
            for fold, (tr, te) in enumerate(outer):
                key = cache / f"v1-seed-{neural_seed}-{fold}.joblib"
                if key.exists():
                    model = joblib.load(key)
                else:
                    model = make_model("mlp", {"dropout": .1, "balanced": False}, neural_seed)
                    model.fit(x[CORE].iloc[tr], y[tr])
                    joblib.dump(model, key, compress=3)
                seed_rows.append({"seed": neural_seed, "fold": fold, **metrics(y[te], model.predict_proba(x[CORE].iloc[te])[:, 1]),
                                  "export_max_error": model[-1].export_max_error_})
        pd.DataFrame(seed_rows).to_csv(OUT / "neural_seed_sensitivity.csv", index=False)

    predictions = pd.DataFrame(prediction_rows)
    pd.DataFrame(rows).to_csv(OUT / "fold_metrics.csv", index=False)
    predictions.to_csv(OUT / "oof_predictions.csv", index=False)
    save_json(OUT / "selection_log.json", selections)
    summary = summarize(rows)
    save_json(OUT / "metrics.json", {"evaluation": CONFIG["interpretation"], "aggregation": "mean and sample SD of 5 outer-fold metrics; SD is not a confidence interval", "results": summary})
    joblib.dump({"version": "1.0.0", "data_sha256": audit["source_sha256"], "schemas": release}, ART / "research_models.joblib", compress=3)

    demo = []
    rng = np.random.default_rng(seed)
    # Fixed random cases by known outcome for a demonstrative UI, not a prevalence sample.
    for label in [0, 1]:
        for r in rng.choice(np.flatnonzero(y == label), 3, replace=False):
            demo.append({"id": f"poland-{int(r):04d}", "row": int(r), "fold": int(fold_ids[r]),
                         "outcome": int(y[r]), "title": f"Historical case {len(demo) + 1}",
                         "features": {k: None if pd.isna(v) else float(v) for k, v in x.iloc[r].items()}})
    save_json(ART / "examples.json", demo)

    # Held-out permutation relevance: descriptive, uncalibrated pipeline; correlated ratios can share relevance.
    importance_rows = []
    with threadpool_limits(limits=2):
        for schema in release:
            cols = release[schema]["columns"]
            for fold, (_, te) in enumerate(outer):
                fitted = fold_bundles[schema][fold]["pipeline"]
                imp = permutation_importance(fitted, x[cols].iloc[te], y[te], scoring="average_precision", n_repeats=3, random_state=seed, n_jobs=1)
                for col, mean, sd in zip(cols, imp.importances_mean, imp.importances_std):
                    importance_rows.append({"schema": schema, "fold": fold, "feature": col, "ap_decrease": float(mean), "repeat_sd": float(sd)})
    pd.DataFrame(importance_rows).to_csv(OUT / "permutation_importance.csv", index=False)
    metadata = {
        "version": "1.0.0", "rag_status": "deferred", "source": audit,
        "protocol": CONFIG, "results": summary,
        "released_models": {k: v["final"]["selection"] for k, v in release.items()},
        "feature_labels": LABELS, "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["Research benchmark, not a validated Egyptian startup predictor", "Exact-vector grouping is not company grouping", "Prior exploratory analysis used this source", "Financial ratios do not identify causal effects", "Demo cases use outer-fold models, not the full-data release estimator"],
    }
    save_json(ART / "metadata.json", metadata)
    save_json(OUT / "release_manifest.json", {"artifact_sha256": digest(ART / "research_models.joblib"), "source_sha256": audit["source_sha256"], "complete": True})
    print(f"Complete in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
