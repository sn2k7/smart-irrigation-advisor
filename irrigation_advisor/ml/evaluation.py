"""Metric computation and report writing.

All numbers written to ``reports/`` are computed here from model predictions
on held-out data; nothing is hand-entered. Reports state explicitly that
they were obtained on SYNTHETIC data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ..utils import write_json

SYNTHETIC_DISCLAIMER = (
    "All metrics were computed on SYNTHETIC data produced by this project's simulator. "
    "They demonstrate that the pipeline works and allow strategies to be compared under "
    "controlled assumptions; they are NOT evidence of real-world accuracy or water savings, "
    "which require validation with field data."
)


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Accuracy, precision, recall, F1, ROC-AUC and confusion matrix."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    y_pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else None
    except ValueError:
        auc = None
    return {
        "n": int(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 4) if len(y_true) else None,
        "threshold": round(float(threshold), 3),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": None if auc is None else round(auc, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """MAE, RMSE and R2 (in mm of net irrigation depth)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"n": 0, "mae_mm": None, "rmse_mm": None, "r2": None}
    return {
        "n": int(len(y_true)),
        "mae_mm": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse_mm": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4) if len(y_true) > 1 else None,
    }


def best_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Decision threshold maximising F1 on the *validation* set."""
    y_true = np.asarray(y_true).astype(int)
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.10, 0.91, 0.025):
        f1 = f1_score(y_true, (prob >= t).astype(int), zero_division=0)
        if f1 > best_f1 + 1e-9:
            best_t, best_f1 = float(t), f1
    return round(best_t, 3)


def rule_net_depth(features: pd.DataFrame) -> np.ndarray:
    """FAO-56 refill formula (``net_irrigation_requirement``) on a feature frame."""
    dr = features["depletion_mm"].clip(lower=0).fillna(0.0)
    etc = features["etc_24h_mm"].fillna(0.0)
    eff = features["forecast_effective_rain_mm"].fillna(0.0)
    return np.minimum(np.maximum(dr - eff, etc), features["taw_mm"]).to_numpy()


def rule_baseline_predictions(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Physics-only rule evaluated on the same features (the engine's rule path).

    Returns:
        (binary decision, net depth mm) arrays.
    """
    dr = features["depletion_mm"].clip(lower=0).fillna(0.0)
    raw = features["raw_mm"]
    etc = features["etc_24h_mm"].fillna(0.0)
    eff = features["forecast_effective_rain_mm"].fillna(0.0)
    needed = ((dr >= raw) | (dr + etc - eff >= raw)).astype(int).to_numpy()
    return needed, np.where(needed == 1, rule_net_depth(features), 0.0)


def write_markdown_report(report: dict[str, Any], path: Path) -> Path:
    """Human-readable Markdown version of the training/evaluation report."""
    lines = [
        "# Model Evaluation Report",
        "",
        f"> **Synthetic data notice:** {SYNTHETIC_DISCLAIMER}",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Feature schema version: {report['feature_schema_version']}",
        f"- Rows after cleaning: {report['dataset']['rows']} from {report['dataset']['fields']} fields",
        f"- Split: {report['split']['method']} (embargo {report['split']['embargo_hours']} h, "
        f"{report['split']['cv_folds']} CV folds, {report['split']['cv_out_of_fold']['rows']} out-of-fold rows)",
        f"- Held-out fields: {', '.join(report['split']['holdout_fields']) or 'none'}",
        f"- Irrigation decision model: **{report['selected']['classifier']}** "
        f"(threshold {report['selected']['threshold']}; chosen on cross-validated F1, the FAO-56 rule is a candidate)",
        f"- Depth model: **{report['selected']['regressor']}** (chosen on cross-validated MAE; ML must beat the "
        f"FAO-56 formula by >= {report['selected']['depth_parsimony_margin_pct']}%)",
        "",
        f"Selection notes: {report['selected']['notes']}",
        "",
        "## Classification - temporal test set (later period, seen fields)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def _cls_rows(section: str) -> list[str]:
        rows = []
        for name, res in report["classification"].items():
            m = res[section]
            cm = m["confusion_matrix"]
            auc = "-" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
            rows.append(
                f"| {name} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
                f"{auc} | {cm['tn']} | {cm['fp']} | {cm['fn']} | {cm['tp']} |"
            )
        return rows

    lines += _cls_rows("test_temporal")
    cv_lines = [
        "",
        "## Classification - rolling-origin cross-validation (used for model selection)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    cv_lines += _cls_rows("cross_validation")
    lines += [
        "",
        "## Classification - unseen fields (never used for training)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += _cls_rows("test_unseen_fields")
    lines += cv_lines
    lines += [
        "",
        "`rule_based_fao56` is the transparent physics rule used by the engine when no model is",
        "available; it is shown as a baseline the ML models must beat.",
        "",
        "## Regression - net irrigation depth (mm), rows where irrigation was truly needed",
        "",
        "| Model | CV MAE | Test MAE | Test RMSE | Test R2 | Unseen MAE | Unseen RMSE | Unseen R2 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, res in report["regression"].items():
        c, t, u = res["cross_validation"], res["test_temporal"], res["test_unseen_fields"]
        lines.append(
            f"| {name} | {c['mae_mm']} | {t['mae_mm']} | {t['rmse_mm']} | {t['r2']} | {u['mae_mm']} | {u['rmse_mm']} | {u['r2']} |"
        )
    lines += [
        "",
        f"## Most influential features ({report['selected']['best_ml_classifier']}, permutation importance, "
        "out-of-fold rows)",
        "",
    ]
    for item in report.get("feature_importance", [])[:15]:
        lines.append(f"- `{item['feature']}`: {item['importance']:.4f}")
    lines += ["", "## Data quality", ""]
    for key, value in report["dataset"]["quality"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_reports(report: dict[str, Any], reports_dir: Path) -> dict[str, str]:
    """Write JSON + Markdown reports, return their paths."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(report, reports_dir / "evaluation_report.json")
    md_path = write_markdown_report(report, reports_dir / "evaluation_report.md")
    return {"json": str(json_path), "markdown": str(md_path)}
