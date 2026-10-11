"""End-to-end training workflow.

    raw readings -> prepare_sensor_frame -> build_features -> split
    (held-out fields, chronological test period, rolling-origin CV folds)
    -> out-of-fold (OOF) predictions for every candidate incl. the FAO-56 rule
    -> select decision model + threshold and depth model on OOF metrics
    -> refit the candidates on the whole development period
    -> evaluate ONCE on the temporal and unseen-field test sets -> save.

The transparent FAO-56 rule is a first-class candidate: an ML component is
only deployed if it beats the rule on the cross-validated development data.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .. import __version__
from ..data.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, NUMERIC_FEATURES, build_features
from ..data.labels import HORIZON_HOURS, TARGET_CLASS, TARGET_REG, derive_labels_from_observations
from ..data.loader import has_labels
from ..data.preprocessing import data_quality_report, prepare_sensor_frame
from ..exceptions import DataValidationError
from . import plots
from .evaluation import (
    SYNTHETIC_DISCLAIMER,
    best_threshold,
    classification_metrics,
    regression_metrics,
    rule_baseline_predictions,
    rule_net_depth,
    save_reports,
)
from .model_store import save_bundle
from .pipelines import build_classifiers, build_regressors
from .splitting import DataSplit, split_dataset

logger = logging.getLogger(__name__)

RULE = "rule_based_fao56"
#: ML depth model must beat the FAO-56 formula's cross-validated MAE by this share.
DEPTH_PARSIMONY_MARGIN = 0.05
#: Minimum rows with a known depth needed to train/evaluate regressors.
MIN_DEPTH_ROWS = 20


@dataclass
class TrainingConfig:
    """Hyper-parameters of the training run."""

    random_state: int = 42
    test_fraction: float = 0.25
    n_folds: int = 3
    embargo_hours: int = HORIZON_HOURS
    holdout_field_fraction: float = 0.2
    fast: bool = False
    #: Use every n-th hour of TRAINING rows (speeds up fitting; evaluation uses all rows).
    train_stride_hours: int = 1
    importance_sample: int = 3000
    data_source: str = "synthetic"


@dataclass
class TrainingResult:
    """Everything produced by :func:`train_models`."""

    bundle: dict[str, Any]
    report: dict[str, Any]
    split: DataSplit
    artifacts: dict[str, str] = field(default_factory=dict)


def build_training_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean raw readings and compute features + targets.

    If the raw data has no targets (external CSV) weak labels are derived from
    the observations; otherwise the simulator's ground-truth labels are used.
    """
    labelled = has_labels(raw)
    passthrough = [TARGET_CLASS, TARGET_REG] if labelled else []
    prepared = prepare_sensor_frame(raw, passthrough=passthrough)
    quality = data_quality_report(prepared)
    feats = build_features(prepared, passthrough=passthrough)
    if not labelled:
        logger.info("Dataset has no labels - deriving weak labels from observations")
        feats = derive_labels_from_observations(feats)
    before = len(feats)
    # Depth may be unknown for some rows (e.g. a custom dataset with only yes/no labels);
    # those rows still train the classifier and are skipped by the regressors.
    feats = feats.dropna(subset=[TARGET_CLASS]).reset_index(drop=True)
    feats[TARGET_CLASS] = feats[TARGET_CLASS].astype(int)
    logger.info("Training frame: %d rows (%d dropped without complete labels)", len(feats), before - len(feats))
    if feats[TARGET_CLASS].nunique() < 2:
        raise DataValidationError("Training data contains only one class - cannot train a classifier")
    return feats, quality


def _subsample(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    return df if stride <= 1 else df[df["timestamp"].dt.hour % stride == 0]


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    sample = pd.util.hash_pandas_object(df[["field_id", "timestamp", "depletion_ratio"]], index=False)
    return hashlib.sha256(sample.to_numpy().tobytes()).hexdigest()[:16]


def _depth(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """Regressors predict depth / RAW; rescale to mm."""
    return np.clip(model.predict(frame[FEATURE_COLUMNS]) * frame["raw_mm"].to_numpy(), 0, None)


def _depth_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for depth regression: irrigation needed and depth known."""
    return frame[(frame[TARGET_CLASS] == 1) & frame[TARGET_REG].notna()]


def _fit_regressor(model: Any, frame: pd.DataFrame) -> Any:
    pos = _depth_rows(frame)
    return model.fit(pos[FEATURE_COLUMNS], (pos[TARGET_REG] / pos["raw_mm"]).to_numpy())


def _permutation_importance(model: Any, X: pd.DataFrame, y: np.ndarray, n: int, seed: int) -> list[dict[str, Any]]:
    if len(X) > n:
        idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
        X, y = X.iloc[idx], y[idx]
    result = permutation_importance(model, X, y, scoring="f1", n_repeats=3, random_state=seed, n_jobs=1)
    order = np.argsort(result.importances_mean)[::-1]
    return [
        {"feature": X.columns[i], "importance": round(float(result.importances_mean[i]), 5),
         "std": round(float(result.importances_std[i]), 5)}
        for i in order
    ]


def _cross_validate(split: DataSplit, config: TrainingConfig) -> tuple[dict, dict, dict, pd.DataFrame]:
    """Out-of-fold predictions for every candidate (classifiers, regressors, rule)."""
    dev = split.development
    oof = dev.loc[np.concatenate([v for _, v in split.folds])]
    cls_prob = {name: np.empty(len(oof)) for name in build_classifiers()}
    reg_pred = {name: np.empty(len(oof)) for name in build_regressors()}
    offset = 0
    for k, (train_idx, val_idx) in enumerate(split.folds, start=1):
        train = _subsample(dev.loc[train_idx], config.train_stride_hours)
        val = dev.loc[val_idx]
        rows = slice(offset, offset + len(val))
        if train[TARGET_CLASS].nunique() < 2:
            raise DataValidationError(f"CV fold {k} training data is single-class; supply more data")
        for name, pipe in build_classifiers(config.random_state, config.fast).items():
            logger.info("CV fold %d/%d: classifier %s on %d rows", k, len(split.folds), name, len(train))
            pipe.fit(train[FEATURE_COLUMNS], train[TARGET_CLASS])
            cls_prob[name][rows] = pipe.predict_proba(val[FEATURE_COLUMNS])[:, 1]
        if len(_depth_rows(train)) >= MIN_DEPTH_ROWS:
            for name, pipe in build_regressors(config.random_state, config.fast).items():
                _fit_regressor(pipe, train)
                reg_pred[name][rows] = _depth(pipe, val)
        else:
            for name in reg_pred:
                reg_pred[name][rows] = np.nan
        offset += len(val)

    y = oof[TARGET_CLASS].to_numpy()
    cls_cv: dict[str, Any] = {name: classification_metrics(y, prob, best_threshold(y, prob)) for name, prob in cls_prob.items()}
    rule_decision, _ = rule_baseline_predictions(oof)
    cls_cv[RULE] = classification_metrics(y, rule_decision.astype(float), 0.5)

    pos = (y == 1) & oof[TARGET_REG].notna().to_numpy()
    y_depth = oof[TARGET_REG].to_numpy()[pos]
    reg_cv: dict[str, Any] = {}
    for name, pred in reg_pred.items():
        ok = np.isfinite(pred[pos])
        reg_cv[name] = regression_metrics(y_depth[ok], pred[pos][ok]) if ok.sum() >= MIN_DEPTH_ROWS else \
            regression_metrics(np.array([]), np.array([]))
    reg_cv[RULE] = regression_metrics(y_depth, rule_net_depth(oof)[pos])
    return cls_cv, reg_cv, {"rows": len(oof), "positive_rate": round(float(y.mean()), 4)}, oof


def train_models(
    raw: pd.DataFrame,
    config: TrainingConfig | None = None,
    model_path: Path | None = None,
    reports_dir: Path | None = None,
) -> TrainingResult:
    """Train, select, evaluate and (optionally) persist the models."""
    config = config or TrainingConfig()
    feats, quality = build_training_frame(raw)
    split = split_dataset(
        feats,
        test_fraction=config.test_fraction,
        n_folds=config.n_folds,
        embargo_hours=config.embargo_hours,
        holdout_field_fraction=config.holdout_field_fraction,
        random_state=config.random_state,
    )
    if not split.folds:
        raise DataValidationError("Not enough data to build cross-validation folds")
    for name in ("development", "test_temporal"):
        part = getattr(split, name)
        if part.empty or part[TARGET_CLASS].nunique() < 2:
            raise DataValidationError(f"Split '{name}' is empty or single-class; supply more data")
    logger.info("Split: %s", split.info["rows"])

    # ---------------- model selection on out-of-fold predictions -------------
    cls_cv, reg_cv, cv_info, oof = _cross_validate(split, config)
    notes: list[str] = []
    best_ml = max((n for n in cls_cv if n != RULE), key=lambda n: (cls_cv[n]["f1"], cls_cv[n]["recall"]))
    if cls_cv[best_ml]["f1"] > cls_cv[RULE]["f1"]:
        decision_model = best_ml
        notes.append(
            f"{best_ml} beat the FAO-56 rule on cross-validated F1 ({cls_cv[best_ml]['f1']:.3f} vs {cls_cv[RULE]['f1']:.3f})."
        )
    else:
        decision_model = RULE
        notes.append(f"No ML classifier beat the FAO-56 rule on cross-validated F1 ({cls_cv[RULE]['f1']:.3f}); the rule is kept.")
    threshold = cls_cv[best_ml]["threshold"] if decision_model != RULE else 0.5

    scored = [n for n in reg_cv if n != RULE and reg_cv[n]["mae_mm"] is not None]
    best_ml_reg = min(scored, key=lambda n: reg_cv[n]["mae_mm"]) if scored else "none"
    rule_mae = reg_cv[RULE]["mae_mm"]
    ml_mae = reg_cv[best_ml_reg]["mae_mm"] if scored else None
    # Parsimony: depth errors cost water directly and the formula is fully explainable.
    if not scored or rule_mae is None:
        depth_model = RULE
        notes.append("Too few rows with known irrigation depth - the FAO-56 formula is used for depth.")
    elif ml_mae < (1 - DEPTH_PARSIMONY_MARGIN) * rule_mae:
        depth_model = best_ml_reg
        notes.append(f"{best_ml_reg} beat the FAO-56 depth formula on cross-validated MAE ({ml_mae:.2f} vs {rule_mae:.2f} mm).")
    else:
        depth_model = RULE
        notes.append(
            f"Best ML regressor ({best_ml_reg}, CV MAE {ml_mae:.2f} mm) did not beat the FAO-56 formula "
            f"({rule_mae:.2f} mm) by the {100 * DEPTH_PARSIMONY_MARGIN:.0f}% margin; the formula is kept for depth."
        )
    for note in notes:
        logger.info("Selection: %s", note)

    # ---------------- refit every candidate on the full development period ----
    dev_train = _subsample(split.development, config.train_stride_hours)
    fitted: dict[str, Any] = {}
    for name, pipe in build_classifiers(config.random_state, config.fast).items():
        logger.info("Final fit: classifier %s on %d rows", name, len(dev_train))
        fitted[name] = pipe.fit(dev_train[FEATURE_COLUMNS], dev_train[TARGET_CLASS])
    reg_fitted = (
        {name: _fit_regressor(pipe, dev_train) for name, pipe in build_regressors(config.random_state, config.fast).items()}
        if len(_depth_rows(dev_train)) >= MIN_DEPTH_ROWS else {}
    )

    # ---------------- single evaluation on the test sets ---------------------
    eval_sets = {"test_temporal": split.test_temporal, "test_unseen_fields": split.test_unseen_fields}
    classification: dict[str, Any] = {}
    for name, model in fitted.items():
        entry = {"cross_validation": cls_cv[name]}
        for set_name, frame in eval_sets.items():
            if len(frame):
                prob = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
                entry[set_name] = classification_metrics(frame[TARGET_CLASS].to_numpy(), prob, cls_cv[name]["threshold"])
        classification[name] = entry
    classification[RULE] = {"cross_validation": cls_cv[RULE]}
    for set_name, frame in eval_sets.items():
        if len(frame):
            decision, _ = rule_baseline_predictions(frame)
            classification[RULE][set_name] = classification_metrics(
                frame[TARGET_CLASS].to_numpy(), decision.astype(float), 0.5
            )

    regression: dict[str, Any] = {}
    for name in [*reg_fitted, RULE]:
        entry = {"cross_validation": reg_cv[name]}
        for set_name, frame in eval_sets.items():
            pos = _depth_rows(frame)
            if not len(pos):
                entry[set_name] = regression_metrics(np.array([]), np.array([]))
                continue
            pred = rule_net_depth(pos) if name == RULE else _depth(reg_fitted[name], pos)
            entry[set_name] = regression_metrics(pos[TARGET_REG].to_numpy(), pred)
        regression[name] = entry

    # Importance of the best ML classifier, measured on out-of-fold rows.
    importance = _permutation_importance(
        fitted[best_ml], oof[FEATURE_COLUMNS], oof[TARGET_CLASS].to_numpy(), config.importance_sample, config.random_state
    )

    trained_at = datetime.now().isoformat(timespec="seconds")
    model_version = f"{decision_model}+{depth_model}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    disclaimer = (
        SYNTHETIC_DISCLAIMER if config.data_source == "synthetic"
        else "Metrics computed on user-supplied data. If labels were derived from the observations "
        "(labels.mode = 'derive'), they are FAO-56 weak labels and the rule baseline is partly scored "
        "against its own definition."
    )
    report: dict[str, Any] = {
        "generated_at": trained_at,
        "disclaimer": disclaimer,
        "data_source": config.data_source,
        "package_version": __version__,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "label_horizon_hours": HORIZON_HOURS,
        "dataset": {
            "rows": len(feats),
            "fields": int(feats["field_id"].nunique()),
            "positive_rate": round(float(feats[TARGET_CLASS].mean()), 4),
            "fingerprint": _dataset_fingerprint(feats),
            "quality": quality,
        },
        "split": {**split.info, "cv_out_of_fold": cv_info},
        "config": dict(config.__dict__),
        "classification": classification,
        "regression": regression,
        "selected": {
            "classifier": decision_model,
            "regressor": depth_model,
            "best_ml_classifier": best_ml,
            "best_ml_regressor": best_ml_reg,
            "threshold": threshold,
            "depth_parsimony_margin_pct": round(100 * DEPTH_PARSIMONY_MARGIN),
            "model_version": model_version,
            "notes": " ".join(notes),
        },
        "feature_importance": importance,
    }
    bundle = {
        "classifier": fitted.get(decision_model),
        "regressor": reg_fitted.get(depth_model),
        "regression_target": "depth_over_raw",
        "threshold": threshold,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "metadata": {
            "model_version": model_version,
            "classifier": decision_model,
            "regressor": depth_model,
            "trained_at": trained_at,
            "data_source": config.data_source,
            "dataset_fingerprint": report["dataset"]["fingerprint"],
            "selection_notes": report["selected"]["notes"],
            "test_temporal": {
                "classification": classification[decision_model].get("test_temporal"),
                "regression": regression[depth_model].get("test_temporal"),
            },
            "disclaimer": disclaimer,
        },
    }

    result = TrainingResult(bundle=bundle, report=report, split=split)
    if model_path is not None:
        if bundle["classifier"] is None and bundle["regressor"] is None:
            logger.warning("The FAO-56 rule won both tasks - no model file written; the engine runs rule-based")
        else:
            result.artifacts["model"] = str(save_bundle(bundle, model_path))
    if reports_dir is not None:
        result.artifacts.update(save_reports(report, reports_dir))
        result.artifacts.update(_write_plots(result, reg_fitted.get(depth_model), reports_dir))
    return result


def _write_plots(result: TrainingResult, regressor: Any, reports_dir: Path) -> dict[str, str]:
    report = result.report
    out = {
        "confusion_temporal": plots.plot_confusion_matrices(
            report["classification"], "test_temporal", reports_dir / "confusion_matrices_test_temporal.png"),
        "model_comparison": plots.plot_model_comparison(report["classification"], reports_dir / "model_comparison.png"),
        "feature_importance": plots.plot_feature_importance(
            report["feature_importance"], reports_dir / "feature_importance.png"),
    }
    if len(result.split.test_unseen_fields):
        out["confusion_unseen"] = plots.plot_confusion_matrices(
            report["classification"], "test_unseen_fields", reports_dir / "confusion_matrices_unseen_fields.png")
    pos = _depth_rows(result.split.test_temporal)
    if len(pos):
        pred = _depth(regressor, pos) if regressor is not None else rule_net_depth(pos)
        out["regression_scatter"] = plots.plot_regression_scatter(
            pos[TARGET_REG].to_numpy(), pred, report["selected"]["regressor"],
            reports_dir / "regression_depth_scatter.png")
    return out
