"""Persisting and loading the trained model bundle.

A bundle is a plain dict saved with joblib:

``classifier`` / ``regressor``  fitted sklearn pipelines, or ``None`` when model
                                selection kept the FAO-56 rule for that task
``threshold``                   decision threshold chosen on validation data
``regression_target``           ``"depth_over_raw"``: the regressor predicts the
                                net depth divided by RAW (dimensionless)
``feature_columns``             ordered list of input columns
``feature_schema_version``      must match ``data.features.FEATURE_SCHEMA_VERSION``
``metadata``                    model names, training date, metrics summary

Loading refuses bundles built with a different feature schema - the engine
then falls back to its rule-based path instead of producing silently wrong
predictions.

Security note: joblib/pickle files execute code when loaded. Only load model
files you produced yourself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..data.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from ..exceptions import ModelNotAvailableError, ModelSchemaMismatchError

logger = logging.getLogger(__name__)


def save_bundle(bundle: dict[str, Any], path: Path) -> Path:
    """Save a model bundle (creates the folder)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    logger.info("Saved model bundle to %s", path)
    return path


@dataclass
class ModelPrediction:
    """Model outputs for one decision point."""

    probability: float
    irrigation_needed: bool
    net_depth_mm: float | None


class IrrigationModel:
    """Thin inference wrapper around a validated bundle."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        version = bundle.get("feature_schema_version")
        if version != FEATURE_SCHEMA_VERSION:
            raise ModelSchemaMismatchError(
                f"Model feature schema {version!r} != code schema {FEATURE_SCHEMA_VERSION!r}; retrain the model"
            )
        if list(bundle.get("feature_columns", [])) != FEATURE_COLUMNS:
            raise ModelSchemaMismatchError("Model feature columns differ from the current feature definition")
        self.classifier = bundle.get("classifier")
        self.regressor = bundle.get("regressor")
        if self.classifier is None and self.regressor is None:
            raise ModelSchemaMismatchError("Bundle contains neither a classifier nor a regressor")
        self.threshold = float(bundle.get("threshold", 0.5))
        self.regression_target = bundle.get("regression_target", "depth_over_raw")
        self.metadata: dict[str, Any] = bundle.get("metadata", {})

    @property
    def has_classifier(self) -> bool:
        return self.classifier is not None

    @property
    def has_regressor(self) -> bool:
        return self.regressor is not None

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    def predict(self, features: pd.DataFrame) -> list[ModelPrediction]:
        """Predict for each row of a ``build_features`` frame.

        Needs :data:`FEATURE_COLUMNS` plus ``raw_mm`` (to rescale the depth).
        Tasks without an ML component return NaN probability / ``None`` depth.
        """
        X = features[FEATURE_COLUMNS]
        prob = self.classifier.predict_proba(X)[:, 1] if self.has_classifier else np.full(len(X), np.nan)
        depth = np.full(len(X), np.nan)
        if self.has_regressor:
            depth = self.regressor.predict(X) * features["raw_mm"].to_numpy()
        return [
            ModelPrediction(
                probability=float(p),
                irrigation_needed=bool(np.isfinite(p) and p >= self.threshold),
                net_depth_mm=None if not np.isfinite(d) else float(max(0.0, d)),
            )
            for p, d in zip(prob, depth)
        ]

    def info(self) -> dict[str, Any]:
        """Metadata exposed by the API."""
        return {"threshold": self.threshold, **self.metadata}


def load_model(path: Path) -> IrrigationModel:
    """Load and validate a bundle. Raises :class:`ModelNotAvailableError` if absent."""
    if not path.exists():
        raise ModelNotAvailableError(f"No model file at {path}. Run scripts/train_models.py")
    bundle = joblib.load(path)
    model = IrrigationModel(bundle)
    logger.info("Loaded model %s from %s", model.version, path)
    return model


def try_load_model(path: Path) -> IrrigationModel | None:
    """Load a model or return ``None`` (logged) so the app can run rule-based only."""
    try:
        return load_model(path)
    except (ModelNotAvailableError, ModelSchemaMismatchError) as exc:
        logger.warning("Model unavailable - using rule-based engine only: %s", exc)
    except Exception as exc:  # corrupted file, incompatible sklearn version, ...
        logger.error("Failed to load model from %s: %s", path, exc)
    return None
