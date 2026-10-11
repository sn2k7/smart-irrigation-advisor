"""Real-dataset workflow: check -> train -> report (new, self-contained module).

Used by ``scripts/train_real_dataset.py`` and the standalone desktop app
``real_dataset_gui.py``. It only *uses* the existing pipeline
(custom-dataset adapter, preprocessing, training) - nothing else changes.

Search for ``PLACEHOLDER(real-dataset)`` to find every value you may need
to adapt for your data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PROJECT_ROOT, get_settings
from .data.custom_dataset import load_custom_dataset, load_dataset_config
from .data.labels import TARGET_CLASS
from .data.preprocessing import data_quality_report, prepare_sensor_frame
from .exceptions import DataValidationError
from .ml.training import TrainingConfig, train_models

logger = logging.getLogger(__name__)

# PLACEHOLDER(real-dataset): path of the JSON config describing your real data.
DEFAULT_REAL_CONFIG = PROJECT_ROOT / "config" / "real_dataset.json"

# PLACEHOLDER(real-dataset): where the model trained on your real data is saved.
# The API loads settings.model_path (models/irrigation_model.joblib). Keep this
# equal to that path if the running API should use the real-data model, or
# choose another file to compare models first.
DEFAULT_REAL_MODEL = get_settings().model_path

# PLACEHOLDER(real-dataset): reports for the real dataset are kept apart from the synthetic ones.
DEFAULT_REAL_REPORTS = PROJECT_ROOT / "reports" / "real_dataset"

# PLACEHOLDER(real-dataset): minimum amount of data before training is allowed.
# These are conservative defaults for hourly data; adjust for your sampling rate.
MIN_ROWS = 24 * 14          # about two weeks of hourly readings
MIN_DAYS = 14               # covered calendar span
MIN_POSITIVE_RATE = 0.02    # at least some "irrigation needed" situations
MAX_POSITIVE_RATE = 0.98    # ...and some "not needed" situations


@dataclass
class CheckResult:
    """Outcome of :func:`check_real_dataset`."""

    ok: bool
    conversion: dict[str, Any]
    quality: dict[str, Any]
    problems: list[str] = field(default_factory=list)   # block training
    warnings: list[str] = field(default_factory=list)   # worth reading, do not block
    preview: pd.DataFrame | None = None
    data: pd.DataFrame | None = None


def check_real_dataset(config_path: str | Path = DEFAULT_REAL_CONFIG) -> CheckResult:
    """Load the real dataset through its config and judge whether it can be trained on.

    Raises:
        DataValidationError: config/file problems (missing file, unfilled placeholders,
            unknown columns, invalid units) - the message says what to fix.
    """
    config = load_dataset_config(config_path)
    df, conversion = load_custom_dataset(config)
    quality = data_quality_report(prepare_sensor_frame(df))

    problems: list[str] = []
    warnings: list[str] = list(conversion.get("warnings", []))
    span_days = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400.0
    if len(df) < MIN_ROWS:
        problems.append(f"Only {len(df)} rows - need at least {MIN_ROWS} (about two weeks of hourly data)")
    if span_days < MIN_DAYS:
        problems.append(f"Data covers {span_days:.1f} days - need at least {MIN_DAYS} days")
    if quality.get("sm_still_missing_pct", 0) > 30:
        problems.append(f"{quality['sm_still_missing_pct']}% of hours have no usable soil moisture after cleaning")

    interval = conversion.get("median_interval_minutes")
    if interval and interval > 180:
        warnings.append(f"Median interval is {interval:.0f} min; hourly (or finer) data works best")
    sm_low, sm_high = conversion["soil_moisture_range"]
    if sm_high > 60 or sm_low < 0:
        warnings.append(f"Soil moisture spans {sm_low}..{sm_high} %; out-of-range values will be flagged invalid. "
                        "Check units.soil_moisture / calibration")
    for col, share in conversion["missing_share"].items():
        if share > 0.5 and col != "soil_moisture":
            warnings.append(f"{col} is missing in {100 * share:.0f}% of rows")
    if config.labels_mode == "columns":
        rate = conversion.get("positive_rate", 0.0)
        if not MIN_POSITIVE_RATE <= rate <= MAX_POSITIVE_RATE:
            problems.append(f"Label column has {100 * rate:.1f}% 'irrigation needed' - both classes are required")
    else:
        warnings.append("labels.mode = 'derive': FAO-56 weak labels are used; the rule baseline partly "
                        "grades itself. Provide your own labels for an independent evaluation")
    if conversion["fields"] < 3:
        warnings.append("Fewer than 3 fields: the 'unseen fields' test set will be small or empty")

    preview_cols = [c for c in ("timestamp", "field_id", "soil_moisture", "temperature_c", "humidity_pct",
                                "rainfall_mm", "crop_type", "soil_type", TARGET_CLASS) if c in df.columns]
    return CheckResult(
        ok=not problems,
        conversion=conversion,
        quality=quality,
        problems=problems,
        warnings=warnings,
        preview=df[preview_cols].head(20),
        data=df,
    )


def train_real_dataset(
    config_path: str | Path = DEFAULT_REAL_CONFIG,
    fast: bool = False,
    model_path: str | Path | None = DEFAULT_REAL_MODEL,
    reports_dir: str | Path = DEFAULT_REAL_REPORTS,
    stride_hours: int = 1,
) -> dict[str, Any]:
    """Check, then train and evaluate on the real dataset.

    Args:
        config_path: dataset config JSON.
        fast: fewer trees (quick look); use False for the final model.
        model_path: where to save the bundle (``None`` = do not save).
        reports_dir: evaluation report / figures folder.
        stride_hours: use every n-th hour for fitting (PLACEHOLDER(real-dataset):
            keep 1 for real data unless the file is very large).

    Returns:
        Summary dict with selection, test metrics, artifact paths and the check result.
    """
    check = check_real_dataset(config_path)
    if not check.ok:
        raise DataValidationError("Dataset not ready for training: " + "; ".join(check.problems))
    result = train_models(
        check.data,
        TrainingConfig(fast=fast, train_stride_hours=stride_hours, data_source="real"),
        model_path=Path(model_path) if model_path else None,
        reports_dir=Path(reports_dir),
    )
    report = result.report
    return {
        "selected": report["selected"],
        "classification": {name: res.get("test_temporal") for name, res in report["classification"].items()},
        "classification_unseen": {name: res.get("test_unseen_fields") for name, res in report["classification"].items()},
        "regression": {name: res.get("test_temporal") for name, res in report["regression"].items()},
        "split": report["split"]["rows"],
        "disclaimer": report["disclaimer"],
        "artifacts": result.artifacts,
        "warnings": check.warnings,
    }
