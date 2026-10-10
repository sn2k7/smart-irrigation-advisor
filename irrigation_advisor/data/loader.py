"""Loading and saving datasets (synthetic or external CSV).

External CSVs (e.g. exported from an ESP32 logger, ThingSpeak or a research
dataset) are mapped onto the internal schema via :data:`COLUMN_ALIASES`.
Field metadata missing from the file can be supplied as defaults
(``crop_type``, ``soil_type``, ...). If the file has no training labels,
:func:`load_external_csv` can derive weak labels from the observations
(see ``data.labels``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..agronomy.crops import CROPS, IRRIGATION_METHODS, SOILS
from ..exceptions import DataValidationError
from .labels import TARGET_CLASS, TARGET_REG
from .preprocessing import REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

#: Lower-case alias -> internal column name.
COLUMN_ALIASES: dict[str, str] = {
    "time": "timestamp",
    "datetime": "timestamp",
    "date_time": "timestamp",
    "created_at": "timestamp",
    "ts": "timestamp",
    "field": "field_id",
    "plot_id": "field_id",
    "farm_id": "field_id",
    "device_id": "field_id",
    "moisture": "soil_moisture",
    "soil_moisture_pct": "soil_moisture",
    "soilmoisture": "soil_moisture",
    "vwc": "soil_moisture",
    "sm": "soil_moisture",
    "temp": "temperature_c",
    "temperature": "temperature_c",
    "air_temperature": "temperature_c",
    "humidity": "humidity_pct",
    "rh": "humidity_pct",
    "relative_humidity": "humidity_pct",
    "rain": "rainfall_mm",
    "rainfall": "rainfall_mm",
    "precipitation": "rainfall_mm",
    "precip_mm": "rainfall_mm",
    "crop": "crop_type",
    "soil": "soil_type",
    "method": "irrigation_method",
    "sowing_date": "planting_date",
    "lat": "latitude",
}

DEFAULT_METADATA: dict[str, Any] = {
    "crop_type": "maize",
    "soil_type": "loam",
    "irrigation_method": "drip",
    "latitude": 18.5,
}


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case/strip headers and apply :data:`COLUMN_ALIASES`."""
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        renamed[col] = COLUMN_ALIASES.get(key, key)
    out = df.rename(columns=renamed)
    duplicated = out.columns[out.columns.duplicated()].tolist()
    if duplicated:
        raise DataValidationError(f"Several input columns map to the same name: {duplicated}")
    return out


def _validate_categories(df: pd.DataFrame) -> None:
    for col, allowed in (("crop_type", CROPS), ("soil_type", SOILS), ("irrigation_method", IRRIGATION_METHODS)):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            unknown = sorted(set(df[col].unique()) - set(allowed))
            if unknown:
                raise DataValidationError(f"Unknown {col} values {unknown}; allowed: {sorted(allowed)}")


def load_external_csv(path: str | Path, defaults: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load an external CSV and map it to the internal raw-reading schema.

    Args:
        path: CSV file path.
        defaults: Metadata values used when a column is missing. If
            ``planting_date`` is missing it defaults to 30 days before the
            first timestamp of each field (documented assumption).

    Returns:
        Raw-reading DataFrame ready for ``prepare_sensor_frame``.
    """
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"CSV file not found: {path}")
    df = standardise_columns(pd.read_csv(path))
    if "field_id" not in df.columns:
        df["field_id"] = "external-1"
    if "rainfall_mm" not in df.columns:
        logger.warning("No rainfall column in %s - assuming 0 mm (rain awareness reduced)", path.name)
        df["rainfall_mm"] = 0.0
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"{path.name}: missing required columns {missing}")

    meta = {**DEFAULT_METADATA, **(defaults or {})}
    for key, value in meta.items():
        if key not in df.columns:
            df[key] = value
        else:
            df[key] = df[key].fillna(value)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    if "planting_date" not in df.columns:
        first = df.groupby("field_id")["timestamp"].transform("min")
        df["planting_date"] = (first - pd.Timedelta(days=30)).dt.date.astype(str)
        logger.warning("planting_date missing - assuming 30 days before first reading")
    _validate_categories(df)
    logger.info("Loaded %d rows / %d fields from %s", len(df), df["field_id"].nunique(), path)
    return df


def save_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a dataset as CSV (creates parent folders)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset previously written by :func:`save_dataset`."""
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"Dataset not found: {path}. Run scripts/generate_dataset.py first.")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df


def has_labels(df: pd.DataFrame) -> bool:
    """True if both training targets are present."""
    return TARGET_CLASS in df.columns and TARGET_REG in df.columns
