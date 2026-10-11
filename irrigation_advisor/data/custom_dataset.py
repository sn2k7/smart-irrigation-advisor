"""Adapter for YOUR OWN dataset (config-driven, no code changes needed for most files).

How to plug in a different dataset
----------------------------------
1. Copy ``config/dataset_config.example.json`` to e.g. ``config/my_dataset.json``.
2. Fill in the placeholders marked ``<...>``: file path, your column names,
   units and field metadata (crop, soil, irrigation method, planting date).
3. If your data needs special handling (merging files, odd encodings,
   sensor-specific calibration, filtering), add it to
   ``irrigation_advisor/data/dataset_hooks.py`` -> ``custom_transform``.
4. Check the mapping:   ``python scripts/prepare_custom_dataset.py --config config/my_dataset.json``
5. Train on it:         ``python scripts/train_models.py --dataset-config config/my_dataset.json``

The adapter converts your data into the internal raw-reading schema
(``timestamp, field_id, soil_moisture [% VWC], temperature_c, humidity_pct,
rainfall_mm`` + field metadata). From there the normal, shared
preprocessing / feature / training pipeline is used unchanged, so models
trained on your data behave exactly like the built-in ones at inference.

Labels: either point ``labels.mode = "columns"`` at your own target columns
or use ``"derive"`` (default) to compute FAO-56 weak labels from the
observations (see ``data/labels.py``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..agronomy.crops import CROPS, IRRIGATION_METHODS, SOILS
from ..config import PROJECT_ROOT
from ..exceptions import DataValidationError
from . import dataset_hooks
from .labels import TARGET_CLASS, TARGET_REG

logger = logging.getLogger(__name__)

#: Internal columns a dataset must provide (directly or via defaults).
REQUIRED_INTERNAL = ("timestamp", "soil_moisture")
#: Internal columns that may be mapped from your file.
MAPPABLE_COLUMNS = (
    "timestamp", "field_id", "soil_moisture", "temperature_c", "humidity_pct", "rainfall_mm",
    "crop_type", "soil_type", "irrigation_method", "planting_date", "latitude",
    TARGET_CLASS, TARGET_REG,
)
METADATA_KEYS = ("crop_type", "soil_type", "irrigation_method", "planting_date", "latitude")
PLACEHOLDER_PREFIX = "<"

SOIL_MOISTURE_UNITS = ("percent", "fraction", "raw")
TEMPERATURE_UNITS = ("celsius", "fahrenheit", "kelvin")
RAINFALL_UNITS = ("mm", "inch")
RAINFALL_MODES = ("interval", "cumulative")


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith(PLACEHOLDER_PREFIX)


@dataclass
class DatasetConfig:
    """Everything needed to read a foreign dataset. Mirrors the JSON template."""

    name: str
    path: Path
    delimiter: str = ","
    encoding: str = "utf-8"
    #: internal column name -> column name in YOUR file (omit what you don't have)
    column_map: dict[str, str] = field(default_factory=dict)
    timestamp_format: str | None = None  # e.g. "%d/%m/%Y %H:%M"; None = auto-detect
    dayfirst: bool = False
    #: hours to ADD to your timestamps to get field-local time (e.g. UTC data in India: 5.5)
    timestamp_offset_hours: float = 0.0
    soil_moisture_unit: str = "percent"
    #: only for soil_moisture_unit == "raw": two-point probe calibration
    soil_raw_calibration: dict[str, float] = field(default_factory=dict)
    temperature_unit: str = "celsius"
    rainfall_unit: str = "mm"
    rainfall_mode: str = "interval"  # "cumulative" = running total that is differenced per field
    default_field_id: str = "field-1"
    #: metadata applied to every field, overridden per field
    field_defaults: dict[str, Any] = field(default_factory=dict)
    per_field_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    labels_mode: str = "derive"  # "derive" | "columns"
    drop_columns: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)  # free space for your hooks

    def __post_init__(self) -> None:
        checks = [
            ("soil_moisture_unit", self.soil_moisture_unit, SOIL_MOISTURE_UNITS),
            ("temperature_unit", self.temperature_unit, TEMPERATURE_UNITS),
            ("rainfall_unit", self.rainfall_unit, RAINFALL_UNITS),
            ("rainfall_mode", self.rainfall_mode, RAINFALL_MODES),
            ("labels_mode", self.labels_mode, ("derive", "columns")),
        ]
        for name, value, allowed in checks:
            if value not in allowed:
                raise DataValidationError(f"Dataset config: {name}={value!r} not in {allowed}")
        unknown = sorted(set(self.column_map) - set(MAPPABLE_COLUMNS))
        if unknown:
            raise DataValidationError(f"Dataset config: unknown internal columns in column_map: {unknown}")
        # PLACEHOLDER(real-dataset): values still looking like "<...>" come from the template.
        placeholders = [f"column_map.{k}" for k, v in self.column_map.items() if _is_placeholder(v)]
        placeholders += [f"fields.defaults.{k}" for k, v in self.field_defaults.items() if _is_placeholder(v)]
        for fid, meta in self.per_field_metadata.items():
            if _is_placeholder(fid):
                placeholders.append(f"fields.per_field.{fid}")
            placeholders += [f"fields.per_field.{fid}.{k}" for k, v in meta.items() if _is_placeholder(v)]
        if placeholders:
            raise DataValidationError(
                "Dataset config still contains template placeholders - fill in (or delete): "
                + ", ".join(placeholders)
            )
        if self.soil_moisture_unit == "raw":
            missing = {"dry_raw", "wet_raw", "dry_pct", "wet_pct"} - set(self.soil_raw_calibration)
            if missing:
                raise DataValidationError(f"Dataset config: soil_raw_calibration needs {sorted(missing)}")
        if self.labels_mode == "columns" and TARGET_CLASS not in self.column_map:
            raise DataValidationError(f"labels_mode='columns' requires column_map['{TARGET_CLASS}']")


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Read a JSON dataset config. Relative data paths resolve from the project root."""
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"Dataset config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}  # "_comment" keys allowed
    if "path" not in raw or str(raw["path"]).startswith(PLACEHOLDER_PREFIX):
        raise DataValidationError("Dataset config: set 'path' to your data file")
    data_path = Path(raw.pop("path"))
    if not data_path.is_absolute():
        data_path = (PROJECT_ROOT / data_path).resolve()
    labels = raw.pop("labels", {}) or {}
    raw["labels_mode"] = labels.get("mode", raw.get("labels_mode", "derive"))
    fields_cfg = raw.pop("fields", {}) or {}
    raw.setdefault("field_defaults", fields_cfg.get("defaults", {}))
    raw.setdefault("per_field_metadata", fields_cfg.get("per_field", {}))
    units = raw.pop("units", {}) or {}
    for key in ("soil_moisture_unit", "temperature_unit", "rainfall_unit", "rainfall_mode"):
        short = key.replace("_unit", "")
        if short in units:
            raw[key] = units[short]
        elif key in units:
            raw[key] = units[key]
    try:
        return DatasetConfig(name=raw.pop("name", path.stem), path=data_path, **raw)
    except TypeError as exc:
        raise DataValidationError(f"Dataset config: {exc}") from exc


# ---------------------------------------------------------------------------
# Conversion steps
# ---------------------------------------------------------------------------
def _read_file(config: DatasetConfig) -> pd.DataFrame:
    if not config.path.exists():
        raise DataValidationError(f"Data file not found: {config.path}")
    suffix = config.path.suffix.lower()
    if suffix in (".csv", ".txt", ".tsv"):
        return pd.read_csv(config.path, sep=config.delimiter, encoding=config.encoding)
    if suffix == ".parquet":  # requires pyarrow (optional)
        return pd.read_parquet(config.path)
    # PLACEHOLDER(real-dataset): other formats (Excel needs `pip install openpyxl` then pd.read_excel).
    raise DataValidationError(f"Unsupported file type '{suffix}'. Convert to CSV or extend _read_file().")


def _apply_column_map(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    missing = [src for src in config.column_map.values() if src not in df.columns]
    if missing:
        raise DataValidationError(
            f"Columns {missing} from column_map are not in {config.path.name}. Available: {list(df.columns)}"
        )
    rename = {src: internal for internal, src in config.column_map.items()}
    out = df.drop(columns=[c for c in config.drop_columns if c in df.columns]).rename(columns=rename)
    keep = [c for c in MAPPABLE_COLUMNS if c in out.columns]
    extras = [c for c in out.columns if c not in keep]
    if extras:
        logger.info("Ignoring unmapped columns: %s", extras)
    return out[keep].copy()


def _convert_timestamps(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    ts = pd.to_datetime(
        df["timestamp"], format=config.timestamp_format or "mixed", dayfirst=config.dayfirst, errors="coerce"
    )
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(None)
    bad = int(ts.isna().sum())
    if bad:
        logger.warning("%d rows have unparseable timestamps and are dropped", bad)
    df = df.assign(timestamp=ts + pd.Timedelta(hours=config.timestamp_offset_hours))
    return df.dropna(subset=["timestamp"])


def _convert_units(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    df = df.copy()
    sm = pd.to_numeric(df["soil_moisture"], errors="coerce")
    if config.soil_moisture_unit == "fraction":
        sm = sm * 100.0
    elif config.soil_moisture_unit == "raw":
        c = config.soil_raw_calibration
        sm = c["dry_pct"] + (sm - c["dry_raw"]) * (c["wet_pct"] - c["dry_pct"]) / (c["wet_raw"] - c["dry_raw"])
    df["soil_moisture"] = sm

    if "temperature_c" in df.columns:
        t = pd.to_numeric(df["temperature_c"], errors="coerce")
        if config.temperature_unit == "fahrenheit":
            t = (t - 32.0) * 5.0 / 9.0
        elif config.temperature_unit == "kelvin":
            t = t - 273.15
        df["temperature_c"] = t
    if "humidity_pct" in df.columns:
        h = pd.to_numeric(df["humidity_pct"], errors="coerce")
        # A 0-1 humidity column is converted automatically.
        df["humidity_pct"] = h * 100.0 if h.dropna().between(0, 1).all() and h.max() <= 1 else h

    if "rainfall_mm" in df.columns:
        r = pd.to_numeric(df["rainfall_mm"], errors="coerce")
        if config.rainfall_unit == "inch":
            r = r * 25.4
        df["rainfall_mm"] = r
        if config.rainfall_mode == "cumulative":
            df = df.sort_values(["field_id", "timestamp"])
            diff = df.groupby("field_id")["rainfall_mm"].diff()
            # Counter resets (negative differences) start a new total.
            df["rainfall_mm"] = diff.where(diff >= 0, df["rainfall_mm"]).fillna(0.0)
    else:
        logger.warning("No rainfall column mapped - assuming 0 mm (rain awareness is reduced)")
        df["rainfall_mm"] = 0.0
    for col in ("temperature_c", "humidity_pct"):
        if col not in df.columns:
            logger.warning("No %s column mapped - ET will fall back to forecast temperatures", col)
            df[col] = np.nan
    return df


def _apply_metadata(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    df = df.copy()
    if "field_id" not in df.columns:
        df["field_id"] = config.default_field_id
    df["field_id"] = df["field_id"].astype(str)
    for key in METADATA_KEYS:
        if key not in df.columns:
            df[key] = np.nan
        df[key] = df[key].astype(object)
        if key in config.field_defaults:
            df[key] = df[key].where(df[key].notna(), config.field_defaults[key])
        for fid, meta in config.per_field_metadata.items():
            if key in meta:
                df.loc[df["field_id"] == str(fid), key] = meta[key]
    if df["planting_date"].isna().any():
        first = df.groupby("field_id")["timestamp"].transform("min")
        fallback = (first - pd.Timedelta(days=30)).dt.strftime("%Y-%m-%d")
        df["planting_date"] = df["planting_date"].where(df["planting_date"].notna(), fallback)
        logger.warning("planting_date missing for some fields - assuming 30 days before their first reading")
    missing = [k for k in ("crop_type", "soil_type", "irrigation_method", "latitude") if df[k].isna().any()]
    if missing:
        raise DataValidationError(f"Field metadata missing for {missing}: set them in fields.defaults/per_field")
    for col, allowed in (("crop_type", CROPS), ("soil_type", SOILS), ("irrigation_method", IRRIGATION_METHODS)):
        df[col] = df[col].astype(str).str.strip().str.lower()
        unknown = sorted(set(df[col]) - set(allowed))
        if unknown:
            raise DataValidationError(f"Unknown {col} {unknown}; allowed {sorted(allowed)}")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["planting_date"] = pd.to_datetime(df["planting_date"], format="mixed").dt.strftime("%Y-%m-%d")
    return df


def _apply_labels(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    if config.labels_mode == "derive":
        # Drop any target columns so training derives weak labels consistently.
        return df.drop(columns=[c for c in (TARGET_CLASS, TARGET_REG) if c in df.columns])
    df = df.copy()
    df[TARGET_CLASS] = pd.to_numeric(df[TARGET_CLASS], errors="coerce")
    values = set(df[TARGET_CLASS].dropna().unique())
    if not values <= {0, 1}:
        raise DataValidationError(f"{TARGET_CLASS} must be 0/1, found {sorted(values)[:10]}")
    if TARGET_REG not in df.columns:
        # PLACEHOLDER(real-dataset): if you have irrigation depths, map them to target_irrigation_mm.
        logger.warning("No depth column mapped - depth regression is skipped (FAO-56 formula used for depth)")
        df[TARGET_REG] = np.where(df[TARGET_CLASS] == 1, np.nan, 0.0)
    df[TARGET_REG] = pd.to_numeric(df[TARGET_REG], errors="coerce")
    return df


def load_custom_dataset(config: DatasetConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read, map, convert and validate a foreign dataset.

    Returns:
        (raw-reading DataFrame in the internal schema, conversion report)
    """
    original = _read_file(config)
    df = dataset_hooks.before_mapping(original, config)
    df = _apply_column_map(df, config)
    missing = [c for c in REQUIRED_INTERNAL if c not in df.columns]
    if missing:
        raise DataValidationError(f"column_map must provide {missing}")
    df = _convert_timestamps(df, config)
    if "field_id" not in df.columns:
        df["field_id"] = config.default_field_id
    df = _convert_units(df, config)
    df = _apply_metadata(df, config)
    df = _apply_labels(df, config)
    df = dataset_hooks.custom_transform(df, config)
    df = df.sort_values(["field_id", "timestamp"]).reset_index(drop=True)
    if df.empty:
        raise DataValidationError("No rows left after conversion")

    report: dict[str, Any] = {
        "name": config.name,
        "file": str(config.path),
        "rows_in_file": int(len(original)),
        "rows_after_conversion": int(len(df)),
        "fields": int(df["field_id"].nunique()),
        "time_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "median_interval_minutes": float(
            df.groupby("field_id")["timestamp"].diff().dt.total_seconds().median() / 60.0
        ) if len(df) > 1 else None,
        "mapped_columns": sorted(config.column_map),
        "missing_share": {c: round(float(df[c].isna().mean()), 4)
                          for c in ("soil_moisture", "temperature_c", "humidity_pct", "rainfall_mm")},
        "soil_moisture_range": [round(float(df["soil_moisture"].min()), 2), round(float(df["soil_moisture"].max()), 2)],
        "labels": config.labels_mode,
    }
    if config.labels_mode == "columns":
        report["positive_rate"] = round(float(df[TARGET_CLASS].mean()), 4)
    if report["soil_moisture_range"][1] <= 1.0:
        report.setdefault("warnings", []).append(
            "Soil moisture looks like a 0-1 fraction - set units.soil_moisture to 'fraction'"
        )
    return df, report
