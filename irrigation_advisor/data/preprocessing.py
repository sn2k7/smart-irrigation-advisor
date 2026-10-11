"""Cleaning and hourly standardisation of sensor time series.

``prepare_sensor_frame`` is the single entry point used by **both** model
training and live inference (the API/engine). Every step only uses current
or past observations of the same field, so the cleaned value at time *t* is
exactly what a real-time system would have computed at *t* (no look-ahead):

1. schema check, type coercion and timestamp normalisation;
2. de-duplication and chronological sorting;
3. physical range checks (impossible values -> NaN);
4. resampling to a regular hourly grid per field (mean / sum aggregation);
5. backward-looking spike and stuck-sensor detection;
6. forward-fill of short gaps (max ``MAX_FFILL_HOURS``) with imputation flags.

Gaps are deliberately *not* interpolated linearly - interpolation would use
the next reading and leak future information into training features.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from ..exceptions import DataValidationError
from .validation import apply_range_checks, flag_frame

logger = logging.getLogger(__name__)

SENSOR_COLUMNS: tuple[str, ...] = ("soil_moisture", "temperature_c", "humidity_pct", "rainfall_mm")
REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "field_id", *SENSOR_COLUMNS)
META_COLUMNS: tuple[str, ...] = ("crop_type", "soil_type", "irrigation_method", "planting_date", "latitude")
MAX_FFILL_HOURS = 3


def check_required_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Raise :class:`DataValidationError` listing any missing columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def normalise_timestamps(values: pd.Series) -> pd.Series:
    """Parse timestamps to naive ``datetime64``; tz-aware values become naive UTC-shifted local."""
    ts = pd.to_datetime(values, errors="coerce", format="mixed")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(None)
    return ts.astype("datetime64[ns]")


def _resample_field(group: pd.DataFrame, passthrough: list[str]) -> pd.DataFrame:
    """Resample one field's readings to an hourly grid."""
    g = group.set_index("timestamp").sort_index()
    agg: dict[str, str] = {
        "soil_moisture": "mean",
        "temperature_c": "mean",
        "humidity_pct": "mean",
        "soil_moisture_out_of_range": "max",
    }
    if "forecast_rain_24h_mm" in g.columns:
        agg["forecast_rain_24h_mm"] = "last"
    if "wind_speed_ms" in g.columns:
        agg["wind_speed_ms"] = "mean"
    for col in passthrough:
        agg[col] = "last"
    # Booleans must be numeric for max-aggregation.
    g["soil_moisture_out_of_range"] = g["soil_moisture_out_of_range"].astype(float)
    hourly = g.resample("1h").agg(agg)
    # Sum of rain; an hour without any rain observation stays NaN (unknown).
    hourly["rainfall_mm"] = g["rainfall_mm"].resample("1h").sum(min_count=1)
    hourly["n_raw_readings"] = g["soil_moisture"].resample("1h").size()
    if "et0_mm" in g.columns:  # measured/model ET0 is a flux: sum per hour
        hourly["et0_mm"] = pd.to_numeric(g["et0_mm"], errors="coerce").resample("1h").sum(min_count=1)
    hourly.index.name = "timestamp"
    return hourly.reset_index()


def prepare_sensor_frame(df: pd.DataFrame, passthrough: Iterable[str] = ()) -> pd.DataFrame:
    """Clean raw readings into a regular hourly frame (one block per field).

    Args:
        df: Raw readings with at least :data:`REQUIRED_COLUMNS`. Field
            metadata (:data:`META_COLUMNS`) is carried over when present.
        passthrough: Extra columns (e.g. training targets) copied using the
            last value in each hour. They are never used as features.

    Returns:
        Hourly DataFrame sorted by ``field_id`` and ``timestamp`` with quality
        flags ``sm_invalid`` (reading unusable), ``sm_imputed`` (gap filled),
        ``spike`` and ``stuck``.
    """
    check_required_columns(df)
    passthrough = [c for c in passthrough if c in df.columns]
    if df.empty:
        raise DataValidationError("No sensor readings supplied")

    data = df.copy()
    data["timestamp"] = normalise_timestamps(data["timestamp"])
    bad_ts = data["timestamp"].isna().sum()
    if bad_ts:
        logger.warning("Dropping %d rows with unparseable timestamps", bad_ts)
        data = data.dropna(subset=["timestamp"])
    data["field_id"] = data["field_id"].astype(str)
    for col in SENSOR_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for optional in ("forecast_rain_24h_mm", "et0_mm", "wind_speed_ms"):
        if optional in data.columns:
            data[optional] = pd.to_numeric(data[optional], errors="coerce")

    before = len(data)
    data = data.drop_duplicates(subset=["field_id", "timestamp"], keep="last")
    if len(data) < before:
        logger.info("Removed %d duplicate readings", before - len(data))

    data = apply_range_checks(data)
    meta_cols = [c for c in META_COLUMNS if c in data.columns]

    blocks: list[pd.DataFrame] = []
    for field_id, group in data.groupby("field_id", sort=True):
        hourly = _resample_field(group, passthrough)
        hourly["field_id"] = field_id
        # Static metadata: last known value for the field.
        for col in meta_cols:
            known = group[col].dropna()
            hourly[col] = known.iloc[-1] if len(known) else np.nan

        flags = flag_frame(hourly["soil_moisture"], hourly["rainfall_mm"])
        hourly["spike"] = flags["spike"].to_numpy()
        hourly["stuck"] = flags["stuck"].to_numpy()
        hourly["soil_moisture"] = hourly["soil_moisture"].mask(hourly["spike"] | hourly["stuck"])

        hourly["sm_invalid"] = hourly["soil_moisture"].isna()
        hourly["soil_moisture"] = hourly["soil_moisture"].ffill(limit=MAX_FFILL_HOURS)
        hourly["sm_imputed"] = hourly["sm_invalid"] & hourly["soil_moisture"].notna()
        for col in ("temperature_c", "humidity_pct"):
            hourly[col] = hourly[col].ffill(limit=MAX_FFILL_HOURS)
        hourly["rain_missing"] = hourly["rainfall_mm"].isna()
        hourly["rainfall_mm"] = hourly["rainfall_mm"].fillna(0.0)
        blocks.append(hourly)

    out = pd.concat(blocks, ignore_index=True)
    out["soil_moisture_out_of_range"] = out["soil_moisture_out_of_range"].fillna(0).astype(bool)
    for col in ("sm_invalid", "sm_imputed", "spike", "stuck", "rain_missing"):
        out[col] = out[col].astype(bool)
    return out.sort_values(["field_id", "timestamp"]).reset_index(drop=True)


def data_quality_report(prepared: pd.DataFrame) -> dict[str, float | int]:
    """Summary statistics of a prepared frame (used in reports and the API)."""
    n = len(prepared)
    if n == 0:
        return {"rows": 0}
    return {
        "rows": n,
        "fields": int(prepared["field_id"].nunique()),
        "sm_invalid_pct": round(100 * float(prepared["sm_invalid"].mean()), 2),
        "sm_imputed_pct": round(100 * float(prepared["sm_imputed"].mean()), 2),
        "sm_still_missing_pct": round(100 * float(prepared["soil_moisture"].isna().mean()), 2),
        "spikes_detected": int(prepared["spike"].sum()),
        "stuck_hours_detected": int(prepared["stuck"].sum()),
        "out_of_range_hours": int(prepared["soil_moisture_out_of_range"].sum()),
    }
