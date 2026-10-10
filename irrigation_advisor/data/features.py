"""Time-series feature engineering shared by training and inference.

``build_features`` turns the hourly frame produced by
:func:`irrigation_advisor.data.preprocessing.prepare_sensor_frame` into the
model matrix. The *same* function (and the same column list
:data:`FEATURE_COLUMNS`) is used by ``ml.training`` and by the recommendation
engine, which guarantees identical feature definitions online and offline.

Leakage rules enforced here:

* every rolling window, lag and "hours since" counter is computed per field
  and looks **backwards only** (``rolling`` on past rows, ``shift(+k)``);
* the only forward-looking input is ``forecast_rain_24h_mm`` - a weather
  forecast that is genuinely available at decision time;
* targets and ``true_*`` simulator columns are never part of
  :data:`FEATURE_COLUMNS` (enforced by :func:`assert_no_target_columns`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..agronomy.crops import CropProfile, get_crop, get_soil
from ..agronomy.water_balance import EFFECTIVE_RAIN_FRACTION, MIN_EFFECTIVE_RAIN_MM, hargreaves_et0
from ..exceptions import DataValidationError

#: Bump whenever a feature definition changes; saved models are checked against it.
FEATURE_SCHEMA_VERSION = "1.0"

#: Model inputs. They are deliberately *dimensionless* (scaled by the crop/soil
#: specific RAW or expressed as ratios) so that one model transfers across
#: soils, crops, root depths and seasons. An ablation (docs/ML_WORKFLOW.md)
#: showed absolute magnitudes (mm, % VWC, day-of-year, crop/soil identity) let
#: models memorise individual fields and seasons and generalise worse.
NUMERIC_FEATURES: list[str] = [
    # weather sensors (current + 24 h history)
    "temperature_c",
    "humidity_pct",
    "rainfall_mm",
    "temp_max_24h",
    "temp_min_24h",
    "humidity_mean_24h",
    "rain_sum_72h",
    "hours_since_rain",
    # root-zone water status relative to the stress threshold (FAO-56)
    "depletion_ratio",
    "dr_mean_6h_ratio",
    "projected_depletion_ratio",
    "available_water_fraction",
    "kc",
    "raw_taw_ratio",
    "etc_raw_ratio",
    "forecast_rain_raw_ratio",
    "rain_24h_raw_ratio",
    # soil-moisture dynamics, converted to mm of water and scaled by RAW
    "sm_delta_1h_ratio",
    "sm_delta_6h_ratio",
    "sm_delta_24h_ratio",
    "sm_std_24h_ratio",
    # time of day
    "hour_sin",
    "hour_cos",
    # data quality
    "sm_imputed",
    "sm_invalid_hours_24h",
]
CATEGORICAL_FEATURES: list[str] = ["growth_stage"]
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
#: Computed alongside the features (same backward-looking rules) for the
#: engine's physics rule, explanations and reports - but NOT model inputs.
CONTEXT_COLUMNS: list[str] = [
    "soil_moisture", "sm_mean_6h", "sm_min_24h", "sm_max_24h", "sm_delta_1h", "sm_delta_6h", "sm_delta_24h",
    "sm_std_24h", "rain_sum_6h", "rain_sum_24h", "days_since_planting", "root_depth_m", "et0_24h_mm",
    "etc_24h_mm", "taw_mm", "raw_mm", "depletion_mm", "forecast_rain_24h_mm", "forecast_effective_rain_mm",
    "doy_sin", "doy_cos", "crop_type", "soil_type",
]
KEY_COLUMNS: list[str] = ["field_id", "timestamp"]
FORBIDDEN_PREFIXES: tuple[str, ...] = ("target_", "true_")
HOURS_SINCE_RAIN_CAP = 168.0
RAIN_EVENT_MM = 0.5


def assert_no_target_columns(columns: list[str]) -> None:
    """Fail loudly if a target / ground-truth column sneaks into the features."""
    leaked = [c for c in columns if c.startswith(FORBIDDEN_PREFIXES)]
    if leaked:
        raise DataValidationError(f"Target/ground-truth columns used as features: {leaked}")


assert_no_target_columns(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Vectorised FAO-56 helpers (identical to the scalar versions in agronomy)
# ---------------------------------------------------------------------------
def kc_array(crop: CropProfile, days: np.ndarray) -> np.ndarray:
    """Vectorised :func:`agronomy.water_balance.crop_coefficient`."""
    ini, dev, mid, late = crop.stage_days
    x = [0, ini, ini + dev, ini + dev + mid, ini + dev + mid + late]
    y = [crop.kc_ini, crop.kc_ini, crop.kc_mid, crop.kc_mid, crop.kc_end]
    return np.interp(np.clip(days, 0, None), x, y)


def root_depth_array(crop: CropProfile, days: np.ndarray) -> np.ndarray:
    """Vectorised :func:`agronomy.water_balance.root_depth`."""
    ini, dev, _, _ = crop.stage_days
    return np.interp(np.clip(days, 0, None), [0, ini + dev], [crop.root_depth_min_m, crop.root_depth_max_m])


def stage_array(crop: CropProfile, days: np.ndarray) -> np.ndarray:
    """Vectorised :func:`agronomy.water_balance.growth_stage`."""
    ini, dev, mid, late = crop.stage_days
    d = np.clip(days, 0, None)
    return np.select(
        [d < ini, d < ini + dev, d < ini + dev + mid, d <= ini + dev + mid + late],
        ["initial", "development", "mid_season", "late_season"],
        default="post_harvest",
    )


def effective_rain_array(rain_mm: np.ndarray) -> np.ndarray:
    """Vectorised :func:`agronomy.water_balance.effective_rainfall`."""
    rain = np.clip(np.asarray(rain_mm, dtype=float), 0, None)
    return np.where(rain < MIN_EFFECTIVE_RAIN_MM, 0.0, EFFECTIVE_RAIN_FRACTION * rain)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def _hours_since_event(ts: pd.Series, event: pd.Series) -> pd.Series:
    last_event = ts.where(event).ffill()
    hours = (ts - last_event).dt.total_seconds() / 3600.0
    return hours.fillna(HOURS_SINCE_RAIN_CAP).clip(upper=HOURS_SINCE_RAIN_CAP)


def _field_features(g: pd.DataFrame) -> pd.DataFrame:
    """Features for ONE field (rows sorted by time on a regular hourly grid)."""
    out = pd.DataFrame(index=g.index)
    sm = g["soil_moisture"]
    ts = g["timestamp"]

    for col in ("soil_moisture", "temperature_c", "humidity_pct", "rainfall_mm"):
        out[col] = g[col]
    out["sm_delta_1h"] = sm - sm.shift(1)
    out["sm_delta_6h"] = sm - sm.shift(6)
    out["sm_delta_24h"] = sm - sm.shift(24)
    out["sm_mean_6h"] = sm.rolling(6, min_periods=1).mean()
    out["sm_min_24h"] = sm.rolling(24, min_periods=1).min()
    out["sm_max_24h"] = sm.rolling(24, min_periods=1).max()
    out["sm_std_24h"] = sm.rolling(24, min_periods=3).std()

    temp = g["temperature_c"]
    out["temp_max_24h"] = temp.rolling(24, min_periods=6).max()
    out["temp_min_24h"] = temp.rolling(24, min_periods=6).min()
    out["humidity_mean_24h"] = g["humidity_pct"].rolling(24, min_periods=6).mean()
    rain = g["rainfall_mm"].fillna(0.0)
    out["rain_sum_6h"] = rain.rolling(6, min_periods=1).sum()
    out["rain_sum_24h"] = rain.rolling(24, min_periods=1).sum()
    out["rain_sum_72h"] = rain.rolling(72, min_periods=1).sum()
    out["hours_since_rain"] = _hours_since_event(ts, rain >= RAIN_EVENT_MM)

    # --- agronomy -------------------------------------------------------
    crop = get_crop(str(g["crop_type"].iloc[0]))
    soil = get_soil(str(g["soil_type"].iloc[0]))
    planting = pd.to_datetime(g["planting_date"].iloc[0])
    latitude = float(g["latitude"].iloc[0])
    days = ((ts - planting).dt.total_seconds() / 86400.0).to_numpy()
    doy = ts.dt.dayofyear.to_numpy()

    out["days_since_planting"] = days
    out["kc"] = kc_array(crop, days)
    zr = root_depth_array(crop, days)
    out["root_depth_m"] = zr
    out["growth_stage"] = stage_array(crop, days)
    et0 = hargreaves_et0(out["temp_min_24h"].to_numpy(), out["temp_max_24h"].to_numpy(), latitude, doy)
    out["et0_24h_mm"] = et0
    out["etc_24h_mm"] = out["kc"] * et0
    taw = 1000.0 * (soil.field_capacity - soil.wilting_point) * zr
    raw = crop.depletion_fraction * taw
    depletion = 1000.0 * (soil.field_capacity - sm.to_numpy() / 100.0) * zr
    out["taw_mm"] = taw
    out["raw_mm"] = raw
    out["depletion_mm"] = depletion
    out["depletion_ratio"] = depletion / raw
    out["available_water_fraction"] = 1.0 - np.clip(depletion, 0, None) / taw

    forecast = g["forecast_rain_24h_mm"] if "forecast_rain_24h_mm" in g.columns else pd.Series(0.0, index=g.index)
    forecast = forecast.fillna(0.0).clip(lower=0.0)
    out["forecast_rain_24h_mm"] = forecast
    out["forecast_effective_rain_mm"] = effective_rain_array(forecast.to_numpy())
    out["projected_depletion_ratio"] = (
        np.clip(depletion, 0, None) + out["etc_24h_mm"].to_numpy() - out["forecast_effective_rain_mm"].to_numpy()
    ) / raw

    # Dimensionless model inputs (1 % VWC over Zr metres = 10 * Zr mm of water).
    mm_per_pct = 10.0 * zr
    out["dr_mean_6h_ratio"] = 1000.0 * (soil.field_capacity - out["sm_mean_6h"].to_numpy() / 100.0) * zr / raw
    out["raw_taw_ratio"] = raw / taw
    out["etc_raw_ratio"] = out["etc_24h_mm"].to_numpy() / raw
    out["forecast_rain_raw_ratio"] = out["forecast_effective_rain_mm"].to_numpy() / raw
    out["rain_24h_raw_ratio"] = out["rain_sum_24h"].to_numpy() / raw
    for col in ("sm_delta_1h", "sm_delta_6h", "sm_delta_24h", "sm_std_24h"):
        out[f"{col}_ratio"] = out[col].to_numpy() * mm_per_pct / raw

    hour = ts.dt.hour.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    invalid = g["sm_invalid"].astype(float) if "sm_invalid" in g.columns else pd.Series(0.0, index=g.index)
    out["sm_imputed"] = g["sm_imputed"].astype(float) if "sm_imputed" in g.columns else 0.0
    out["sm_invalid_hours_24h"] = invalid.rolling(24, min_periods=1).sum()

    out["crop_type"] = crop.name
    out["soil_type"] = soil.name
    return out


def build_features(prepared: pd.DataFrame, passthrough: list[str] | tuple[str, ...] = ()) -> pd.DataFrame:
    """Compute :data:`FEATURE_COLUMNS` (+ :data:`CONTEXT_COLUMNS`) for every row.

    Args:
        prepared: Output of ``prepare_sensor_frame`` (must include field metadata).
        passthrough: Columns copied unchanged (targets, ground truth, flags).

    Returns:
        DataFrame with :data:`KEY_COLUMNS`, :data:`FEATURE_COLUMNS`,
        :data:`CONTEXT_COLUMNS` and passthrough columns.
    """
    needed = ["field_id", "timestamp", "soil_moisture", "temperature_c", "humidity_pct", "rainfall_mm",
              "crop_type", "soil_type", "planting_date", "latitude"]
    missing = [c for c in needed if c not in prepared.columns]
    if missing:
        raise DataValidationError(f"Prepared frame lacks columns needed for features: {missing}")
    if prepared.empty:
        raise DataValidationError("Cannot build features from an empty frame")

    frames = []
    for _, group in prepared.sort_values(["field_id", "timestamp"]).groupby("field_id", sort=True):
        group = group.reset_index(drop=True)
        feats = _field_features(group)
        feats.insert(0, "timestamp", group["timestamp"].to_numpy())
        feats.insert(0, "field_id", group["field_id"].to_numpy())
        for col in passthrough:
            if col in group.columns:
                feats[col] = group[col].to_numpy()
        frames.append(feats)
    result = pd.concat(frames, ignore_index=True)
    for col in CATEGORICAL_FEATURES + ["crop_type", "soil_type"]:
        result[col] = result[col].astype(str)
    return result
