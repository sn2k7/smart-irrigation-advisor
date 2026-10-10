"""Feature engineering: schema, physics consistency and NO look-ahead leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from irrigation_advisor.agronomy.crops import get_crop, get_soil
from irrigation_advisor.agronomy.water_balance import soil_water_status
from irrigation_advisor.data.features import (
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    assert_no_target_columns,
    build_features,
)
from irrigation_advisor.data.preprocessing import prepare_sensor_frame
from irrigation_advisor.exceptions import DataValidationError


@pytest.fixture(scope="module")
def raw(small_dataset) -> pd.DataFrame:
    field = small_dataset["field_id"].iloc[0]
    return small_dataset[small_dataset["field_id"] == field].iloc[:400].reset_index(drop=True)


def features_of(raw_df: pd.DataFrame) -> pd.DataFrame:
    return build_features(prepare_sensor_frame(raw_df))


def test_all_feature_and_context_columns_present(raw):
    feats = features_of(raw)
    for col in FEATURE_COLUMNS + CONTEXT_COLUMNS:
        assert col in feats.columns, col
    assert len(feats) == len(raw)


def test_no_targets_or_ground_truth_in_features():
    assert not [c for c in FEATURE_COLUMNS if c.startswith(("target_", "true_"))]
    with pytest.raises(DataValidationError):
        assert_no_target_columns(FEATURE_COLUMNS + ["true_soil_moisture"])


def test_features_do_not_use_future_data(raw):
    """Altering every row after t must not change any feature at or before t."""
    cut = 250
    base = features_of(raw)
    tampered = raw.copy()
    rng = np.random.default_rng(0)
    for col in ("soil_moisture", "temperature_c", "humidity_pct", "rainfall_mm"):
        tampered.loc[cut + 1 :, col] = rng.uniform(0, 50, len(tampered) - cut - 1)
    altered = features_of(tampered)
    pd.testing.assert_frame_equal(
        base.loc[:cut, FEATURE_COLUMNS + CONTEXT_COLUMNS], altered.loc[:cut, FEATURE_COLUMNS + CONTEXT_COLUMNS]
    )


def test_truncated_history_gives_identical_last_row(raw):
    """Inference on a 72 h window reproduces the training feature row (identical preprocessing)."""
    full = features_of(raw)
    t = 300
    window = raw.iloc[t - 71 : t + 1]
    online = features_of(window).iloc[-1]
    offline = full.iloc[t]
    # Features with windows <= 72 h must match exactly.
    for col in ("depletion_ratio", "projected_depletion_ratio", "etc_raw_ratio", "sm_delta_24h_ratio",
                "humidity_mean_24h", "rain_sum_72h", "kc", "growth_stage"):
        assert online[col] == pytest.approx(offline[col]) if not isinstance(offline[col], str) else online[col] == offline[col]


def test_depletion_matches_scalar_agronomy(raw):
    feats = features_of(raw)
    row = feats.dropna(subset=["soil_moisture"]).iloc[100]
    status = soil_water_status(
        row["soil_moisture"], get_soil(raw["soil_type"].iloc[0]), get_crop(raw["crop_type"].iloc[0]), row["days_since_planting"]
    )
    assert row["depletion_mm"] == pytest.approx(status.depletion_mm)
    assert row["raw_mm"] == pytest.approx(status.raw_mm)
    assert row["depletion_ratio"] == pytest.approx(status.depletion_mm / status.raw_mm)
    assert row["kc"] == pytest.approx(status.kc)


def test_forecast_missing_defaults_to_zero(raw):
    feats = features_of(raw.drop(columns=["forecast_rain_24h_mm"]))
    assert (feats["forecast_rain_24h_mm"] == 0).all()


def test_missing_metadata_raises(raw):
    prepared = prepare_sensor_frame(raw).drop(columns=["crop_type"])
    with pytest.raises(DataValidationError):
        build_features(prepared)
