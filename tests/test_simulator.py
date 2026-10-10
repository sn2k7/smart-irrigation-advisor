"""Synthetic data generator, weather generator and label definitions."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from irrigation_advisor.agronomy.crops import get_crop, get_soil
from irrigation_advisor.data.climate import generate_hourly_weather, get_climate, simulate_rain_forecast
from irrigation_advisor.data.labels import TARGET_CLASS, TARGET_REG
from irrigation_advisor.data.simulator import FieldSpec, generate_dataset, simulate_field
from irrigation_advisor.data.weather_provider import SimulatedWeatherProvider


def test_dataset_is_reproducible(small_dataset):
    again, _ = generate_dataset(n_fields=8, seed=7, year=2025)
    pd.testing.assert_frame_equal(small_dataset, again)


def test_dataset_schema_and_ranges(small_dataset):
    for col in ("timestamp", "field_id", "soil_moisture", "temperature_c", "humidity_pct", "rainfall_mm",
                "forecast_rain_24h_mm", TARGET_CLASS, TARGET_REG, "true_soil_moisture", "sensor_fault"):
        assert col in small_dataset.columns
    assert set(small_dataset[TARGET_CLASS].unique()) <= {0, 1}
    assert 0.05 < small_dataset[TARGET_CLASS].mean() < 0.8
    assert (small_dataset["rainfall_mm"] >= 0).all()
    assert small_dataset["humidity_pct"].between(0, 100).all()
    assert (small_dataset["sensor_fault"] != "none").any()


def test_targets_consistent_with_definition(small_dataset):
    df = small_dataset
    # Not needed -> zero depth; needed -> positive depth.
    assert (df.loc[df[TARGET_CLASS] == 0, TARGET_REG] == 0).all()
    assert (df.loc[df[TARGET_CLASS] == 1, TARGET_REG] > 0).all()
    # Whenever the TRUE depletion already exceeds RAW, irrigation is needed.
    stressed = df["true_depletion_mm"] >= df["true_raw_mm"]
    assert (df.loc[stressed, TARGET_CLASS] == 1).all()


def test_sensor_tracks_truth_when_healthy(small_dataset):
    ok = small_dataset[small_dataset["sensor_fault"] == "none"]
    err = (ok["soil_moisture"] - ok["true_soil_moisture"]).abs()
    assert err.median() < 2.0


def test_no_faults_when_disabled():
    spec = FieldSpec("X", "maize", "loam", "drip", date(2025, 3, 1), 18.5, 1000, 60, "fixed_interval", 4, 20, 0.0, 1)
    df = simulate_field(spec, fault_rate=0.0)
    assert (df["sensor_fault"] == "none").all()
    assert df["soil_moisture"].notna().all()
    assert len(df) == get_crop("maize").season_days * 24 + 24  # season + labelled part of the extra days


def test_weather_generator_plausible():
    w = generate_hourly_weather(get_climate("semi_arid_monsoon"), 18.5, date(2025, 1, 1), 365, seed=3)
    monsoon = w[pd.to_datetime(w["timestamp"]).dt.month.isin([7, 8])]["rainfall_mm"].sum()
    dry = w[pd.to_datetime(w["timestamp"]).dt.month.isin([1, 2])]["rainfall_mm"].sum()
    assert monsoon > 3 * max(dry, 1)
    assert w["temperature_c"].between(-5, 50).all()
    daily_et0 = w.groupby(pd.to_datetime(w["timestamp"]).dt.date)["et0_true_mm"].sum()
    assert 2 < daily_et0.mean() < 8


def test_forecast_error_model_statistics():
    rng = np.random.default_rng(0)
    actual = np.where(rng.random(20000) < 0.3, rng.gamma(1, 10, 20000) + 1, 0.0)
    fc = simulate_rain_forecast(actual, rng)
    rain = actual >= 1
    assert 0.08 < (fc[rain] == 0).mean() < 0.16  # ~12 % misses
    assert 0.05 < (fc[~rain] > 0).mean() < 0.11  # ~8 % false alarms
    assert (fc >= 0).all()


def test_weather_provider_is_deterministic_and_offline():
    provider = SimulatedWeatherProvider()
    a = provider.forecast(18.5, datetime(2026, 7, 15, 6), 24)
    b = provider.forecast(18.5, datetime(2026, 7, 15, 6), 24)
    assert a == b
    assert len(a.hourly) == 24 and a.source == "simulated-offline"
    # crosses a year boundary without error
    assert len(provider.hourly(18.5, datetime(2026, 12, 31, 20), 10)) == 10
    with pytest.raises(ValueError):
        provider.hourly(18.5, datetime(2026, 1, 1), 0)
    assert get_soil("clay").saturation > 0
