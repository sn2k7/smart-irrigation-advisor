"""Cleaning, hourly resampling and gap handling."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from irrigation_advisor.data.preprocessing import MAX_FFILL_HOURS, data_quality_report, prepare_sensor_frame
from irrigation_advisor.exceptions import DataValidationError

T0 = datetime(2026, 7, 1, 0, 0)


def raw_frame(n_hours=30, sm=None, step_min=60):
    n = int(n_hours * 60 / step_min)
    ts = [T0 + timedelta(minutes=step_min * i) for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "field_id": "A",
            "soil_moisture": sm if sm is not None else np.linspace(25, 20, n),
            "temperature_c": 28.0,
            "humidity_pct": 60.0,
            "rainfall_mm": 0.0,
            "crop_type": "maize",
            "soil_type": "loam",
            "irrigation_method": "drip",
            "planting_date": "2026-05-01",
            "latitude": 18.5,
        }
    )


def test_missing_columns_raise():
    with pytest.raises(DataValidationError):
        prepare_sensor_frame(raw_frame().drop(columns=["soil_moisture"]))


def test_empty_frame_raises():
    with pytest.raises(DataValidationError):
        prepare_sensor_frame(raw_frame().iloc[:0])


def test_resamples_15_minute_data_to_hourly_means_and_rain_sums():
    df = raw_frame(n_hours=4, step_min=15)
    df["rainfall_mm"] = 0.5
    out = prepare_sensor_frame(df)
    assert len(out) == 4
    assert out["rainfall_mm"].tolist() == pytest.approx([2.0] * 4)
    assert out["soil_moisture"].iloc[0] == pytest.approx(df["soil_moisture"].iloc[:4].mean())


def test_duplicates_removed_and_order_restored():
    df = raw_frame(10)
    shuffled = pd.concat([df, df.iloc[[3, 5]]]).sample(frac=1, random_state=1)
    out = prepare_sensor_frame(shuffled)
    assert len(out) == 10
    assert out["timestamp"].is_monotonic_increasing


def test_out_of_range_values_become_missing_and_are_flagged():
    sm = np.full(10, 22.0) + np.arange(10) * 0.1
    sm[5] = 150.0
    out = prepare_sensor_frame(raw_frame(10, sm=sm))
    assert bool(out.loc[5, "soil_moisture_out_of_range"])
    assert bool(out.loc[5, "sm_invalid"]) and bool(out.loc[5, "sm_imputed"])
    assert out.loc[5, "soil_moisture"] == pytest.approx(sm[4])  # carried forward, not interpolated


def test_forward_fill_is_limited():
    sm = np.linspace(25, 20, 20)
    sm[5:15] = np.nan
    out = prepare_sensor_frame(raw_frame(20, sm=sm))
    filled = out.loc[5 : 5 + MAX_FFILL_HOURS - 1, "soil_moisture"]
    assert filled.notna().all()
    assert out.loc[5 + MAX_FFILL_HOURS : 14, "soil_moisture"].isna().all()


def test_spike_removed():
    sm = np.linspace(25, 22, 24)
    sm[12] = 45.0
    out = prepare_sensor_frame(raw_frame(24, sm=sm))
    assert bool(out.loc[12, "spike"])
    assert out.loc[12, "soil_moisture"] < 30


def test_timezone_aware_timestamps_are_accepted():
    df = raw_frame(5)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
    out = prepare_sensor_frame(df)
    assert out["timestamp"].dt.tz is None and len(out) == 5


def test_passthrough_columns_kept_and_quality_report():
    df = raw_frame(6)
    df["target_irrigation_needed"] = 1
    out = prepare_sensor_frame(df, passthrough=["target_irrigation_needed"])
    assert (out["target_irrigation_needed"] == 1).all()
    report = data_quality_report(out)
    assert report["rows"] == 6 and report["fields"] == 1
