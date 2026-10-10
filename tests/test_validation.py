"""Sensor validation rules (row level and frame level)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from irrigation_advisor.data.validation import apply_range_checks, flag_frame, summarize_issues, validate_reading

NOW = datetime(2026, 7, 1, 12, 0)


def reading(**overrides):
    base = {"timestamp": NOW, "soil_moisture": 22.0, "temperature_c": 30.0, "humidity_pct": 55.0, "rainfall_mm": 0.0}
    base.update(overrides)
    return base


def history(values, step_h=1.0):
    return [
        {"timestamp": NOW - timedelta(hours=step_h * (len(values) - i)), "soil_moisture": v, "rainfall_mm": 0.0}
        for i, v in enumerate(values)
    ]


def test_good_reading_is_valid():
    res = validate_reading(reading(), history([22.4, 22.3, 22.1]), now=NOW)
    assert res.is_valid and res.issues == []


def test_out_of_range_soil_moisture_invalidates():
    res = validate_reading(reading(soil_moisture=85.0), now=NOW)
    assert not res.is_valid
    assert "out_of_range" in res.codes


def test_out_of_range_temperature_keeps_reading_usable():
    res = validate_reading(reading(temperature_c=85.0), now=NOW)
    assert res.is_valid  # soil moisture still usable
    assert "temperature_c" in res.invalid_fields


def test_missing_soil_moisture_invalidates():
    res = validate_reading(reading(soil_moisture=None), now=NOW)
    assert not res.is_valid and "missing_value" in res.codes


def test_future_timestamp_invalidates_and_stale_warns():
    assert "future_timestamp" in validate_reading(reading(timestamp=NOW + timedelta(hours=2)), now=NOW).codes
    stale = validate_reading(reading(timestamp=NOW - timedelta(hours=10)), now=NOW, stale_hours=6)
    assert "stale_reading" in stale.codes and stale.is_valid


def test_spike_detected_without_rain_but_not_with_rain():
    hist = history([20.0, 20.1, 20.0])
    assert "spike" in validate_reading(reading(soil_moisture=40.0), hist, now=NOW).codes
    explained = validate_reading(reading(soil_moisture=40.0, rainfall_mm=12.0), hist, now=NOW)
    assert "spike" not in explained.codes
    # a sudden drop is never explained by rain
    assert "spike" in validate_reading(reading(soil_moisture=2.0, rainfall_mm=12.0), hist, now=NOW).codes


def test_stuck_sensor_detected():
    res = validate_reading(reading(soil_moisture=21.0), history([21.0] * 11), now=NOW)
    assert "stuck_sensor" in res.codes and not res.is_valid


def test_low_battery_is_warning():
    res = validate_reading(reading(battery_v=3.1), now=NOW)
    assert res.is_valid and "low_battery" in res.codes


def test_summarize_issues_counts_codes():
    results = [validate_reading(reading(soil_moisture=None), now=NOW), validate_reading(reading(soil_moisture=99), now=NOW)]
    assert summarize_issues(results) == {"missing_value": 1, "out_of_range": 1}


def test_apply_range_checks_masks_values():
    df = pd.DataFrame({"soil_moisture": [20, -5, 70], "temperature_c": [25, 85, 20]})
    out = apply_range_checks(df)
    assert out["soil_moisture"].isna().tolist() == [False, True, True]
    assert out["temperature_c_out_of_range"].tolist() == [False, True, False]
    assert df["soil_moisture"].tolist() == [20, -5, 70]  # input untouched


def test_flag_frame_spike_and_stuck_are_backward_looking():
    sm = pd.Series([20.0, 20.1, 19.9, 20.0, 45.0, 20.0] + [18.0] * 12 + [17.5])
    flags = flag_frame(sm)
    assert bool(flags["spike"].iloc[4])
    assert not bool(flags["spike"].iloc[3])
    assert bool(flags["stuck"].iloc[17])
    # changing the future does not change past flags
    sm2 = sm.copy()
    sm2.iloc[10:] = np.nan
    np.testing.assert_array_equal(flag_frame(sm2)["spike"].iloc[:10], flags["spike"].iloc[:10])
