"""Utility helpers and environment-based configuration."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from irrigation_advisor.config import PROJECT_ROOT, get_settings
from irrigation_advisor.utils import (
    clamp,
    is_missing,
    mean_or_none,
    parse_date,
    round_or_none,
    safe_float,
    stable_seed,
    to_json,
    to_local_naive,
)


def test_numeric_helpers():
    assert clamp(5, 0, 3) == 3 and clamp(-1, 0, 3) == 0
    with pytest.raises(ValueError):
        clamp(1, 3, 0)
    assert is_missing(None) and is_missing(float("nan")) and is_missing("abc") and not is_missing(0)
    assert safe_float("2.5") == 2.5 and safe_float(None, 1.0) == 1.0
    assert round_or_none(np.float64(1.23456), 2) == 1.23 and round_or_none(np.nan) is None
    assert mean_or_none([1, None, 3]) == 2 and mean_or_none([]) is None


def test_stable_seed_is_deterministic():
    assert stable_seed("weather", 18.5, 2026) == stable_seed("weather", 18.5, 2026)
    assert stable_seed("a") != stable_seed("b")


def test_time_helpers():
    naive = datetime(2026, 7, 1, 6, 0)
    assert to_local_naive(naive, 5.5) == naive
    aware = datetime(2026, 7, 1, 0, 30, tzinfo=timezone.utc)
    assert to_local_naive(aware, 5.5) == datetime(2026, 7, 1, 6, 0)
    assert to_local_naive(datetime(2026, 7, 1, 2, 30, tzinfo=timezone(timedelta(hours=2))), 0) == datetime(2026, 7, 1, 0, 30)
    assert parse_date("2026-07-01T10:00:00") == date(2026, 7, 1)


def test_json_encoder_handles_numpy_and_dates():
    data = json.loads(to_json({"a": np.int64(3), "b": np.float32(1.5), "c": np.array([1, 2]), "d": date(2026, 1, 2),
                               "e": np.float64("nan"), "f": Path("x")}))
    assert data == {"a": 3, "b": 1.5, "c": [1, 2], "d": "2026-01-02", "e": None, "f": "x"}


def test_settings_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SIA_DATABASE_PATH", str(tmp_path / "x.db"))
    monkeypatch.setenv("SIA_MODEL_PATH", "models/custom.joblib")
    monkeypatch.setenv("SIA_HISTORY_HOURS", "48")
    monkeypatch.setenv("SIA_LOG_LEVEL", "debug")
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert s.database_path == tmp_path / "x.db"
        assert s.model_path == (PROJECT_ROOT / "models" / "custom.joblib").resolve()
        assert s.history_hours == 48 and s.log_level == "DEBUG"
        monkeypatch.setenv("SIA_HISTORY_HOURS", "lots")
        get_settings.cache_clear()
        with pytest.raises(ValueError):
            get_settings()
    finally:
        get_settings.cache_clear()
