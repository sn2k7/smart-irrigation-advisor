"""Open-Meteo integration, tested against a MOCKED Open-Meteo server.

The mock returns physically plausible but artificial numbers; these tests
verify request parameters, parsing, unit handling, season splitting, the
forecast logic and the pipeline wiring - not anything about the real site.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from irrigation_advisor.data.custom_dataset import load_custom_dataset, load_dataset_config
from irrigation_advisor.data.features import build_features
from irrigation_advisor.data.openmeteo import (
    FORECAST_SOIL_LAYERS,
    OpenMeteoClient,
    OpenMeteoConfig,
    OpenMeteoError,
    OpenMeteoWeatherProvider,
    Season,
    build_dataset,
    estimate_field_capacity,
    load_openmeteo_config,
    next_24h_sum,
    suggest_soil_type,
    weighted_soil_moisture,
    write_dataset_config,
)
from irrigation_advisor.data.preprocessing import prepare_sensor_frame
from irrigation_advisor.engine.recommender import FieldConfig, RecommendationEngine

LAT, LON = 11.93975, 79.49244


def _series(start: date, end: date) -> pd.DataFrame:
    """Deterministic artificial hourly weather + a simple bucket soil moisture."""
    times = pd.date_range(f"{start}T00:00", f"{end}T23:00", freq="h")
    rng = np.random.default_rng(int(times[0].timestamp()) % 2**32)
    hour = times.hour.to_numpy()
    temp = 27 + 5 * np.sin((hour - 9) / 24 * 2 * np.pi)
    rh = 70 - 15 * np.sin((hour - 9) / 24 * 2 * np.pi)
    rain = np.where(rng.random(len(times)) < 0.02, rng.gamma(1.0, 8.0, len(times)), 0.0)
    et0 = np.clip(0.6 * np.sin((hour - 6) / 12 * np.pi), 0, None)
    sm = np.empty(len(times))
    theta = 0.30
    for i in range(len(times)):
        theta = min(0.45, theta + rain[i] / 280.0) - et0[i] / 280.0
        theta = max(theta, 0.12)
        if theta > 0.32:
            theta -= 0.05 * (theta - 0.32)
        sm[i] = theta
    return pd.DataFrame({"time": times.strftime("%Y-%m-%dT%H:%M"), "temperature_2m": temp.round(1),
                         "relative_humidity_2m": rh.round(0), "precipitation": rain.round(1),
                         "et0_fao_evapotranspiration": et0.round(3), "wind_speed_10m": np.full(len(times), 2.5),
                         "soil_moisture_0_to_7cm": (sm - 0.01).round(3), "soil_moisture_7_to_28cm": sm.round(3),
                         "soil_moisture_0_to_1cm": sm.round(3), "soil_moisture_1_to_3cm": sm.round(3),
                         "soil_moisture_3_to_9cm": sm.round(3), "soil_moisture_9_to_27cm": sm.round(3)})


class FakeOpenMeteo:
    """httpx MockTransport handler that records requests."""

    def __init__(self, historical_forecast_ok: bool = True) -> None:
        self.calls: list[httpx.Request] = []
        self.historical_forecast_ok = historical_forecast_ok

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        p = request.url.params
        host = request.url.host
        if host.startswith("historical-forecast") and not self.historical_forecast_ok:
            return httpx.Response(400, json={"error": True, "reason": "out of range"})
        if host == "api.open-meteo.com":
            now = datetime.now().replace(minute=0, second=0, microsecond=0)
            start = (now - timedelta(days=int(p["past_days"]))).date()
            end = (now + timedelta(days=int(p["forecast_days"]))).date()
        else:
            start, end = date.fromisoformat(p["start_date"]), date.fromisoformat(p["end_date"])
        df = _series(start, end)
        wanted = p["hourly"].split(",")
        return httpx.Response(200, json={"latitude": float(p["latitude"]), "longitude": float(p["longitude"]),
                                         "hourly": {"time": df["time"].tolist(),
                                                    **{v: df[v].tolist() for v in wanted}}})


def make_client(fake: FakeOpenMeteo, cache: Path | None = None) -> OpenMeteoClient:
    return OpenMeteoClient(httpx.Client(transport=httpx.MockTransport(fake)), retries=1, cache_dir=cache)


def make_config(tmp_path: Path, **overrides) -> OpenMeteoConfig:
    base = dict(name="t", latitude=LAT, longitude=LON, start_date=date(2023, 1, 1), end_date=date(2024, 12, 31),
                seasons=[Season("groundnut", "06-15"), Season("groundnut", "12-15")],
                output_csv=tmp_path / "om.csv", dataset_config_out=tmp_path / "om_cfg.json",
                cache_dir=tmp_path / "cache")
    base.update(overrides)
    return OpenMeteoConfig(**base)


def test_shipped_config_targets_requested_location():
    from irrigation_advisor.config import PROJECT_ROOT

    cfg = load_openmeteo_config(PROJECT_ROOT / "config" / "openmeteo_dataset.json")
    assert (cfg.latitude, cfg.longitude) == (LAT, LON)
    assert cfg.timezone == "Asia/Kolkata" and cfg.seasons


def test_requests_use_location_variables_and_yearly_chunks(tmp_path):
    fake = FakeOpenMeteo()
    df, report = build_dataset(make_config(tmp_path), make_client(fake))
    archive = [c for c in fake.calls if c.url.host.startswith("archive")]
    assert len(archive) == 2  # one request per calendar year
    params = archive[0].url.params
    assert params["latitude"] == str(LAT) and params["longitude"] == str(LON)
    assert params["timezone"] == "Asia/Kolkata" and params["wind_speed_unit"] == "ms"
    for var in ("soil_moisture_0_to_7cm", "temperature_2m", "relative_humidity_2m", "precipitation",
                "et0_fao_evapotranspiration", "wind_speed_10m"):
        assert var in params["hourly"]
    assert report["forecast_proxy_share"] < 0.01  # archived forecasts used


def test_dataset_columns_units_and_seasons(tmp_path):
    df, report = build_dataset(make_config(tmp_path), make_client(FakeOpenMeteo()))
    for col in ("soil_moisture", "temperature_c", "humidity_pct", "rainfall_mm", "rain_last_24h_mm",
                "forecast_rain_24h_mm", "et0_mm", "wind_speed_ms"):
        assert col in df.columns
    assert df["soil_moisture"].between(5, 50).all()  # m3/m3 -> %
    assert set(df["crop_type"]) == {"groundnut"}
    # 2023-06-15, 2023-12-15, 2024-06-15 fully inside; 2024-12-15 partly (>= 14 days) inside
    assert "groundnut-2023-06-15" in report["seasons"] and len(report["seasons"]) >= 3
    one = df[df["field_id"] == "groundnut-2023-06-15"]
    assert len(one) == 130 * 24
    assert report["soil_type"] in {"sandy", "sandy_loam", "loam", "clay_loam", "clay"}


def test_forecast_falls_back_to_observed_proxy(tmp_path):
    df, report = build_dataset(make_config(tmp_path), make_client(FakeOpenMeteo(historical_forecast_ok=False)))
    assert (df["forecast_source"] == "observed_next_24h_proxy").all()
    assert report["forecast_proxy_share"] == 1.0


def test_cache_avoids_repeat_downloads(tmp_path):
    fake = FakeOpenMeteo()
    cfg = make_config(tmp_path)
    build_dataset(cfg, make_client(fake, tmp_path / "cache"))
    first = len(fake.calls)
    build_dataset(cfg, make_client(fake, tmp_path / "cache"))
    assert len(fake.calls) == first


def test_http_errors_are_reported(tmp_path):
    client = OpenMeteoClient(httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(400, json={"error": True, "reason": "Parameter 'hourly' is invalid"}))), retries=1)
    with pytest.raises(OpenMeteoError, match="invalid"):
        client.archive(LAT, LON, date(2024, 1, 1), date(2024, 1, 2), "Asia/Kolkata", ["soil_moisture_0_to_7cm"])


def test_helpers():
    df = pd.DataFrame({"a": [0.2, 0.3], "b": [0.4, np.nan]})
    np.testing.assert_allclose(weighted_soil_moisture(df, {"a": 1, "b": 3}), [35.0, 30.0])
    s = pd.Series(np.arange(30, dtype=float))
    assert next_24h_sum(s).iloc[0] == pytest.approx(sum(range(1, 25)))
    assert np.isnan(next_24h_sum(s).iloc[10])
    assert suggest_soil_type(0.36, 0.22) == "clay" and suggest_soil_type(0.12, 0.045) == "sandy"
    assert estimate_field_capacity(pd.Series([30.0] * 10), pd.Series([0.0] * 10))[0] is None
    with pytest.raises(Exception):
        OpenMeteoConfig(name="x", latitude=LAT, longitude=LON, start_date=date(2024, 1, 1),
                        end_date=date(2023, 1, 1), seasons=[Season("groundnut", "06-15")])


def test_measured_et0_replaces_hargreaves(tmp_path):
    df, _ = build_dataset(make_config(tmp_path), make_client(FakeOpenMeteo()))
    one = df[df["field_id"] == "groundnut-2023-06-15"]
    with_et0 = build_features(prepare_sensor_frame(one))
    without = build_features(prepare_sensor_frame(one.drop(columns=["et0_mm"])))
    expected = one["et0_mm"].rolling(24, min_periods=18).sum().to_numpy()
    np.testing.assert_allclose(with_et0["et0_24h_mm"].to_numpy()[30:], expected[30:], rtol=1e-6)
    assert not np.allclose(with_et0["et0_24h_mm"].to_numpy()[30:], without["et0_24h_mm"].to_numpy()[30:])
    assert "wind_speed_ms" in with_et0.columns


def test_generated_dataset_config_checks_and_trains(tmp_path):
    from irrigation_advisor.real_dataset_workflow import check_real_dataset, train_real_dataset

    cfg = make_config(tmp_path)
    df, _ = build_dataset(cfg, make_client(FakeOpenMeteo()))
    df.to_csv(cfg.output_csv, index=False)
    ds_cfg = write_dataset_config(cfg, cfg.output_csv)
    loaded, _ = load_custom_dataset(load_dataset_config(ds_cfg))
    assert {"et0_mm", "wind_speed_ms", "forecast_rain_24h_mm"} <= set(loaded.columns)
    check = check_real_dataset(ds_cfg)
    assert check.ok, check.problems
    summary = train_real_dataset(ds_cfg, fast=True, model_path=None, reports_dir=tmp_path / "rep", stride_hours=3)
    assert summary["selected"]["classifier"]


def test_live_provider_forecast_et0_and_fallback():
    provider = OpenMeteoWeatherProvider(LON, client=make_client(FakeOpenMeteo()))
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    fc = provider.forecast(LAT, now, 24)
    assert fc.source == "open-meteo" and len(fc.hourly) == 24 and "et0_mm" in fc.hourly[0]
    obs = provider.observed_hourly(LAT, now - timedelta(hours=48), now)
    assert len(obs) == 49 and obs["et0_mm"].notna().all()
    readings = provider.recent_readings(LAT, 24)
    assert len(readings) == 24 and readings["soil_moisture"].between(5, 50).all()
    assert set(FORECAST_SOIL_LAYERS) <= {"soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm",
                                         "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm"}

    offline = OpenMeteoWeatherProvider(LON, client=OpenMeteoClient(httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(503))), retries=1))
    fc2 = offline.forecast(LAT, now, 24)
    assert "Open-Meteo unreachable" in fc2.source
    assert offline.observed_hourly(LAT, now - timedelta(hours=5), now).empty


def test_engine_uses_open_meteo_et0_and_field_longitude():
    fake = FakeOpenMeteo()
    provider = OpenMeteoWeatherProvider(0.0, client=make_client(fake))
    engine = RecommendationEngine(weather=provider)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    field = FieldConfig("1", "groundnut", "sandy_loam", "drip", (now - timedelta(days=60)).date(), 1000, 60, LAT,
                        longitude=LON)
    ts = [now - timedelta(hours=h) for h in range(47, -1, -1)]
    readings = pd.DataFrame({"timestamp": ts, "soil_moisture": np.linspace(16, 15, 48), "temperature_c": 30.0,
                             "humidity_pct": 60.0, "rainfall_mm": 0.0})
    rec = engine.recommend(field, readings, now)
    assert rec.forecast_source == "open-meteo"
    assert all(c.url.params["longitude"] == str(LON) for c in fake.calls)
    # ETc uses the mocked Penman-Monteith ET0 (24 h sum ~ 4.6 mm), not Hargreaves on a flat 30 °C day
    obs = provider.observed_hourly(LAT, now - timedelta(hours=23), now)
    assert rec.etc_24h_mm == pytest.approx(rec.kc * obs["et0_mm"].sum(), rel=0.05)


def test_api_provider_switch(settings):
    from dataclasses import replace

    from irrigation_advisor.api.main import build_weather_provider
    from irrigation_advisor.data.weather_provider import SimulatedWeatherProvider

    assert isinstance(build_weather_provider(settings), SimulatedWeatherProvider)
    assert isinstance(build_weather_provider(replace(settings, weather_provider="openmeteo")),
                      OpenMeteoWeatherProvider)


def test_studio_keeps_open_meteo_columns(tmp_path):
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from irrigation_gui.tasks import TaskRunner
    from real_dataset_gui import RealDatasetStudio

    app = QApplication.instance() or QApplication([])
    cfg = make_config(tmp_path)
    df, _ = build_dataset(cfg, make_client(FakeOpenMeteo()))
    df.to_csv(cfg.output_csv, index=False)
    ds_cfg = write_dataset_config(cfg, cfg.output_csv)
    win = RealDatasetStudio(ds_cfg, tasks=TaskRunner(synchronous=True))
    assert win.config_tab.save() is True
    saved = json.loads(ds_cfg.read_text())
    assert {"et0_mm", "wind_speed_ms", "forecast_rain_24h_mm"} <= set(saved["column_map"])
    win.check_tab.run_check()
    assert win.check_tab.verdict.text() == "READY FOR TRAINING"
    win.deleteLater()
    app.processEvents()
