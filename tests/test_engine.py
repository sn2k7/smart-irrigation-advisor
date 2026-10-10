"""Recommendation engine: rule path, ML hybrid, safety rules and fallbacks."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from irrigation_advisor.data.features import build_features
from irrigation_advisor.data.preprocessing import prepare_sensor_frame
from irrigation_advisor.engine.recommender import FieldConfig, RecommendationEngine, field_config_from_row
from irrigation_advisor.ml.evaluation import rule_baseline_predictions

from .conftest import StubModel, make_readings

NOW = datetime(2025, 9, 10, 7, 0)  # maize planted 2025-06-01 -> mid season, Zr 1.2 m
# loam FC 27 %, WP 12 %; Zr 1.2 m -> TAW 180 mm, RAW 99 mm.
# Dr = 12 * (27 - SM) mm  ->  RAW reached at SM = 18.75 %.


@pytest.fixture()
def engine() -> RecommendationEngine:
    return RecommendationEngine()


def test_dry_field_needs_irrigation(engine, field_cfg):
    rec = engine.recommend(field_cfg, make_readings(NOW, 48, 18.6, 17.0), NOW, forecast_rain_24h_mm=0)
    assert rec.irrigation_needed
    assert rec.decision_source == "rule_based"
    assert rec.depletion_mm == pytest.approx(12 * (27 - 17.0), abs=0.5)
    # net = max(Dr - 0, ETc) = Dr ; gross = net / 0.9 ; litres = gross * 2000 ; minutes = litres / 120
    assert rec.gross_depth_mm == pytest.approx(rec.net_depth_mm / 0.9, rel=1e-3)
    assert rec.water_volume_l == pytest.approx(rec.gross_depth_mm * 2000, rel=1e-3)
    assert rec.duration_min == pytest.approx(rec.water_volume_l / 120, rel=1e-3)
    assert any("RAW" in r for r in rec.reasons)


def test_moist_field_does_not_need_irrigation(engine, field_cfg):
    rec = engine.recommend(field_cfg, make_readings(NOW, 48, 25.5, 25.0), NOW, forecast_rain_24h_mm=0)
    assert not rec.irrigation_needed
    assert rec.water_volume_l == 0 and rec.duration_min == 0
    assert "No irrigation" in rec.summary


def test_forecast_rain_cancels_borderline_irrigation(engine, field_cfg):
    readings = make_readings(NOW, 48, 19.6, 19.0)  # Dr ~ 96 mm, RAW 99 mm reached within 24 h
    dry = engine.recommend(field_cfg, readings, NOW, forecast_rain_24h_mm=0)
    wet = engine.recommend(field_cfg, readings, NOW, forecast_rain_24h_mm=30)
    assert dry.irrigation_needed and not wet.irrigation_needed
    assert wet.forecast_effective_rain_mm == pytest.approx(24.0)


def test_rain_in_progress_defers(engine, field_cfg):
    readings = make_readings(NOW, 48, 18.6, 17.0)
    readings.loc[readings.index[-3:], "rainfall_mm"] = 3.0
    rec = engine.recommend(field_cfg, readings, NOW, forecast_rain_24h_mm=0)
    assert not rec.irrigation_needed
    assert rec.next_check_hours == 3
    assert any("Rain-aware deferral" in r for r in rec.reasons)


def test_never_irrigate_above_field_capacity(field_cfg):
    engine = RecommendationEngine(model=StubModel(probability=0.99))
    rec = engine.recommend(field_cfg, make_readings(NOW, 48, 31.0, 30.0), NOW, forecast_rain_24h_mm=0)
    assert not rec.irrigation_needed
    assert any("field capacity" in r for r in rec.reasons)


def test_severe_stress_overrides_ml_no(field_cfg):
    engine = RecommendationEngine(model=StubModel(probability=0.05))
    rec = engine.recommend(field_cfg, make_readings(NOW, 48, 15.0, 13.5), NOW, forecast_rain_24h_mm=0)
    assert rec.ml_decision is False
    assert rec.irrigation_needed
    assert any("Safety override" in r for r in rec.reasons)


def test_ml_hybrid_uses_model_probability_and_depth(field_cfg):
    engine = RecommendationEngine(model=StubModel(probability=0.9, depth=20.0))
    rec = engine.recommend(field_cfg, make_readings(NOW, 48, 20.5, 20.0), NOW, forecast_rain_24h_mm=0)
    assert rec.decision_source == "ml_hybrid"
    assert rec.irrigation_needed and rec.probability == pytest.approx(0.9)
    assert rec.net_depth_mm == pytest.approx(20.0)
    assert rec.model_version == "stub-1"
    # rule says no at Dr ~ 84 mm -> disagreement is surfaced
    assert rec.rule_decision is False
    assert any("disagree" in w for w in rec.warnings)


def test_estimated_fallback_when_latest_soil_reading_invalid(engine, field_cfg):
    readings = make_readings(NOW, 48, 19.5, 18.0)
    readings.loc[readings.index[-5:], "soil_moisture"] = np.nan
    rec = engine.recommend(field_cfg, readings, NOW, forecast_rain_24h_mm=0)
    assert rec.decision_source == "fallback_estimated"
    assert rec.soil_moisture_estimated and rec.confidence == "low"
    # extrapolated depletion is larger than at the last valid reading
    assert rec.depletion_mm > 12 * (27 - 18.4)


def test_out_of_range_latest_reading_triggers_fallback(engine, field_cfg):
    readings = make_readings(NOW, 48, 20.0, 19.0)
    readings.loc[readings.index[-1], "soil_moisture"] = 95.0
    rec = engine.recommend(field_cfg, readings, NOW, forecast_rain_24h_mm=0)
    assert rec.decision_source == "fallback_estimated"
    assert any(i["code"] == "out_of_range" for i in rec.sensor_issues)


def test_conservative_fallback_without_any_data(engine, field_cfg):
    rec = engine.recommend(field_cfg, make_readings(NOW, 1, 20, 20).iloc[:0], NOW, forecast_rain_24h_mm=0)
    assert rec.decision_source == "fallback_conservative"
    assert rec.irrigation_needed  # replaces one day of ETc when no rain is expected
    assert rec.net_depth_mm == pytest.approx(max(2.0, rec.etc_24h_mm), rel=1e-2)
    rainy = engine.recommend(field_cfg, make_readings(NOW, 1, 20, 20).iloc[:0], NOW, forecast_rain_24h_mm=40)
    assert not rainy.irrigation_needed


def test_stale_data_uses_conservative_fallback(engine, field_cfg):
    readings = make_readings(datetime(2025, 9, 8, 7), 24, 20, 19)
    rec = engine.recommend(field_cfg, readings, NOW, forecast_rain_24h_mm=0)
    assert rec.decision_source == "fallback_conservative"


def test_duration_is_capped(engine):
    cfg = FieldConfig("9", "maize", "loam", "furrow", NOW.date().replace(month=6, day=1), 5000, 50, 18.5,
                      max_irrigation_minutes=60)
    rec = engine.recommend(cfg, make_readings(NOW, 48, 16, 15), NOW, forecast_rain_24h_mm=0)
    assert rec.irrigation_needed and rec.capped
    assert rec.duration_min == pytest.approx(60)
    assert rec.water_volume_l == pytest.approx(60 * 50)


def test_simulated_forecast_is_labelled(engine, field_cfg):
    rec = engine.recommend(field_cfg, make_readings(NOW, 48, 22, 21), NOW)
    assert rec.forecast_source == "simulated-offline"
    assert any("SIMULATED" in w for w in rec.warnings)


def test_engine_rule_matches_vectorised_training_baseline(engine, field_cfg):
    """The engine's rule and the rule baseline reported in training are the same function."""
    for start, end, rain in [(18.6, 17.0, 0), (19.6, 19.0, 0), (19.6, 19.0, 30), (25.0, 24.0, 0)]:
        readings = make_readings(NOW, 48, start, end)
        rec = engine.recommend(field_cfg, readings, NOW, forecast_rain_24h_mm=rain)
        frame = readings.assign(field_id="1", crop_type="maize", soil_type="loam", irrigation_method="drip",
                                planting_date="2025-06-01", latitude=18.5, forecast_rain_24h_mm=rain)
        feats = build_features(prepare_sensor_frame(frame))
        decision, depth = rule_baseline_predictions(feats.iloc[[-1]])
        assert bool(decision[0]) == rec.rule_decision


def test_field_config_validation():
    with pytest.raises(KeyError):
        FieldConfig("1", "rice", "loam", "drip", NOW.date(), 100, 10, 18)
    with pytest.raises(ValueError):
        FieldConfig("1", "maize", "loam", "drip", NOW.date(), -1, 10, 18)
    cfg = field_config_from_row({"id": 3, "crop_type": "onion", "soil_type": "clay", "irrigation_method": "furrow",
                                 "planting_date": "2025-01-01", "area_m2": 100, "flow_rate_lpm": 10, "latitude": 18})
    assert cfg.field_id == "3" and cfg.max_irrigation_minutes == 180


def test_engine_with_real_trained_model(trained_model, field_cfg):
    engine = RecommendationEngine(model=trained_model)
    rec = engine.recommend(field_cfg, make_readings(NOW, 72, 19.5, 17.5), NOW, forecast_rain_24h_mm=0)
    assert rec.decision_source in {"ml_hybrid", "rule_based"}
    assert rec.irrigation_needed  # far beyond RAW: model or safety rule must irrigate
    assert rec.to_dict()["decision_time"] == NOW.isoformat()
