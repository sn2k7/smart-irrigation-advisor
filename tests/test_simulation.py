"""Closed-loop simulation and strategy comparison."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from irrigation_advisor.data.simulator import FieldSpec
from irrigation_advisor.engine.recommender import RecommendationEngine
from irrigation_advisor.simulation.comparison import run_comparison, summarise, write_simulation_reports
from irrigation_advisor.simulation.field_env import VirtualField
from irrigation_advisor.simulation.strategies import AdvisorStrategy, DecisionContext, FixedScheduleStrategy


@pytest.fixture(scope="module")
def comparison():
    return run_comparison(n_fields=2, seed=11, model=None, fault_rate=1.0)


def test_all_strategies_run_on_every_field(comparison):
    table, results = comparison
    assert set(table["strategy"]) == {"fixed_schedule", "fixed_skip_after_rain", "calendar_kc_adjusted", "advisor_rules"}
    assert len(table) == 2 * 4
    assert (table["irrigation_gross_mm"] >= 0).all()
    assert table["relative_transpiration"].between(0, 1).all()
    assert all(len(r.trace) > 0 for r in results)


def test_strategies_share_identical_weather(comparison):
    table, _ = comparison
    assert table.groupby("field_id")["rain_mm"].nunique().eq(1).all()


def test_summary_and_reports(comparison, tmp_path):
    table, results = comparison
    summary, comp = summarise(table)
    assert "advisor_rules" in comp["vs_baseline"]
    paths = write_simulation_reports(table, results, tmp_path, {"seed": 11, "fixed_interval_days": 3, "fault_rate": 1.0})
    text = (tmp_path / "simulation_report.md").read_text(encoding="utf-8")
    assert "SIMULATED RESULTS" in text
    assert all(pd.io.common.file_exists(p) for p in paths.values())


def test_fixed_schedule_depth_and_rain_skip():
    spec = FieldSpec("X", "maize", "loam", "drip", date(2025, 6, 1), 18.5, 1000, 60, "fixed_interval", 3, 20, 0.0, 5)
    vf = VirtualField(spec, fault_rate=0)
    strategy = FixedScheduleStrategy(3, skip_after_rain_mm=10)
    expected = 3 * 1.20 * 5.0 / 0.9
    assert strategy.depth_for(vf.field_cfg) == pytest.approx(expected)
    history = pd.DataFrame({"timestamp": pd.date_range("2025-07-01", periods=24, freq="h"), "rainfall_mm": 1.0})
    ctx = DecisionContext(datetime(2025, 7, 2, 0), vf.field_cfg, 3, history, 0.0)
    assert strategy.decide(ctx).gross_depth_mm == 0  # 24 mm rain in last 24 h
    assert FixedScheduleStrategy(3).decide(ctx).gross_depth_mm == pytest.approx(expected)
    assert FixedScheduleStrategy(3).decide(DecisionContext(datetime(2025, 7, 2), vf.field_cfg, 4, history, 0)).gross_depth_mm == 0
    kc_strategy = FixedScheduleStrategy(3, kc_adjusted=True)
    assert kc_strategy.depth_for(vf.field_cfg, 0) == pytest.approx(3 * 0.30 * 5.0 / 0.9)  # initial-stage Kc
    with pytest.raises(ValueError):
        FixedScheduleStrategy(0)


def test_advisor_reduces_water_vs_pure_fixed_schedule_on_a_wet_season_field():
    # Monsoon-season field: a calendar schedule irrigates through the rain.
    spec = FieldSpec("W", "maize", "loam", "drip", date(2025, 6, 10), 18.5, 1000, 60, "fixed_interval", 3, 20, 0.0, 9)
    vf = VirtualField(spec, fault_rate=0)
    fixed = vf.run(FixedScheduleStrategy(3))
    advisor = vf.run(AdvisorStrategy(RecommendationEngine(), "advisor_rules"))
    assert advisor.metrics["irrigation_gross_mm"] < fixed.metrics["irrigation_gross_mm"]
    assert advisor.metrics["relative_transpiration"] > 0.9
