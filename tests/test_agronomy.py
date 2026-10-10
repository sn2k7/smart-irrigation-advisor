"""FAO-56 calculations against hand-computed / published values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from irrigation_advisor.agronomy.crops import CROPS, SOILS, get_crop, get_method, get_soil
from irrigation_advisor.agronomy.water_balance import (
    crop_coefficient,
    depletion_from_moisture,
    effective_rainfall,
    extraterrestrial_radiation,
    growth_stage,
    hargreaves_et0,
    hourly_et0_fraction,
    litres_to_mm,
    moisture_from_depletion,
    net_irrigation_requirement,
    plan_irrigation,
    projected_depletion,
    root_depth,
    soil_water_status,
    water_stress_coefficient,
)
from irrigation_advisor.data.features import kc_array, root_depth_array, stage_array


def test_extraterrestrial_radiation_matches_fao56_example_8():
    # FAO-56 Example 8: latitude 20 deg S, 3 September (day 246) -> Ra = 32.2 MJ m-2 day-1
    assert float(extraterrestrial_radiation(-20.0, 246)) == pytest.approx(32.2, abs=0.1)


def test_hargreaves_is_plausible_for_hot_dry_day():
    et0 = float(hargreaves_et0(24.0, 38.0, 18.5, 130))
    assert 5.0 < et0 < 8.5
    # wider temperature range -> higher ET0
    assert float(hargreaves_et0(20.0, 38.0, 18.5, 130)) > et0
    # never negative
    assert float(hargreaves_et0(30.0, 29.0, 18.5, 130)) == 0.0


def test_crop_coefficient_follows_stage_curve():
    maize = get_crop("maize")  # stages 30/40/50/30, Kc 0.30/1.20/0.60
    assert crop_coefficient(maize, 0) == pytest.approx(0.30)
    assert crop_coefficient(maize, 50) == pytest.approx(0.30 + 0.9 * 20 / 40)
    assert crop_coefficient(maize, 100) == pytest.approx(1.20)
    assert crop_coefficient(maize, 135) == pytest.approx(1.20 - 0.6 * 15 / 30)
    assert crop_coefficient(maize, 500) == pytest.approx(0.60)
    assert growth_stage(maize, 10) == "initial"
    assert growth_stage(maize, 45) == "development"
    assert growth_stage(maize, 80) == "mid_season"
    assert growth_stage(maize, 140) == "late_season"
    assert growth_stage(maize, 200) == "post_harvest"


@pytest.mark.parametrize("crop_name", sorted(CROPS))
def test_vectorised_helpers_equal_scalar_versions(crop_name):
    crop = CROPS[crop_name]
    days = np.linspace(-5, crop.season_days + 20, 97)
    np.testing.assert_allclose(kc_array(crop, days), [crop_coefficient(crop, d) for d in days], atol=1e-9)
    np.testing.assert_allclose(root_depth_array(crop, days), [root_depth(crop, d) for d in days], atol=1e-9)
    assert list(stage_array(crop, days)) == [growth_stage(crop, d) for d in days]


def test_soil_water_status_numbers():
    soil, crop = get_soil("loam"), get_crop("maize")
    status = soil_water_status(20.0, soil, crop, 100)  # full root depth 1.2 m
    assert status.root_depth_m == pytest.approx(1.2)
    assert status.taw_mm == pytest.approx(1000 * (0.27 - 0.12) * 1.2)  # 180 mm
    assert status.raw_mm == pytest.approx(0.55 * 180)
    assert status.depletion_mm == pytest.approx(1000 * (0.27 - 0.20) * 1.2)  # 84 mm
    assert status.depletion_ratio == pytest.approx(84 / 99)
    assert moisture_from_depletion(status.depletion_mm, soil, 1.2) == pytest.approx(20.0)


def test_depletion_negative_above_field_capacity():
    assert depletion_from_moisture(30.0, get_soil("loam"), 1.0) < 0


def test_water_stress_coefficient():
    assert water_stress_coefficient(10, 100, 50) == 1.0
    assert water_stress_coefficient(75, 100, 50) == pytest.approx(0.5)
    assert water_stress_coefficient(100, 100, 50) == 0.0


def test_effective_rainfall_rule():
    assert effective_rainfall(1.5) == 0.0
    assert effective_rainfall(10.0) == pytest.approx(8.0)
    assert effective_rainfall(-3) == 0.0


def test_net_requirement_and_projection():
    assert projected_depletion(40, 5, 8) == pytest.approx(37)
    # refill 40 mm minus 8 mm rain credit
    assert net_irrigation_requirement(40, 5, 8, 100) == pytest.approx(32)
    # at least one day of ETc
    assert net_irrigation_requirement(3, 5, 0, 100) == pytest.approx(5)
    # never above TAW
    assert net_irrigation_requirement(150, 5, 0, 100) == pytest.approx(100)


def test_plan_irrigation_arithmetic():
    plan = plan_irrigation(9.0, "drip", area_m2=1000, flow_rate_lpm=100)
    assert plan.gross_depth_mm == pytest.approx(10.0)
    assert plan.water_volume_l == pytest.approx(10_000)
    assert plan.duration_min == pytest.approx(100)
    assert not plan.capped


def test_plan_irrigation_caps_duration_proportionally():
    plan = plan_irrigation(9.0, "drip", area_m2=1000, flow_rate_lpm=100, max_duration_min=50)
    assert plan.capped
    assert plan.duration_min == pytest.approx(50)
    assert plan.water_volume_l == pytest.approx(5_000)
    assert plan.net_depth_mm == pytest.approx(4.5)


def test_plan_irrigation_rejects_bad_inputs():
    with pytest.raises(ValueError):
        plan_irrigation(5, "drip", area_m2=0, flow_rate_lpm=10)
    with pytest.raises(ValueError):
        plan_irrigation(5, "drip", area_m2=10, flow_rate_lpm=0)
    with pytest.raises(KeyError):
        plan_irrigation(5, "teleport", area_m2=10, flow_rate_lpm=1)


def test_hourly_et0_fraction_sums_to_one():
    assert sum(hourly_et0_fraction(h) for h in range(24)) == pytest.approx(1.0)
    assert hourly_et0_fraction(2) == 0.0


def test_reference_tables_are_consistent():
    for soil in SOILS.values():
        assert 0 < soil.wilting_point < soil.field_capacity < soil.saturation < 1
    for crop in CROPS.values():
        assert 0 < crop.depletion_fraction < 1
        assert crop.root_depth_min_m < crop.root_depth_max_m
    assert 0 < get_method("furrow").efficiency < get_method("drip").efficiency <= 1
    assert litres_to_mm(10_000, 1000) == pytest.approx(10)
    assert math.isclose(get_soil("LOAM").field_capacity, 0.27)
    with pytest.raises(KeyError):
        get_crop("rice")
