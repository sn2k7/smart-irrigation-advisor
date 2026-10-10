"""Ground-truth bucket model used by the simulators."""

from __future__ import annotations

import numpy as np
import pytest

from irrigation_advisor.agronomy.crops import get_crop, get_soil
from irrigation_advisor.agronomy.soil_model import SoilWaterModel, infiltrated_rain, project_max_depletion


@pytest.fixture()
def model() -> SoilWaterModel:
    return SoilWaterModel(get_soil("loam"), get_crop("maize"), theta=0.22)


def test_et_dries_the_soil(model):
    before = model.theta
    result = model.step(rain_mm=0, irrigation_net_mm=0, et0_mm=0.6, days_since_planting=80)
    assert model.theta < before
    assert result.et_actual_mm == pytest.approx(result.et_potential_mm)  # no stress at 0.22


def test_rain_and_irrigation_wet_the_soil(model):
    before = model.theta
    model.step(rain_mm=10, irrigation_net_mm=5, et0_mm=0, days_since_planting=80)
    assert model.theta > before


def test_water_above_field_capacity_drains(model):
    model.theta = 0.35
    res = model.step(0, 0, 0, 80)
    assert res.drainage_mm > 0
    assert model.theta < 0.35


def test_state_stays_within_physical_bounds(model):
    for _ in range(50):
        model.step(200, 0, 0, 80)
    assert model.theta <= model.soil.saturation
    for _ in range(2000):
        model.step(0, 0, 1.0, 80)
    assert model.theta >= 0.85 * model.soil.wilting_point


def test_stress_reduces_actual_et():
    m = SoilWaterModel(get_soil("loam"), get_crop("maize"), theta=0.14)
    res = m.step(0, 0, 0.6, 80)
    assert res.ks < 1
    assert res.et_actual_mm < res.et_potential_mm


def test_interception_and_runoff():
    assert infiltrated_rain(0.05) == 0
    assert infiltrated_rain(10) == pytest.approx(9.9)
    assert infiltrated_rain(40) < 40 - 0.1


def test_projection_without_rain_only_increases_depletion():
    soil, crop = get_soil("loam"), get_crop("maize")
    max_dr, end_dr = project_max_depletion(0.22, soil, crop, 80, np.zeros(24), np.full(24, 0.25))
    start_dr = 1000 * (0.27 - 0.22) * 1.2
    assert end_dr > start_dr
    assert max_dr == pytest.approx(end_dr)
