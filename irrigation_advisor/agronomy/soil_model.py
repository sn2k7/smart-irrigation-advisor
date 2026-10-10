"""Hourly 'ground-truth' soil water bucket model.

Used by the synthetic data generator and by the closed-loop strategy
simulator to evolve the *true* soil moisture of a virtual field. Models and
the recommendation engine never access this state directly - they only see
noisy sensor readings derived from it.

Processes per hour:
* rain infiltration with canopy interception and an intensity-dependent
  runoff loss,
* irrigation (net depth that reaches the root zone),
* actual ET = Ks * Kc * ET0 (FAO-56 water-stress reduction),
* drainage of water held above field capacity (exponential, soil-specific).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crops import CropProfile, SoilProfile
from .water_balance import crop_coefficient, root_depth, water_stress_coefficient

INTERCEPTION_MM_PER_HOUR = 0.1
RUNOFF_INTENSITY_MM_PER_HOUR = 15.0
RUNOFF_FRACTION_ABOVE_THRESHOLD = 0.3


def infiltrated_rain(rain_mm: float) -> float:
    """Rain entering the soil in one hour after interception and runoff."""
    water = max(0.0, rain_mm - INTERCEPTION_MM_PER_HOUR)
    if water > RUNOFF_INTENSITY_MM_PER_HOUR:
        water = RUNOFF_INTENSITY_MM_PER_HOUR + (1 - RUNOFF_FRACTION_ABOVE_THRESHOLD) * (
            water - RUNOFF_INTENSITY_MM_PER_HOUR
        )
    return water


@dataclass
class HourResult:
    """Fluxes (mm) computed for one simulated hour."""

    infiltration_mm: float
    et_actual_mm: float
    et_potential_mm: float
    drainage_mm: float
    ks: float


@dataclass
class SoilWaterModel:
    """Mutable bucket state of one field (``theta`` in m3/m3)."""

    soil: SoilProfile
    crop: CropProfile
    theta: float

    def _bounds(self) -> tuple[float, float]:
        return 0.85 * self.soil.wilting_point, self.soil.saturation

    def status(self, days_since_planting: float) -> tuple[float, float, float, float]:
        """Return (root depth m, TAW mm, RAW mm, depletion mm) for the current state."""
        zr = root_depth(self.crop, days_since_planting)
        taw = 1000.0 * (self.soil.field_capacity - self.soil.wilting_point) * zr
        raw = self.crop.depletion_fraction * taw
        dr = 1000.0 * (self.soil.field_capacity - self.theta) * zr
        return zr, taw, raw, dr

    def step(self, rain_mm: float, irrigation_net_mm: float, et0_mm: float, days_since_planting: float) -> HourResult:
        """Advance the bucket by one hour."""
        zr, taw, raw, _ = self.status(days_since_planting)
        infiltration = infiltrated_rain(rain_mm) + max(0.0, irrigation_net_mm)
        self.theta += infiltration / (1000.0 * zr)
        dr = 1000.0 * (self.soil.field_capacity - self.theta) * zr
        ks = water_stress_coefficient(dr, taw, raw)
        et_pot = crop_coefficient(self.crop, days_since_planting) * max(0.0, et0_mm)
        et_act = ks * et_pot
        self.theta -= et_act / (1000.0 * zr)
        drainage = 0.0
        if self.theta > self.soil.field_capacity:
            excess = self.theta - self.soil.field_capacity
            drained = self.soil.drainage_rate_per_hour * excess
            self.theta -= drained
            drainage = drained * 1000.0 * zr
        low, high = self._bounds()
        if self.theta > high:  # saturated: extra water runs off
            self.theta = high
        self.theta = max(self.theta, low)
        return HourResult(infiltration, et_act, et_pot, drainage, ks)


def project_max_depletion(
    theta: float,
    soil: SoilProfile,
    crop: CropProfile,
    days_since_planting: float,
    future_rain_mm: np.ndarray,
    future_et0_mm: np.ndarray,
) -> tuple[float, float]:
    """Counterfactual run *without irrigation* over the given future hours.

    Returns:
        (maximum depletion reached incl. the start state, depletion at the end)
    """
    model = SoilWaterModel(soil, crop, theta)
    _, _, _, dr = model.status(days_since_planting)
    max_dr = dr
    for i, (rain, et0) in enumerate(zip(future_rain_mm, future_et0_mm)):
        model.step(float(rain), 0.0, float(et0), days_since_planting + (i + 1) / 24.0)
        dr = model.status(days_since_planting)[3]
        max_dr = max(max_dr, dr)
    return max_dr, dr
