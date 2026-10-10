"""Crop, soil and irrigation-method reference parameters.

Values are typical figures taken from FAO Irrigation and Drainage Paper 56
(Allen et al., 1998):

* Crop coefficients (Kc) - Table 12, single crop coefficients.
* Growth-stage lengths - Table 11 (one representative climate/region each).
* Maximum effective root depth and depletion fraction ``p`` - Table 22.
* Soil water characteristics (field capacity, wilting point) - Table 19,
  mid-points of the published ranges.

They are generic defaults for a decision-support prototype. A real
deployment should calibrate them with local agronomists / soil tests.

Paddy rice and perennial crops (sugarcane) are deliberately out of scope:
flooded paddies are managed by ponding depth, not root-zone depletion.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CropProfile:
    """FAO-56 style description of a crop."""

    name: str
    kc_ini: float
    kc_mid: float
    kc_end: float
    #: Length (days) of the initial, development, mid-season and late stages.
    stage_days: tuple[int, int, int, int]
    root_depth_min_m: float
    root_depth_max_m: float
    #: Fraction of total available water that can be depleted before stress (p).
    depletion_fraction: float

    @property
    def season_days(self) -> int:
        """Total growing-season length in days."""
        return sum(self.stage_days)


@dataclass(frozen=True)
class SoilProfile:
    """Volumetric soil water characteristics (m3/m3)."""

    name: str
    field_capacity: float
    wilting_point: float
    saturation: float
    #: Fraction of water above field capacity that drains each hour.
    drainage_rate_per_hour: float

    @property
    def available_water_per_m(self) -> float:
        """Plant-available water per metre of root depth (mm/m)."""
        return 1000.0 * (self.field_capacity - self.wilting_point)


@dataclass(frozen=True)
class IrrigationMethod:
    """Irrigation application method and its typical application efficiency."""

    name: str
    efficiency: float


CROPS: dict[str, CropProfile] = {
    "maize": CropProfile("maize", 0.30, 1.20, 0.60, (30, 40, 50, 30), 0.30, 1.20, 0.55),
    "wheat": CropProfile("wheat", 0.40, 1.15, 0.40, (20, 25, 60, 30), 0.30, 1.20, 0.55),
    "tomato": CropProfile("tomato", 0.60, 1.15, 0.80, (30, 40, 40, 25), 0.25, 1.00, 0.40),
    "potato": CropProfile("potato", 0.50, 1.15, 0.75, (25, 30, 45, 30), 0.25, 0.60, 0.35),
    "cotton": CropProfile("cotton", 0.35, 1.18, 0.60, (30, 50, 60, 55), 0.30, 1.40, 0.65),
    "onion": CropProfile("onion", 0.70, 1.05, 0.75, (15, 25, 70, 40), 0.20, 0.50, 0.30),
    "groundnut": CropProfile("groundnut", 0.40, 1.15, 0.60, (25, 35, 45, 25), 0.30, 0.80, 0.50),
}

SOILS: dict[str, SoilProfile] = {
    "sandy": SoilProfile("sandy", 0.12, 0.045, 0.38, 0.50),
    "sandy_loam": SoilProfile("sandy_loam", 0.23, 0.11, 0.41, 0.30),
    "loam": SoilProfile("loam", 0.27, 0.12, 0.43, 0.20),
    "clay_loam": SoilProfile("clay_loam", 0.32, 0.18, 0.47, 0.12),
    "clay": SoilProfile("clay", 0.36, 0.22, 0.50, 0.08),
}

IRRIGATION_METHODS: dict[str, IrrigationMethod] = {
    "drip": IrrigationMethod("drip", 0.90),
    "sprinkler": IrrigationMethod("sprinkler", 0.75),
    "furrow": IrrigationMethod("furrow", 0.60),
}

GROWTH_STAGES: tuple[str, ...] = ("initial", "development", "mid_season", "late_season", "post_harvest")


def get_crop(name: str) -> CropProfile:
    """Return a crop profile or raise ``KeyError`` with the valid options."""
    key = str(name).strip().lower()
    if key not in CROPS:
        raise KeyError(f"Unknown crop '{name}'. Valid crops: {sorted(CROPS)}")
    return CROPS[key]


def get_soil(name: str) -> SoilProfile:
    """Return a soil profile or raise ``KeyError`` with the valid options."""
    key = str(name).strip().lower()
    if key not in SOILS:
        raise KeyError(f"Unknown soil '{name}'. Valid soils: {sorted(SOILS)}")
    return SOILS[key]


def get_method(name: str) -> IrrigationMethod:
    """Return an irrigation method or raise ``KeyError``."""
    key = str(name).strip().lower()
    if key not in IRRIGATION_METHODS:
        raise KeyError(f"Unknown irrigation method '{name}'. Valid: {sorted(IRRIGATION_METHODS)}")
    return IRRIGATION_METHODS[key]
