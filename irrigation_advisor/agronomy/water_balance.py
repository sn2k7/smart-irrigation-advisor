"""Root-zone water balance and irrigation arithmetic (FAO-56 methodology).

Key quantities (all depths in millimetres of water):

* ``TAW`` - total available water, ``1000 * (FC - WP) * Zr``.
* ``RAW`` - readily available water, ``p * TAW``. Crops start to suffer
  water stress once the root-zone depletion exceeds RAW.
* ``Dr``  - root-zone depletion, ``1000 * (FC - theta) * Zr``; 0 means the
  root zone is at field capacity.
* ``ETc`` - crop evapotranspiration ``Kc * ET0``.

Unit identity used throughout: 1 mm of water over 1 m2 = 1 litre.

All functions are pure and are shared by the synthetic data generator, the
label generator, the rule-based engine and the closed-loop simulator so the
same physics is used everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .crops import CropProfile, SoilProfile, get_method

SOLAR_CONSTANT_MJ = 0.0820  # MJ m-2 min-1 (FAO-56 eq. 21)
MIN_EFFECTIVE_RAIN_MM = 2.0  # daily rain below this is mostly intercepted / evaporated
EFFECTIVE_RAIN_FRACTION = 0.8  # share of larger events that reaches the root zone


# ---------------------------------------------------------------------------
# Reference evapotranspiration
# ---------------------------------------------------------------------------
def extraterrestrial_radiation(latitude_deg: float | np.ndarray, day_of_year: int | np.ndarray) -> np.ndarray:
    """Daily extraterrestrial radiation Ra in MJ m-2 day-1 (FAO-56 eqs. 21-25)."""
    lat = np.radians(np.asarray(latitude_deg, dtype=float))
    doy = np.asarray(day_of_year, dtype=float)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    decl = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(decl), -1.0, 1.0))
    return (24.0 * 60.0 / np.pi) * SOLAR_CONSTANT_MJ * dr * (
        ws * np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.sin(ws)
    )


def hargreaves_et0(
    t_min: float | np.ndarray,
    t_max: float | np.ndarray,
    latitude_deg: float | np.ndarray,
    day_of_year: int | np.ndarray,
) -> np.ndarray:
    """Hargreaves-Samani reference ET0 in mm/day (FAO-56 eq. 52).

    Only temperature is required, which suits low-cost farm sensors without
    radiation or wind measurements.
    """
    t_min_a = np.asarray(t_min, dtype=float)
    t_max_a = np.asarray(t_max, dtype=float)
    t_mean = (t_min_a + t_max_a) / 2.0
    t_range = np.clip(t_max_a - t_min_a, 0.0, None)
    ra_mm = 0.408 * extraterrestrial_radiation(latitude_deg, day_of_year)  # MJ -> mm equivalent
    return np.clip(0.0023 * (t_mean + 17.8) * np.sqrt(t_range) * ra_mm, 0.0, None)


# ---------------------------------------------------------------------------
# Crop development
# ---------------------------------------------------------------------------
def growth_stage(crop: CropProfile, days_since_planting: float) -> str:
    """Return the FAO-56 growth stage name for the given crop age."""
    ini, dev, mid, late = crop.stage_days
    d = max(0.0, float(days_since_planting))
    if d < ini:
        return "initial"
    if d < ini + dev:
        return "development"
    if d < ini + dev + mid:
        return "mid_season"
    if d <= ini + dev + mid + late:
        return "late_season"
    return "post_harvest"


def crop_coefficient(crop: CropProfile, days_since_planting: float) -> float:
    """Single crop coefficient Kc with FAO-56 linear interpolation between stages."""
    ini, dev, mid, late = crop.stage_days
    d = max(0.0, float(days_since_planting))
    if d < ini:
        return crop.kc_ini
    if d < ini + dev:
        return crop.kc_ini + (crop.kc_mid - crop.kc_ini) * (d - ini) / dev
    if d < ini + dev + mid:
        return crop.kc_mid
    if d <= ini + dev + mid + late:
        return crop.kc_mid + (crop.kc_end - crop.kc_mid) * (d - ini - dev - mid) / late
    return crop.kc_end


def root_depth(crop: CropProfile, days_since_planting: float) -> float:
    """Effective root depth Zr (m), growing linearly until full canopy."""
    ini, dev, _, _ = crop.stage_days
    frac = min(1.0, max(0.0, float(days_since_planting)) / float(ini + dev))
    return crop.root_depth_min_m + (crop.root_depth_max_m - crop.root_depth_min_m) * frac


# ---------------------------------------------------------------------------
# Soil water status
# ---------------------------------------------------------------------------
def total_available_water(soil: SoilProfile, root_depth_m: float) -> float:
    """TAW in mm."""
    return 1000.0 * (soil.field_capacity - soil.wilting_point) * root_depth_m


def readily_available_water(soil: SoilProfile, crop: CropProfile, root_depth_m: float) -> float:
    """RAW in mm."""
    return crop.depletion_fraction * total_available_water(soil, root_depth_m)


def depletion_from_moisture(soil_moisture_pct: float, soil: SoilProfile, root_depth_m: float) -> float:
    """Root-zone depletion Dr (mm) from volumetric moisture in % (negative = above FC)."""
    return 1000.0 * (soil.field_capacity - soil_moisture_pct / 100.0) * root_depth_m


def water_stress_coefficient(depletion_mm: float, taw_mm: float, raw_mm: float) -> float:
    """FAO-56 Ks (eq. 84): 1 when Dr <= RAW, falling linearly to 0 at TAW."""
    if depletion_mm <= raw_mm:
        return 1.0
    if taw_mm <= raw_mm:
        return 0.0
    return max(0.0, (taw_mm - depletion_mm) / (taw_mm - raw_mm))


def effective_rainfall(rain_total_mm: float) -> float:
    """Planning estimate of rain that reaches the root zone (24 h totals).

    Small events are lost to canopy interception and evaporation, larger
    events lose a share to runoff/percolation. A simple, transparent rule is
    preferred over a site-specific curve-number model for a prototype.
    """
    rain = max(0.0, float(rain_total_mm))
    if rain < MIN_EFFECTIVE_RAIN_MM:
        return 0.0
    return EFFECTIVE_RAIN_FRACTION * rain


@dataclass(frozen=True)
class WaterStatus:
    """Snapshot of root-zone water status derived from a moisture reading."""

    root_depth_m: float
    taw_mm: float
    raw_mm: float
    depletion_mm: float
    kc: float
    growth_stage: str

    @property
    def depletion_ratio(self) -> float:
        """Dr / RAW; values >= 1 mean the stress threshold has been reached."""
        return self.depletion_mm / self.raw_mm if self.raw_mm > 0 else 0.0

    @property
    def available_fraction(self) -> float:
        """Share of TAW still available to the crop (1 = field capacity)."""
        return 1.0 - max(0.0, self.depletion_mm) / self.taw_mm if self.taw_mm > 0 else 0.0


def soil_water_status(
    soil_moisture_pct: float, soil: SoilProfile, crop: CropProfile, days_since_planting: float
) -> WaterStatus:
    """Compute TAW, RAW, Dr, Kc and stage for a single moisture reading."""
    zr = root_depth(crop, days_since_planting)
    return WaterStatus(
        root_depth_m=zr,
        taw_mm=total_available_water(soil, zr),
        raw_mm=readily_available_water(soil, crop, zr),
        depletion_mm=depletion_from_moisture(soil_moisture_pct, soil, zr),
        kc=crop_coefficient(crop, days_since_planting),
        growth_stage=growth_stage(crop, days_since_planting),
    )


def moisture_from_depletion(depletion_mm: float, soil: SoilProfile, root_depth_m: float) -> float:
    """Inverse of :func:`depletion_from_moisture` - returns % VWC."""
    return 100.0 * (soil.field_capacity - depletion_mm / (1000.0 * root_depth_m))


# ---------------------------------------------------------------------------
# Irrigation requirement & scheduling arithmetic
# ---------------------------------------------------------------------------
def projected_depletion(depletion_mm: float, etc_next_mm: float, effective_rain_next_mm: float) -> float:
    """Depletion expected at the end of the planning horizon without irrigation."""
    return max(0.0, depletion_mm) + max(0.0, etc_next_mm) - max(0.0, effective_rain_next_mm)


def net_irrigation_requirement(
    depletion_mm: float, etc_next_mm: float, effective_rain_next_mm: float, taw_mm: float
) -> float:
    """Net depth (mm) to apply when irrigation is needed.

    Refill the root zone to field capacity, credit the effective rain expected
    within the horizon, but apply at least the next day's crop water use so a
    single irrigation is worthwhile. Capped at TAW (more would only drain).
    """
    need = max(max(0.0, depletion_mm) - max(0.0, effective_rain_next_mm), max(0.0, etc_next_mm))
    return float(min(max(need, 0.0), taw_mm))


@dataclass(frozen=True)
class IrrigationPlan:
    """Concrete pump/valve instructions for a field."""

    net_depth_mm: float
    gross_depth_mm: float
    water_volume_l: float
    duration_min: float
    capped: bool
    efficiency: float

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "net_depth_mm": round(self.net_depth_mm, 2),
            "gross_depth_mm": round(self.gross_depth_mm, 2),
            "water_volume_l": round(self.water_volume_l, 1),
            "duration_min": round(self.duration_min, 1),
            "capped": self.capped,
            "efficiency": self.efficiency,
        }


def plan_irrigation(
    net_depth_mm: float,
    irrigation_method: str,
    area_m2: float,
    flow_rate_lpm: float,
    max_duration_min: float | None = None,
) -> IrrigationPlan:
    """Convert a net depth into gross depth, litres and pump run-time.

    gross_mm = net_mm / efficiency
    volume_l = gross_mm * area_m2           (1 mm on 1 m2 = 1 L)
    duration = volume_l / flow_rate_lpm     (minutes)

    If ``max_duration_min`` is exceeded the run is capped and the applied
    depths are reduced proportionally (``capped=True``).
    """
    if area_m2 <= 0:
        raise ValueError("area_m2 must be positive")
    if flow_rate_lpm <= 0:
        raise ValueError("flow_rate_lpm must be positive")
    efficiency = get_method(irrigation_method).efficiency
    net = max(0.0, float(net_depth_mm))
    gross = net / efficiency
    volume = gross * area_m2
    duration = volume / flow_rate_lpm
    capped = False
    if max_duration_min is not None and duration > max_duration_min > 0:
        scale = max_duration_min / duration
        duration, volume, gross, net = max_duration_min, volume * scale, gross * scale, net * scale
        capped = True
    return IrrigationPlan(
        net_depth_mm=net,
        gross_depth_mm=gross,
        water_volume_l=volume,
        duration_min=duration,
        capped=capped,
        efficiency=efficiency,
    )


def litres_to_mm(volume_l: float, area_m2: float) -> float:
    """Convert a water volume applied over ``area_m2`` into depth (mm)."""
    if area_m2 <= 0:
        raise ValueError("area_m2 must be positive")
    return volume_l / area_m2


def hourly_et0_fraction(hour: int) -> float:
    """Share of daily ET0 occurring in a clock hour (half-sine over 06:00-18:00)."""
    if hour < 6 or hour >= 18:
        return 0.0
    # Integral of sin over each daylight hour, normalised to sum to 1.
    a = math.pi * (hour - 6) / 12.0
    b = math.pi * (hour - 5) / 12.0
    return (math.cos(a) - math.cos(b)) / 2.0
