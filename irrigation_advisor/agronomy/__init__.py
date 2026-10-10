"""Agronomic reference data and FAO-56 water-balance calculations."""

from .crops import CROPS, IRRIGATION_METHODS, SOILS, CropProfile, SoilProfile, get_crop, get_method, get_soil
from .water_balance import (
    IrrigationPlan,
    WaterStatus,
    crop_coefficient,
    effective_rainfall,
    hargreaves_et0,
    net_irrigation_requirement,
    plan_irrigation,
    root_depth,
    soil_water_status,
)

__all__ = [
    "CROPS",
    "SOILS",
    "IRRIGATION_METHODS",
    "CropProfile",
    "SoilProfile",
    "IrrigationPlan",
    "WaterStatus",
    "get_crop",
    "get_soil",
    "get_method",
    "crop_coefficient",
    "effective_rainfall",
    "hargreaves_et0",
    "net_irrigation_requirement",
    "plan_irrigation",
    "root_depth",
    "soil_water_status",
]
