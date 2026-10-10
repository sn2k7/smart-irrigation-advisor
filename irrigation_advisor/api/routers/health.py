"""Service health and reference data."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends

from ... import __version__
from ...agronomy.crops import CROPS, IRRIGATION_METHODS, SOILS
from ..dependencies import AppState, get_state
from ..schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/", summary="API information")
def root() -> dict[str, str]:
    return {
        "name": "Smart Irrigation Advisor",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", response_model=HealthOut, summary="Liveness and dependency status")
def health(state: AppState = Depends(get_state)) -> HealthOut:
    model = state.model
    return HealthOut(
        status="ok",
        version=__version__,
        database=state.db.ping(),
        model_loaded=model is not None,
        model_version=model.version if model else None,
        weather_source=getattr(state.engine.weather, "source_name", type(state.engine.weather).__name__),
        time=datetime.now().isoformat(timespec="seconds"),
    )


@router.get("/api/v1/reference/crops", tags=["reference"], summary="Supported crops and FAO-56 parameters")
def crops() -> dict[str, dict]:
    return {
        name: {
            "kc_ini": c.kc_ini,
            "kc_mid": c.kc_mid,
            "kc_end": c.kc_end,
            "stage_days": list(c.stage_days),
            "season_days": c.season_days,
            "root_depth_m": [c.root_depth_min_m, c.root_depth_max_m],
            "depletion_fraction_p": c.depletion_fraction,
        }
        for name, c in CROPS.items()
    }


@router.get("/api/v1/reference/soils", tags=["reference"], summary="Supported soils (volumetric water content)")
def soils() -> dict[str, dict]:
    return {
        name: {
            "field_capacity": s.field_capacity,
            "wilting_point": s.wilting_point,
            "saturation": s.saturation,
            "available_water_mm_per_m": round(s.available_water_per_m, 1),
        }
        for name, s in SOILS.items()
    }


@router.get("/api/v1/reference/irrigation-methods", tags=["reference"], summary="Irrigation methods and efficiency")
def methods() -> dict[str, float]:
    return {name: m.efficiency for name, m in IRRIGATION_METHODS.items()}
