"""Stateless FAO-56 irrigation calculator."""

from __future__ import annotations

from fastapi import APIRouter

from ...agronomy.crops import get_crop, get_soil
from ...agronomy.water_balance import (
    crop_coefficient,
    effective_rainfall,
    growth_stage,
    net_irrigation_requirement,
    plan_irrigation,
    projected_depletion,
    root_depth,
)
from ..schemas import CalculatorRequest, CalculatorResponse

router = APIRouter(prefix="/api/v1/tools", tags=["tools"])


@router.post("/irrigation-calculator", response_model=CalculatorResponse,
             summary="Compute water status and irrigation plan from a single moisture value")
def irrigation_calculator(req: CalculatorRequest) -> CalculatorResponse:
    crop, soil = get_crop(req.crop_type.value), get_soil(req.soil_type.value)
    days = req.days_since_planting
    kc = crop_coefficient(crop, days)
    zr = root_depth(crop, days)
    taw = 1000 * (soil.field_capacity - soil.wilting_point) * zr
    raw = crop.depletion_fraction * taw
    dr = 1000 * (soil.field_capacity - req.soil_moisture_pct / 100) * zr
    etc = kc * req.et0_mm_day
    eff = effective_rainfall(req.forecast_rain_24h_mm)
    projected = projected_depletion(dr, etc, eff)
    needed = dr > 0 and (dr >= raw or projected >= raw)
    net = net_irrigation_requirement(dr, etc, eff, taw) if needed else 0.0
    plan = plan_irrigation(net, req.irrigation_method.value, req.area_m2, req.flow_rate_lpm, req.max_irrigation_minutes)
    explanation = [
        f"Stage {growth_stage(crop, days)}: Kc {kc:.2f}, root depth {zr:.2f} m",
        f"TAW = 1000 x ({soil.field_capacity} - {soil.wilting_point}) x {zr:.2f} = {taw:.1f} mm; "
        f"RAW = {crop.depletion_fraction} x TAW = {raw:.1f} mm",
        f"Depletion Dr = 1000 x ({soil.field_capacity} - {req.soil_moisture_pct / 100:.3f}) x {zr:.2f} = {dr:.1f} mm",
        f"ETc = {kc:.2f} x {req.et0_mm_day} = {etc:.1f} mm/day; effective rain {eff:.1f} mm; "
        f"projected Dr in 24 h = {projected:.1f} mm",
        "Irrigate: depletion reaches RAW within 24 h" if needed else "No irrigation: depletion stays below RAW",
    ]
    if needed:
        explanation.append(
            f"Net {plan.net_depth_mm:.1f} mm / efficiency {plan.efficiency} = gross {plan.gross_depth_mm:.1f} mm "
            f"-> {plan.water_volume_l:.0f} L at {req.flow_rate_lpm} L/min = {plan.duration_min:.0f} min"
        )
    return CalculatorResponse(
        irrigation_needed=needed,
        growth_stage=growth_stage(crop, days),
        kc=round(kc, 3),
        root_depth_m=round(zr, 3),
        taw_mm=round(taw, 2),
        raw_mm=round(raw, 2),
        depletion_mm=round(dr, 2),
        depletion_ratio=round(dr / raw, 3) if raw > 0 else 0.0,
        etc_mm_day=round(etc, 2),
        effective_rain_mm=round(eff, 2),
        projected_depletion_mm=round(projected, 2),
        net_depth_mm=round(plan.net_depth_mm, 2),
        gross_depth_mm=round(plan.gross_depth_mm, 2),
        water_volume_l=round(plan.water_volume_l, 1),
        duration_min=round(plan.duration_min, 1),
        capped=plan.capped,
        explanation=explanation,
    )
