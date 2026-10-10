"""Offline (simulated) weather forecast."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from fastapi import APIRouter, Depends, Query

from ...db import repository as repo
from ..dependencies import AppState, get_conn, get_state
from ..schemas import ForecastOut

router = APIRouter(prefix="/api/v1/fields/{field_id}/weather", tags=["weather"])


@router.get("/forecast", response_model=ForecastOut, summary="Simulated forecast used by the engine (offline)")
def forecast(
    field_id: int,
    hours: int = Query(24, ge=1, le=168),
    start: datetime | None = None,
    conn: sqlite3.Connection = Depends(get_conn),
    state: AppState = Depends(get_state),
) -> dict:
    field = repo.get_field(conn, field_id)
    start = (start or datetime.now()).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    fc = state.engine.weather.forecast(float(field["latitude"]), start, hours)
    return {
        "field_id": field_id,
        "start": fc.start.isoformat(),
        "hours": fc.hours,
        "total_rain_mm": fc.total_rain_mm,
        "rain_probability": fc.rain_probability,
        "temp_max_c": fc.temp_max_c,
        "temp_min_c": fc.temp_min_c,
        "mean_humidity_pct": fc.mean_humidity_pct,
        "source": fc.source,
        "disclaimer": "SIMULATED weather from an offline stochastic generator - not a real forecast. "
        "Send forecast_rain_24h_mm with recommendation requests to use a real forecast.",
        "hourly": fc.hourly,
    }
