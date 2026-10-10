"""Compact endpoint for ESP32 / Arduino sensor nodes."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, Query

from ...db import repository as repo
from ..dependencies import AppState, get_conn, get_state, verify_device_key
from ..schemas import IoTCommand, IoTPayload, IoTResponse
from ..services import ingest_reading, make_recommendation, parse_iot_timestamp

router = APIRouter(prefix="/api/v1/iot", tags=["iot"])


@router.post(
    "/ingest",
    response_model=IoTResponse,
    dependencies=[Depends(verify_device_key)],
    summary="Ingest a compact device message; optionally return a valve command",
)
def iot_ingest(
    payload: IoTPayload,
    recommend: bool = Query(False, description="Also compute a recommendation and return a valve command"),
    conn: sqlite3.Connection = Depends(get_conn),
    state: AppState = Depends(get_state),
) -> IoTResponse:
    device = repo.get_device(conn, payload.device_id)  # 404 if not registered
    field_id = int(device["field_id"])
    values = {
        "timestamp": parse_iot_timestamp(payload.ts, state.settings),
        "soil_moisture": payload.sm,
        "temperature_c": payload.t,
        "humidity_pct": payload.h,
        "rainfall_mm": payload.r,
        "battery_v": payload.bat,
        "rssi_dbm": payload.rssi,
        "device_id": payload.device_id,
    }
    result = ingest_reading(conn, state.settings, field_id, values, source="iot")
    cmd = None
    if recommend:
        rec = make_recommendation(conn, state.settings, state.engine, field_id, at=values["timestamp"])
        cmd = IoTCommand(
            irrigate=rec["irrigation_needed"],
            duration_min=rec["duration_min"] if rec["irrigation_needed"] else 0.0,
            next_check_min=int(rec["next_check_hours"] * 60),
            recommendation_id=rec["id"],
        )
    return IoTResponse(
        accepted=True,
        duplicate=result["duplicate"],
        valid=result["is_valid"],
        issues=[i["code"] for i in result["issues"]],
        field_id=field_id,
        cmd=cmd,
    )
