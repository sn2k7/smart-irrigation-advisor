"""Sensor reading ingestion and history."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from fastapi import APIRouter, Depends, Query, status

from ...db import repository as repo
from ..dependencies import AppState, get_conn, get_state
from ..schemas import BatchIngestResult, BatchReadingsIn, ReadingIngestResult, ReadingOut, SensorReadingIn
from ..services import ingest_reading

router = APIRouter(prefix="/api/v1", tags=["readings"])


@router.post("/readings", response_model=ReadingIngestResult, status_code=status.HTTP_201_CREATED,
             summary="Store one sensor reading (validated and quality-flagged)")
def create_reading(
    payload: SensorReadingIn, conn: sqlite3.Connection = Depends(get_conn), state: AppState = Depends(get_state)
) -> dict:
    values = payload.model_dump(exclude={"field_id"})
    return ingest_reading(conn, state.settings, payload.field_id, values, source="api")


@router.post("/readings/batch", response_model=BatchIngestResult, status_code=status.HTTP_201_CREATED,
             summary="Store many readings (e.g. offline logger upload); processed in time order")
def create_readings_batch(
    payload: BatchReadingsIn, conn: sqlite3.Connection = Depends(get_conn), state: AppState = Depends(get_state)
) -> dict:
    # Process oldest first so spike/stuck checks see the right history (tz info ignored for ordering only).
    ordered = sorted(
        payload.readings,
        key=lambda r: (r.field_id, r.timestamp.replace(tzinfo=None) if r.timestamp else datetime.max),
    )
    results = [
        ingest_reading(conn, state.settings, r.field_id, r.model_dump(exclude={"field_id"}), source="batch")
        for r in ordered
    ]
    return {
        "received": len(results),
        "stored": sum(1 for r in results if not r["duplicate"]),
        "duplicates": sum(1 for r in results if r["duplicate"]),
        "invalid": sum(1 for r in results if not r["is_valid"]),
        "results": results,
    }


@router.get("/fields/{field_id}/readings", response_model=list[ReadingOut], summary="Reading history (newest first)")
def list_readings(
    field_id: int,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = Query(500, ge=1, le=10_000),
    conn: sqlite3.Connection = Depends(get_conn),
) -> list[dict]:
    repo.get_field(conn, field_id)
    return repo.list_readings(conn, field_id, start, end, limit)
