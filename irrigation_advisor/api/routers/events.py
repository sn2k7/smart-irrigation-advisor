"""Irrigation events actually carried out (manual log or valve controller)."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, Query, status

from ...db import repository as repo
from ..dependencies import get_conn
from ..schemas import IrrigationEventIn, IrrigationEventOut

router = APIRouter(prefix="/api/v1/fields/{field_id}/irrigation-events", tags=["irrigation events"])


@router.post("", response_model=IrrigationEventOut, status_code=status.HTTP_201_CREATED,
             summary="Log an irrigation that was carried out")
def create_event(field_id: int, payload: IrrigationEventIn, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    return repo.insert_irrigation_event(conn, field_id, payload.model_dump())


@router.get("", response_model=list[IrrigationEventOut], summary="Irrigation log (newest first)")
def list_events(field_id: int, limit: int = Query(100, ge=1, le=5000), conn: sqlite3.Connection = Depends(get_conn)) -> list[dict]:
    repo.get_field(conn, field_id)
    return repo.list_irrigation_events(conn, field_id, limit)
