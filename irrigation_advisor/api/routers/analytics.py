"""Historical analytics per field."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query

from ...db import repository as repo
from ..dependencies import get_conn
from ..schemas import DailyAnalytics, FieldSummary

router = APIRouter(prefix="/api/v1/fields/{field_id}/analytics", tags=["analytics"])


def _period(end: date | None, days: int) -> tuple[date, date]:
    end = end or date.today()
    return end - timedelta(days=days - 1), end


@router.get("/daily", response_model=list[DailyAnalytics], summary="Daily sensor, recommendation and water statistics")
def daily(
    field_id: int,
    days: int = Query(14, ge=1, le=366),
    end: date | None = Query(None, description="Last day (default today)"),
    conn: sqlite3.Connection = Depends(get_conn),
) -> list[dict]:
    start, stop = _period(end, days)
    return repo.daily_analytics(conn, field_id, start, stop)


@router.get("/summary", response_model=FieldSummary, summary="Period totals with an ESTIMATED fixed-schedule reference")
def summary(
    field_id: int,
    days: int = Query(30, ge=1, le=366),
    end: date | None = None,
    fixed_interval_days: int = Query(3, ge=1, le=30),
    conn: sqlite3.Connection = Depends(get_conn),
) -> dict:
    if days < 1:
        raise HTTPException(status_code=422, detail="days must be >= 1")
    start, stop = _period(end, days)
    return repo.field_summary(conn, field_id, start, stop, fixed_interval_days)
