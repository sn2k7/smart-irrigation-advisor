"""Irrigation recommendations."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, Query, status

from ...db import repository as repo
from ..dependencies import AppState, get_conn, get_state
from ..schemas import RecommendationOut, RecommendationRequest
from ..services import ingest_reading, make_recommendation

router = APIRouter(prefix="/api/v1", tags=["recommendations"])


@router.post("/recommendations", response_model=RecommendationOut, status_code=status.HTTP_201_CREATED,
             summary="Decide whether/how long to irrigate a field now")
def create_recommendation(
    payload: RecommendationRequest, conn: sqlite3.Connection = Depends(get_conn), state: AppState = Depends(get_state)
) -> dict:
    if payload.reading is not None:
        ingest_reading(conn, state.settings, payload.field_id, payload.reading.model_dump(), source="api")
    # Decision time defaults to the newest (non-future) reading - see services.make_recommendation.
    return make_recommendation(
        conn, state.settings, state.engine, payload.field_id, at=payload.at,
        forecast_rain_24h_mm=payload.forecast_rain_24h_mm, persist=payload.persist,
    )


@router.get("/recommendations/{recommendation_id}", response_model=RecommendationOut, summary="Get a stored recommendation")
def get_recommendation(recommendation_id: int, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    return repo.get_recommendation(conn, recommendation_id)


@router.get("/fields/{field_id}/recommendations", response_model=list[RecommendationOut],
            summary="Recommendation history of a field (newest first)")
def list_recommendations(
    field_id: int, limit: int = Query(50, ge=1, le=1000), conn: sqlite3.Connection = Depends(get_conn)
) -> list[dict]:
    repo.get_field(conn, field_id)
    return repo.list_recommendations(conn, field_id, limit)
