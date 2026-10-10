"""Use-case functions shared by several routers (HTTP-agnostic)."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import Settings
from ..data.validation import validate_reading
from ..db import repository as repo
from ..db.database import to_iso
from ..engine.recommender import RecommendationEngine, field_config_from_row
from ..utils import to_local_naive

logger = logging.getLogger(__name__)
VALIDATION_HISTORY = 24
FUTURE_TOLERANCE = timedelta(minutes=10)


def normalise_timestamp(ts: datetime | None, settings: Settings) -> datetime:
    """Field-local naive timestamp, seconds precision (default: now)."""
    ts = ts or datetime.now()
    return to_local_naive(ts, settings.utc_offset_hours).replace(microsecond=0)


def parse_iot_timestamp(ts: int | str | None, settings: Settings) -> datetime:
    """Epoch seconds (UTC) or ISO string -> field-local naive time."""
    if ts is None:
        return normalise_timestamp(None, settings)
    if isinstance(ts, int):
        return normalise_timestamp(datetime.fromtimestamp(ts, tz=timezone.utc), settings)
    return normalise_timestamp(parse_iso_timestamp(ts), settings)


def parse_iso_timestamp(value: str) -> datetime:
    """ISO-8601 parser that also accepts a trailing 'Z' (not supported by Python 3.10)."""
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def ingest_reading(
    conn: sqlite3.Connection, settings: Settings, field_id: int, values: dict[str, Any], source: str
) -> dict[str, Any]:
    """Validate a reading against the field's recent history and store it.

    Invalid readings are stored too (flagged) - they are useful for sensor
    diagnostics and are blanked when the engine loads history.
    """
    repo.get_field(conn, field_id)
    reading = dict(values)
    reading["timestamp"] = normalise_timestamp(reading.get("timestamp"), settings)
    history_rows = repo.recent_readings(conn, field_id, reading["timestamp"] - timedelta(seconds=1), hours=24)
    history = [
        {**r, "timestamp": datetime.fromisoformat(r["timestamp"])}
        for r in history_rows[-VALIDATION_HISTORY:]
        if r["is_valid"]
    ]
    result = validate_reading(reading, history, now=datetime.now(), stale_hours=settings.stale_reading_hours)
    # Stale/old readings are fine to store (back-filled data); only errors make them invalid.
    reading_id = repo.insert_reading(conn, field_id, reading, result.is_valid, result.codes, source)
    if reading.get("device_id") and source == "iot":
        repo.touch_device(conn, reading["device_id"])
    return {
        "id": reading_id,
        "field_id": field_id,
        "timestamp": to_iso(reading["timestamp"]),
        "duplicate": reading_id is None,
        "is_valid": result.is_valid,
        "issues": [i.as_dict() for i in result.issues],
    }


def make_recommendation(
    conn: sqlite3.Connection,
    settings: Settings,
    engine: RecommendationEngine,
    field_id: int,
    at: datetime | None = None,
    forecast_rain_24h_mm: float | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Load history from SQLite, run the engine and optionally store the result."""
    field_row = repo.get_field(conn, field_id)
    field_cfg = field_config_from_row(field_row)
    if at is None:
        # Latest reading time, but never a (flagged) future timestamp from a misconfigured clock.
        now = normalise_timestamp(None, settings)
        latest = repo.latest_reading_time(conn, field_id)
        at = latest if latest is not None and latest <= now + FUTURE_TOLERANCE else now
    else:
        at = normalise_timestamp(at, settings)
    history = repo.readings_frame(conn, field_id, at, settings.history_hours)
    rec = engine.recommend(field_cfg, history, now=at, forecast_rain_24h_mm=forecast_rain_24h_mm).to_dict()
    rec["id"] = repo.insert_recommendation(conn, field_id, rec) if persist else None
    logger.info(
        "Field %s: %s (%s, %s confidence)", field_id, rec["summary"], rec["decision_source"], rec["confidence"]
    )
    return rec
