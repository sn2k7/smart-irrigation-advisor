"""Data-access functions (one function per query, explicit SQL).

Every function takes an open ``sqlite3.Connection`` so callers control the
transaction boundary (one per API request). Rows are returned as plain dicts.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from ..agronomy.crops import get_crop, get_method
from ..exceptions import ConflictError, NotFoundError
from ..utils import to_json
from .database import now_iso, to_iso

FIELD_COLUMNS = (
    "name", "crop_type", "soil_type", "irrigation_method", "planting_date", "area_m2",
    "flow_rate_lpm", "latitude", "longitude", "max_irrigation_minutes",
)


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return None if row is None else dict(row)


# ---------------------------------------------------------------- fields
def create_field(conn: sqlite3.Connection, data: dict[str, Any]) -> dict[str, Any]:
    values = {k: data.get(k) for k in FIELD_COLUMNS}
    if isinstance(values["planting_date"], date):
        values["planting_date"] = values["planting_date"].isoformat()
    if values["max_irrigation_minutes"] is None:
        values["max_irrigation_minutes"] = 180.0
    ts = now_iso()
    cur = conn.execute(
        f"INSERT INTO fields ({', '.join(FIELD_COLUMNS)}, created_at, updated_at) "
        f"VALUES ({', '.join('?' for _ in FIELD_COLUMNS)}, ?, ?)",
        [*values.values(), ts, ts],
    )
    return get_field(conn, int(cur.lastrowid))


def get_field(conn: sqlite3.Connection, field_id: int) -> dict[str, Any]:
    row = _row(conn.execute("SELECT * FROM fields WHERE id = ?", (field_id,)).fetchone())
    if row is None:
        raise NotFoundError(f"Field {field_id} not found")
    return row


def list_fields(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute("SELECT * FROM fields ORDER BY id").fetchall()]


def update_field(conn: sqlite3.Connection, field_id: int, changes: dict[str, Any]) -> dict[str, Any]:
    get_field(conn, field_id)
    updates = {k: v for k, v in changes.items() if k in FIELD_COLUMNS}
    if isinstance(updates.get("planting_date"), date):
        updates["planting_date"] = updates["planting_date"].isoformat()
    if updates:
        assignments = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(
            f"UPDATE fields SET {assignments}, updated_at = ? WHERE id = ?",
            [*updates.values(), now_iso(), field_id],
        )
    return get_field(conn, field_id)


def delete_field(conn: sqlite3.Connection, field_id: int) -> None:
    get_field(conn, field_id)
    conn.execute("DELETE FROM fields WHERE id = ?", (field_id,))


# ---------------------------------------------------------------- devices
def register_device(conn: sqlite3.Connection, device_id: str, field_id: int, description: str | None) -> dict[str, Any]:
    get_field(conn, field_id)
    conn.execute(
        "INSERT INTO devices (device_id, field_id, description, created_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(device_id) DO UPDATE SET field_id = excluded.field_id, description = excluded.description",
        (device_id, field_id, description, now_iso()),
    )
    return get_device(conn, device_id)


def get_device(conn: sqlite3.Connection, device_id: str) -> dict[str, Any]:
    row = _row(conn.execute("SELECT * FROM devices WHERE device_id = ?", (device_id,)).fetchone())
    if row is None:
        raise NotFoundError(f"Device '{device_id}' is not registered")
    return row


def list_devices(conn: sqlite3.Connection, field_id: int | None = None) -> list[dict[str, Any]]:
    if field_id is None:
        rows = conn.execute("SELECT * FROM devices ORDER BY device_id").fetchall()
    else:
        rows = conn.execute("SELECT * FROM devices WHERE field_id = ? ORDER BY device_id", (field_id,)).fetchall()
    return [dict(r) for r in rows]


def touch_device(conn: sqlite3.Connection, device_id: str) -> None:
    conn.execute("UPDATE devices SET last_seen_at = ? WHERE device_id = ?", (now_iso(), device_id))


# ---------------------------------------------------------------- readings
def insert_reading(
    conn: sqlite3.Connection,
    field_id: int,
    reading: dict[str, Any],
    is_valid: bool,
    quality_flags: list[str],
    source: str = "api",
) -> int | None:
    """Insert one reading. Returns the new id, or ``None`` if it is a duplicate."""
    cur = conn.execute(
        """INSERT INTO sensor_readings
           (field_id, device_id, timestamp, soil_moisture, temperature_c, humidity_pct, rainfall_mm,
            battery_v, rssi_dbm, source, is_valid, quality_flags, received_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(field_id, device_id, timestamp) DO NOTHING""",
        (
            field_id,
            reading.get("device_id") or "manual",
            to_iso(reading["timestamp"]),
            reading.get("soil_moisture"),
            reading.get("temperature_c"),
            reading.get("humidity_pct"),
            reading.get("rainfall_mm"),
            reading.get("battery_v"),
            reading.get("rssi_dbm"),
            source,
            int(bool(is_valid)),
            json.dumps(sorted(set(quality_flags))),
            now_iso(),
        ),
    )
    return int(cur.lastrowid) if cur.rowcount == 1 else None


def _reading_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["is_valid"] = bool(data["is_valid"])
    data["quality_flags"] = json.loads(data["quality_flags"] or "[]")
    return data


def list_readings(
    conn: sqlite3.Connection,
    field_id: int,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Readings of a field, newest first."""
    sql = "SELECT * FROM sensor_readings WHERE field_id = ?"
    params: list[Any] = [field_id]
    if start is not None:
        sql += " AND timestamp >= ?"
        params.append(to_iso(start))
    if end is not None:
        sql += " AND timestamp <= ?"
        params.append(to_iso(end))
    sql += " ORDER BY timestamp DESC, id DESC LIMIT ?"
    params.append(int(limit))
    return [_reading_dict(r) for r in conn.execute(sql, params).fetchall()]


def recent_readings(conn: sqlite3.Connection, field_id: int, end: datetime, hours: int) -> list[dict[str, Any]]:
    """Readings in ``(end - hours, end]`` oldest first (input for the engine)."""
    rows = conn.execute(
        "SELECT * FROM sensor_readings WHERE field_id = ? AND timestamp > ? AND timestamp <= ? ORDER BY timestamp, id",
        (field_id, to_iso(end - timedelta(hours=hours)), to_iso(end)),
    ).fetchall()
    return [_reading_dict(r) for r in rows]


def readings_frame(conn: sqlite3.Connection, field_id: int, end: datetime, hours: int) -> pd.DataFrame:
    """Recent readings as a DataFrame for the recommendation engine.

    Readings that failed validation are kept but their soil-moisture value is
    blanked, so preprocessing treats them as missing (same as training).
    """
    rows = recent_readings(conn, field_id, end, hours)
    cols = ["timestamp", "soil_moisture", "temperature_c", "humidity_pct", "rainfall_mm", "battery_v"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df.loc[~df["is_valid"], "soil_moisture"] = None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[cols]


def latest_reading_time(conn: sqlite3.Connection, field_id: int) -> datetime | None:
    row = conn.execute("SELECT MAX(timestamp) AS ts FROM sensor_readings WHERE field_id = ?", (field_id,)).fetchone()
    return datetime.fromisoformat(row["ts"]) if row and row["ts"] else None


# ---------------------------------------------------------------- recommendations
def insert_recommendation(conn: sqlite3.Connection, field_id: int, rec: dict[str, Any]) -> int:
    cur = conn.execute(
        """INSERT INTO recommendations
           (field_id, created_at, decision_time, irrigation_needed, decision_source, confidence, probability,
            net_depth_mm, gross_depth_mm, water_volume_l, duration_min, soil_moisture_pct, depletion_mm,
            forecast_rain_24h_mm, model_version, summary, payload)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            field_id,
            now_iso(),
            rec["decision_time"][:19],
            int(bool(rec["irrigation_needed"])),
            rec["decision_source"],
            rec["confidence"],
            rec.get("probability"),
            rec.get("net_depth_mm") or 0.0,
            rec.get("gross_depth_mm") or 0.0,
            rec.get("water_volume_l") or 0.0,
            rec.get("duration_min") or 0.0,
            rec.get("soil_moisture_pct"),
            rec.get("depletion_mm"),
            rec.get("forecast_rain_24h_mm"),
            rec.get("model_version"),
            rec["summary"],
            to_json(rec),
        ),
    )
    return int(cur.lastrowid)


def _rec_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = json.loads(row["payload"])
    data["id"] = row["id"]
    data["created_at"] = row["created_at"]
    return data


def get_recommendation(conn: sqlite3.Connection, rec_id: int) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM recommendations WHERE id = ?", (rec_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"Recommendation {rec_id} not found")
    return _rec_dict(row)


def list_recommendations(conn: sqlite3.Connection, field_id: int, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM recommendations WHERE field_id = ? ORDER BY decision_time DESC, id DESC LIMIT ?",
        (field_id, int(limit)),
    ).fetchall()
    return [_rec_dict(r) for r in rows]


# ---------------------------------------------------------------- irrigation events
def insert_irrigation_event(conn: sqlite3.Connection, field_id: int, event: dict[str, Any]) -> dict[str, Any]:
    field = get_field(conn, field_id)
    rec_id = event.get("recommendation_id")
    if rec_id is not None:
        rec = conn.execute("SELECT field_id FROM recommendations WHERE id = ?", (rec_id,)).fetchone()
        if rec is None:
            raise NotFoundError(f"Recommendation {rec_id} not found")
        if rec["field_id"] != field_id:
            raise ConflictError(f"Recommendation {rec_id} belongs to another field")
    volume = event.get("water_volume_l")
    if volume is None:  # derive from pump run time
        volume = float(event["duration_min"]) * float(field["flow_rate_lpm"])
    cur = conn.execute(
        """INSERT INTO irrigation_events
           (field_id, recommendation_id, start_time, duration_min, water_volume_l, source, notes, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (field_id, rec_id, to_iso(event["start_time"]), float(event["duration_min"]), float(volume),
         event.get("source") or "manual", event.get("notes"), now_iso()),
    )
    return dict(conn.execute("SELECT * FROM irrigation_events WHERE id = ?", (cur.lastrowid,)).fetchone())


def list_irrigation_events(conn: sqlite3.Connection, field_id: int, limit: int = 100) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM irrigation_events WHERE field_id = ? ORDER BY start_time DESC LIMIT ?", (field_id, int(limit))
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- analytics
def daily_analytics(conn: sqlite3.Connection, field_id: int, start: date, end: date) -> list[dict[str, Any]]:
    """Per-day sensor statistics, recommended water and applied water."""
    get_field(conn, field_id)
    lo, hi = start.isoformat(), (end + timedelta(days=1)).isoformat()
    readings = {
        r["day"]: dict(r)
        for r in conn.execute(
            """SELECT substr(timestamp, 1, 10) AS day,
                      COUNT(*) AS readings,
                      SUM(CASE WHEN is_valid = 0 THEN 1 ELSE 0 END) AS invalid_readings,
                      AVG(CASE WHEN is_valid = 1 THEN soil_moisture END) AS avg_soil_moisture,
                      MIN(CASE WHEN is_valid = 1 THEN soil_moisture END) AS min_soil_moisture,
                      MAX(temperature_c) AS max_temperature_c,
                      AVG(humidity_pct) AS avg_humidity_pct,
                      COALESCE(SUM(rainfall_mm), 0) AS rainfall_mm
               FROM sensor_readings WHERE field_id = ? AND timestamp >= ? AND timestamp < ?
               GROUP BY day""",
            (field_id, lo, hi),
        ).fetchall()
    }
    recs = {
        r["day"]: dict(r)
        for r in conn.execute(
            """SELECT substr(decision_time, 1, 10) AS day,
                      COUNT(*) AS recommendations,
                      SUM(irrigation_needed) AS irrigate_recommendations,
                      SUM(CASE WHEN irrigation_needed = 1 THEN water_volume_l ELSE 0 END) AS recommended_water_l
               FROM recommendations WHERE field_id = ? AND decision_time >= ? AND decision_time < ?
               GROUP BY day""",
            (field_id, lo, hi),
        ).fetchall()
    }
    events = {
        r["day"]: dict(r)
        for r in conn.execute(
            """SELECT substr(start_time, 1, 10) AS day, COUNT(*) AS irrigation_events,
                      SUM(water_volume_l) AS applied_water_l, SUM(duration_min) AS pump_minutes
               FROM irrigation_events WHERE field_id = ? AND start_time >= ? AND start_time < ?
               GROUP BY day""",
            (field_id, lo, hi),
        ).fetchall()
    }
    out = []
    day = start
    while day <= end:
        key = day.isoformat()
        r, c, e = readings.get(key, {}), recs.get(key, {}), events.get(key, {})
        out.append(
            {
                "date": key,
                "readings": int(r.get("readings") or 0),
                "invalid_readings": int(r.get("invalid_readings") or 0),
                "avg_soil_moisture": _r(r.get("avg_soil_moisture")),
                "min_soil_moisture": _r(r.get("min_soil_moisture")),
                "max_temperature_c": _r(r.get("max_temperature_c")),
                "avg_humidity_pct": _r(r.get("avg_humidity_pct")),
                "rainfall_mm": _r(r.get("rainfall_mm") or 0.0),
                "recommendations": int(c.get("recommendations") or 0),
                "irrigate_recommendations": int(c.get("irrigate_recommendations") or 0),
                "recommended_water_l": _r(c.get("recommended_water_l") or 0.0, 1),
                "irrigation_events": int(e.get("irrigation_events") or 0),
                "applied_water_l": _r(e.get("applied_water_l") or 0.0, 1),
                "pump_minutes": _r(e.get("pump_minutes") or 0.0, 1),
            }
        )
        day += timedelta(days=1)
    return out


def _r(value: Any, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)


def field_summary(
    conn: sqlite3.Connection, field_id: int, start: date, end: date, fixed_interval_days: int = 3, design_et0_mm: float = 5.0
) -> dict[str, Any]:
    """Totals for a period plus an *estimated* fixed-schedule comparison.

    The fixed-schedule figure is a calculation (what a calendar schedule
    sized for peak demand would have used), not a measurement.
    """
    field = get_field(conn, field_id)
    days = daily_analytics(conn, field_id, start, end)
    n_days = len(days)
    crop, method = get_crop(field["crop_type"]), get_method(field["irrigation_method"])
    baseline_gross_mm = fixed_interval_days * crop.kc_mid * design_et0_mm / method.efficiency
    baseline_events = (n_days + fixed_interval_days - 1) // fixed_interval_days
    baseline_l = baseline_events * baseline_gross_mm * field["area_m2"]
    recommended_l = sum(d["recommended_water_l"] or 0 for d in days)
    applied_l = sum(d["applied_water_l"] or 0 for d in days)
    return {
        "field_id": field_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": n_days,
        "readings": sum(d["readings"] for d in days),
        "invalid_readings": sum(d["invalid_readings"] for d in days),
        "rainfall_mm": round(sum(d["rainfall_mm"] or 0 for d in days), 1),
        "recommendations": sum(d["recommendations"] for d in days),
        "irrigate_recommendations": sum(d["irrigate_recommendations"] for d in days),
        "recommended_water_l": round(recommended_l, 1),
        "applied_water_l": round(applied_l, 1),
        "irrigation_events": sum(d["irrigation_events"] for d in days),
        "fixed_schedule_estimate_l": round(baseline_l, 1),
        "estimated_saving_vs_fixed_pct": (
            round(100 * (1 - applied_l / baseline_l), 1) if baseline_l > 0 and applied_l > 0 else None
        ),
        "note": "fixed_schedule_estimate_l is a calculated reference (every "
        f"{fixed_interval_days} days, sized for peak demand), not a measurement; savings are estimates.",
    }
