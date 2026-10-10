"""SQLite repository functions."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from irrigation_advisor.db import repository as repo
from irrigation_advisor.db.database import Database
from irrigation_advisor.exceptions import ConflictError, NotFoundError

FIELD = {
    "name": "Plot A", "crop_type": "maize", "soil_type": "loam", "irrigation_method": "drip",
    "planting_date": date(2026, 6, 1), "area_m2": 1000, "flow_rate_lpm": 50, "latitude": 18.5,
    "longitude": None, "max_irrigation_minutes": None,
}
T0 = datetime(2026, 7, 1, 6, 0)


@pytest.fixture()
def db(tmp_path) -> Database:
    database = Database(tmp_path / "db.sqlite")
    database.initialise()
    database.initialise()  # idempotent
    return database


def test_memory_database_is_rejected():
    with pytest.raises(ValueError):
        Database(":memory:")


def test_field_crud(db):
    with db.connection() as conn:
        field = repo.create_field(conn, FIELD)
        assert field["id"] == 1 and field["max_irrigation_minutes"] == 180 and field["planting_date"] == "2026-06-01"
        updated = repo.update_field(conn, 1, {"name": "Plot B", "area_m2": 1500, "ignored": 1})
        assert updated["name"] == "Plot B" and updated["area_m2"] == 1500
        assert len(repo.list_fields(conn)) == 1
        repo.delete_field(conn, 1)
        with pytest.raises(NotFoundError):
            repo.get_field(conn, 1)


def test_check_constraints_reject_bad_rows(db):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        with db.connection() as conn:
            repo.create_field(conn, {**FIELD, "area_m2": -5})


def test_readings_duplicates_and_history(db):
    with db.connection() as conn:
        repo.create_field(conn, FIELD)
        for h in range(5):
            rid = repo.insert_reading(conn, 1, {"timestamp": T0 + timedelta(hours=h), "soil_moisture": 20 + h,
                                                "temperature_c": 25, "humidity_pct": 50, "rainfall_mm": 0},
                                      is_valid=h != 2, quality_flags=["spike"] if h == 2 else [])
            assert rid is not None
        dup = repo.insert_reading(conn, 1, {"timestamp": T0, "soil_moisture": 1}, True, [])
        assert dup is None
        rows = repo.list_readings(conn, 1)
        assert len(rows) == 5 and rows[0]["timestamp"] > rows[-1]["timestamp"]
        assert rows[2]["quality_flags"] == ["spike"] and rows[2]["is_valid"] is False
        frame = repo.readings_frame(conn, 1, T0 + timedelta(hours=4), hours=24)
        assert len(frame) == 5
        assert frame["soil_moisture"].isna().sum() == 1  # invalid reading blanked for the engine
        assert repo.latest_reading_time(conn, 1) == T0 + timedelta(hours=4)
        assert len(repo.list_readings(conn, 1, start=T0 + timedelta(hours=3))) == 2


def test_devices(db):
    with db.connection() as conn:
        repo.create_field(conn, FIELD)
        repo.register_device(conn, "esp-1", 1, "node")
        repo.touch_device(conn, "esp-1")
        assert repo.get_device(conn, "esp-1")["last_seen_at"] is not None
        with pytest.raises(NotFoundError):
            repo.get_device(conn, "esp-2")
        with pytest.raises(NotFoundError):
            repo.register_device(conn, "esp-3", 99, None)


def test_recommendations_events_and_analytics(db):
    with db.connection() as conn:
        repo.create_field(conn, FIELD)
        repo.create_field(conn, {**FIELD, "name": "Other"})
        repo.insert_reading(conn, 1, {"timestamp": T0, "soil_moisture": 18, "temperature_c": 30, "humidity_pct": 40,
                                      "rainfall_mm": 2.5}, True, [])
        rec = {"decision_time": T0.isoformat(), "irrigation_needed": True, "decision_source": "rule_based",
               "confidence": "high", "water_volume_l": 5000.0, "duration_min": 100.0, "summary": "Irrigate", "reasons": []}
        rec_id = repo.insert_recommendation(conn, 1, rec)
        assert repo.get_recommendation(conn, rec_id)["summary"] == "Irrigate"
        assert len(repo.list_recommendations(conn, 1)) == 1
        event = repo.insert_irrigation_event(conn, 1, {"start_time": T0, "duration_min": 60, "recommendation_id": rec_id})
        assert event["water_volume_l"] == pytest.approx(60 * 50)  # derived from flow rate
        with pytest.raises(ConflictError):
            repo.insert_irrigation_event(conn, 2, {"start_time": T0, "duration_min": 10, "recommendation_id": rec_id})
        days = repo.daily_analytics(conn, 1, T0.date() - timedelta(days=1), T0.date())
        assert len(days) == 2 and days[0]["readings"] == 0
        today = days[1]
        assert today["rainfall_mm"] == 2.5 and today["recommended_water_l"] == 5000 and today["applied_water_l"] == 3000
        summary = repo.field_summary(conn, 1, T0.date() - timedelta(days=5), T0.date(), fixed_interval_days=3)
        # 6 days -> 2 scheduled irrigations of 3 * 1.2 * 5 / 0.9 = 20 mm on 1000 m2
        assert summary["fixed_schedule_estimate_l"] == pytest.approx(2 * 20 * 1000)
        assert summary["estimated_saving_vs_fixed_pct"] == pytest.approx(100 * (1 - 3000 / 40000), abs=0.1)
        assert "not a measurement" in summary["note"]


def test_cascade_delete(db):
    with db.connection() as conn:
        repo.create_field(conn, FIELD)
        repo.insert_reading(conn, 1, {"timestamp": T0, "soil_moisture": 20}, True, [])
        repo.delete_field(conn, 1)
        assert conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0] == 0


def test_transaction_rolls_back_on_error(db):
    with pytest.raises(RuntimeError):
        with db.connection() as conn:
            repo.create_field(conn, FIELD)
            raise RuntimeError("boom")
    with db.connection() as conn:
        assert repo.list_fields(conn) == []
