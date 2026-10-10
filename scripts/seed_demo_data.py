"""Populate the SQLite database with demo fields and 5 days of SIMULATED sensor history.

Three fields exercise different engine paths:
  1. tomato / sandy / drip        - under-irrigated history (likely needs water)
  2. maize / loam / sprinkler     - normal sensor data
  3. onion / clay loam / furrow   - soil sensor fails for the last 6 hours (fallback path)

Usage (PowerShell):
    python scripts/seed_demo_data.py --reset
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta

import _bootstrap  # noqa: F401

from irrigation_advisor.config import get_settings
from irrigation_advisor.data.simulator import FieldSpec, simulate_field
from irrigation_advisor.data.weather_provider import SimulatedWeatherProvider
from irrigation_advisor.db import repository as repo
from irrigation_advisor.db.database import Database
from irrigation_advisor.engine.recommender import RecommendationEngine
from irrigation_advisor.api.services import ingest_reading, make_recommendation
from irrigation_advisor.logging_config import configure_logging
from irrigation_advisor.ml.model_store import try_load_model

logger = logging.getLogger("seed_demo_data")
HISTORY_HOURS = 120

DEMO_FIELDS = [
    {"name": "Demo 1 - Tomato (drip, under-irrigated)", "crop_type": "tomato", "soil_type": "sandy",
     "irrigation_method": "drip", "days_ago": 55, "area_m2": 1500, "rate_mm_h": 3.5, "farmer_interval": 12,
     "farmer_depth": 6, "fail_sensor": False},
    {"name": "Demo 2 - Maize (sprinkler)", "crop_type": "maize", "soil_type": "loam", "irrigation_method": "sprinkler",
     "days_ago": 40, "area_m2": 4000, "rate_mm_h": 8.0, "farmer_interval": 4, "farmer_depth": 30, "fail_sensor": False},
    {"name": "Demo 3 - Onion (furrow, faulty sensor)", "crop_type": "onion", "soil_type": "clay_loam",
     "irrigation_method": "furrow", "days_ago": 70, "area_m2": 1000, "rate_mm_h": 25.0, "farmer_interval": 5,
     "farmer_depth": 25, "fail_sensor": True},
]


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="delete the existing database first")
    parser.add_argument("--latitude", type=float, default=settings.default_latitude)
    args = parser.parse_args()
    configure_logging(settings.log_level)

    if args.reset and settings.database_path.exists():
        for suffix in ("", "-wal", "-shm"):
            path = settings.database_path.with_name(settings.database_path.name + suffix)
            if path.exists():
                path.unlink()
        logger.info("Removed existing database %s", settings.database_path)
    db = Database(settings.database_path)
    db.initialise()
    engine = RecommendationEngine(try_load_model(settings.model_path), SimulatedWeatherProvider(settings.climate_profile))
    now = datetime.now().replace(minute=0, second=0, microsecond=0)

    for i, demo in enumerate(DEMO_FIELDS, start=1):
        planting = date.today() - timedelta(days=demo["days_ago"])
        flow = round(demo["rate_mm_h"] * demo["area_m2"] / 60.0, 1)
        spec = FieldSpec(
            field_id=f"demo{i}", crop_type=demo["crop_type"], soil_type=demo["soil_type"],
            irrigation_method=demo["irrigation_method"], planting_date=planting, latitude=args.latitude,
            area_m2=demo["area_m2"], flow_rate_lpm=flow, farmer_policy="fixed_interval",
            farmer_interval_days=demo["farmer_interval"], farmer_depth_mm=demo["farmer_depth"],
            sensor_bias=0.0, seed=1000 + i,
        )
        sim = simulate_field(spec, settings.climate_profile, fault_rate=0.0)
        window = sim[(sim["timestamp"] > now - timedelta(hours=HISTORY_HOURS)) & (sim["timestamp"] <= now)]

        with db.connection() as conn:
            field = repo.create_field(conn, {
                "name": demo["name"], "crop_type": demo["crop_type"], "soil_type": demo["soil_type"],
                "irrigation_method": demo["irrigation_method"], "planting_date": planting, "area_m2": demo["area_m2"],
                "flow_rate_lpm": flow, "latitude": args.latitude, "longitude": 73.85, "max_irrigation_minutes": 240,
            })
            device_id = f"esp32-demo-{field['id']:02d}"
            repo.register_device(conn, device_id, field["id"], f"Simulated node for {demo['name']}")
            n_rows = len(window)
            for k, row in enumerate(window.itertuples(index=False)):
                sm = None if demo["fail_sensor"] and k >= n_rows - 6 else float(row.soil_moisture)
                ingest_reading(conn, settings, field["id"], {
                    "timestamp": row.timestamp.to_pydatetime(),
                    "soil_moisture": sm, "temperature_c": float(row.temperature_c),
                    "humidity_pct": float(row.humidity_pct), "rainfall_mm": float(row.rainfall_mm),
                    "battery_v": 3.95, "device_id": device_id,
                }, source="seed")
                if row.irrigation_applied_mm > 0 and k < n_rows - 1:
                    gross_mm = row.irrigation_applied_mm
                    repo.insert_irrigation_event(conn, field["id"], {
                        "start_time": row.timestamp.to_pydatetime(),
                        "duration_min": 60.0, "water_volume_l": gross_mm * demo["area_m2"],
                        "source": "manual", "notes": "simulated farmer irrigation (seed data)",
                    })
            rec = make_recommendation(conn, settings, engine, field["id"])
        logger.info("Field %d '%s': %d readings -> %s [%s]", field["id"], demo["name"], n_rows, rec["summary"], rec["decision_source"])
    logger.info("Demo database ready: %s (all sensor data is SIMULATED)", settings.database_path)


if __name__ == "__main__":
    main()
