-- Smart Irrigation Advisor - SQLite schema (see docs/DATABASE.md)
-- Timestamps are ISO-8601 text in field-local time ("YYYY-MM-DDTHH:MM:SS").

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS fields (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    name                   TEXT    NOT NULL,
    crop_type              TEXT    NOT NULL,
    soil_type              TEXT    NOT NULL,
    irrigation_method      TEXT    NOT NULL,
    planting_date          TEXT    NOT NULL,
    area_m2                REAL    NOT NULL CHECK (area_m2 > 0),
    flow_rate_lpm          REAL    NOT NULL CHECK (flow_rate_lpm > 0),
    latitude               REAL    NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude              REAL             CHECK (longitude IS NULL OR longitude BETWEEN -180 AND 180),
    max_irrigation_minutes REAL    NOT NULL DEFAULT 180 CHECK (max_irrigation_minutes > 0),
    created_at             TEXT    NOT NULL,
    updated_at             TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS devices (
    device_id    TEXT PRIMARY KEY,
    field_id     INTEGER NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
    description  TEXT,
    created_at   TEXT NOT NULL,
    last_seen_at TEXT
);

CREATE TABLE IF NOT EXISTS sensor_readings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    field_id      INTEGER NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
    device_id     TEXT    NOT NULL DEFAULT 'manual',
    timestamp     TEXT    NOT NULL,
    soil_moisture REAL,
    temperature_c REAL,
    humidity_pct  REAL,
    rainfall_mm   REAL,
    battery_v     REAL,
    rssi_dbm      INTEGER,
    source        TEXT    NOT NULL DEFAULT 'api',
    is_valid      INTEGER NOT NULL CHECK (is_valid IN (0, 1)),
    quality_flags TEXT    NOT NULL DEFAULT '[]',
    received_at   TEXT    NOT NULL,
    UNIQUE (field_id, device_id, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_readings_field_time ON sensor_readings (field_id, timestamp);

CREATE TABLE IF NOT EXISTS recommendations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    field_id              INTEGER NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
    created_at            TEXT    NOT NULL,
    decision_time         TEXT    NOT NULL,
    irrigation_needed     INTEGER NOT NULL CHECK (irrigation_needed IN (0, 1)),
    decision_source       TEXT    NOT NULL,
    confidence            TEXT    NOT NULL,
    probability           REAL,
    net_depth_mm          REAL    NOT NULL DEFAULT 0,
    gross_depth_mm        REAL    NOT NULL DEFAULT 0,
    water_volume_l        REAL    NOT NULL DEFAULT 0,
    duration_min          REAL    NOT NULL DEFAULT 0,
    soil_moisture_pct     REAL,
    depletion_mm          REAL,
    forecast_rain_24h_mm  REAL,
    model_version         TEXT,
    summary               TEXT    NOT NULL,
    payload               TEXT    NOT NULL           -- full JSON (reasons, warnings, sensor issues)
);
CREATE INDEX IF NOT EXISTS idx_recs_field_time ON recommendations (field_id, decision_time);

CREATE TABLE IF NOT EXISTS irrigation_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    field_id          INTEGER NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
    recommendation_id INTEGER REFERENCES recommendations(id) ON DELETE SET NULL,
    start_time        TEXT    NOT NULL,
    duration_min      REAL    NOT NULL CHECK (duration_min >= 0),
    water_volume_l    REAL    NOT NULL CHECK (water_volume_l >= 0),
    source            TEXT    NOT NULL DEFAULT 'manual',
    notes             TEXT,
    created_at        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_field_time ON irrigation_events (field_id, start_time);
