# Database schema

SQLite file (default `data/irrigation.db`), created automatically on API start-up or by
`scripts/seed_demo_data.py`. The DDL lives in `irrigation_advisor/db/schema.sql`.

* `PRAGMA foreign_keys = ON` on every connection; deleting a field cascades to its data.
* `PRAGMA journal_mode = WAL` so IoT writes and dashboard reads do not block each other.
* One connection + transaction per API request (commit on success, rollback on error).
* Timestamps are ISO-8601 text in **field-local time** (`YYYY-MM-DDTHH:MM:SS`). Lexicographic order
  equals chronological order, so range queries and `substr(timestamp, 1, 10)` daily grouping work.

## Entity relationships

```
fields 1 ──< devices
fields 1 ──< sensor_readings
fields 1 ──< recommendations 1 ──< irrigation_events (optional link)
fields 1 ──< irrigation_events
```

## Tables

### `fields`
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | autoincrement |
| name | TEXT | |
| crop_type | TEXT | one of the crops in `agronomy/crops.py` (validated by the API) |
| soil_type | TEXT | sandy, sandy_loam, loam, clay_loam, clay |
| irrigation_method | TEXT | drip (0.90), sprinkler (0.75), furrow (0.60) efficiency |
| planting_date | TEXT | drives Kc, root depth and growth stage |
| area_m2 | REAL | CHECK > 0 |
| flow_rate_lpm | REAL | CHECK > 0, pump/valve flow used for run time |
| latitude / longitude | REAL | CHECK ranges; latitude used for ET0 and simulated weather |
| max_irrigation_minutes | REAL | safety cap per run (default 180) |
| created_at / updated_at | TEXT | |

### `devices`
IoT nodes mapped to a field. `device_id` (PK, e.g. `esp32-node-01`), `field_id` (FK), `description`,
`created_at`, `last_seen_at` (updated on every IoT message).

### `sensor_readings`
| Column | Notes |
|---|---|
| field_id, device_id, timestamp | **UNIQUE** together - re-sent messages are ignored (reported as duplicates) |
| soil_moisture | % volumetric water content |
| temperature_c, humidity_pct, rainfall_mm | rainfall is the amount since the previous reading |
| battery_v, rssi_dbm | device health |
| source | `api`, `batch`, `iot`, `seed` |
| is_valid | 0/1 - soil moisture usable for decisions (see `data/validation.py`) |
| quality_flags | JSON list of issue codes (`out_of_range`, `spike`, `stuck_sensor`, `stale_reading`, ...) |
| received_at | server receive time |

Index: `(field_id, timestamp)` - the engine always reads one field's recent window.

### `recommendations`
Key numbers as columns for analytics (`irrigation_needed`, `decision_source`, `confidence`,
`probability`, `net_depth_mm`, `gross_depth_mm`, `water_volume_l`, `duration_min`, `soil_moisture_pct`,
`depletion_mm`, `forecast_rain_24h_mm`, `model_version`, `summary`) plus `payload`: the full JSON response
(reasons, warnings, sensor issues) so every decision can be audited later.
Index: `(field_id, decision_time)`.

### `irrigation_events`
What was actually applied (manual log or valve controller): `start_time`, `duration_min`,
`water_volume_l` (if omitted, `duration_min × flow_rate_lpm`), optional `recommendation_id`
(must belong to the same field), `source`, `notes`. Separating *recommended* from *applied* water lets
analytics measure adherence.

## Analytics queries (`repository.py`)

* `daily_analytics` - per day: number of readings and invalid readings, mean/min valid soil moisture,
  max temperature, mean humidity, rain sum, recommendations, recommended water, applied water, pump minutes.
  Days without data are returned with zeros so charts have no gaps.
* `field_summary` - totals for a period and a **calculated** fixed-schedule reference
  (`ceil(days / interval) × interval × Kc_mid × 5 mm / efficiency × area`). It is labelled as an estimate,
  never as a measured saving.

## Backup / reset

The database is a single file: copy `data/irrigation.db` (with the API stopped) to back it up;
`python scripts/seed_demo_data.py --reset` recreates a demo database.
