# Deployment and operations

## 1. Local Windows setup (PowerShell)

Requirements: Windows 10/11, **Python 3.11 or 3.12** (3.10-3.13 supported) from python.org with
"Add python.exe to PATH" ticked. No internet is needed after `pip install`.

```powershell
cd C:\path\to\smart-irrigation-advisor
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env        # optional - defaults work without it
```

Full pipeline (data -> training -> simulation -> demo DB -> tests):

```powershell
.\scripts\run_full_pipeline.ps1          # or: .\scripts\run_full_pipeline.ps1 -Fast
```

Or step by step:

```powershell
python scripts/generate_dataset.py --fields 30 --seed 42
python scripts/train_models.py
python scripts/run_simulation.py --fields 10
python scripts/seed_demo_data.py --reset
python -m pytest -q
uvicorn irrigation_advisor.api.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs. In a second PowerShell window (venv activated):

```powershell
.\scripts\demo_api.ps1
python scripts/iot_device_simulator.py --field-id 1 --messages 8 --inject-faults
```

## 2. Serving on a farm LAN (Raspberry Pi / mini-PC / laptop)

```powershell
uvicorn irrigation_advisor.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

* Set `SIA_IOT_API_KEY` in `.env` and the same value in the ESP32 sketch (`DEVICE_KEY`).
* Open the port in Windows Defender Firewall (private network only):
  `New-NetFirewallRule -DisplayName "Irrigation API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private`
* Use **one worker**: SQLite handles concurrent readers with WAL, but a single writer process is the
  simplest correct setup for a farm-scale deployment (hundreds of readings per hour are trivial).
* Keep the server clock correct; readings without a timestamp are stamped with server time.

### Run as a Windows service (optional)
Use NSSM (`nssm install IrrigationAPI`) with
`Path = C:\...\.venv\Scripts\python.exe`,
`Arguments = -m uvicorn irrigation_advisor.api.main:app --host 0.0.0.0 --port 8000`,
`Startup directory = C:\...\smart-irrigation-advisor`.

### Linux (systemd) equivalent
```
ExecStart=/opt/sia/.venv/bin/uvicorn irrigation_advisor.api.main:app --host 0.0.0.0 --port 8000
WorkingDirectory=/opt/sia
Restart=always
```

## 3. Configuration reference (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `SIA_DATABASE_PATH` | `data/irrigation.db` | SQLite file |
| `SIA_MODEL_PATH` | `models/irrigation_model.joblib` | model bundle; missing -> rule-based engine |
| `SIA_DATA_DIR` / `SIA_REPORTS_DIR` / `SIA_LOG_DIR` | `data` / `reports` / `logs` | output folders |
| `SIA_LOG_LEVEL` | `INFO` | `DEBUG` logs every request with timing |
| `SIA_IOT_API_KEY` | empty | required `X-Device-Key` for `/api/v1/iot/ingest` when set |
| `SIA_CLIMATE_PROFILE` | `semi_arid_monsoon` | simulated weather (`humid_subtropical` also available) |
| `SIA_DEFAULT_LATITUDE` | `18.5` | default for scripts |
| `SIA_UTC_OFFSET_HOURS` | `5.5` | converts tz-aware timestamps (ESP32 epoch) to field-local time |
| `SIA_HISTORY_HOURS` | `72` | history window loaded per recommendation |
| `SIA_STALE_READING_HOURS` | `6` | older latest reading -> fallback path |

## 4. Operations

* **Logs**: console + `logs/app.log` (rotating, 3 × 2 MB).
* **Health**: `GET /health` reports DB connectivity and whether a model is loaded.
* **Model updates**: retrain (`scripts/train_models.py`), then `POST /api/v1/model/reload` - no restart.
  A model with an incompatible feature schema is refused and the engine stays rule-based.
* **Backups**: copy `data/irrigation.db` while the API is stopped (or use `sqlite3 .backup`).
* **Real weather**: pass `forecast_rain_24h_mm` per request (e.g. from an SMS/radio bulletin or IMD), or
  implement the `WeatherProvider` protocol (`data/weather_provider.py`) for an online service.

## 5. Security notes

* The API has no user authentication; run it on a trusted LAN or behind a reverse proxy with auth/TLS
  (e.g. Caddy or nginx) if exposed.
* IoT ingest can be protected with `SIA_IOT_API_KEY` (constant-time comparison).
* Model files are joblib/pickle - only load models you trained yourself.
* All SQL uses parameter binding; dynamic column names come from a fixed allow-list.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `/health` shows `model_loaded: false` | run `python scripts/train_models.py`, then `POST /api/v1/model/reload` |
| Every recommendation is `fallback_conservative` | readings are older than `SIA_STALE_READING_HOURS` or missing; check device time / `ts` |
| IoT returns 404 | register the device: `POST /api/v1/devices` |
| IoT returns 401 | `X-Device-Key` header does not match `SIA_IOT_API_KEY` |
| `database is locked` | run a single uvicorn worker; avoid opening the DB in another tool while writing |
