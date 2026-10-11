# Open-Meteo dataset (11.93975 N, 79.49244 E)

Real hourly data for the requested location from [Open-Meteo](https://open-meteo.com) (free, no API key,
non-commercial use). Configuration: `config/openmeteo_dataset.json`. Code: `irrigation_advisor/data/openmeteo.py`.

## Parameters

| Parameter | Open-Meteo source | How it is used |
|---|---|---|
| Soil moisture | archive `soil_moisture_0_to_7cm` + `soil_moisture_7_to_28cm` (ERA5-Land), depth-weighted, m³/m³ → % VWC | root-zone depletion, the core of every decision |
| Temperature | `temperature_2m` | weather features, Hargreaves fallback |
| Relative humidity | `relative_humidity_2m` | weather features |
| Rain in last 24 h | `precipitation`, rolling 24 h sum (column `rain_last_24h_mm`, feature `rain_sum_24h`) | rain awareness, labels |
| Rain forecast (24 h) | **Historical Forecast API** `precipitation` (archived model forecasts), summed over (t, t+24 h] | effective-rain credit; real forecasts, not hindsight. Hours without an archived forecast use observed next-24 h rain and are flagged in `forecast_source` |
| ET0 | `et0_fao_evapotranspiration` (FAO-56 Penman-Monteith) | replaces the temperature-only Hargreaves ET0 for ETc, in training **and** live decisions |
| Wind speed | `wind_speed_10m` (m/s) | stored, shown and quality-checked; its effect enters via Penman-Monteith ET0 (not double-counted as a model input) |

All requests use `timezone=Asia/Kolkata`, so timestamps are field-local.

## Steps (PowerShell, needs internet once)

```powershell
python scripts/fetch_openmeteo.py                    # -> data/real/openmeteo_dataset.csv (+ .report.json)
python scripts/fetch_openmeteo.py --train --fast     # download + quick training/evaluation
python real_dataset_gui.py --config config/openmeteo_generated_dataset.json   # check/train in the GUI
```

Responses are cached in `data/real/openmeteo_cache/`, so a repeated run does not download again (`--no-cache` to refresh).

### Live use with the API and desktop app

```powershell
$env:SIA_WEATHER_PROVIDER = "openmeteo"     # live rain forecast + Penman-Monteith ET0 (falls back to simulated offline)
uvicorn irrigation_advisor.api.main:app --port 8000
python scripts/openmeteo_live_ingest.py --field-id 1     # last 72 h as 'virtual sensor' readings for a field
python -m irrigation_gui
```

Create the field with latitude 11.93975 and longitude 79.49244. The engine requests forecasts and ET0 for
each field's own coordinates.

## Values to confirm (`PLACEHOLDER(openmeteo)` in the config)

* **Period**: default 2022-01-01 to 2025-12-31 (archived forecasts start around 2022).
* **Soil type**: `auto` picks the class whose field capacity (moisture 48 h after ≥ 10 mm rain) and wilting
  proxy best match the data. Replace it with the farm's measured texture when known.
* **Crop seasons**: example groundnut sown 15 June and 15 December. Replace with the real crop calendar.
  Each season becomes one "field" (gives the held-out-field test).
* **Irrigation method**: drip by default.

## Limitations (say these in the viva)

* Open-Meteo soil moisture is a **reanalysis for a ~9 km grid cell without irrigation**, not a probe in the
  field. It is real and physically consistent, which is good for developing and validating the pipeline,
  but deployment should use or calibrate against in-field sensors.
* Labels are derived (FAO-56 weak labels) because there is no irrigation log, so the rule baseline partly
  grades itself. Real irrigation decisions or stress observations give an independent evaluation.
* Archive soil layers (0-7/7-28 cm) differ from the live forecast layers (0-1/1-3/3-9/9-27 cm); the live
  ingest averages 0-27 cm to stay comparable.
