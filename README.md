# Smart Irrigation Advisor for Small Farmers

An offline Python backend that tells a small farmer **whether to irrigate now, how long to run the pump
and how much water that uses**. It combines soil moisture, temperature, humidity, rainfall, a 24 h rain
forecast, crop type/stage and the field's recent sensor history.

* **FAO-56 water balance at the core:** crop coefficients, root depth, readily available water, depletion.
* **Machine learning where it earns its place:** Logistic Regression, Random Forest and Gradient Boosting
  classifiers, plus regressors for irrigation depth. They are selected by leakage-safe cross-validation
  against the physics rule.
* **Transparent and safe:** every recommendation lists its reasons. Sensor validation, rain awareness,
  safety overrides and two fallback levels keep it working when sensors fail.
* **Runs anywhere:** FastAPI and SQLite, simulated offline weather and an ESP32-ready IoT endpoint.
  No paid APIs, internet access or hardware are needed.
* **Desktop app:** a PySide6 (Qt) frontend for dashboards, readings, CSV import, history, analytics and the
  calculator ([`docs/FRONTEND.md`](docs/FRONTEND.md)).
* **Bring your own dataset:** a JSON config plus a hooks file let you plug in a different dataset without
  touching the pipeline ([`config/README.md`](config/README.md)).

> **Honesty notice.** All training data, weather and water-savings figures in this repository are
> **synthetic/simulated**. They show that the method works and let strategies be compared under stated
> assumptions. They are **not** measured real-world results.

---

## 1. Quick start (Windows PowerShell)

```powershell
# 1. environment (Python 3.11 or 3.12 recommended; 3.10-3.13 supported)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2. data -> models -> simulation -> demo database -> tests (one command, ~5-10 min)
.\scripts\run_full_pipeline.ps1            # add -Fast for a ~2 min version

# 3. start the API + desktop app
.\scripts\run_gui.ps1
# or only the API: uvicorn irrigation_advisor.api.main:app --host 127.0.0.1 --port 8000  (docs at /docs)
```

Individual steps:

```powershell
python scripts/generate_dataset.py --fields 30 --seed 42      # SYNTHETIC dataset -> data/
python scripts/train_models.py                                # train + evaluate -> models/, reports/
python scripts/run_simulation.py --fields 10                  # closed-loop comparison -> reports/
python scripts/seed_demo_data.py --reset                      # 3 demo fields with 5 days of data
python -m pytest -q                                           # unit/integration/API/GUI tests
python -m irrigation_gui --api-url http://127.0.0.1:8000      # desktop app (API must be running)
uvicorn irrigation_advisor.api.main:app --port 8000           # API
.\scripts\demo_api.ps1                                        # scripted API walkthrough (2nd window)
python scripts/iot_device_simulator.py --field-id 1 --inject-faults   # fake ESP32 node (2nd window)
```

Linux/macOS: identical commands with `source .venv/bin/activate`.

---

## 2. Architecture

```
ESP32 node ──compact JSON──┐            ┌── Swagger UI / app / curl
                           ▼            ▼
                 FastAPI routers (iot, readings, recommendations, fields, analytics, weather, model, tools)
                           │
          services: ingest_reading (validate + store) | make_recommendation (load 72 h + engine)
                           │
     RecommendationEngine ─┼─ forecast (client value or offline simulated provider)
                           ├─ sensor validation  ──► fallback_estimated / fallback_conservative
                           ├─ prepare_sensor_frame + build_features   (SAME code as training)
                           ├─ FAO-56 rule  +  ML model (only if it won cross-validation)
                           ├─ safety rules: field capacity, rain in progress, severe stress
                           └─ plan: net mm → gross mm → litres → pump minutes (capped)
                           │
                       SQLite: fields · devices · sensor_readings · recommendations · irrigation_events

Offline: simulator → preprocessing → features → split (held-out fields, temporal test, rolling CV, embargo)
         → LR / RF / GB vs FAO-56 rule → model bundle + reports;  closed-loop strategy simulation → reports
```

Full directory tree and design decisions: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

| Package | Responsibility |
|---|---|
| `irrigation_advisor/agronomy` | FAO-56 tables (7 crops, 5 soils, 3 methods), ET0, Kc, TAW/RAW, depletion, irrigation plan, true-state bucket model |
| `irrigation_advisor/data` | weather generator, offline weather provider, synthetic dataset, CSV loader, validation, preprocessing, features, labels |
| `irrigation_advisor/ml` | pipelines, leakage-safe splitting, training/selection, metrics, reports, plots, model store |
| `irrigation_advisor/engine` | recommendation engine |
| `irrigation_advisor/simulation` | fixed-schedule baselines vs advisor, closed loop |
| `irrigation_advisor/db` | schema, connections, repository/analytics SQL |
| `irrigation_advisor/api` | FastAPI app, schemas, routers |
| `irrigation_gui` | PySide6 desktop frontend (API client, threaded tasks, pages, charts, CSV import) |
| `config/` | dataset config template + working sample for plugging in your own data |

---

## 3. How a decision is made

For a field with root depth `Zr`, field capacity `FC` and wilting point `WP` (all FAO-56):

```
TAW = 1000 (FC − WP) Zr              total available water (mm)
RAW = p · TAW                        stress starts beyond this depletion (p is crop-specific)
Dr  = 1000 (FC − θ) Zr               current depletion from the soil-moisture sensor
ETc = Kc · ET0(Hargreaves)           crop water use over the next 24 h
Peff = 0.8 · forecast rain (if ≥ 2 mm)

Irrigate if  Dr ≥ RAW  or  Dr + ETc − Peff ≥ RAW
Net depth  = min(TAW, max(Dr − Peff, ETc))
Gross      = net / efficiency (drip 0.90, sprinkler 0.75, furrow 0.60)
Litres     = gross × area_m²          (1 mm on 1 m² = 1 L)
Minutes    = litres / flow_rate_lpm   (capped by the field's max run time)
```

If an ML model won model selection, its probability replaces the rule's yes/no and its depth replaces the
formula. The rule is still computed, and disagreements are reported and lower the confidence. Safety
rules always apply:

| Situation | Behaviour |
|---|---|
| Soil at/above field capacity | never irrigate |
| ≥ 1 mm rain this hour or ≥ 5 mm in 6 h (and stress not severe) | defer, re-check in 3 h |
| Depletion past halfway between RAW and TAW | irrigate even if the model says no |
| Latest soil reading invalid/missing/stale, valid one < 24 h old | `fallback_estimated`: water-balance extrapolation |
| No usable soil data in 24 h | `fallback_conservative`: at most one day of ETc, skipped if rain is expected |

Real response for demo field 1 (sandy soil, tomato, drip) after `scripts/seed_demo_data.py`, abridged.
The exact numbers depend on the date you seed, because the demo history is simulated up to "now":

```json
{
  "irrigation_needed": true,
  "decision_source": "ml_hybrid",
  "confidence": "high",
  "summary": "Irrigate now: run 240 min (~21,000 L, 14.0 mm)",
  "probability": 0.969,
  "net_depth_mm": 12.6, "gross_depth_mm": 14.0, "water_volume_l": 21000.0, "duration_min": 240.0, "capped": true,
  "soil_moisture_pct": 8.1, "depletion_mm": 33.03, "raw_mm": 25.41, "taw_mm": 63.52, "depletion_ratio": 1.3,
  "etc_24h_mm": 4.03, "kc": 0.953, "growth_stage": "development",
  "forecast_rain_24h_mm": 0.0, "forecast_source": "client",
  "rule_decision": true, "ml_decision": true,
  "reasons": [
    "Root-zone depletion 33.0 mm = 130% of the stress threshold RAW 25.4 mm (TAW 63.5 mm)",
    "Estimated crop water use ETc 4.0 mm/day (Kc 0.95, stage development)",
    "ML model (random_forest+gradient_boosting-...) irrigation probability 0.97 (decision threshold 0.40)",
    "Net depth from the ML regressor: 26.4 mm (FAO-56 formula: 33.0 mm)",
    "Apply 12.6 mm net = 14.0 mm gross at 90% drip efficiency over 1500 m2"
  ],
  "warnings": ["Run time capped at 240 min; remaining need will be re-evaluated at the next check"]
}
```

---

## 4. Machine learning in one page

Details: [`docs/ML_WORKFLOW.md`](docs/ML_WORKFLOW.md). Generated report: [`reports/evaluation_report.md`](reports/evaluation_report.md).

* **Target:** "will depletion reach RAW within 24 h without irrigation?" (from the true simulated soil state),
  plus the net depth needed.
* **Same code online and offline:** `prepare_sensor_frame` → `build_features` for both. Backward-looking
  windows only. Tests prove that changing future rows leaves past features unchanged, and that the 72 h
  inference window reproduces the training features.
* **No leakage:** 20 % of fields held out entirely, a chronological test period, rolling-origin CV for
  selection, and a 24 h embargo at every boundary (asserted in code).
* **Features are dimensionless** (depletion/RAW, ETc/RAW, ...). This lets one model transfer across soils,
  crops and seasons. Absolute values overfitted, as an ablation showed.
* **The FAO-56 rule is a candidate model.** ML is deployed only if it beats the rule in CV (depth needs
  a ≥ 5 % margin).

Results on the synthetic data (30 fields, 106k hourly rows):

| | CV F1 | Temporal-test F1 | Unseen-fields F1 |
|---|---|---|---|
| Random Forest (selected) | 0.890 | 0.883 | 0.858 |
| Gradient Boosting | 0.881 | 0.906 | 0.871 |
| Logistic Regression | 0.875 | 0.877 | 0.867 |
| FAO-56 rule | 0.884 | 0.915 | 0.890 |

| Net depth | CV MAE | Temporal-test MAE | Unseen-fields MAE |
|---|---|---|---|
| Gradient Boosting (selected) | 6.6 mm | 7.6 mm | 8.3 mm |
| FAO-56 formula | 8.6 mm | 9.1 mm | 7.0 mm |

Plain reading: on data generated from FAO-56 physics, the physics rule is very hard to beat at the
yes/no decision. ML matches it in CV, has higher recall and ROC-AUC, but is slightly lower on the test
sets. ML improves depth estimates on the temporal test. That is why the system is a hybrid with the
rule always visible. Figures: `reports/model_comparison.png`, `reports/confusion_matrices_*.png`,
`reports/feature_importance.png`, `reports/regression_depth_scatter.png`.

---

## 5. Water savings: simulated, with a fair baseline

`scripts/run_simulation.py` runs a **closed loop**: virtual fields (unseen seed, synthetic weather, noisy and
faulty sensors) where each strategy's decisions change the simulated soil. Every strategy sees identical
weather and sensor noise. Report: [`reports/simulation_report.md`](reports/simulation_report.md).

Mean per field and season over 10 virtual fields:

| Strategy | Water (m³/ha) | Stress days | Rel. transpiration | Drainage (mm) |
|---|---|---|---|---|
| Fixed schedule (every 3 days, peak depth) | 12,724 | 0.3 | 1.000 | 780 |
| Fixed, skipped after ≥ 10 mm rain | 11,292 | 0.3 | 1.000 | 683 |
| Crop-stage (Kc) calendar, skipped after rain | 8,516 | 6.1 | 0.990 | 496 |
| Advisor - rules only | 4,643 | 5.1 | 0.998 | 258 |
| Advisor - ML hybrid | 4,537 | 7.2 | 0.997 | 248 |

Against the **strongest** no-sensor baseline (crop-stage calendar), the advisor used about 45-47 % less
water in this simulation, with comparable stress and higher relative transpiration. Savings against
naive calendars look larger (about 60 %), mainly because those calendars over-irrigate (see drainage).
**These are model outputs under stated assumptions, not field measurements.**

---

## 6. API overview

Interactive docs at `/docs` (Swagger) and `/redoc`. Example payloads: [`examples/`](examples/).

| Method & path | Purpose |
|---|---|
| `GET /health` | service, DB and model status |
| `GET /api/v1/reference/{crops,soils,irrigation-methods}` | agronomic parameters |
| `POST/GET/PATCH/DELETE /api/v1/fields[/{id}]` | field management |
| `POST /api/v1/devices`, `GET /api/v1/devices` | register IoT nodes |
| `POST /api/v1/readings`, `POST /api/v1/readings/batch` | store readings (validated, flagged) |
| `GET /api/v1/fields/{id}/readings` | reading history |
| `POST /api/v1/iot/ingest?recommend=true` | compact ESP32 payload → optional valve command (`X-Device-Key` if configured) |
| `POST /api/v1/recommendations` | decide now (optional inline reading, optional own rain forecast) |
| `GET /api/v1/recommendations/{id}`, `GET /api/v1/fields/{id}/recommendations` | history/audit |
| `POST/GET /api/v1/fields/{id}/irrigation-events` | log irrigation actually applied |
| `GET /api/v1/fields/{id}/analytics/daily`, `/summary` | daily statistics, totals vs estimated fixed schedule |
| `GET /api/v1/fields/{id}/weather/forecast` | simulated offline forecast (labelled) |
| `GET /api/v1/model`, `POST /api/v1/model/reload` | model metadata and test metrics, hot reload |
| `POST /api/v1/tools/irrigation-calculator` | stateless FAO-56 calculator with worked explanation |

PowerShell example:

```powershell
$body = Get-Content examples\recommendation_request.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/recommendations -ContentType "application/json" -Body $body
```

---

## 7. Using a different dataset

The code leaves a defined slot for your data, so no pipeline changes are needed:

| Where | What you fill in |
|---|---|
| `config/dataset_config.example.json` | copy it and replace the `<...>` placeholders: file path, your column names, units (% / fraction / raw ADC, °C/°F/K, mm/inch, per-row or cumulative rain), timestamp format, field metadata (crop, soil, method, planting date, latitude), label mode |
| `irrigation_advisor/data/dataset_hooks.py` | optional code for things a mapping cannot express (merge files, pivot long data, per-probe calibration, your own labels). Search for `PLACEHOLDER(real-dataset)` |

```powershell
Copy-Item config\dataset_config.example.json config\my_dataset.json      # edit the <...> values
python scripts/prepare_custom_dataset.py --config config\my_dataset.json   # check mapping, units, quality
python scripts/train_models.py --dataset-config config\my_dataset.json     # train + evaluate on your data
```

Labels: `"derive"` computes FAO-56 weak labels from your observations, or use `"columns"` with your own
`target_irrigation_needed` (and optionally `target_irrigation_mm`). With derived labels the rule baseline
is partly scored against its own definition, so treat its score with care. A worked example with a
deliberately foreign format: `python scripts/generate_dataset.py --fields 12 --export-foreign data/foreign_format_sample.csv`,
then use `config/dataset_config.sample.json`. The desktop app's CSV import uses the same adapter.

## 8. Data, storage and IoT

* **Database:** [`docs/DATABASE.md`](docs/DATABASE.md). Five tables, foreign keys with cascade, WAL mode,
  unique `(field, device, timestamp)` so re-sent messages are ignored. Invalid readings are kept
  (flagged) for diagnostics.
* **Quick external CSV:** `python scripts/train_models.py --external-csv my.csv --crop maize --soil loam`
  maps common column names automatically. For anything else, use the dataset config in section 7.
* **IoT:** [`firmware/`](firmware/) contains a reference ESP32 sketch (capacitive probe, DHT22, rain gauge,
  relay). `scripts/iot_device_simulator.py` sends the same payloads without hardware.

---

## 9. Deployment

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md): LAN serving, Windows service (NSSM), systemd,
configuration reference (`.env.example`), model hot-reload, backups, security notes and troubleshooting.

---

## 10. Verification

* `python -m pytest -q` runs the unit, integration, API and GUI tests (agronomy maths against FAO-56 worked
  examples, validation rules, preprocessing, leakage tests, splitting, training/selection, engine paths
  and fallbacks, simulation, repository SQL, every API endpoint, the custom-dataset adapter and the
  desktop app driven headless).
* Static check: `python -m pyflakes irrigation_advisor scripts tests` (only intentional `_bootstrap` imports reported).
* Implementation checklist: [`docs/CHECKLIST.md`](docs/CHECKLIST.md).

## 11. Limitations and next steps

* Validate on real field data (one season of paired plots: advisor vs farmer practice).
* Calibrate each capacitive probe; per-probe bias cannot be learned from its own data.
* Penman-Monteith ET0 when radiation/wind sensors are available; a real forecast provider when online.
* Paddy rice and perennial crops are out of scope (different water management).
