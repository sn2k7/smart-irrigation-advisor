# Architecture

## 1. Goals and constraints

| Constraint | How it is met |
|---|---|
| Runs offline on a Windows laptop | Pure Python, SQLite file DB, simulated weather generator, no cloud/paid APIs |
| No physical sensors required | Synthetic data generator, seed script, IoT device simulator |
| Scientifically defensible | FAO-56 water balance at the core; ML only where it beats the physics on held-out data |
| Safe | Sensor validation, safety overrides, two fallback levels, run-time caps, confidence levels |
| Explainable | Every recommendation carries human-readable reasons, warnings and its decision source |

## 2. Directory structure

```
smart-irrigation-advisor/
├── irrigation_advisor/            # the Python package
│   ├── config.py                  # SIA_* settings from env/.env (safe defaults)
│   ├── logging_config.py          # console + rotating file logging
│   ├── exceptions.py              # domain errors -> HTTP codes
│   ├── utils.py                   # clamp, seeds, JSON encoder, time helpers
│   ├── agronomy/
│   │   ├── crops.py               # FAO-56 crop, soil and irrigation-method tables
│   │   ├── water_balance.py       # ET0 (Hargreaves), Kc, TAW/RAW, depletion, irrigation plan
│   │   └── soil_model.py          # hourly "ground truth" bucket model (simulators only)
│   ├── data/
│   │   ├── climate.py             # stochastic weather generator + forecast error model
│   │   ├── weather_provider.py    # offline WeatherProvider (pluggable interface)
│   │   ├── simulator.py           # synthetic dataset generator (fields, farmers, sensors, faults)
│   │   ├── loader.py              # external CSV loading with column aliases
│   │   ├── custom_dataset.py      # YOUR dataset: JSON-config driven mapping/units/metadata adapter
│   │   ├── dataset_hooks.py       # YOUR code: before_mapping / custom_transform (TODO(dataset))
│   │   ├── validation.py          # sensor plausibility rules (row + frame level)
│   │   ├── preprocessing.py       # cleaning + hourly standardisation  ┐ shared by
│   │   ├── features.py            # time-series feature engineering    ┘ training & inference
│   │   └── labels.py              # target definitions (+ weak labels for external data)
│   ├── ml/
│   │   ├── pipelines.py           # sklearn Pipelines (impute/scale/encode + model)
│   │   ├── splitting.py           # held-out fields, temporal test, rolling-origin CV, embargo
│   │   ├── training.py            # CV model selection (rule is a candidate), refit, test, save
│   │   ├── evaluation.py          # metrics, rule baseline, JSON/Markdown reports
│   │   ├── plots.py               # PNG figures
│   │   └── model_store.py         # bundle save/load with schema check
│   ├── engine/
│   │   └── recommender.py         # RecommendationEngine (ML hybrid + rules + fallbacks)
│   ├── simulation/
│   │   ├── strategies.py          # fixed schedules, advisor strategies
│   │   ├── field_env.py           # closed-loop virtual field
│   │   └── comparison.py          # multi-field comparison + reports
│   ├── db/
│   │   ├── schema.sql             # tables, constraints, indexes
│   │   ├── database.py            # connection management (WAL, FK, per-request transactions)
│   │   └── repository.py          # explicit SQL functions incl. analytics
│   └── api/
│       ├── main.py                # app factory, lifespan, error handlers, middleware
│       ├── schemas.py             # Pydantic models (shown in /docs)
│       ├── dependencies.py        # shared state, DB connection, IoT key check
│       ├── services.py            # ingest + recommend use cases
│       └── routers/               # health, fields, readings, iot, recommendations,
│                                  # events, analytics, weather, model, tools
├── irrigation_gui/                # PySide6 desktop frontend (talks to the API only)
│   ├── app.py  main_window.py     # bootstrap, navigation, field selector, status
│   ├── api_client.py  tasks.py    # HTTP client, thread-pool task runner
│   ├── import_dialog.py           # CSV import using data/custom_dataset.py
│   ├── charts.py  widgets.py  theme.py
│   └── pages/                     # dashboard, readings, history, analytics, fields, calculator, model
├── config/                        # dataset_config.example.json (template) + sample
├── scripts/                       # generate_dataset, train_models, prepare_custom_dataset, run_simulation,
│                                  # seed_demo_data, iot_device_simulator, *.ps1 (run_gui.ps1)
├── tests/                         # unit, integration and API tests (pytest)
├── examples/                      # example JSON payloads + external CSV format
├── firmware/esp32_soil_node/      # reference ESP32 sketch
├── reports/                       # generated evaluation + simulation reports and figures
├── docs/                          # this documentation
├── data/  models/  logs/          # generated at runtime (git-ignored)
├── requirements.txt  .env.example  pytest.ini  README.md
```

## 3. Component diagram

```
          ESP32 / Arduino node        PySide6 desktop app (irrigation_gui) / curl / Swagger UI
                 │  compact JSON                         │  REST JSON
                 ▼                                       ▼
┌────────────────────────────────── FastAPI (irrigation_advisor.api) ─────────────────────────────┐
│ routers: iot · readings · recommendations · fields · events · analytics · weather · model · tools│
│                │ services.ingest_reading                 │ services.make_recommendation         │
│                ▼                                         ▼                                      │
│      data.validation (row checks)            db.repository.readings_frame (last 72 h)          │
│                │                                         ▼                                      │
│                ▼                       ┌────────── engine.RecommendationEngine ───────────┐     │
│        SQLite (db/*.py)  ◄─────────────┤ 1 forecast  (client value or WeatherProvider)    │     │
│   fields · devices · sensor_readings   │ 2 validate latest reading                        │     │
│   recommendations · irrigation_events  │ 3 prepare_sensor_frame + build_features  ◄───────┼──┐  │
│                                        │ 4 rule (FAO-56) + ML model (if selected)        │  │  │
│                                        │ 5 safety overrides / fallbacks                  │  │  │
│                                        │ 6 plan: net → gross → litres → minutes          │  │  │
│                                        └──────────────────────────────────────────────────┘  │  │
└──────────────────────────────────────────────────────────────────────────────────────────────┼──┘
                                                                                               │
   OFFLINE PIPELINE (scripts/)                                                   same code ────┘
   simulator.generate_dataset ─► prepare_sensor_frame ─► build_features ─► splitting (CV, embargo)
        ─► training (LR / RF / GB vs FAO-56 rule) ─► model bundle (joblib) + reports/
   simulation.comparison: fixed schedules vs advisor on unseen virtual fields ─► reports/
```

## 4. Request flow: `POST /api/v1/recommendations`

1. `services.make_recommendation` loads the field and the last `SIA_HISTORY_HOURS` (72 h) of readings.
   Readings stored as invalid keep their row but their soil-moisture value is blanked.
2. `RecommendationEngine.recommend`:
   * gets the 24 h rain forecast (client-supplied value wins; otherwise the simulated provider - labelled);
   * validates the newest raw reading (range, spike, stuck, stale, future);
   * runs **the same** `prepare_sensor_frame` → `build_features` as training and takes the last row;
   * computes the FAO-56 rule decision and depth, and the ML probability/depth if a model was selected;
   * applies safety rules (field capacity, rain in progress, severe stress) or the fallbacks;
   * converts the net depth to gross depth, litres and minutes, capped by `max_irrigation_minutes`.
3. The result (incl. reasons/warnings) is stored in `recommendations` and returned.

## 5. Decision sources

| `decision_source` | When | Confidence |
|---|---|---|
| `ml_hybrid` | model loaded with an ML classifier, latest soil reading valid | from probability margin; reduced if ML and rule disagree or value was gap-filled |
| `rule_based` | no model, or model selection kept the rule | from distance to the RAW threshold |
| `fallback_estimated` | latest soil reading invalid/missing/stale but a valid one exists within 24 h | always low |
| `fallback_conservative` | no usable soil data in 24 h | always low; at most one day of ETc, skipped if rain expected |

## 6. Design decisions

* **Physics first, ML second.** The FAO-56 water balance is well established and explainable; ML is a
  candidate that has to beat it on cross-validated data (see `ML_WORKFLOW.md`).
* **One preprocessing/feature path.** `prepare_sensor_frame` and `build_features` are the only way data
  becomes model input, online and offline. A test verifies the 72 h inference window reproduces the
  training feature values.
* **Plain `sqlite3`.** Zero extra dependencies, explicit SQL that is easy to explain, file-based backups.
  The repository layer isolates SQL, so swapping to PostgreSQL later is contained.
* **Pluggable weather.** `WeatherProvider` protocol; the offline generator is the default, a real forecast
  can be passed per request or implemented as another provider.
* **Store invalid readings.** They are evidence for sensor diagnostics; the engine ignores their values.
