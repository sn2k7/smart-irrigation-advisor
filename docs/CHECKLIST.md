# Implementation checklist

Status of every requirement in the project brief. "Test" names the test module that covers the behaviour.

## Data
- [x] Synthetic agricultural dataset generator (weather, soil physics, farmers, sensors, faults, labels) - `data/simulator.py`, `data/climate.py` - test: `test_simulator.py`
- [x] External CSV loading with column aliases, defaults and weak-label derivation - `data/loader.py`, `data/labels.py` - test: `test_splitting_training.py::test_weak_labels_for_external_data`
- [x] Data cleaning, validation and hourly standardisation - `data/validation.py`, `data/preprocessing.py` - tests: `test_validation.py`, `test_preprocessing.py`
- [x] Time-series feature engineering (lags, rolling windows, hours since rain, FAO-56 ratios, forecast) - `data/features.py` - test: `test_features.py`

## Machine learning
- [x] Logistic Regression, Random Forest, Gradient Boosting classifiers - `ml/pipelines.py`
- [x] Optional regression of irrigation depth (Ridge / RF / GB) -> duration and volume - `ml/training.py`
- [x] Accuracy, precision, recall, F1, ROC-AUC, confusion matrices, MAE, RMSE, R² - `ml/evaluation.py`
- [x] Identical preprocessing/features for training and inference; schema-versioned bundle - `ml/model_store.py` - tests: `test_features.py`, `test_splitting_training.py::test_schema_mismatch_is_rejected`
- [x] Leakage prevention: held-out fields, chronological test, rolling-origin CV, 24 h embargo, target-column guard - `ml/splitting.py` - test: `test_splitting_training.py`
- [x] Rule baseline is a selection candidate; selection documented in report - `ml/training.py`
- [x] Generated evaluation reports + figures - `reports/evaluation_report.{md,json}`, `reports/*.png`

## Baseline and water savings
- [x] Fixed-schedule baselines (pure, skip-after-rain, crop-stage adjusted) - `simulation/strategies.py`
- [x] Closed-loop measured comparison on unseen virtual fields, clearly labelled SIMULATED - `simulation/comparison.py`, `reports/simulation_report.md` - test: `test_simulation.py`
- [x] No real-world savings claimed anywhere; analytics label the fixed-schedule figure as an estimate

## Recommendation engine
- [x] Transparent reasons/warnings, decision source and confidence - `engine/recommender.py`
- [x] Rainfall awareness (forecast credit, rain-in-progress deferral) - test: `test_engine.py`
- [x] Sensor validation (range, spike, stuck, stale, future, battery) - test: `test_engine.py`, `test_validation.py`
- [x] Safety rules (field capacity, severe stress override, run-time cap) and fallbacks (estimated, conservative) - test: `test_engine.py`

## Backend
- [x] FastAPI REST API with Pydantic schemas and `/docs` - `api/` - test: `test_api.py`
- [x] Endpoints: health, reference, fields CRUD, devices, readings (single/batch/history), IoT ingest, recommendations, irrigation events, daily analytics, summary, weather forecast, model info/reload, calculator
- [x] SQLite storage of readings, recommendations, events, analytics - `db/` - test: `test_database.py`
- [x] Offline operation with simulated weather; IoT-ready ESP32 interface (+ API key) - `data/weather_provider.py`, `api/routers/iot.py`, `firmware/`
- [x] Error handling (domain exceptions -> 404/409/422/400/500), logging (console + rotating file), type hints, docstrings

## Follow-up: custom dataset slot and desktop frontend
- [x] Config-driven adapter for a different dataset (column map, units incl. raw ADC/fraction/°F/inches/cumulative rain, timestamp format/offset, per-field metadata, label mode) - `data/custom_dataset.py`, `config/` - test: `test_custom_dataset.py`
- [x] Code hooks for dataset-specific logic (`PLACEHOLDER(real-dataset)`) - `data/dataset_hooks.py`
- [x] `scripts/prepare_custom_dataset.py`, `train_models.py --dataset-config`, `generate_dataset.py --export-foreign`
- [x] Trainer tolerates labels without depth (regression skipped, FAO-56 depth used)
- [x] PySide6 desktop app: dashboard, readings + CSV import, history, analytics, fields/devices, calculator, model/system - `irrigation_gui/` - test: `test_gui.py`; screenshots in `docs/screenshots/`

## Deliverables
- [x] Example payloads - `examples/`
- [x] Reproducible scripts - `scripts/generate_dataset.py`, `train_models.py`, `run_simulation.py`, `seed_demo_data.py`, `iot_device_simulator.py`, `run_full_pipeline.ps1`, `demo_api.ps1`
- [x] `requirements.txt`, `.env.example`, `README.md`
- [x] Architecture, database schema, ML workflow, deployment docs - `docs/`
- [x] Final review of imports, dependencies, signatures, error handling and tests (see README "Verification")
