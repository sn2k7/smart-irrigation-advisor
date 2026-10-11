# Desktop frontend (PySide6)

`irrigation_gui/` is a Qt desktop application (PySide6) for farmers, field officers and demo
judges. It talks to the REST API only, so it can run on the same PC or on another machine in the
farm network.

## Start

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\run_gui.ps1                     # starts the API (minimised) and the app
# or manually, in two windows:
uvicorn irrigation_advisor.api.main:app --port 8000
python -m irrigation_gui --api-url http://127.0.0.1:8000
```

For a demo with data: `python scripts/seed_demo_data.py --reset` first (and `python scripts/train_models.py`
for the ML model). The API URL can also be changed in the top bar (Connect) or set via `SIA_API_URL`.

## Pages

| Page | What it does | Screenshot |
|---|---|---|
| Dashboard | Get a recommendation (optionally with your own 24 h rain forecast), decision/source/confidence chips, run time, litres, depth, moisture, depletion vs RAW, ETc, reasons and warnings, 72 h moisture chart, "Log irrigation as done", optional 5-min auto-refresh | `screenshots/gui_dashboard.png` |
| Sensor readings | Reading history (flagged rows tinted red with issue codes), moisture and temperature charts, manual entry, **CSV import with column/unit mapping** | `screenshots/gui_readings.png` |
| History | Stored recommendations with their reasons (audit trail), irrigation log, log an irrigation | `screenshots/gui_history.png` |
| Analytics | Period totals, recommended vs applied water, *estimated* fixed-schedule reference, daily charts and table, the simulated forecast with its disclaimer | `screenshots/gui_analytics.png` |
| Fields | Create / edit / delete fields, register ESP32 devices | `screenshots/gui_fields.png` |
| Irrigation calculator | Stateless FAO-56 calculation with step-by-step explanation | `screenshots/gui_calculator.png` |
| Model & system | API/DB status, active model, its held-out test metrics, hot reload | `screenshots/gui_model.png` |

## CSV import (your dataset → a field)

**Sensor readings → Import CSV / your dataset…** opens a dialog that:

1. previews the file (`,` `;` or tab separated) and guesses the column mapping from common names
   (`time`, `moisture`, `temp`, `rh`, `rain`, ...);
2. lets you choose units: soil moisture % or 0-1 fraction, °C/°F/K, rain in mm or inches, rain per row
   or cumulative, plus a timestamp format;
3. can pre-fill everything from a dataset config JSON (`config/*.json`, see `config/README.md`);
4. converts with the **same adapter used for training** (`irrigation_advisor/data/custom_dataset.py`) and
   uploads in batches of 1,000 to `POST /api/v1/readings/batch`. The server validates every reading,
   skips duplicates and flags implausible values.

All rows go into the field selected in the top bar.

## Design

* `api_client.py` - one method per endpoint; errors become `ApiError` with the server's message.
* `tasks.py` - `TaskRunner` runs HTTP calls in Qt's thread pool and delivers results on the GUI thread
  (queued signals), so the window never freezes. Tests use `TaskRunner(synchronous=True)`.
* `pages/` - one class per page (`Page` base: scroll area, header, `on_field_changed`, `refresh`).
  Only the visible page loads data; others reload when shown.
* `charts.py` - matplotlib embedded via `FigureCanvasQTAgg`, one y-axis per chart, fixed series colours.
* `theme.py` - style sheet and status colours. Status is always text plus colour, never colour alone.

## Tests

`tests/test_gui.py` drives the real window headless (`QT_QPA_PLATFORM=offscreen`) against the real API
(in-process TestClient): dashboard recommendation and logging, field CRUD and devices, readings,
history, analytics, model page, calculator, CSV import with unit conversion, and the threaded task runner.
On Linux servers without a display, PySide6 needs `libegl1 libgl1 libxkbcommon0 libfontconfig1`;
the tests are skipped if PySide6 cannot be imported.
