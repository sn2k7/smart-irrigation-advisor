"""PySide6 desktop frontend, driven against the real API (in-process TestClient).

Runs headless (QT_QPA_PLATFORM=offscreen). Skipped when PySide6 or its system
libraries are not available.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

from irrigation_gui.api_client import ApiClient, ApiError  # noqa: E402
from irrigation_gui.import_dialog import ImportDialog, guess_column, readings_payload  # noqa: E402
from irrigation_gui.main_window import MainWindow  # noqa: E402
from irrigation_gui.pages.analytics import AnalyticsPage  # noqa: E402
from irrigation_gui.pages.calculator import CalculatorPage  # noqa: E402
from irrigation_gui.pages.dashboard import DashboardPage  # noqa: E402
from irrigation_gui.pages.fields import FieldsPage  # noqa: E402
from irrigation_gui.pages.history import HistoryPage  # noqa: E402
from irrigation_gui.pages.model_page import ModelPage  # noqa: E402
from irrigation_gui.pages.readings import ReadingsPage  # noqa: E402
from irrigation_gui.tasks import TaskRunner  # noqa: E402

FIELD = {"name": "GUI plot", "crop_type": "maize", "soil_type": "loam", "irrigation_method": "drip",
         "planting_date": "2026-06-01", "area_m2": 2000, "flow_rate_lpm": 120, "latitude": 18.5}


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def api(client):
    return ApiClient("http://testserver", client=client)


@pytest.fixture()
def window(qapp, api):
    win = MainWindow(api, TaskRunner(synchronous=True))
    win.confirm = lambda question: True  # no modal dialogs in tests
    yield win
    win.health_timer.stop()
    win.deleteLater()


def seed(api: ApiClient, start: float = 18.5, end: float = 17.0) -> dict:
    field = api.create_field(FIELD)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    api.add_readings_batch([
        {"field_id": field["id"], "timestamp": (now - timedelta(hours=47 - i)).isoformat(),
         "soil_moisture": round(start + (end - start) * i / 47, 2), "temperature_c": 30.0, "humidity_pct": 50.0,
         "rainfall_mm": 0.0}
        for i in range(48)
    ])
    return field


def test_api_client_maps_errors(api):
    with pytest.raises(ApiError) as err:
        api.create_field({**FIELD, "crop_type": "rice"})
    assert err.value.status == 422 and "crop_type" in str(err.value)
    with pytest.raises(ApiError, match="not found"):
        api.list_readings(999)
    offline = ApiClient("http://127.0.0.1:9", timeout=0.5)
    with pytest.raises(ApiError, match="Cannot reach"):
        offline.health()


def test_window_loads_fields_reference_and_health(window, api):
    seed(api)
    window.start()
    assert window.field_combo.count() == 1 and window.current_field["name"] == "GUI plot"
    assert "online" in window.conn_chip.text()
    fields_page = window.page(FieldsPage)
    assert fields_page.crop.count() == 7 and fields_page.method.count() == 3


def test_dashboard_recommendation_and_logging(window, api):
    seed(api)
    window.start()
    page = window.page(DashboardPage)
    page.use_forecast.setChecked(True)
    page.forecast.setValue(0.0)
    page.request_recommendation()
    rec = page.last_rec
    assert rec["irrigation_needed"] is True
    assert page.decision_chip.text() == "IRRIGATE"
    assert "min" in page.tiles.tiles["duration"].value.text()
    assert page.reasons.count() >= 3
    assert page.irrigated_btn.isEnabled()
    page.log_irrigation()
    events = api.list_events(window.current_field["id"])
    assert len(events) == 1 and events[0]["recommendation_id"] == rec["id"]


def test_fields_page_create_update_delete(window, api):
    window.start()
    page = window.page(FieldsPage)
    page.name.setText("New plot")
    page.crop.setCurrentText("onion")
    page.create_field()
    assert [f["name"] for f in window.fields] == ["New plot"]
    page.table.fill(window.fields)
    page.table.selectRow(0)
    page.area.setValue(3000)
    page.update_field()
    assert window.fields[0]["area_m2"] == 3000 and window.fields[0]["crop_type"] == "onion"
    page.device_id.setText("esp32-gui")
    page.register_device()
    assert api.list_devices()[0]["device_id"] == "esp32-gui"
    page.table.selectRow(0)
    page.delete_field()
    assert window.fields == [] and window.current_field is None


def test_fields_page_requires_name(window):
    window.start()
    page = window.page(FieldsPage)
    page.name.setText("")
    page.create_field()
    assert "field name" in window.statusBar().currentMessage()


def test_readings_history_analytics_model_calculator_pages(window, api):
    seed(api)
    window.start()
    readings = window.page(ReadingsPage)
    window.nav.setCurrentRow(window.pages.index(readings))
    assert readings.table.rowCount() == 48
    readings.sm.setValue(85.0)  # flagged by server range check? 85 > 60 -> invalid
    readings.when.setDateTime(readings.when.dateTime().addSecs(60))
    readings.add_reading()
    assert "FLAGGED" in window.statusBar().currentMessage()

    window.page(DashboardPage).request_recommendation()
    history = window.page(HistoryPage)
    window.nav.setCurrentRow(window.pages.index(history))
    assert history.recs.rowCount() == 1
    history.recs.selectRow(0)
    assert history.reasons.count() >= 1
    history.add_event()
    assert history.events.rowCount() == 1

    analytics = window.page(AnalyticsPage)
    window.nav.setCurrentRow(window.pages.index(analytics))
    assert analytics.table.rowCount() == 14
    assert analytics.tiles.tiles["baseline"].value.text().endswith(" L")
    assert "SIMULATED" in analytics.forecast.text()

    model = window.page(ModelPage)
    window.nav.setCurrentRow(window.pages.index(model))
    assert "rule-based" in model.model_chip.text()

    calc = window.page(CalculatorPage)
    calc.sm.setValue(12.0)
    calc.calculate()
    assert calc.chip.text() in {"IRRIGATE", "NO IRRIGATION NEEDED"} and calc.steps.count() >= 5


def test_csv_import_dialog_converts_units_and_uploads(window, api, tmp_path):
    seed(api)
    window.start()
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    path = tmp_path / "logger.csv"
    lines = ["Time;Moisture;Temp;RH;Rain"]
    for i in range(10):
        ts = (now - timedelta(days=5, hours=9 - i)).strftime("%d/%m/%Y %H:%M")
        lines.append(f"{ts};0.21;86;55;0")
    path.write_text("\n".join(lines))
    dialog = ImportDialog(window, window.current_field)
    dialog.load_file(path)
    assert dialog.combos["soil_moisture"].currentText() == "Moisture"  # alias auto-detected
    assert dialog.combos["timestamp"].currentText() == "Time"
    dialog.ts_format.setText("%d/%m/%Y %H:%M")
    dialog.sm_unit.setCurrentText("fraction")
    dialog.temp_unit.setCurrentText("fahrenheit")
    dialog.start_import()
    assert "10 stored" in dialog.status.text()
    stored = [r for r in api.list_readings(window.current_field["id"], 500) if r["device_id"] == "csv-import"]
    assert len(stored) == 10
    assert stored[0]["soil_moisture"] == pytest.approx(21.0) and stored[0]["temperature_c"] == pytest.approx(30.0)


def test_import_helpers():
    assert guess_column("rainfall_mm", ["Date", "precipitation"]) == "precipitation"
    assert guess_column("humidity_pct", ["x"]) == "(not in file)"
    import pandas as pd

    df = pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01 05:00")], "soil_moisture": [float("nan")],
                       "temperature_c": [20.0], "humidity_pct": [50.0], "rainfall_mm": [float("nan")]})
    payload = readings_payload(df, 3)
    assert payload[0]["soil_moisture"] is None and payload[0]["rainfall_mm"] == 0.0
    assert payload[0]["timestamp"] == "2026-01-01T05:00:00"


def test_threaded_task_runner_delivers_on_gui_thread(qapp):
    runner = TaskRunner()
    results, errors = [], []
    runner.run(lambda x: x * 2, 21, on_done=results.append)
    runner.run(lambda: 1 / 0, on_error=errors.append)
    runner.wait()
    deadline = datetime.now() + timedelta(seconds=5)
    while (not results or not errors) and datetime.now() < deadline:
        qapp.processEvents()
    assert results == [42]
    assert "ZeroDivisionError" in errors[0]
    assert not runner.busy
