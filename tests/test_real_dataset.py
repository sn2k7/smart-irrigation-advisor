"""Real-dataset workflow, CLI defaults and the Real Dataset Studio GUI (new files)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from irrigation_advisor.config import PROJECT_ROOT
from irrigation_advisor.exceptions import DataValidationError
from irrigation_advisor.real_dataset_workflow import check_real_dataset, train_real_dataset

REAL_TEMPLATE = PROJECT_ROOT / "config" / "real_dataset.json"


def write_real_like(tmp_path: Path, small_dataset: pd.DataFrame, fields: int = 8) -> Path:
    """A stand-in 'real' export: own column names, °F, fraction moisture."""
    keep = sorted(small_dataset["field_id"].unique())[:fields]
    src = small_dataset[small_dataset["field_id"].isin(keep)]
    out = pd.DataFrame({
        "Time": pd.to_datetime(src["timestamp"]).dt.strftime("%Y-%m-%d %H:%M"),
        "Plot": src["field_id"], "VWC": src["soil_moisture"] / 100, "TempF": src["temperature_c"] * 9 / 5 + 32,
        "RH": src["humidity_pct"], "Rain": src["rainfall_mm"], "Crop": src["crop_type"], "Soil": src["soil_type"],
        "Sown": src["planting_date"],
    })
    path = tmp_path / "real.csv"
    out.to_csv(path, index=False)
    return path


def write_config(tmp_path: Path, data_path: Path) -> Path:
    cfg = json.loads(REAL_TEMPLATE.read_text())
    cfg["path"] = str(data_path)
    cfg["column_map"] = {"timestamp": "Time", "field_id": "Plot", "soil_moisture": "VWC", "temperature_c": "TempF",
                         "humidity_pct": "RH", "rainfall_mm": "Rain", "crop_type": "Crop", "soil_type": "Soil",
                         "planting_date": "Sown"}
    cfg["units"] = {"soil_moisture": "fraction", "temperature": "fahrenheit", "rainfall": "mm",
                    "rainfall_mode": "interval"}
    cfg["fields"] = {"defaults": {"irrigation_method": "drip", "latitude": 18.5}, "per_field": {}}
    path = tmp_path / "real_dataset.json"
    path.write_text(json.dumps(cfg))
    return path


def test_shipped_config_is_all_placeholders():
    text = REAL_TEMPLATE.read_text()
    assert text.count("PLACEHOLDER(real-dataset)") >= 8
    with pytest.raises(DataValidationError, match="path"):
        check_real_dataset(REAL_TEMPLATE)


def test_unfilled_field_details_are_named(tmp_path, small_dataset):
    cfg = json.loads(write_config(tmp_path, write_real_like(tmp_path, small_dataset)).read_text())
    cfg["fields"]["defaults"]["crop_type"] = "<crop>"
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(DataValidationError, match="fields.defaults.crop_type"):
        check_real_dataset(path)


def test_too_little_data_blocks_training(tmp_path, small_dataset):
    data = write_real_like(tmp_path, small_dataset.groupby("field_id").head(48), fields=1)
    check = check_real_dataset(write_config(tmp_path, data))
    assert not check.ok and any("rows" in p for p in check.problems)
    with pytest.raises(DataValidationError, match="not ready"):
        train_real_dataset(write_config(tmp_path, data), fast=True, model_path=None, reports_dir=tmp_path / "r")


def test_check_and_train_on_real_like_file(tmp_path, small_dataset):
    cfg = write_config(tmp_path, write_real_like(tmp_path, small_dataset))
    check = check_real_dataset(cfg)
    assert check.ok, check.problems
    assert check.conversion["fields"] == 8
    assert 10 < check.data["temperature_c"].median() < 40  # °F converted (85 °C sensor errors are injected)
    summary = train_real_dataset(cfg, fast=True, model_path=tmp_path / "m.joblib", reports_dir=tmp_path / "rep",
                                 stride_hours=3)
    assert summary["selected"]["classifier"]
    assert {"rule_based_fao56", "random_forest"} <= set(summary["classification"])
    assert (tmp_path / "rep" / "evaluation_report.md").exists()
    assert "user-supplied" in summary["disclaimer"]


# --------------------------------------------------------------------------- GUI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_real_dataset_studio_configure_check_train(tmp_path, small_dataset):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from irrigation_gui.tasks import TaskRunner
    from real_dataset_gui import NONE, RealDatasetStudio

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    data = write_real_like(tmp_path, small_dataset)
    cfg_path = tmp_path / "studio.json"
    cfg_path.write_text(REAL_TEMPLATE.read_text())  # start from the placeholder template

    win = RealDatasetStudio(cfg_path, tasks=TaskRunner(synchronous=True))
    tab = win.config_tab
    assert tab.path_edit.text() == ""  # placeholders shown as empty
    assert tab.combos["timestamp"].currentText() == NONE
    assert tab.save() is False  # still incomplete
    assert "Still to fill in" in tab.status.text()

    tab.path_edit.setText(str(data))
    tab.read_header()
    assert "VWC" in tab.file_info.text()
    for key, col in {"timestamp": "Time", "field_id": "Plot", "soil_moisture": "VWC", "temperature_c": "TempF",
                     "humidity_pct": "RH", "rainfall_mm": "Rain", "crop_type": "Crop", "soil_type": "Soil",
                     "planting_date": "Sown"}.items():
        tab.combos[key].setCurrentText(col)
    tab.sm_unit.setCurrentText("fraction")
    tab.temp_unit.setCurrentText("fahrenheit")
    tab.method.setCurrentText("drip")
    assert tab.save() is True
    saved = json.loads(cfg_path.read_text())
    assert saved["column_map"]["soil_moisture"] == "VWC" and "_README" in saved  # comments preserved

    win.check_tab.run_check()
    assert win.check_tab.verdict.text() == "READY FOR TRAINING"
    assert win.check_tab.preview.rowCount() == 20
    assert win.train_tab.train_btn.isEnabled()

    win.train_tab.model_out.setText(str(tmp_path / "studio_model.joblib"))
    win.train_tab.reports_out.setText(str(tmp_path / "studio_reports"))
    win.train_tab.reload.setChecked(False)
    win.train_tab.run_training()
    assert win.train_tab.cls_table.rowCount() == 4
    assert "Deployed decision model" in win.train_tab.selected.text()
    assert (tmp_path / "studio_model.joblib").exists() or "rule_based_fao56" in win.train_tab.selected.text()
    win.deleteLater()
    app.processEvents()


def test_studio_shows_config_errors(tmp_path):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from irrigation_gui.tasks import TaskRunner
    from real_dataset_gui import RealDatasetStudio

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text(REAL_TEMPLATE.read_text())
    win = RealDatasetStudio(cfg_path, tasks=TaskRunner(synchronous=True))
    win.check_tab.run_check()
    assert win.check_tab.verdict.text() == "CONFIG / FILE PROBLEM"
    assert not win.train_tab.train_btn.isEnabled()
    win.deleteLater()
    app.processEvents()
