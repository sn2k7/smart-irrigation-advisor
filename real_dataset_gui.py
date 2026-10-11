"""Real Dataset Studio - standalone PySide6 app for YOUR REAL dataset (new file).

Run from the project root (virtual environment active):

    python real_dataset_gui.py
    python real_dataset_gui.py --config config/real_dataset.json --api-url http://127.0.0.1:8000

What it does
------------
1. **Configure**  - pick your data file, map its columns, units and field details
                    in a form; saves ``config/real_dataset.json`` (comments kept).
2. **Check**      - converts the file with the project's dataset adapter and shows
                    readiness, problems, warnings, quality numbers, a preview and a
                    soil-moisture chart.
3. **Train**      - trains + evaluates the models on your data in the background
                    and shows the metrics; optionally tells a running API to reload
                    the new model.

It runs on the machine that holds the data and the project code (training is
local). Search for ``PLACEHOLDER(real-dataset)`` to adapt defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from irrigation_advisor.agronomy.crops import CROPS, IRRIGATION_METHODS, SOILS  # noqa: E402
from irrigation_advisor.data.labels import TARGET_CLASS, TARGET_REG  # noqa: E402
from irrigation_advisor.real_dataset_workflow import (  # noqa: E402
    DEFAULT_REAL_CONFIG,
    DEFAULT_REAL_MODEL,
    DEFAULT_REAL_REPORTS,
    CheckResult,
    check_real_dataset,
    train_real_dataset,
)
from irrigation_gui.api_client import DEFAULT_URL, ApiClient  # noqa: E402
from irrigation_gui.charts import ChartCanvas  # noqa: E402
from irrigation_gui.tasks import TaskRunner  # noqa: E402
from irrigation_gui.theme import STYLE_SHEET  # noqa: E402
from irrigation_gui.widgets import Card, Chip, DataTable, MessageList, TileRow, fmt, page_header  # noqa: E402

NONE = "(not in file)"
# PLACEHOLDER(real-dataset): internal columns offered in the mapping form, with their labels.
MAPPING_ROWS: list[tuple[str, str]] = [
    ("timestamp", "Date / time *"),
    ("soil_moisture", "Soil moisture *"),
    ("temperature_c", "Air temperature"),
    ("humidity_pct", "Relative humidity"),
    ("rainfall_mm", "Rainfall"),
    ("field_id", "Field / plot / node id"),
    ("crop_type", "Crop (column)"),
    ("soil_type", "Soil (column)"),
    ("planting_date", "Planting date (column)"),
    (TARGET_CLASS, "Your label: irrigation needed (0/1)"),
    (TARGET_REG, "Your label: irrigation depth (mm)"),
    ("et0_mm", "ET0 per row (mm, e.g. Open-Meteo)"),
    ("wind_speed_ms", "Wind speed (m/s)"),
    ("forecast_rain_24h_mm", "Rain forecast next 24 h (mm)"),
    ("irrigation_method", "Irrigation method (column)"),
    ("latitude", "Latitude (column)"),
]
PLACEHOLDER_TEXT = "<"


def _scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setWidget(widget)
    return area


def _clean(value: Any) -> Any:
    """Template placeholders ('<...>') are shown as empty fields."""
    return "" if isinstance(value, str) and value.startswith(PLACEHOLDER_TEXT) else value


class ConfigTab(QWidget):
    """Form editor for the dataset config JSON (keeps '_' comment keys)."""

    def __init__(self, app: "RealDatasetStudio") -> None:
        super().__init__()
        self.app = app
        self.raw: dict[str, Any] = {}
        self.columns: list[str] = []
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.addWidget(page_header(
            "1 · Configure your real dataset",
            "Choose the data file, then tell the app which column is which and what units they use. "
            "Values you leave empty are kept as PLACEHOLDER and block training until filled in."))

        file_card = Card("Data file")
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("PLACEHOLDER(real-dataset): e.g. data/real/real_dataset.csv")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse_file)
        self.delimiter = QComboBox()
        self.delimiter.addItems([",", ";", "\\t"])
        row.addWidget(self.path_edit, 1)
        row.addWidget(QLabel("Delimiter"))
        row.addWidget(self.delimiter)
        row.addWidget(browse)
        file_card.body.addLayout(row)
        self.file_info = QLabel("")
        self.file_info.setObjectName("subtitle")
        file_card.body.addWidget(self.file_info)
        layout.addWidget(file_card)

        map_card = Card("Column mapping (your header → meaning)")
        map_form = QFormLayout()
        self.combos: dict[str, QComboBox] = {}
        for key, label in MAPPING_ROWS:
            combo = QComboBox()
            combo.setEditable(True)  # allow typing a header before the file exists
            self.combos[key] = combo
            map_form.addRow(label, combo)
        map_card.body.addLayout(map_form)
        layout.addWidget(map_card)

        unit_card = Card("Units and time")
        unit_form = QFormLayout()
        self.sm_unit = self._combo(["percent", "fraction", "raw"])
        self.temp_unit = self._combo(["celsius", "fahrenheit", "kelvin"])
        self.rain_unit = self._combo(["mm", "inch"])
        self.rain_mode = self._combo(["interval", "cumulative"])
        self.ts_format = QLineEdit()
        self.ts_format.setPlaceholderText("auto-detect, or e.g. %d/%m/%Y %H:%M")
        self.dayfirst = QCheckBox("Day first in dates (31/12/2024)")
        self.offset = QDoubleSpinBox()
        self.offset.setRange(-14, 14)
        self.offset.setDecimals(2)
        self.offset.setSuffix(" h added to timestamps")
        self.raw_dry = self._spin(0, 100000, 3300, 0)
        self.raw_wet = self._spin(0, 100000, 1350, 0)
        self.raw_dry_pct = self._spin(0, 100, 3, 1)
        self.raw_wet_pct = self._spin(0, 100, 45, 1)
        for label, w in (("Soil moisture unit", self.sm_unit), ("Temperature unit", self.temp_unit),
                         ("Rainfall unit", self.rain_unit), ("Rainfall values are", self.rain_mode),
                         ("Timestamp format", self.ts_format), ("", self.dayfirst), ("Time offset", self.offset),
                         ("Raw sensor: dry reading", self.raw_dry), ("Raw sensor: wet reading", self.raw_wet),
                         ("Raw sensor: dry = % VWC", self.raw_dry_pct), ("Raw sensor: wet = % VWC", self.raw_wet_pct)):
            unit_form.addRow(label, w)
        unit_card.body.addLayout(unit_form)
        layout.addWidget(unit_card)

        field_card = Card("Field details (used when not a column in your file)")
        field_form = QFormLayout()
        self.crop = self._combo(["", *sorted(CROPS)])
        self.soil = self._combo(["", *sorted(SOILS)])
        self.method = self._combo(["", *sorted(IRRIGATION_METHODS)])
        self.planting = QLineEdit()
        self.planting.setPlaceholderText("YYYY-MM-DD")
        self.latitude = self._spin(-90, 90, 18.5, 4)
        self.labels_mode = self._combo(["derive", "columns"])
        for label, w in (("Crop", self.crop), ("Soil", self.soil), ("Irrigation method", self.method),
                         ("Planting date", self.planting), ("Latitude", self.latitude),
                         ("Labels", self.labels_mode)):
            field_form.addRow(label, w)
        field_card.body.addLayout(field_form)
        layout.addWidget(field_card)

        buttons = QHBoxLayout()
        self.save_btn = QPushButton("Save config")
        self.save_btn.setObjectName("primary")
        self.save_btn.clicked.connect(self.save)
        open_btn = QPushButton("Open JSON in editor")
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.app.config_path))))
        buttons.addWidget(open_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.save_btn)
        layout.addLayout(buttons)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scroll(inner))

    @staticmethod
    def _combo(items: list[str]) -> QComboBox:
        combo = QComboBox()
        combo.addItems(items)
        return combo

    @staticmethod
    def _spin(lo: float, hi: float, value: float, decimals: int) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setDecimals(decimals)
        box.setValue(value)
        return box

    # ------------------------------------------------------------------ load / save
    def load(self, path: Path) -> None:
        if path.exists():
            self.raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            template = PROJECT_ROOT / "config" / "dataset_config.example.json"
            self.raw = json.loads(template.read_text(encoding="utf-8"))
        r = self.raw
        self.path_edit.setText(_clean(r.get("path", "")))
        delim = r.get("delimiter", ",")
        self.delimiter.setCurrentText("\\t" if delim == "\t" else delim)
        self.read_header()
        cmap = r.get("column_map", {})
        for key, combo in self.combos.items():
            value = _clean(cmap.get(key, ""))
            combo.setCurrentText(value if value else NONE)
        units = r.get("units", {})
        self.sm_unit.setCurrentText(units.get("soil_moisture", "percent"))
        self.temp_unit.setCurrentText(units.get("temperature", "celsius"))
        self.rain_unit.setCurrentText(units.get("rainfall", "mm"))
        self.rain_mode.setCurrentText(units.get("rainfall_mode", "interval"))
        self.ts_format.setText(r.get("timestamp_format") or "")
        self.dayfirst.setChecked(bool(r.get("dayfirst", False)))
        self.offset.setValue(float(r.get("timestamp_offset_hours", 0.0)))
        cal = r.get("soil_raw_calibration", {})
        self.raw_dry.setValue(cal.get("dry_raw", 3300))
        self.raw_wet.setValue(cal.get("wet_raw", 1350))
        self.raw_dry_pct.setValue(cal.get("dry_pct", 3.0))
        self.raw_wet_pct.setValue(cal.get("wet_pct", 45.0))
        defaults = r.get("fields", {}).get("defaults", {})
        self.crop.setCurrentText(_clean(defaults.get("crop_type", "")))
        self.soil.setCurrentText(_clean(defaults.get("soil_type", "")))
        self.method.setCurrentText(_clean(defaults.get("irrigation_method", "")))
        self.planting.setText(_clean(defaults.get("planting_date", "")))
        self.latitude.setValue(float(defaults.get("latitude", 18.5) or 18.5))
        self.labels_mode.setCurrentText(r.get("labels", {}).get("mode", "derive"))
        self.status.setText(f"Loaded {path}" if path.exists() else f"New config from template - will be saved to {path}")

    def _resolve(self, text: str) -> Path:
        p = Path(text)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def browse_file(self) -> None:
        start = str(PROJECT_ROOT / "data" / "real")
        path, _ = QFileDialog.getOpenFileName(self, "Your real dataset", start, "Data (*.csv *.txt *.tsv);;All (*)")
        if path:
            try:
                rel = Path(path).resolve().relative_to(PROJECT_ROOT)
                self.path_edit.setText(rel.as_posix())
            except ValueError:
                self.path_edit.setText(path)
            self.read_header()

    def _delimiter(self) -> str:
        return "\t" if self.delimiter.currentText() == "\\t" else self.delimiter.currentText()

    def read_header(self) -> None:
        text = self.path_edit.text().strip()
        path = self._resolve(text) if text else None
        if not path or not path.exists():
            self.file_info.setText("File not found yet - put it in data/real/ and press Browse")
            self.columns = []
        else:
            try:
                sample = pd.read_csv(path, sep=self._delimiter(), nrows=5)
            except Exception as exc:  # noqa: BLE001
                self.file_info.setText(f"Cannot read file: {exc}")
                return
            self.columns = [str(c) for c in sample.columns]
            self.file_info.setText(f"{len(self.columns)} columns: {', '.join(self.columns)}")
        for combo in self.combos.values():
            current = combo.currentText()
            combo.clear()
            combo.addItems([NONE, *self.columns])
            combo.setCurrentText(current or NONE)

    def to_config(self) -> dict[str, Any]:
        """Merge the form into the raw JSON (comments and unknown keys preserved)."""
        r = dict(self.raw)
        # PLACEHOLDER(real-dataset): empty required values are written back as placeholders.
        r["path"] = self.path_edit.text().strip() or "<data/real/real_dataset.csv>"
        r["delimiter"] = self._delimiter()
        cmap = {}
        for key, combo in self.combos.items():
            value = combo.currentText().strip()
            if value and value != NONE:
                cmap[key] = value
        for required in ("timestamp", "soil_moisture"):
            cmap.setdefault(required, f"<your {required} column>")
        r["column_map"] = cmap
        r["units"] = {"soil_moisture": self.sm_unit.currentText(), "temperature": self.temp_unit.currentText(),
                      "rainfall": self.rain_unit.currentText(), "rainfall_mode": self.rain_mode.currentText()}
        r["timestamp_format"] = self.ts_format.text().strip() or None
        r["dayfirst"] = self.dayfirst.isChecked()
        r["timestamp_offset_hours"] = self.offset.value()
        r["soil_raw_calibration"] = {"dry_raw": self.raw_dry.value(), "wet_raw": self.raw_wet.value(),
                                     "dry_pct": self.raw_dry_pct.value(), "wet_pct": self.raw_wet_pct.value()}
        fields = dict(r.get("fields", {}))
        defaults = {
            "crop_type": self.crop.currentText() or "<crop>",
            "soil_type": self.soil.currentText() or "<soil>",
            "irrigation_method": self.method.currentText() or "<drip|sprinkler|furrow>",
            "latitude": self.latitude.value(),
        }
        if self.planting.text().strip():
            defaults["planting_date"] = self.planting.text().strip()
        # Values supplied by mapped columns don't need defaults.
        for key in ("crop_type", "soil_type", "irrigation_method"):
            if key in cmap and defaults[key].startswith(PLACEHOLDER_TEXT):
                del defaults[key]
        fields["defaults"] = defaults
        fields.setdefault("per_field", {})
        fields["per_field"] = {k: v for k, v in fields["per_field"].items() if not k.startswith(PLACEHOLDER_TEXT)}
        r["fields"] = fields
        r["labels"] = {"mode": self.labels_mode.currentText()}
        r.setdefault("name", "real_dataset")
        return r

    def save(self) -> bool:
        data = self.to_config()
        path = self.app.config_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.raw = data
        left = [k for k, v in data["column_map"].items() if str(v).startswith(PLACEHOLDER_TEXT)]
        left += [f"fields.{k}" for k, v in data["fields"]["defaults"].items() if str(v).startswith(PLACEHOLDER_TEXT)]
        if str(data["path"]).startswith(PLACEHOLDER_TEXT):
            left.insert(0, "path")
        self.status.setText(f"Saved {path}." + (f" Still to fill in: {', '.join(left)}" if left else
                                                " Next: tab 2 'Check'."))
        return not left


class CheckTab(QWidget):
    """Readiness check: problems, warnings, numbers, preview, chart."""

    def __init__(self, app: "RealDatasetStudio") -> None:
        super().__init__()
        self.app = app
        self.result: CheckResult | None = None
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.addWidget(page_header("2 · Check the data",
                                     "Converts your file exactly as training will and reports whether it is usable."))
        row = QHBoxLayout()
        self.run_btn = QPushButton("Save config and check")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.run_check)
        self.verdict = Chip("not checked yet", "neutral")
        row.addWidget(self.verdict)
        row.addStretch(1)
        row.addWidget(self.run_btn)
        layout.addLayout(row)
        self.tiles = TileRow([("rows", "Rows"), ("fields", "Fields"), ("span", "Time span"),
                              ("interval", "Median interval"), ("invalid", "Soil readings flagged"),
                              ("missing", "Hours without soil data")])
        layout.addWidget(self.tiles)
        grid = QGridLayout()
        prob_card = Card("Problems (must fix before training)")
        self.problems = MessageList("critical")
        prob_card.body.addWidget(self.problems)
        warn_card = Card("Warnings")
        self.warnings = MessageList("warning")
        warn_card.body.addWidget(self.warnings)
        grid.addWidget(prob_card, 0, 0)
        grid.addWidget(warn_card, 0, 1)
        holder = QWidget()
        holder.setLayout(grid)
        layout.addWidget(holder)
        chart_card = Card("Soil moisture after conversion (first field, % VWC)")
        self.chart = ChartCanvas(2.4)
        chart_card.body.addWidget(self.chart)
        layout.addWidget(chart_card)
        prev_card = Card("Preview in the internal schema")
        self.preview = DataTable([])
        self.preview.setMinimumHeight(240)
        prev_card.body.addWidget(self.preview)
        layout.addWidget(prev_card)
        layout.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scroll(inner))

    def run_check(self) -> None:
        self.app.config_tab.save()
        self.run_btn.setEnabled(False)
        self.verdict.set("checking…", "neutral")
        self.app.tasks.run(check_real_dataset, self.app.config_path, on_done=self.show_result, on_error=self.show_error)

    def show_error(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.result = None
        self.verdict.set("CONFIG / FILE PROBLEM", "critical")
        self.problems.set_items([message])
        self.warnings.set_items([])
        self.tiles.clear()
        self.app.train_tab.update_state()

    def show_result(self, result: CheckResult) -> None:
        self.run_btn.setEnabled(True)
        self.result = result
        c, q = result.conversion, result.quality
        self.verdict.set("READY FOR TRAINING" if result.ok else "NOT READY", "good" if result.ok else "critical")
        start, end = (pd.Timestamp(t) for t in c["time_range"])
        self.tiles.set("rows", f"{c['rows_after_conversion']:,}")
        self.tiles.set("fields", str(c["fields"]))
        self.tiles.set("span", f"{(end - start).total_seconds() / 86400:.1f} days")
        self.tiles.set("interval", fmt(c.get("median_interval_minutes"), 0, " min"))
        self.tiles.set("invalid", f"{q.get('sm_invalid_pct', 0)} %")
        self.tiles.set("missing", f"{q.get('sm_still_missing_pct', 0)} %")
        self.problems.set_items(result.problems or ["None"])
        self.warnings.set_items(result.warnings or ["None"])
        preview = result.preview.copy()
        preview["timestamp"] = preview["timestamp"].astype(str)
        self.preview.columns = [(col, col) for col in preview.columns]
        self.preview.setColumnCount(len(preview.columns))
        self.preview.setHorizontalHeaderLabels(list(preview.columns))
        self.preview.fill(preview.to_dict("records"))
        data = result.data
        first = data[data["field_id"] == data["field_id"].iloc[0]].tail(24 * 30)
        x = [t.to_pydatetime() for t in first["timestamp"]]
        self.chart.plot_lines(x, {str(first["field_id"].iloc[0]): first["soil_moisture"].tolist()}, "", "% VWC")
        self.app.train_tab.update_state()


class TrainTab(QWidget):
    """Train on the checked data and show held-out metrics."""

    def __init__(self, app: "RealDatasetStudio") -> None:
        super().__init__()
        self.app = app
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.addWidget(page_header(
            "3 · Train and evaluate",
            "Uses the project's leakage-safe procedure (held-out fields, chronological test, rolling CV). "
            "The FAO-56 rule competes with the ML models; the better one on cross-validation is deployed."))
        form_card = Card("Options")
        form = QFormLayout()
        self.fast = QCheckBox("Quick run (fewer trees) - untick for the final model")
        self.fast.setChecked(True)
        # PLACEHOLDER(real-dataset): default output locations (see real_dataset_workflow.py).
        self.model_out = QLineEdit(str(DEFAULT_REAL_MODEL))
        self.reports_out = QLineEdit(str(DEFAULT_REAL_REPORTS))
        self.api_url = QLineEdit(app.api_url)
        self.reload = QCheckBox("Ask the running API to reload the model afterwards")
        self.reload.setChecked(True)
        for label, w in (("", self.fast), ("Save model to", self.model_out), ("Reports folder", self.reports_out),
                         ("API URL", self.api_url), ("", self.reload)):
            form.addRow(label, w)
        form_card.body.addLayout(form)
        row = QHBoxLayout()
        self.hint = QLabel("Run the check (tab 2) first.")
        self.hint.setObjectName("subtitle")
        self.train_btn = QPushButton("Train on my data")
        self.train_btn.setObjectName("primary")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self.run_training)
        row.addWidget(self.hint, 1)
        row.addWidget(self.train_btn)
        form_card.body.addLayout(row)
        layout.addWidget(form_card)

        res_card = Card("Results on held-out test data (later period, seen fields)")
        self.selected = QLabel("-")
        self.selected.setWordWrap(True)
        res_card.body.addWidget(self.selected)
        self.cls_table = DataTable([("model", "Decision model"), ("f1", "F1"), ("precision", "Precision"),
                                    ("recall", "Recall"), ("accuracy", "Accuracy"), ("n", "Rows")])
        self.cls_table.setMinimumHeight(170)
        res_card.body.addWidget(self.cls_table)
        self.reg_table = DataTable([("model", "Depth model"), ("mae_mm", "MAE mm"), ("rmse_mm", "RMSE mm"),
                                    ("n", "Rows")])
        self.reg_table.setMinimumHeight(170)
        res_card.body.addWidget(self.reg_table)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(140)
        res_card.body.addWidget(self.log)
        open_reports = QPushButton("Open reports folder")
        open_reports.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(Path(self.reports_out.text())))))
        res_card.body.addWidget(open_reports)
        layout.addWidget(res_card)
        layout.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scroll(inner))

    def update_state(self) -> None:
        result = self.app.check_tab.result
        ready = bool(result and result.ok)
        self.train_btn.setEnabled(ready)
        self.hint.setText("Ready." if ready else "Run the check (tab 2) and fix all problems first.")

    def _log(self, text: str) -> None:
        self.log.appendPlainText(f"[{datetime.now():%H:%M:%S}] {text}")

    def run_training(self) -> None:
        self.train_btn.setEnabled(False)
        self._log("Training started - this can take a few minutes …")
        self.app.tasks.run(
            train_real_dataset, self.app.config_path, self.fast.isChecked(), self.model_out.text().strip() or None,
            self.reports_out.text().strip(), on_done=self.show_summary, on_error=self.show_error,
        )

    def show_error(self, message: str) -> None:
        self.train_btn.setEnabled(True)
        self._log(f"FAILED: {message}")

    def show_summary(self, summary: dict) -> None:
        self.train_btn.setEnabled(True)
        sel = summary["selected"]
        self.selected.setText(f"<b>Deployed decision model:</b> {sel['classifier']} · "
                              f"<b>depth model:</b> {sel['regressor']}<br>{sel['notes']}<br>"
                              f"<i>{summary['disclaimer']}</i>")
        self.cls_table.fill([{"model": k, **v} for k, v in summary["classification"].items() if v],
                            {k: (lambda v: fmt(v, 3)) for k in ("f1", "precision", "recall", "accuracy")})
        self.reg_table.fill([{"model": k, **v} for k, v in summary["regression"].items() if v],
                            {k: (lambda v: fmt(v, 2)) for k in ("mae_mm", "rmse_mm")})
        self._log(f"Done. Artifacts: {json.dumps(summary['artifacts'])}")
        if self.reload.isChecked() and "model" in summary["artifacts"]:
            client = ApiClient(self.api_url.text().strip())
            self.app.tasks.run(client.reload_model,
                               on_done=lambda info: self._log(f"API reloaded model: {info['message']}"),
                               on_error=lambda m: self._log(f"API not reloaded ({m}) - start it and press "
                                                            "'Reload model' in the main app"))


class RealDatasetStudio(QMainWindow):
    """Three tabs: configure -> check -> train."""

    def __init__(self, config_path: Path = DEFAULT_REAL_CONFIG, api_url: str = DEFAULT_URL,
                 tasks: TaskRunner | None = None) -> None:
        super().__init__()
        self.config_path = Path(config_path)
        self.api_url = api_url
        self.tasks = tasks or TaskRunner(parent=self)
        self.setWindowTitle(f"Real Dataset Studio - {self.config_path.name}")
        self.resize(1200, 880)
        self.tabs = QTabWidget()
        self.config_tab = ConfigTab(self)
        self.check_tab = CheckTab(self)
        self.train_tab = TrainTab(self)
        self.tabs.addTab(self.config_tab, "1 · Configure")
        self.tabs.addTab(self.check_tab, "2 · Check")
        self.tabs.addTab(self.train_tab, "3 · Train")
        self.setCentralWidget(self.tabs)
        self.config_tab.load(self.config_path)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.tasks.wait(2000)
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real Dataset Studio (PySide6)")
    # PLACEHOLDER(real-dataset): default config and API locations.
    parser.add_argument("--config", default=str(DEFAULT_REAL_CONFIG))
    parser.add_argument("--api-url", default=os.getenv("SIA_API_URL", DEFAULT_URL))
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyleSheet(STYLE_SHEET)
    window = RealDatasetStudio(Path(args.config), args.api_url)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

