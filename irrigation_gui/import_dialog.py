"""CSV import dialog - maps YOUR columns/units onto the API schema.

It reuses the backend's custom-dataset adapter
(``irrigation_advisor.data.custom_dataset``), so the same mapping and unit
conversions apply whether you train on a file or upload it here. A saved
JSON dataset config (``config/*.json``) can be loaded to pre-fill the form.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from irrigation_advisor.data.custom_dataset import DatasetConfig, load_custom_dataset, load_dataset_config
from irrigation_advisor.data.loader import COLUMN_ALIASES

from .widgets import DataTable

NONE = "(not in file)"
MAPPED = (
    ("timestamp", "Timestamp *"),
    ("soil_moisture", "Soil moisture *"),
    ("temperature_c", "Air temperature"),
    ("humidity_pct", "Relative humidity"),
    ("rainfall_mm", "Rainfall"),
)
BATCH_SIZE = 1000


def guess_column(internal: str, columns: list[str]) -> str:
    """Best guess of the file column for an internal name (exact name or known alias)."""
    for col in columns:
        key = col.strip().lower().replace(" ", "_")
        if key == internal or COLUMN_ALIASES.get(key) == internal:
            return col
    return NONE


def readings_payload(df: pd.DataFrame, field_id: int) -> list[dict[str, Any]]:
    """Internal-schema frame -> list of API reading dicts (NaN -> null)."""
    def clean(value: Any) -> Any:
        return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)

    return [
        {
            "field_id": field_id,
            "timestamp": pd.Timestamp(row.timestamp).isoformat(timespec="seconds"),
            "soil_moisture": clean(row.soil_moisture),
            "temperature_c": clean(row.temperature_c),
            "humidity_pct": clean(row.humidity_pct),
            "rainfall_mm": clean(row.rainfall_mm) or 0.0,
            "device_id": "csv-import",
        }
        for row in df.itertuples(index=False)
    ]


class ImportDialog(QDialog):
    """Pick a file, map columns and units, convert, upload to the selected field."""

    imported = Signal(dict)

    def __init__(self, window, field: dict) -> None:
        super().__init__(window)
        self.window_ = window
        self.field = field
        self.path: Path | None = None
        self.setWindowTitle(f"Import readings into '{field['name']}'")
        self.resize(820, 620)
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        pick = QPushButton("Choose CSV…")
        pick.clicked.connect(self._pick_file)
        config_btn = QPushButton("Load dataset config (JSON)…")
        config_btn.clicked.connect(self._pick_config)
        top.addWidget(self.file_label, 1)
        top.addWidget(config_btn)
        top.addWidget(pick)
        layout.addLayout(top)

        self.preview = DataTable([])
        self.preview.setMaximumHeight(170)
        layout.addWidget(self.preview)

        form = QFormLayout()
        self.combos: dict[str, QComboBox] = {}
        for key, label in MAPPED:
            combo = QComboBox()
            self.combos[key] = combo
            form.addRow(label, combo)
        self.ts_format = QLineEdit()
        self.ts_format.setPlaceholderText("auto-detect, or e.g. %d/%m/%Y %H:%M")
        self.dayfirst = QCheckBox("Day comes first in dates (31/12/2026)")
        self.sm_unit = self._combo(["percent", "fraction"])
        self.temp_unit = self._combo(["celsius", "fahrenheit", "kelvin"])
        self.rain_unit = self._combo(["mm", "inch"])
        self.rain_mode = self._combo(["interval", "cumulative"])
        form.addRow("Timestamp format", self.ts_format)
        form.addRow("", self.dayfirst)
        form.addRow("Soil moisture unit", self.sm_unit)
        form.addRow("Temperature unit", self.temp_unit)
        form.addRow("Rainfall unit", self.rain_unit)
        form.addRow("Rainfall values are", self.rain_mode)
        layout.addLayout(form)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Close")
        cancel.clicked.connect(self.reject)
        self.import_btn = QPushButton("Convert and upload")
        self.import_btn.setObjectName("primary")
        self.import_btn.setEnabled(False)
        self.import_btn.clicked.connect(self.start_import)
        buttons.addWidget(cancel)
        buttons.addWidget(self.import_btn)
        layout.addLayout(buttons)

    @staticmethod
    def _combo(items: list[str]) -> QComboBox:
        combo = QComboBox()
        combo.addItems(items)
        return combo

    # ------------------------------------------------------------------ file / config
    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose CSV", "", "CSV files (*.csv *.txt *.tsv);;All files (*)")
        if path:
            self.load_file(Path(path))

    def _pick_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Dataset config", "config", "JSON (*.json)")
        if path:
            self.load_config(Path(path))

    def load_file(self, path: Path) -> None:
        try:
            sample = pd.read_csv(path, nrows=200, sep=None, engine="python")
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Cannot read file: {exc}")
            return
        self.path = path
        self.file_label.setText(str(path))
        columns = [str(c) for c in sample.columns]
        self.preview.columns = [(c, c) for c in columns]
        self.preview.setColumnCount(len(columns))
        self.preview.setHorizontalHeaderLabels(columns)
        self.preview.fill(sample.head(8).to_dict("records"))
        for key, combo in self.combos.items():
            combo.clear()
            combo.addItems([NONE, *columns])
            combo.setCurrentText(guess_column(key, columns))
        self.import_btn.setEnabled(True)
        self.status.setText("Check the column mapping and units, then press 'Convert and upload'.")

    def load_config(self, path: Path) -> None:
        try:
            cfg = load_dataset_config(path)
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Config problem: {exc}")
            return
        self.load_file(cfg.path)
        for key, combo in self.combos.items():
            if key in cfg.column_map:
                combo.setCurrentText(cfg.column_map[key])
        self.ts_format.setText(cfg.timestamp_format or "")
        self.dayfirst.setChecked(cfg.dayfirst)
        self.sm_unit.setCurrentText(cfg.soil_moisture_unit)
        self.temp_unit.setCurrentText(cfg.temperature_unit)
        self.rain_unit.setCurrentText(cfg.rainfall_unit)
        self.rain_mode.setCurrentText(cfg.rainfall_mode)
        self.status.setText(f"Mapping loaded from {path.name}")

    # ------------------------------------------------------------------ convert + upload
    def build_config(self) -> DatasetConfig:
        if self.path is None:
            raise ValueError("Choose a file first")
        column_map = {k: c.currentText() for k, c in self.combos.items() if c.currentText() not in ("", NONE)}
        for required in ("timestamp", "soil_moisture"):
            if required not in column_map:
                raise ValueError(f"Map a column for '{required}'")
        f = self.field
        return DatasetConfig(
            name=f"gui-import-{f['id']}",
            path=self.path,
            delimiter=self._delimiter(),
            column_map=column_map,
            timestamp_format=self.ts_format.text().strip() or None,
            dayfirst=self.dayfirst.isChecked(),
            soil_moisture_unit=self.sm_unit.currentText(),
            temperature_unit=self.temp_unit.currentText(),
            rainfall_unit=self.rain_unit.currentText(),
            rainfall_mode=self.rain_mode.currentText(),
            default_field_id=str(f["id"]),
            # All rows go to the selected field: its metadata satisfies the adapter.
            field_defaults={k: f[k] for k in ("crop_type", "soil_type", "irrigation_method", "planting_date",
                                              "latitude")},
        )

    def _delimiter(self) -> str:
        if self.path is None:
            return ","
        first = self.path.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
        line = first[0] if first else ""
        return max((",", ";", "\t"), key=line.count)

    def start_import(self) -> None:
        try:
            config = self.build_config()
            df, report = load_custom_dataset(config)
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Conversion failed: {exc}")
            return
        payload = readings_payload(df, int(self.field["id"]))
        self.status.setText(f"Uploading {len(payload):,} readings "
                            f"({report['time_range'][0]} → {report['time_range'][1]})…")
        self.import_btn.setEnabled(False)
        api = self.window_.api

        def upload() -> dict:
            totals = {"received": 0, "stored": 0, "duplicates": 0, "invalid": 0}
            for start in range(0, len(payload), BATCH_SIZE):
                result = api.add_readings_batch(payload[start:start + BATCH_SIZE])
                for key in totals:
                    totals[key] += result[key]
            return totals

        def done(totals: dict) -> None:
            self.status.setText(
                f"Done: {totals['stored']:,} stored, {totals['duplicates']:,} duplicates skipped, "
                f"{totals['invalid']:,} flagged invalid by the server."
            )
            self.import_btn.setEnabled(True)
            self.imported.emit(totals)

        def failed(message: str) -> None:
            self.status.setText(f"Upload failed: {message}")
            self.import_btn.setEnabled(True)

        self.window_.tasks.run(upload, on_done=done, on_error=failed)
