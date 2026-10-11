"""Readings: history table + chart, manual entry and CSV import."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QDateTime
from PySide6.QtWidgets import (
    QDateTimeEdit,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
)

from ..charts import ChartCanvas
from ..import_dialog import ImportDialog
from ..widgets import Card, DataTable, fmt
from .base import Page


class ReadingsPage(Page):
    title = "Sensor readings"
    subtitle = ("Every reading is validated by the server; flagged readings are kept for diagnostics "
                "but ignored by the decision engine.")

    def __init__(self, window) -> None:
        super().__init__(window)
        top = QHBoxLayout()
        self.limit = QSpinBox()
        self.limit.setRange(10, 10_000)
        self.limit.setValue(300)
        self.limit.setPrefix("show last ")
        self.limit.setSuffix(" readings")
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        self.import_btn = QPushButton("Import CSV / your dataset…")
        self.import_btn.clicked.connect(self.open_import)
        top.addWidget(self.limit)
        top.addWidget(refresh)
        top.addStretch(1)
        top.addWidget(self.import_btn)
        self.layout_.addLayout(top)

        chart_card = Card("Soil moisture and air temperature")
        self.moisture_chart = ChartCanvas(2.3)
        self.temp_chart = ChartCanvas(2.0)
        chart_card.body.addWidget(self.moisture_chart)
        chart_card.body.addWidget(self.temp_chart)
        self.layout_.addWidget(chart_card)

        table_card = Card("Readings (newest first)")
        self.table = DataTable([
            ("timestamp", "Time"), ("soil_moisture", "Soil %"), ("temperature_c", "Temp °C"),
            ("humidity_pct", "RH %"), ("rainfall_mm", "Rain mm"), ("battery_v", "Battery V"),
            ("device_id", "Device"), ("source", "Source"), ("is_valid", "Valid"), ("quality_flags", "Issues"),
        ])
        self.table.setMinimumHeight(260)
        table_card.body.addWidget(self.table)
        self.layout_.addWidget(table_card)

        form_card = Card("Add a reading manually")
        form = QFormLayout()
        self.when = QDateTimeEdit(QDateTime.currentDateTime())
        self.when.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.when.setCalendarPopup(True)
        self.sm = self._spin(0, 100, 22, " % VWC")
        self.temp = self._spin(-20, 60, 28, " °C")
        self.rh = self._spin(0, 100, 60, " %")
        self.rain = self._spin(0, 500, 0, " mm")
        for label, widget in (("Time", self.when), ("Soil moisture", self.sm), ("Air temperature", self.temp),
                              ("Relative humidity", self.rh), ("Rain since last reading", self.rain)):
            form.addRow(label, widget)
        form_card.body.addLayout(form)
        self.add_btn = QPushButton("Add reading")
        self.add_btn.setObjectName("primary")
        self.add_btn.clicked.connect(self.add_reading)
        form_card.body.addWidget(self.add_btn)
        self.layout_.addWidget(form_card)
        self.layout_.addStretch(1)

    @staticmethod
    def _spin(lo: float, hi: float, value: float, suffix: str) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setDecimals(1)
        box.setValue(value)
        box.setSuffix(suffix)
        return box

    def on_field_changed(self, field: dict | None) -> None:
        if field is None:
            self.table.fill([])
            self.moisture_chart.show_message("No field selected")
            self.temp_chart.show_message("")
            return
        self.refresh()

    def refresh(self) -> None:
        if self.field:
            self.run(self.api.list_readings, self.field["id"], self.limit.value(), on_done=self._show)

    def _show(self, rows: list[dict]) -> None:
        self.table.fill(
            rows,
            {"timestamp": lambda v: v.replace("T", " "), "quality_flags": lambda v: ", ".join(v) if v else "",
             "is_valid": lambda v: "yes" if v else "NO", "soil_moisture": lambda v: fmt(v, 1),
             "temperature_c": lambda v: fmt(v, 1), "humidity_pct": lambda v: fmt(v, 0),
             "rainfall_mm": lambda v: fmt(v, 1), "battery_v": lambda v: fmt(v, 2)},
            highlight=lambda r: None if r["is_valid"] else "critical",
        )
        ordered = sorted(rows, key=lambda r: r["timestamp"])
        x = [datetime.fromisoformat(r["timestamp"]) for r in ordered]
        valid_sm = [r["soil_moisture"] if r["is_valid"] else None for r in ordered]
        rain_points = [(datetime.fromisoformat(r["timestamp"]), r["soil_moisture"]) for r in ordered
                       if r["is_valid"] and (r["rainfall_mm"] or 0) >= 1 and r["soil_moisture"] is not None]
        self.moisture_chart.plot_lines(x, {"soil moisture": valid_sm}, "Soil moisture (valid readings)", "% VWC",
                                       markers={"rain ≥ 1 mm": rain_points} if rain_points else None)
        self.temp_chart.plot_lines(x, {"air temperature": [r["temperature_c"] for r in ordered]},
                                   "Air temperature", "°C")

    def add_reading(self) -> None:
        if not self.field:
            self.window_.show_error("Select a field first")
            return
        reading = {
            "field_id": self.field["id"],
            "timestamp": self.when.dateTime().toString("yyyy-MM-ddTHH:mm:00"),
            "soil_moisture": self.sm.value(),
            "temperature_c": self.temp.value(),
            "humidity_pct": self.rh.value(),
            "rainfall_mm": self.rain.value(),
            "device_id": "desktop-app",
        }

        def done(result: dict) -> None:
            issues = ", ".join(i["message"] for i in result["issues"]) or "no issues"
            state = "duplicate (ignored)" if result["duplicate"] else ("valid" if result["is_valid"] else "FLAGGED")
            self.notify(f"Reading stored: {state} - {issues}")
            self.refresh()

        self.run(self.api.add_reading, reading, on_done=done)

    def open_import(self) -> None:
        if not self.field:
            self.window_.show_error("Select a field first - imported readings are added to it")
            return
        dialog = ImportDialog(self.window_, self.field)
        dialog.imported.connect(lambda _: self.refresh())
        dialog.open()
