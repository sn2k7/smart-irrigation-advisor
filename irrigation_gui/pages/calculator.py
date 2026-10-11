"""Stateless FAO-56 calculator (works without sensors or stored data)."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QPushButton

from ..widgets import Card, Chip, MessageList, TileRow, fmt
from .base import Page


class CalculatorPage(Page):
    title = "Irrigation calculator"
    subtitle = "Enter one soil-moisture value and see the full FAO-56 calculation step by step."
    needs_field = False

    def __init__(self, window) -> None:
        super().__init__(window)
        card = Card("Inputs")
        form = QFormLayout()
        self.crop, self.soil, self.method = QComboBox(), QComboBox(), QComboBox()
        self.days = self._spin(0, 400, 60, 0, " days")
        self.sm = self._spin(0, 60, 18, 1, " % VWC")
        self.area = self._spin(1, 1_000_000, 1000, 0, " m²")
        self.flow = self._spin(0.1, 100_000, 60, 1, " L/min")
        self.et0 = self._spin(0, 15, 5, 1, " mm/day")
        self.rain = self._spin(0, 500, 0, 1, " mm")
        for label, w in (("Crop", self.crop), ("Soil", self.soil), ("Irrigation method", self.method),
                         ("Days since planting", self.days), ("Soil moisture", self.sm), ("Area", self.area),
                         ("Flow rate", self.flow), ("Reference ET0", self.et0), ("Rain forecast (24 h)", self.rain)):
            form.addRow(label, w)
        card.body.addLayout(form)
        calc = QPushButton("Calculate")
        calc.setObjectName("primary")
        calc.clicked.connect(self.calculate)
        card.body.addWidget(calc)
        self.layout_.addWidget(card)

        result = Card("Result")
        self.chip = Chip("", "neutral")
        result.body.addWidget(self.chip)
        self.tiles = TileRow([("duration", "Run time"), ("volume", "Water"), ("gross", "Gross depth"),
                              ("depletion", "Depletion / RAW"), ("taw", "TAW"), ("etc", "ETc")])
        result.body.addWidget(self.tiles)
        self.steps = MessageList()
        self.steps.setMinimumHeight(150)
        result.body.addWidget(self.steps)
        self.layout_.addWidget(result)
        self.layout_.addStretch(1)

    @staticmethod
    def _spin(lo, hi, value, decimals, suffix) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setDecimals(decimals)
        box.setValue(value)
        box.setSuffix(suffix)
        return box

    def set_reference(self, reference: dict) -> None:
        for combo, key, default in ((self.crop, "crops", "tomato"), (self.soil, "soils", "loam"),
                                    (self.method, "methods", "drip")):
            combo.clear()
            combo.addItems(sorted(reference.get(key, {})))
            combo.setCurrentText(default)

    def calculate(self) -> None:
        data = {
            "crop_type": self.crop.currentText(), "soil_type": self.soil.currentText(),
            "irrigation_method": self.method.currentText(), "days_since_planting": self.days.value(),
            "soil_moisture_pct": self.sm.value(), "area_m2": self.area.value(), "flow_rate_lpm": self.flow.value(),
            "et0_mm_day": self.et0.value(), "forecast_rain_24h_mm": self.rain.value(),
        }
        self.run(self.api.calculator, data, on_done=self._show)

    def _show(self, r: dict) -> None:
        self.chip.set("IRRIGATE" if r["irrigation_needed"] else "NO IRRIGATION NEEDED",
                      "info" if r["irrigation_needed"] else "good")
        self.tiles.set("duration", fmt(r["duration_min"], 0, " min"))
        self.tiles.set("volume", fmt(r["water_volume_l"], 0, " L"))
        self.tiles.set("gross", fmt(r["gross_depth_mm"], 1, " mm"))
        self.tiles.set("depletion", f"{100 * r['depletion_ratio']:.0f} %")
        self.tiles.set("taw", fmt(r["taw_mm"], 0, " mm"))
        self.tiles.set("etc", fmt(r["etc_mm_day"], 1, " mm/day"))
        self.steps.set_items(r["explanation"], "→ ")
