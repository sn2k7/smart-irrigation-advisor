"""Fields: create, edit, delete fields and register IoT devices."""

from __future__ import annotations

from PySide6.QtCore import QDate
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
)

from ..widgets import Card, DataTable, fmt
from .base import Page


class FieldsPage(Page):
    title = "Fields"
    subtitle = "Each field has a crop, soil, irrigation method and pump flow rate - they drive every calculation."
    needs_field = False

    def __init__(self, window) -> None:
        super().__init__(window)
        table_card = Card("Your fields")
        self.table = DataTable([
            ("id", "ID"), ("name", "Name"), ("crop_type", "Crop"), ("soil_type", "Soil"),
            ("irrigation_method", "Method"), ("planting_date", "Planted"), ("area_m2", "Area m²"),
            ("flow_rate_lpm", "Flow L/min"), ("max_irrigation_minutes", "Max min"),
        ])
        self.table.itemSelectionChanged.connect(self._load_selected)
        table_card.body.addWidget(self.table)
        self.layout_.addWidget(table_card)

        form_card = Card("Field details")
        form = QFormLayout()
        self.name = QLineEdit()
        self.crop = QComboBox()
        self.soil = QComboBox()
        self.method = QComboBox()
        self.planting = QDateEdit(QDate.currentDate().addDays(-30))
        self.planting.setCalendarPopup(True)
        self.planting.setDisplayFormat("yyyy-MM-dd")
        self.area = self._spin(1, 1_000_000, 2000, 0, " m²")
        self.flow = self._spin(0.1, 100_000, 120, 1, " L/min")
        self.lat = self._spin(-90, 90, 18.5, 4, "°")
        self.lon = self._spin(-180, 180, 73.85, 4, "°")
        self.max_min = self._spin(1, 1440, 180, 0, " min")
        for label, widget in (
            ("Name", self.name), ("Crop", self.crop), ("Soil", self.soil), ("Irrigation method", self.method),
            ("Planting date", self.planting), ("Area", self.area), ("Pump / valve flow", self.flow),
            ("Latitude", self.lat), ("Longitude", self.lon), ("Max run time per irrigation", self.max_min),
        ):
            form.addRow(label, widget)
        form_card.body.addLayout(form)
        buttons = QHBoxLayout()
        self.create_btn = QPushButton("Create new field")
        self.create_btn.setObjectName("primary")
        self.update_btn = QPushButton("Save changes to selected")
        self.delete_btn = QPushButton("Delete selected")
        self.delete_btn.setObjectName("danger")
        self.create_btn.clicked.connect(self.create_field)
        self.update_btn.clicked.connect(self.update_field)
        self.delete_btn.clicked.connect(self.delete_field)
        buttons.addWidget(self.create_btn)
        buttons.addWidget(self.update_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.delete_btn)
        form_card.body.addLayout(buttons)
        self.layout_.addWidget(form_card)

        device_card = Card("Register an IoT sensor node (ESP32) to the selected field")
        row = QHBoxLayout()
        self.device_id = QLineEdit()
        self.device_id.setPlaceholderText("device id, e.g. esp32-node-01")
        self.device_btn = QPushButton("Register device")
        self.device_btn.clicked.connect(self.register_device)
        row.addWidget(self.device_id)
        row.addWidget(self.device_btn)
        device_card.body.addLayout(row)
        self.devices = DataTable([("device_id", "Device"), ("field_id", "Field"), ("last_seen_at", "Last seen")])
        self.devices.setMaximumHeight(140)
        device_card.body.addWidget(self.devices)
        self.layout_.addWidget(device_card)
        self.layout_.addStretch(1)

    @staticmethod
    def _spin(lo: float, hi: float, value: float, decimals: int, suffix: str) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setDecimals(decimals)
        box.setValue(value)
        box.setSuffix(suffix)
        return box

    # ---------------------------------------------------------------------
    def set_reference(self, reference: dict) -> None:
        for combo, key in ((self.crop, "crops"), (self.soil, "soils"), (self.method, "methods")):
            current = combo.currentText()
            combo.clear()
            combo.addItems(sorted(reference.get(key, {})))
            if current:
                combo.setCurrentText(current)

    def refresh(self) -> None:
        self.table.fill(self.window_.fields, {"area_m2": lambda v: fmt(v, 0), "flow_rate_lpm": lambda v: fmt(v, 1),
                                              "max_irrigation_minutes": lambda v: fmt(v, 0)})
        self.run(self.api.list_devices, None, on_done=lambda rows: self.devices.fill(rows))

    def on_field_changed(self, field: dict | None) -> None:
        self.refresh()

    def form_data(self) -> dict:
        return {
            "name": self.name.text().strip(),
            "crop_type": self.crop.currentText(),
            "soil_type": self.soil.currentText(),
            "irrigation_method": self.method.currentText(),
            "planting_date": self.planting.date().toString("yyyy-MM-dd"),
            "area_m2": self.area.value(),
            "flow_rate_lpm": self.flow.value(),
            "latitude": self.lat.value(),
            "longitude": self.lon.value(),
            "max_irrigation_minutes": self.max_min.value(),
        }

    def _load_selected(self) -> None:
        row = self.table.selected_row()
        if not row:
            return
        self.name.setText(row["name"])
        self.crop.setCurrentText(row["crop_type"])
        self.soil.setCurrentText(row["soil_type"])
        self.method.setCurrentText(row["irrigation_method"])
        self.planting.setDate(QDate.fromString(row["planting_date"], "yyyy-MM-dd"))
        self.area.setValue(row["area_m2"])
        self.flow.setValue(row["flow_rate_lpm"])
        self.lat.setValue(row["latitude"])
        self.lon.setValue(row["longitude"] or 0.0)
        self.max_min.setValue(row["max_irrigation_minutes"])

    def create_field(self) -> None:
        data = self.form_data()
        if not data["name"]:
            self.window_.show_error("Please enter a field name")
            return
        self.run(self.api.create_field, data, on_done=lambda f: self.window_.reload_fields(select_id=f["id"]))

    def update_field(self) -> None:
        row = self.table.selected_row()
        if not row:
            self.window_.show_error("Select a field in the table first")
            return
        self.run(self.api.update_field, row["id"], self.form_data(),
                 on_done=lambda f: self.window_.reload_fields(select_id=f["id"]))

    def delete_field(self) -> None:
        row = self.table.selected_row()
        if not row:
            self.window_.show_error("Select a field in the table first")
            return
        if not self.window_.confirm(f"Delete field '{row['name']}' and ALL its readings and history?"):
            return
        self.run(self.api.delete_field, row["id"], on_done=lambda _: self.window_.reload_fields())

    def register_device(self) -> None:
        device = self.device_id.text().strip()
        if not device or not self.field:
            self.window_.show_error("Enter a device id and select a field (top bar)")
            return
        self.run(self.api.register_device, device, self.field["id"], "registered from desktop app",
                 on_done=lambda d: (self.notify(f"Device {d['device_id']} -> field {d['field_id']}"), self.refresh()))


__all__ = ["FieldsPage"]
