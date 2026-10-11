"""History: stored recommendations (audit trail) and the irrigation log."""

from __future__ import annotations

from PySide6.QtCore import QDateTime
from PySide6.QtWidgets import QDateTimeEdit, QDoubleSpinBox, QFormLayout, QLineEdit, QPushButton

from ..widgets import Card, DataTable, MessageList, fmt
from .base import Page


class HistoryPage(Page):
    title = "History"
    subtitle = "Every recommendation is stored with its reasons; compare what was recommended with what was applied."

    def __init__(self, window) -> None:
        super().__init__(window)
        recs = Card("Recommendations (newest first) - select one to see its reasons")
        self.recs = DataTable([
            ("id", "#"), ("decision_time", "Decision time"), ("irrigation_needed", "Irrigate"),
            ("duration_min", "Minutes"), ("water_volume_l", "Litres"), ("decision_source", "Source"),
            ("confidence", "Confidence"), ("summary", "Summary"),
        ])
        self.recs.setMinimumHeight(220)
        self.recs.itemSelectionChanged.connect(self._show_reasons)
        recs.body.addWidget(self.recs)
        self.reasons = MessageList()
        self.warnings = MessageList("warning")
        recs.body.addWidget(self.reasons)
        recs.body.addWidget(self.warnings)
        self.layout_.addWidget(recs)

        events = Card("Irrigation log (water actually applied)")
        self.events = DataTable([
            ("start_time", "Start"), ("duration_min", "Minutes"), ("water_volume_l", "Litres"),
            ("recommendation_id", "Rec #"), ("source", "Source"), ("notes", "Notes"),
        ])
        self.events.setMinimumHeight(160)
        events.body.addWidget(self.events)
        form = QFormLayout()
        self.start = QDateTimeEdit(QDateTime.currentDateTime())
        self.start.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.start.setCalendarPopup(True)
        self.minutes = QDoubleSpinBox()
        self.minutes.setRange(0, 1440)
        self.minutes.setValue(30)
        self.minutes.setSuffix(" min")
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("optional - e.g. 'pump failed after 20 min'")
        form.addRow("Started", self.start)
        form.addRow("Duration", self.minutes)
        form.addRow("Notes", self.notes)
        events.body.addLayout(form)
        add = QPushButton("Log irrigation (litres = minutes × field flow rate)")
        add.clicked.connect(self.add_event)
        events.body.addWidget(add)
        self.layout_.addWidget(events)
        self.layout_.addStretch(1)

    def on_field_changed(self, field: dict | None) -> None:
        if field is None:
            self.recs.fill([])
            self.events.fill([])
            return
        self.refresh()

    def refresh(self) -> None:
        if not self.field:
            return
        fid = self.field["id"]
        self.run(self.api.list_recommendations, fid, 200, on_done=self._fill_recs)
        self.run(self.api.list_events, fid, 200, on_done=lambda rows: self.events.fill(
            rows, {"start_time": lambda v: v.replace("T", " "), "water_volume_l": lambda v: fmt(v, 0)}))

    def _fill_recs(self, rows: list[dict]) -> None:
        self.recs.fill(
            rows,
            {"decision_time": lambda v: v.replace("T", " "), "irrigation_needed": lambda v: "YES" if v else "no",
             "duration_min": lambda v: fmt(v, 0), "water_volume_l": lambda v: fmt(v, 0)},
            highlight=lambda r: "warning" if r["decision_source"].startswith("fallback") else None,
        )
        self.reasons.set_items([])
        self.warnings.set_items([])

    def _show_reasons(self) -> None:
        row = self.recs.selected_row()
        if row:
            self.reasons.set_items(row.get("reasons", []), "• ")
            self.warnings.set_items(row.get("warnings", []), "⚠ ")

    def add_event(self) -> None:
        if not self.field:
            self.window_.show_error("Select a field first")
            return
        event = {
            "start_time": self.start.dateTime().toString("yyyy-MM-ddTHH:mm:00"),
            "duration_min": self.minutes.value(),
            "source": "manual",
            "notes": self.notes.text().strip() or None,
        }
        selected = self.recs.selected_row()
        if selected and selected.get("irrigation_needed"):
            event["recommendation_id"] = selected["id"]
        self.run(self.api.add_event, self.field["id"], event,
                 on_done=lambda e: (self.notify(f"Logged {e['water_volume_l']:,.0f} L"), self.refresh()))
