"""Dashboard: current recommendation for the selected field."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from ..charts import ChartCanvas
from ..widgets import Card, Chip, MessageList, TileRow, fmt
from .base import Page

AUTO_REFRESH_MS = 5 * 60 * 1000

SOURCE_LABELS = {
    "ml_hybrid": ("ML + FAO-56 rules", "info"),
    "rule_based": ("FAO-56 rule", "info"),
    "fallback_estimated": ("Fallback: estimated moisture", "warning"),
    "fallback_conservative": ("Fallback: no sensor data", "critical"),
}
CONFIDENCE_KIND = {"high": "good", "medium": "warning", "low": "critical"}


class DashboardPage(Page):
    title = "Dashboard"
    subtitle = "Should this field be irrigated now? The decision, the numbers behind it and the sensor trend."

    def __init__(self, window) -> None:
        super().__init__(window)
        self.last_rec: dict | None = None

        # --- controls --------------------------------------------------------
        controls = Card()
        row = QHBoxLayout()
        self.use_forecast = QCheckBox("Use my rain forecast (mm, next 24 h):")
        self.forecast = QDoubleSpinBox()
        self.forecast.setRange(0, 500)
        self.forecast.setDecimals(1)
        self.forecast.setEnabled(False)
        self.use_forecast.toggled.connect(self.forecast.setEnabled)
        self.recommend_btn = QPushButton("Get recommendation")
        self.recommend_btn.setObjectName("primary")
        self.recommend_btn.clicked.connect(self.request_recommendation)
        self.irrigated_btn = QPushButton("Log irrigation as done")
        self.irrigated_btn.setEnabled(False)
        self.irrigated_btn.clicked.connect(self.log_irrigation)
        self.auto = QCheckBox("Auto-refresh every 5 min")
        self.auto.toggled.connect(self._toggle_auto)
        for w in (self.use_forecast, self.forecast):
            row.addWidget(w)
        row.addStretch(1)
        for w in (self.auto, self.irrigated_btn, self.recommend_btn):
            row.addWidget(w)
        controls.body.addLayout(row)
        self.layout_.addWidget(controls)

        # --- decision card ---------------------------------------------------
        decision = Card("Recommendation")
        head = QHBoxLayout()
        self.decision_chip = Chip("No recommendation yet", "neutral")
        self.source_chip = Chip("", "neutral")
        self.confidence_chip = Chip("", "neutral")
        head.addWidget(self.decision_chip)
        head.addWidget(self.source_chip)
        head.addWidget(self.confidence_chip)
        head.addStretch(1)
        decision.body.addLayout(head)
        self.summary = QLabel("Select a field and press 'Get recommendation'.")
        self.summary.setObjectName("summary")
        self.summary.setWordWrap(True)
        decision.body.addWidget(self.summary)
        self.meta = QLabel("")
        self.meta.setObjectName("subtitle")
        decision.body.addWidget(self.meta)
        self.tiles = TileRow([
            ("duration", "Pump run time"),
            ("volume", "Water volume"),
            ("depth", "Gross depth"),
            ("moisture", "Soil moisture"),
            ("depletion", "Depletion vs stress threshold (RAW)"),
            ("etc", "Crop water use (ETc)"),
            ("rain", "Rain forecast 24 h"),
        ])
        decision.body.addWidget(self.tiles)
        self.layout_.addWidget(decision)

        # --- reasons & warnings ----------------------------------------------
        grid = QGridLayout()
        reasons_card = Card("Why")
        self.reasons = MessageList()
        reasons_card.body.addWidget(self.reasons)
        warnings_card = Card("Warnings")
        self.warnings = MessageList("warning")
        warnings_card.body.addWidget(self.warnings)
        grid.addWidget(reasons_card, 0, 0)
        grid.addWidget(warnings_card, 0, 1)
        holder = QWidget()
        holder.setLayout(grid)
        self.layout_.addWidget(holder)

        chart_card = Card("Soil moisture - last 72 h (valid readings)")
        self.chart = ChartCanvas(2.4)
        chart_card.body.addWidget(self.chart)
        self.layout_.addWidget(chart_card)
        self.layout_.addStretch(1)

        self.timer = QTimer(self)
        self.timer.setInterval(AUTO_REFRESH_MS)
        self.timer.timeout.connect(self.request_recommendation)

    # ---------------------------------------------------------------------
    def _toggle_auto(self, on: bool) -> None:
        self.timer.start() if on else self.timer.stop()

    def on_field_changed(self, field: dict | None) -> None:
        self.last_rec = None
        self.irrigated_btn.setEnabled(False)
        self._show(None)
        if field is None:
            self.summary.setText("Create a field on the 'Fields' page first.")
            self.chart.show_message("No field selected")
            return
        self.refresh()

    def refresh(self) -> None:
        if not self.field:
            return
        fid = self.field["id"]
        self.run(self.api.list_recommendations, fid, 1, on_done=lambda recs: self._show(recs[0] if recs else None))
        self.run(self.api.list_readings, fid, 300, on_done=self._plot_readings)

    def request_recommendation(self) -> None:
        if not self.field:
            self.notify("Select a field first")
            return
        rain = self.forecast.value() if self.use_forecast.isChecked() else None
        self.recommend_btn.setEnabled(False)

        def done(rec: dict) -> None:
            self.recommend_btn.setEnabled(True)
            self._show(rec)
            self.notify(f"Recommendation stored (#{rec.get('id')})")
            self.run(self.api.list_readings, self.field["id"], 300, on_done=self._plot_readings)

        def failed(message: str) -> None:
            self.recommend_btn.setEnabled(True)
            self.window_.show_error(message)

        self.run(self.api.recommend, self.field["id"], rain, on_done=done, on_error=failed)

    def log_irrigation(self) -> None:
        rec = self.last_rec
        if not rec or not rec.get("irrigation_needed") or not self.field:
            return
        event = {
            "start_time": datetime.now().replace(microsecond=0).isoformat(),
            "duration_min": rec["duration_min"],
            "water_volume_l": rec["water_volume_l"],
            "recommendation_id": rec.get("id"),
            "source": "manual",
            "notes": "Logged from desktop dashboard",
        }
        self.run(self.api.add_event, self.field["id"], event,
                 on_done=lambda ev: self.notify(f"Irrigation logged: {ev['duration_min']:.0f} min, "
                                                f"{ev['water_volume_l']:,.0f} L"))
        self.irrigated_btn.setEnabled(False)

    # ---------------------------------------------------------------------
    def _show(self, rec: dict | None) -> None:
        self.last_rec = rec
        if rec is None:
            self.decision_chip.set("No recommendation yet", "neutral")
            self.source_chip.set("")
            self.confidence_chip.set("")
            self.summary.setText("Press 'Get recommendation'.")
            self.meta.setText("")
            self.tiles.clear()
            self.reasons.set_items([])
            self.warnings.set_items([])
            return
        needed = rec["irrigation_needed"]
        self.decision_chip.set("IRRIGATE" if needed else "NO IRRIGATION", "info" if needed else "good")
        label, kind = SOURCE_LABELS.get(rec["decision_source"], (rec["decision_source"], "neutral"))
        self.source_chip.set(label, kind)
        self.confidence_chip.set(f"{rec['confidence']} confidence", CONFIDENCE_KIND.get(rec["confidence"], "neutral"))
        self.summary.setText(rec["summary"])
        prob = rec.get("probability")
        self.meta.setText(
            f"Decision time {rec['decision_time'].replace('T', ' ')}"
            + (f"  ·  ML probability {prob:.2f}" if prob is not None else "")
            + f"  ·  next check in {rec['next_check_hours']} h  ·  forecast source: {rec['forecast_source']}"
        )
        ratio = rec.get("depletion_ratio")
        self.tiles.set("duration", fmt(rec["duration_min"], 0, " min"))
        self.tiles.set("volume", fmt(rec["water_volume_l"], 0, " L"))
        self.tiles.set("depth", fmt(rec["gross_depth_mm"], 1, " mm"))
        moisture = fmt(rec.get("soil_moisture_pct"), 1, " %")
        self.tiles.set("moisture", moisture + (" (est.)" if rec.get("soil_moisture_estimated") else ""))
        self.tiles.set("depletion", "-" if ratio is None else f"{100 * ratio:.0f} %")
        self.tiles.set("etc", fmt(rec.get("etc_24h_mm"), 1, " mm/day"))
        self.tiles.set("rain", fmt(rec.get("forecast_rain_24h_mm"), 1, " mm"))
        self.reasons.set_items(rec.get("reasons", []), "• ")
        self.warnings.set_items(rec.get("warnings", []) or ["None"], "⚠ " if rec.get("warnings") else "")
        self.irrigated_btn.setEnabled(bool(needed and rec.get("id")))

    def _plot_readings(self, readings: list[dict]) -> None:
        valid = [r for r in readings if r["is_valid"] and r["soil_moisture"] is not None]
        if not valid:
            self.chart.show_message("No valid soil readings yet")
            return
        valid.sort(key=lambda r: r["timestamp"])
        latest = datetime.fromisoformat(valid[-1]["timestamp"])
        valid = [r for r in valid if (latest - datetime.fromisoformat(r["timestamp"])).total_seconds() <= 72 * 3600]
        x = [datetime.fromisoformat(r["timestamp"]) for r in valid]
        invalid = [(datetime.fromisoformat(r["timestamp"]), r["soil_moisture"]) for r in readings
                   if not r["is_valid"] and r["soil_moisture"] is not None and 0 <= r["soil_moisture"] <= 60
                   and (latest - datetime.fromisoformat(r["timestamp"])).total_seconds() <= 72 * 3600]
        self.chart.plot_lines(x, {"soil moisture (% VWC)": [r["soil_moisture"] for r in valid]},
                              "", "% VWC", markers={"flagged reading": invalid} if invalid else None)
