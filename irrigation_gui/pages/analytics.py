"""Analytics: daily history, water totals and the simulated forecast."""

from __future__ import annotations

from datetime import date, datetime

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSpinBox

from ..charts import ChartCanvas
from ..widgets import Card, DataTable, TileRow, fmt
from .base import Page


class AnalyticsPage(Page):
    title = "Analytics"
    subtitle = ("Daily soil moisture, rain and water. The fixed-schedule figure is a calculated reference, "
                "not a measurement - savings shown here are estimates.")

    def __init__(self, window) -> None:
        super().__init__(window)
        row = QHBoxLayout()
        self.days = QSpinBox()
        self.days.setRange(1, 366)
        self.days.setValue(14)
        self.days.setPrefix("last ")
        self.days.setSuffix(" days")
        go = QPushButton("Refresh")
        go.clicked.connect(self.refresh)
        row.addWidget(self.days)
        row.addWidget(go)
        row.addStretch(1)
        self.layout_.addLayout(row)

        self.tiles = TileRow([
            ("readings", "Readings (flagged)"), ("rain", "Rain"), ("recommended", "Water recommended"),
            ("applied", "Water applied (logged)"), ("baseline", "Fixed schedule (estimate)"),
            ("saving", "Estimated saving vs fixed"),
        ])
        self.layout_.addWidget(self.tiles)
        self.note = QLabel("")
        self.note.setObjectName("subtitle")
        self.note.setWordWrap(True)
        self.layout_.addWidget(self.note)

        charts = Card("Daily trends")
        self.moisture_chart = ChartCanvas(2.2)
        self.water_chart = ChartCanvas(2.2)
        self.rain_chart = ChartCanvas(1.8)
        for c in (self.moisture_chart, self.water_chart, self.rain_chart):
            charts.body.addWidget(c)
        self.layout_.addWidget(charts)

        table_card = Card("Daily table")
        self.table = DataTable([
            ("date", "Date"), ("readings", "Readings"), ("invalid_readings", "Flagged"),
            ("avg_soil_moisture", "Avg soil %"), ("min_soil_moisture", "Min soil %"),
            ("max_temperature_c", "Max °C"), ("rainfall_mm", "Rain mm"), ("irrigate_recommendations", "Irrigate recs"),
            ("recommended_water_l", "Recommended L"), ("applied_water_l", "Applied L"),
        ])
        self.table.setMinimumHeight(220)
        table_card.body.addWidget(self.table)
        self.layout_.addWidget(table_card)

        fc_card = Card("Weather forecast used by the engine (next 24 h)")
        self.forecast = QLabel("-")
        self.forecast.setWordWrap(True)
        fc_card.body.addWidget(self.forecast)
        self.layout_.addWidget(fc_card)
        self.layout_.addStretch(1)

    def on_field_changed(self, field: dict | None) -> None:
        if field is None:
            self.tiles.clear()
            self.table.fill([])
            return
        self.refresh()

    def refresh(self) -> None:
        if not self.field:
            return
        fid, days = self.field["id"], self.days.value()
        self.run(self.api.daily_analytics, fid, days, on_done=self._daily)
        self.run(self.api.summary, fid, days, on_done=self._summary)
        self.run(self.api.forecast, fid, 24, on_done=self._forecast)

    def _summary(self, s: dict) -> None:
        self.tiles.set("readings", f"{s['readings']:,} ({s['invalid_readings']:,})")
        self.tiles.set("rain", fmt(s["rainfall_mm"], 1, " mm"))
        self.tiles.set("recommended", fmt(s["recommended_water_l"], 0, " L"))
        self.tiles.set("applied", fmt(s["applied_water_l"], 0, " L"))
        self.tiles.set("baseline", fmt(s["fixed_schedule_estimate_l"], 0, " L"))
        saving = s.get("estimated_saving_vs_fixed_pct")
        self.tiles.set("saving", "-" if saving is None else f"{saving:+.0f} % (est.)")
        self.note.setText(s.get("note", ""))

    def _daily(self, rows: list[dict]) -> None:
        self.table.fill(rows)
        days = [date.fromisoformat(r["date"]) for r in rows]
        labels = [d.strftime("%d %b") for d in days]
        x = [datetime(d.year, d.month, d.day) for d in days]
        self.moisture_chart.plot_lines(
            x, {"average": [r["avg_soil_moisture"] for r in rows], "minimum": [r["min_soil_moisture"] for r in rows]},
            "Soil moisture per day", "% VWC")
        self.water_chart.plot_grouped_bars(
            labels, {"recommended": [r["recommended_water_l"] or 0 for r in rows],
                     "applied": [r["applied_water_l"] or 0 for r in rows]},
            "Irrigation water per day", "litres")
        rain = [r["rainfall_mm"] or 0 for r in rows]
        if any(rain):
            self.rain_chart.plot_grouped_bars(labels, {"rain": rain}, "Rain per day", "mm")
        else:
            self.rain_chart.show_message("No rain recorded in this period")

    def _forecast(self, fc: dict) -> None:
        self.forecast.setText(
            f"Rain {fc['total_rain_mm']:.1f} mm (probability {fc['rain_probability']:.0%}), temperature "
            f"{fc['temp_min_c']:.0f}-{fc['temp_max_c']:.0f} °C, humidity {fc['mean_humidity_pct']:.0f} %.  "
            f"Source: {fc['source']}. {fc['disclaimer']}"
        )
