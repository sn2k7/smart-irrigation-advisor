"""Model & system: health, loaded model, its test metrics and hot reload."""

from __future__ import annotations

import json

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QPushButton

from ..widgets import Card, Chip, TileRow, fmt
from .base import Page


class ModelPage(Page):
    title = "Model & system"
    subtitle = ("Which decision model is active and how it scored on held-out SYNTHETIC test data. "
                "Without a model the engine uses the FAO-56 rule.")
    needs_field = False

    def __init__(self, window) -> None:
        super().__init__(window)
        card = Card("Status")
        row = QHBoxLayout()
        self.api_chip = Chip("API: unknown", "neutral")
        self.model_chip = Chip("Model: unknown", "neutral")
        row.addWidget(self.api_chip)
        row.addWidget(self.model_chip)
        row.addStretch(1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        reload_btn = QPushButton("Reload model from disk")
        reload_btn.clicked.connect(lambda: self.run(self.api.reload_model, on_done=self._show_model))
        row.addWidget(refresh)
        row.addWidget(reload_btn)
        card.body.addLayout(row)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        card.body.addWidget(self.message)
        self.tiles = TileRow([("f1", "Test F1 (temporal)"), ("precision", "Precision"), ("recall", "Recall"),
                              ("mae", "Depth MAE"), ("threshold", "Decision threshold")])
        card.body.addWidget(self.tiles)
        self.layout_.addWidget(card)
        details = Card("Model metadata (raw)")
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(260)
        details.body.addWidget(self.details)
        self.layout_.addWidget(details)
        self.layout_.addStretch(1)

    def on_field_changed(self, field: dict | None) -> None:
        self.refresh()

    def refresh(self) -> None:
        self.run(self.api.health, on_done=self._show_health,
                 on_error=lambda m: self.api_chip.set("API: offline", "critical"))
        self.run(self.api.model_info, on_done=self._show_model)

    def _show_health(self, h: dict) -> None:
        self.api_chip.set(f"API {h['version']}: {'database ok' if h['database'] else 'DB ERROR'}",
                          "good" if h["database"] else "critical")

    def _show_model(self, info: dict) -> None:
        self.message.setText(info["message"])
        self.tiles.clear()
        if not info["loaded"]:
            self.model_chip.set("Model: not loaded (rule-based)", "warning")
            self.details.setPlainText(f"Model path: {info['model_path']}")
            return
        meta = info["info"] or {}
        self.model_chip.set(f"Model: {meta.get('classifier')} + {meta.get('regressor')}", "good")
        test = meta.get("test_temporal") or {}
        cls, reg = test.get("classification") or {}, test.get("regression") or {}
        self.tiles.set("f1", fmt(cls.get("f1"), 3))
        self.tiles.set("precision", fmt(cls.get("precision"), 3))
        self.tiles.set("recall", fmt(cls.get("recall"), 3))
        self.tiles.set("mae", fmt(reg.get("mae_mm"), 1, " mm"))
        self.tiles.set("threshold", fmt(meta.get("threshold"), 3))
        self.details.setPlainText(json.dumps(info, indent=2))
