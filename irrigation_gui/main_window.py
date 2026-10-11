"""Main window: navigation, field selector, connection status."""

from __future__ import annotations

import logging

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .api_client import ApiClient
from .pages.analytics import AnalyticsPage
from .pages.base import Page
from .pages.calculator import CalculatorPage
from .pages.dashboard import DashboardPage
from .pages.fields import FieldsPage
from .pages.history import HistoryPage
from .pages.model_page import ModelPage
from .pages.readings import ReadingsPage
from .tasks import TaskRunner
from .theme import STATUS
from .widgets import Chip

logger = logging.getLogger(__name__)
HEALTH_POLL_MS = 30_000


class MainWindow(QMainWindow):
    """Top-level window. Pages talk to the API through ``self.api`` + ``self.tasks``."""

    field_changed = Signal(object)

    def __init__(self, api: ApiClient, tasks: TaskRunner | None = None) -> None:
        super().__init__()
        self.api = api
        self.tasks = tasks or TaskRunner(parent=self)
        self.tasks.error.connect(self.show_error)
        self.fields: list[dict] = []
        self.current_field: dict | None = None
        self.reference: dict = {}
        self.setWindowTitle("Smart Irrigation Advisor")
        self.resize(1280, 860)

        # --- top bar ---------------------------------------------------------
        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(12, 8, 12, 8)
        brand = QLabel("💧 Smart Irrigation Advisor")
        brand.setStyleSheet("font-size:13pt; font-weight:600;")
        top_layout.addWidget(brand)
        top_layout.addSpacing(24)
        top_layout.addWidget(QLabel("Field:"))
        self.field_combo = QComboBox()
        self.field_combo.setMinimumWidth(280)
        self.field_combo.currentIndexChanged.connect(self._on_field_index)
        top_layout.addWidget(self.field_combo)
        top_layout.addStretch(1)
        top_layout.addWidget(QLabel("API:"))
        self.url_edit = QLineEdit(api.base_url)
        self.url_edit.setMinimumWidth(220)
        connect = QPushButton("Connect")
        connect.clicked.connect(self.reconnect)
        self.conn_chip = Chip("connecting…", "neutral")
        for w in (self.url_edit, connect, self.conn_chip):
            top_layout.addWidget(w)

        # --- navigation + pages ----------------------------------------------
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(190)
        self.stack = QStackedWidget()
        self.pages: list[Page] = [
            DashboardPage(self), ReadingsPage(self), HistoryPage(self), AnalyticsPage(self),
            FieldsPage(self), CalculatorPage(self), ModelPage(self),
        ]
        for page in self.pages:
            self.nav.addItem(page.title)
            self.stack.addWidget(page)
        self.nav.currentRowChanged.connect(self._on_nav)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self.nav)
        body_layout.addWidget(self.stack, 1)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(top)
        layout.addWidget(body, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")
        self.tasks.busy_changed.connect(self._on_busy)

        self.field_changed.connect(self._broadcast_field)
        self.health_timer = QTimer(self)
        self.health_timer.setInterval(HEALTH_POLL_MS)
        self.health_timer.timeout.connect(self.check_health)
        self.health_timer.start()
        self.nav.setCurrentRow(0)

    # ------------------------------------------------------------------ public helpers
    def page(self, cls: type[Page]) -> Page:
        return next(p for p in self.pages if isinstance(p, cls))

    def show_message(self, text: str) -> None:
        self.statusBar().setStyleSheet("")
        self.statusBar().showMessage(text, 8000)

    def show_error(self, text: str) -> None:
        logger.warning("UI error: %s", text)
        self.statusBar().setStyleSheet(f"color:{STATUS['critical'][1]};")
        self.statusBar().showMessage(f"Error: {text}", 15000)

    def _on_busy(self, busy: bool) -> None:
        if busy:
            self.statusBar().showMessage("Working…")
        elif self.statusBar().currentMessage() == "Working…":
            self.statusBar().showMessage("Ready")

    def confirm(self, question: str) -> bool:
        answer = QMessageBox.question(self, "Please confirm", question)
        return answer == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------------ startup / connection
    def start(self) -> None:
        """Load reference data, fields and status (call after show())."""
        self.check_health()
        self.tasks.run(self.api.reference, on_done=self._set_reference, on_error=self.show_error)
        self.reload_fields()

    def reconnect(self) -> None:
        self.api.set_base_url(self.url_edit.text().strip())
        self.start()

    def check_health(self) -> None:
        def ok(h: dict) -> None:
            model = "ML model" if h["model_loaded"] else "rule-based"
            self.conn_chip.set(f"online · {model}", "good" if h["database"] else "warning")

        self.tasks.run(self.api.health, on_done=ok, on_error=lambda m: self.conn_chip.set("offline", "critical"))

    def _set_reference(self, reference: dict) -> None:
        self.reference = reference
        for page in self.pages:
            if hasattr(page, "set_reference"):
                page.set_reference(reference)

    def reload_fields(self, select_id: int | None = None) -> None:
        def done(fields: list[dict]) -> None:
            self.fields = fields
            keep = select_id if select_id is not None else (self.current_field or {}).get("id")
            self.field_combo.blockSignals(True)
            self.field_combo.clear()
            for f in fields:
                self.field_combo.addItem(f"#{f['id']}  {f['name']}  ({f['crop_type']}, {f['soil_type']})", f["id"])
            index = next((i for i, f in enumerate(fields) if f["id"] == keep), 0 if fields else -1)
            self.field_combo.setCurrentIndex(index)
            self.field_combo.blockSignals(False)
            self._on_field_index(index, force=True)

        self.tasks.run(self.api.list_fields, on_done=done, on_error=self.show_error)

    # ------------------------------------------------------------------ events
    def _on_field_index(self, index: int, force: bool = False) -> None:
        field = self.fields[index] if 0 <= index < len(self.fields) else None
        if force or field != self.current_field:
            self.current_field = field
            self.field_changed.emit(field)

    def _broadcast_field(self, field: dict | None) -> None:
        # Only the visible page loads immediately; others refresh when shown.
        self._stale = {id(p) for p in self.pages}
        self._refresh_visible()

    def _on_nav(self, row: int) -> None:
        self.stack.setCurrentIndex(row)
        self._refresh_visible()

    def _refresh_visible(self) -> None:
        """Field changed since last shown -> full reset; otherwise just reload data."""
        page = self.stack.currentWidget()
        stale = getattr(self, "_stale", None)
        if not isinstance(page, Page) or stale is None:
            return  # nothing loaded yet (start() not called)
        if id(page) in stale:
            stale.discard(id(page))
            page.on_field_changed(self.current_field)
        else:
            page.refresh()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.health_timer.stop()
        self.tasks.wait(3000)
        self.api.close()
        super().closeEvent(event)


__all__ = ["MainWindow"]
