"""Base class shared by all pages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from ..widgets import page_header

if TYPE_CHECKING:  # pragma: no cover
    from ..main_window import MainWindow


class Page(QWidget):
    """A scrollable page with access to the API, task runner and selected field."""

    title = "Page"
    subtitle = ""
    needs_field = True

    def __init__(self, window: "MainWindow") -> None:
        super().__init__()
        self.window_ = window
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)
        inner = QWidget()
        scroll.setWidget(inner)
        self.layout_ = QVBoxLayout(inner)
        self.layout_.setContentsMargins(20, 16, 20, 16)
        self.layout_.setSpacing(12)
        self.layout_.addWidget(page_header(self.title, self.subtitle))

    # convenience -----------------------------------------------------------
    @property
    def api(self):
        return self.window_.api

    @property
    def field(self) -> dict | None:
        return self.window_.current_field

    def run(self, fn: Callable[..., Any], *args: Any, on_done: Callable[[Any], None] | None = None,
            on_error: Callable[[str], None] | None = None) -> None:
        self.window_.tasks.run(fn, *args, on_done=on_done, on_error=on_error or self.window_.show_error)

    def notify(self, text: str) -> None:
        self.window_.show_message(text)

    # hooks -------------------------------------------------------------------
    def on_field_changed(self, field: dict | None) -> None:
        """Called when the selected field changes (and on first show)."""

    def refresh(self) -> None:
        """Reload data from the API."""
