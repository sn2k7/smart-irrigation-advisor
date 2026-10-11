"""Reusable widgets: cards, metric tiles, status chips, tables and forms."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .theme import STATUS, chip_style


def fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    """Format numbers for display; '-' for missing values."""
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        text = f"{value:,.{digits}f}" if isinstance(value, float) else f"{value:,}"
        return f"{text}{suffix}"
    return str(value)


class Card(QFrame):
    """White rounded panel with an optional heading."""

    def __init__(self, heading: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(14, 12, 14, 12)
        self.body.setSpacing(8)
        if heading:
            label = QLabel(heading)
            label.setStyleSheet("font-weight:600;")
            self.body.addWidget(label)


class Tile(QFrame):
    """Metric tile: small label + big value."""

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        self.value = QLabel("-")
        self.value.setObjectName("tileValue")
        caption = QLabel(label)
        caption.setObjectName("tileLabel")
        caption.setWordWrap(True)
        layout.addWidget(self.value)
        layout.addWidget(caption)

    def set(self, text: str) -> None:
        self.value.setText(text)


class TileRow(QWidget):
    """Horizontal row of tiles addressed by key."""

    def __init__(self, items: Sequence[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tiles: dict[str, Tile] = {}
        for key, label in items:
            tile = Tile(label)
            self.tiles[key] = tile
            layout.addWidget(tile)

    def set(self, key: str, text: str) -> None:
        self.tiles[key].set(text)

    def clear(self) -> None:
        for tile in self.tiles.values():
            tile.set("-")


class Chip(QLabel):
    """Status chip; always text + colour (never colour alone)."""

    def __init__(self, text: str = "", kind: str = "neutral", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.set(text, kind)

    def set(self, text: str, kind: str = "neutral") -> None:
        self.setText(text)
        self.setStyleSheet(chip_style(kind))
        self.setVisible(bool(text))


class DataTable(QTableWidget):
    """Read-only table filled from a list of dicts."""

    def __init__(self, columns: Sequence[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(0, len(columns), parent)
        self.columns = list(columns)
        self.rows: list[dict] = []
        self.setHorizontalHeaderLabels([title for _, title in columns])
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setStretchLastSection(True)

    def fill(self, rows: Iterable[dict], formatters: dict[str, Callable[[Any], str]] | None = None,
             highlight: Callable[[dict], str | None] | None = None) -> None:
        """Replace content. ``highlight(row)`` may return a STATUS kind to tint the row."""
        formatters = formatters or {}
        self.rows = list(rows)
        self.setRowCount(len(self.rows))
        for r, row in enumerate(self.rows):
            kind = highlight(row) if highlight else None
            for c, (key, _) in enumerate(self.columns):
                value = row.get(key)
                text = formatters[key](value) if key in formatters else fmt(value, 2)
                item = QTableWidgetItem(text)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if kind:
                    item.setBackground(QColor(STATUS[kind][0]))
                self.setItem(r, c, item)
        self.resizeColumnsToContents()

    def selected_row(self) -> dict | None:
        index = self.currentRow()
        return self.rows[index] if 0 <= index < len(self.rows) else None


class MessageList(QListWidget):
    """List of reason / warning strings."""

    def __init__(self, kind: str = "neutral", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.setWordWrap(True)
        self.setMinimumHeight(80)

    def set_items(self, items: Iterable[str], prefix: str = "") -> None:
        self.clear()
        for text in items:
            item = QListWidgetItem(f"{prefix}{text}")
            if self.kind != "neutral":
                item.setForeground(QColor(STATUS[self.kind][1]))
            self.addItem(item)


def page_header(title: str, subtitle: str = "") -> QWidget:
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 6)
    t = QLabel(title)
    t.setObjectName("title")
    layout.addWidget(t)
    if subtitle:
        s = QLabel(subtitle)
        s.setObjectName("subtitle")
        s.setWordWrap(True)
        layout.addWidget(s)
    return box
