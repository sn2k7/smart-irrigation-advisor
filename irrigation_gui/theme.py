"""Colours and the application style sheet (light theme)."""

from __future__ import annotations

# Categorical series colours (fixed order) and status colours.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
SURFACE = "#fcfcfb"
PANEL = "#ffffff"
BORDER = "#e4e3df"
ACCENT = "#2a78d6"
STATUS = {
    "good": ("#e6f4ea", "#137333"),      # background, text
    "info": ("#e8f0fe", "#1a56b0"),
    "warning": ("#fef7e0", "#8a5300"),
    "critical": ("#fce8e6", "#b3261e"),
    "neutral": ("#f1f0ec", "#52514e"),
}

STYLE_SHEET = f"""
QWidget {{ font-size: 10pt; color: {TEXT}; }}
QMainWindow, QStackedWidget > QWidget {{ background: {SURFACE}; }}
QListWidget#nav {{ background: #f4f3ef; border: none; border-right: 1px solid {BORDER}; padding-top: 8px; }}
QListWidget#nav::item {{ padding: 10px 14px; border-radius: 6px; margin: 2px 6px; }}
QListWidget#nav::item:selected {{ background: {ACCENT}; color: white; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; }}
QLabel#title {{ font-size: 15pt; font-weight: 600; }}
QLabel#subtitle {{ color: {TEXT_MUTED}; }}
QLabel#tileValue {{ font-size: 16pt; font-weight: 600; }}
QLabel#tileLabel {{ color: {TEXT_MUTED}; font-size: 9pt; }}
QLabel#summary {{ font-size: 14pt; font-weight: 600; }}
QPushButton {{ padding: 6px 14px; border: 1px solid {BORDER}; border-radius: 6px; background: {PANEL}; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton#primary {{ background: {ACCENT}; color: white; border: none; }}
QPushButton#primary:disabled {{ background: #9dbbe6; }}
QPushButton#danger {{ color: #b3261e; }}
QTableWidget {{ background: {PANEL}; gridline-color: {BORDER}; border: 1px solid {BORDER}; }}
QHeaderView::section {{ background: #f4f3ef; padding: 4px; border: none; border-bottom: 1px solid {BORDER}; }}
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QDateEdit, QDateTimeEdit, QPlainTextEdit {{
    padding: 4px; border: 1px solid {BORDER}; border-radius: 4px; background: {PANEL}; }}
"""


def chip_style(kind: str) -> str:
    bg, fg = STATUS.get(kind, STATUS["neutral"])
    return f"background:{bg}; color:{fg}; border-radius:10px; padding:3px 10px; font-weight:600;"
