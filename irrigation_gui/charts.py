"""Matplotlib charts embedded in Qt (one y-axis per chart, fixed series colours)."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

# The Qt canvas is used directly with a Figure (no pyplot), so no global
# backend switch is needed and headless report plotting keeps working.
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure

from .theme import BORDER, SERIES, TEXT, TEXT_MUTED


class ChartCanvas(FigureCanvasQTAgg):
    """A single-axes matplotlib figure styled for the app."""

    def __init__(self, height: float = 2.6) -> None:
        self.figure = Figure(figsize=(6, height), tight_layout=True)
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.setMinimumHeight(int(height * 70))
        self.show_message("No data yet")

    def _style(self, title: str, ylabel: str) -> None:
        ax = self.ax
        ax.set_title(title, loc="left", fontsize=10, color=TEXT)
        ax.set_ylabel(ylabel, fontsize=8, color=TEXT_MUTED)
        ax.tick_params(labelsize=8, colors=TEXT_MUTED)
        ax.grid(True, color=BORDER, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BORDER)

    def show_message(self, text: str) -> None:
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, text, ha="center", va="center", color=TEXT_MUTED, transform=self.ax.transAxes)
        self.draw_idle()

    def plot_lines(self, x: Sequence[datetime], series: dict[str, Sequence[float | None]], title: str,
                   ylabel: str, markers: dict[str, Sequence[tuple[datetime, float]]] | None = None) -> None:
        self.ax.clear()
        self.ax.set_axis_on()
        if not x:
            self.show_message("No data yet")
            return
        for k, (name, values) in enumerate(series.items()):
            self.ax.plot(x, [float("nan") if v is None else v for v in values], color=SERIES[k % len(SERIES)],
                         linewidth=1.8, label=name)
        for k, (name, points) in enumerate((markers or {}).items()):
            if points:
                xs, ys = zip(*points)
                self.ax.scatter(xs, ys, s=28, color=SERIES[(k + 1) % len(SERIES)], zorder=3, label=name,
                                edgecolors="white", linewidths=1)
        locator = AutoDateLocator()
        self.ax.xaxis.set_major_locator(locator)
        self.ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        self._style(title, ylabel)
        if len(series) + len(markers or {}) > 1:
            self.ax.legend(frameon=False, fontsize=8, loc="upper left")
        self.draw_idle()

    def plot_grouped_bars(self, labels: Sequence[str], series: dict[str, Sequence[float]], title: str,
                          ylabel: str) -> None:
        self.ax.clear()
        self.ax.set_axis_on()
        if not labels:
            self.show_message("No data yet")
            return
        n = len(series)
        width = 0.8 / max(n, 1)
        positions = range(len(labels))
        for k, (name, values) in enumerate(series.items()):
            offset = (k - (n - 1) / 2) * width
            self.ax.bar([p + offset for p in positions], values, width * 0.92, color=SERIES[k % len(SERIES)],
                        label=name)
        step = max(1, len(labels) // 10)
        self.ax.set_xticks(list(positions)[::step], list(labels)[::step], rotation=0)
        self._style(title, ylabel)
        if n > 1:
            self.ax.legend(frameon=False, fontsize=8, loc="upper left")
        self.draw_idle()
