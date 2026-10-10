"""Static report figures (PNG) written to ``reports/``.

Uses the non-interactive Agg backend so it runs headless on Windows/Linux.
Colours follow a fixed categorical order (never cycled) with thin marks,
recessive grids and one y-axis per chart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed categorical order (validated reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQUENTIAL = "Blues"


def _style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=9)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.99, -0.03, "Synthetic data - not field-validated", ha="right", va="top", fontsize=7, color=TEXT_MUTED)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def plot_confusion_matrices(classification: dict[str, Any], section: str, path: Path) -> str:
    """One confusion-matrix heatmap per model (counts + row percentages)."""
    names = list(classification)
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.4))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        cm = classification[name][section]["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
        row_pct = mat / np.clip(mat.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(row_pct, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{int(mat[i, j])}\n({row_pct[i, j]:.0%})", ha="center", va="center",
                    color="white" if row_pct[i, j] > 0.6 else TEXT, fontsize=9,
                )
        ax.set_xticks([0, 1], ["no irrigation", "irrigate"], fontsize=8)
        ax.set_yticks([0, 1], ["no irrigation", "irrigate"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8, color=TEXT_MUTED)
        ax.set_ylabel("Actual", fontsize=8, color=TEXT_MUTED)
        ax.set_title(f"{name}\nF1 {classification[name][section]['f1']:.3f}", fontsize=9, color=TEXT)
    fig.suptitle(f"Confusion matrices - {section.replace('_', ' ')}", fontsize=11, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, path)


def plot_model_comparison(classification: dict[str, Any], path: Path) -> str:
    """Grouped bars of precision / recall / F1 per model on the temporal test set."""
    names = list(classification)
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k, metric in enumerate(metrics):
        values = [classification[n]["test_temporal"][metric] for n in names]
        bars = ax.bar(x + (k - 1) * width, values, width - 0.02, color=SERIES[k], label=metric)
        if metric == "f1":
            ax.bar_label(bars, fmt="%.2f", fontsize=7, color=TEXT, padding=2)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylim(0, 1.0)
    _style(ax, "Classifier comparison - temporal test set", "", "score")
    ax.legend(frameon=False, fontsize=8, ncols=3, loc="lower left", bbox_to_anchor=(0.0, 1.01), borderaxespad=0)
    ax.set_title("Classifier comparison - temporal test set", color=TEXT, fontsize=11, loc="left", pad=30)
    return _save(fig, path)


def plot_feature_importance(importance: list[dict[str, Any]], path: Path, top_n: int = 15) -> str:
    """Horizontal bars of permutation importance."""
    items = importance[:top_n][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(items) + 1.2))
    ax.barh([i["feature"] for i in items], [i["importance"] for i in items], color=SERIES[0], height=0.6)
    _style(ax, "Permutation importance (drop in F1, out-of-fold rows)", "importance", "")
    return _save(fig, path)


def plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, path: Path) -> str:
    """Predicted vs. true net irrigation depth."""
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, color=SERIES[0], edgecolors="none", label="test rows")
    lim = float(max(np.nanmax(y_true), np.nanmax(y_pred), 1.0)) * 1.05
    ax.plot([0, lim], [0, lim], color=TEXT_MUTED, linewidth=1, linestyle="--", label="perfect")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    _style(ax, f"Net irrigation depth - {model_name}", "true depth (mm)", "predicted depth (mm)")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_strategy_comparison(summary: pd.DataFrame, path: Path) -> str:
    """Water use and crop stress per strategy (two separate panels, one axis each)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    strategies = list(summary.index)
    colors = SERIES[: len(strategies)]
    b1 = ax1.bar(strategies, summary["water_m3_per_ha"], color=colors, width=0.6)
    ax1.bar_label(b1, fmt="%.0f", fontsize=8, color=TEXT, padding=2)
    _style(ax1, "Total irrigation water", "", "m³ per hectare")
    b2 = ax2.bar(strategies, summary["stress_days"], color=colors, width=0.6)
    ax2.bar_label(b2, fmt="%.1f", fontsize=8, color=TEXT, padding=2)
    _style(ax2, "Crop water-stress days (Ks < 1)", "", "days per field (mean)")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Closed-loop simulation (synthetic)", fontsize=11, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, path)


def plot_moisture_trace(traces: dict[str, pd.DataFrame], raw_threshold: pd.Series, path: Path, title: str) -> str:
    """True soil moisture over a season for each strategy, with the stress threshold."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for k, (name, trace) in enumerate(traces.items()):
        ax.plot(trace["timestamp"], trace["true_soil_moisture"], color=SERIES[k], linewidth=1.4, label=name)
    ax.plot(raw_threshold.index, raw_threshold.values, color=TEXT_MUTED, linewidth=1, linestyle="--",
            label="stress threshold (RAW)")
    _style(ax, title, "", "true soil moisture (% VWC)")
    ax.legend(frameon=False, fontsize=8, ncols=4, loc="upper left")
    fig.autofmt_xdate()
    return _save(fig, path)
