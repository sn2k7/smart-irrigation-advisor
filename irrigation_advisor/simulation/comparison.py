"""Measured (simulated) comparison of irrigation strategies.

Runs every strategy on the same set of virtual fields and reports water use,
crop water stress and drainage losses. Fields are drawn with a different
seed from the training data, so the ML model is evaluated on unseen
fields and weather.

IMPORTANT: savings reported here are *simulated* under the documented model
assumptions. They are not measured real-world savings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.simulator import random_field_specs
from ..engine.recommender import RecommendationEngine
from ..ml.model_store import IrrigationModel
from ..ml import plots
from ..utils import write_json
from .field_env import SimulationResult, VirtualField
from .strategies import AdvisorStrategy, FixedScheduleStrategy, Strategy

logger = logging.getLogger(__name__)

SIMULATION_DISCLAIMER = (
    "SIMULATED RESULTS - produced by a closed-loop soil water balance model with synthetic weather "
    "and sensor noise. They compare strategies under identical, documented assumptions and are NOT "
    "measured real-world water savings. Field trials are required before claiming savings."
)


def build_strategies(model: IrrigationModel | None, fixed_interval_days: int = 3) -> list[Strategy]:
    """Three no-sensor baselines + advisor variants (ML variant only if a model is supplied)."""
    strategies: list[Strategy] = [
        FixedScheduleStrategy(fixed_interval_days),
        FixedScheduleStrategy(fixed_interval_days, "fixed_skip_after_rain", skip_after_rain_mm=10.0),
        FixedScheduleStrategy(fixed_interval_days, "calendar_kc_adjusted", skip_after_rain_mm=10.0, kc_adjusted=True),
        AdvisorStrategy(RecommendationEngine(model=None), "advisor_rules"),
    ]
    if model is not None:
        strategies.append(AdvisorStrategy(RecommendationEngine(model=model), "advisor_ml_hybrid"))
    return strategies


def run_comparison(
    n_fields: int = 8,
    seed: int = 2024,
    model: IrrigationModel | None = None,
    climate_name: str = "semi_arid_monsoon",
    fault_rate: float = 1.0,
    fixed_interval_days: int = 3,
    year: int = 2026,
) -> tuple[pd.DataFrame, list[SimulationResult]]:
    """Run all strategies on ``n_fields`` virtual fields.

    Returns:
        (per field x strategy metrics table, list of detailed results)
    """
    rng = np.random.default_rng(seed)
    specs = random_field_specs(n_fields, rng, year)
    strategies = build_strategies(model, fixed_interval_days)
    results: list[SimulationResult] = []
    for spec in specs:
        field = VirtualField(spec, climate_name, fault_rate)
        for strategy in strategies:
            logger.info("Simulating %s with %s", spec.field_id, strategy.name)
            results.append(field.run(strategy))
    rows = []
    for spec_res in results:
        spec = next(s for s in specs if s.field_id == spec_res.field_id)
        rows.append(
            {
                "field_id": spec_res.field_id,
                "strategy": spec_res.strategy,
                "crop_type": spec.crop_type,
                "soil_type": spec.soil_type,
                "irrigation_method": spec.irrigation_method,
                "planting_date": spec.planting_date.isoformat(),
                **spec_res.metrics,
            }
        )
    return pd.DataFrame(rows), results


def summarise(table: pd.DataFrame, baseline: str = "fixed_schedule") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mean metrics per strategy and paired water savings vs. ``baseline``."""
    metric_cols = [
        "water_m3_per_ha", "irrigation_gross_mm", "irrigation_events", "pump_hours", "stress_days",
        "severe_stress_days", "relative_transpiration", "deep_drainage_mm",
    ]
    summary = table.groupby("strategy", sort=False)[metric_cols].mean().round(3)
    base = table[table["strategy"] == baseline].set_index("field_id")
    comparisons: dict[str, Any] = {}
    for strategy in summary.index:
        if strategy == baseline:
            continue
        cur = table[table["strategy"] == strategy].set_index("field_id")
        paired = (1 - cur["irrigation_gross_mm"] / base["irrigation_gross_mm"].replace(0, np.nan)) * 100
        total_saving = 100 * (1 - cur["irrigation_gross_mm"].sum() / max(base["irrigation_gross_mm"].sum(), 1e-9))
        comparisons[strategy] = {
            "water_saving_pct_total": round(float(total_saving), 1),
            "water_saving_pct_median_field": round(float(paired.median()), 1),
            "water_saving_pct_range": [round(float(paired.min()), 1), round(float(paired.max()), 1)],
            "fields_with_saving": int((paired > 0).sum()),
            "stress_days_change": round(float(cur["stress_days"].mean() - base["stress_days"].mean()), 2),
            "relative_transpiration_change": round(
                float(cur["relative_transpiration"].mean() - base["relative_transpiration"].mean()), 4
            ),
            "deep_drainage_change_mm": round(float(cur["deep_drainage_mm"].mean() - base["deep_drainage_mm"].mean()), 1),
        }
    return summary, {"baseline": baseline, "n_fields": int(table["field_id"].nunique()), "vs_baseline": comparisons}


def write_simulation_reports(
    table: pd.DataFrame, results: list[SimulationResult], reports_dir: Path, config: dict[str, Any]
) -> dict[str, str]:
    """Persist CSV/JSON/Markdown/PNG outputs of a comparison run."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary, comparison = summarise(table, "fixed_schedule")
    _, comparison_fair = summarise(table, "fixed_skip_after_rain")
    _, comparison_kc = summarise(table, "calendar_kc_adjusted")
    paths = {"table": str(reports_dir / "simulation_results.csv")}
    table.to_csv(paths["table"], index=False)
    payload = {
        "disclaimer": SIMULATION_DISCLAIMER,
        "config": config,
        "summary_by_strategy": summary.reset_index().to_dict(orient="records"),
        **comparison,
        "vs_fixed_skip_after_rain": comparison_fair["vs_baseline"],
        "vs_calendar_kc_adjusted": comparison_kc["vs_baseline"],
    }
    paths["json"] = str(write_json(payload, reports_dir / "simulation_summary.json"))
    paths["chart"] = plots.plot_strategy_comparison(summary, reports_dir / "simulation_comparison.png")

    # Moisture trace of the first field, every strategy.
    first = results[0].field_id
    traces = {r.strategy: r.trace for r in results if r.field_id == first}
    any_trace = next(iter(traces.values()))
    threshold = any_trace.set_index("timestamp")["stress_threshold_sm"]
    paths["trace"] = plots.plot_moisture_trace(
        traces, threshold, reports_dir / "simulation_moisture_trace.png",
        f"True soil moisture by strategy - field {first} (synthetic)",
    )

    lines = [
        "# Irrigation Strategy Comparison (Closed-Loop Simulation)",
        "",
        f"> **{SIMULATION_DISCLAIMER}**",
        "",
        f"- Virtual fields: {comparison['n_fields']} (seed {config.get('seed')}, different from the training data)",
        f"- Fixed schedules irrigate every {config.get('fixed_interval_days')} days with a depth sized for peak demand",
        f"- Sensor faults injected: {'yes' if config.get('fault_rate', 0) > 0 else 'no'}",
        "",
        "## Mean per field and season",
        "",
        "Baselines (no sensors): `fixed_schedule` (pure calendar, depth for peak demand), `fixed_skip_after_rain` "
        "(same, skipped after >= 10 mm rain) and `calendar_kc_adjusted` (depth follows the crop-stage Kc curve, "
        "skipped after rain).",
        "",
        "| Strategy | Water (m³/ha) | Events | Pump h | Stress days | Severe stress days | Rel. transpiration | Drainage (mm) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in summary.iterrows():
        lines.append(
            f"| {name} | {row['water_m3_per_ha']:.0f} | {row['irrigation_events']:.1f} | {row['pump_hours']:.1f} | "
            f"{row['stress_days']:.1f} | {row['severe_stress_days']:.1f} | {row['relative_transpiration']:.3f} | "
            f"{row['deep_drainage_mm']:.0f} |"
        )
    sections = [
        ("Versus the pure fixed schedule (ignores rain)", comparison["vs_baseline"]),
        ("Versus the fixed schedule that skips after >= 10 mm rain", comparison_fair["vs_baseline"]),
        ("Versus the crop-stage (Kc) adjusted calendar that also skips after rain (strongest no-sensor baseline)",
         comparison_kc["vs_baseline"]),
    ]
    for title, comps in sections:
        lines += ["", f"## {title}", ""]
        for name, comp in comps.items():
            if not name.startswith("advisor"):
                continue
            lines.append(
                f"- **{name}**: simulated water use {comp['water_saving_pct_total']:+.1f}% saving overall "
                f"(median field {comp['water_saving_pct_median_field']:+.1f}%, range "
                f"{comp['water_saving_pct_range'][0]:+.1f}% to {comp['water_saving_pct_range'][1]:+.1f}%, "
                f"{comp['fields_with_saving']}/{comparison['n_fields']} fields saved water); "
                f"stress days {comp['stress_days_change']:+.2f}; relative transpiration "
                f"{comp['relative_transpiration_change']:+.4f}; drainage {comp['deep_drainage_change_mm']:+.0f} mm"
            )
    lines += [
        "",
        "## How to read these numbers",
        "",
        "- The size of the 'saving' depends strongly on the baseline. Calendar schedules sized for peak demand",
        "  over-irrigate early and late in the season and during rain (see the drainage column); the",
        "  crop-stage adjusted calendar is the most demanding comparison and gives the most conservative figure.",
        "- Stress days count hours with Ks < 1, i.e. depletion beyond RAW. The advisor deliberately lets the root",
        "  zone dry to the RAW threshold before refilling, so short mild stress is expected; relative",
        "  transpiration (actual / potential crop ET, a yield proxy) shows its size.",
        "- Weather, soil physics, sensor noise and faults are simulated; real savings must be measured in field trials.",
        "",
    ]
    md = reports_dir / "simulation_report.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    paths["markdown"] = str(md)
    return paths
