"""Irrigation strategies compared in the closed-loop simulation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd

from ..agronomy.crops import get_crop, get_method
from ..agronomy.water_balance import crop_coefficient
from ..engine.recommender import FieldConfig, RecommendationEngine

#: Peak-season reference ET (mm/day) a fixed schedule is typically designed for.
DESIGN_ET0_MM = 5.0


@dataclass
class DecisionContext:
    """What a strategy may see at decision time (never the true soil state)."""

    now: datetime
    field_cfg: FieldConfig
    day_index: int
    history: pd.DataFrame  # recent SENSOR readings only
    forecast_rain_24h_mm: float


@dataclass
class Decision:
    """Gross irrigation depth to apply (mm) plus the pump minutes it implies."""

    gross_depth_mm: float
    duration_min: float
    source: str = ""


class Strategy(Protocol):
    name: str

    def decide(self, ctx: DecisionContext) -> Decision: ...


class FixedScheduleStrategy:
    """Baseline: irrigate every ``interval_days`` with a fixed depth, ignoring sensors and rain.

    The depth is sized like a typical calendar schedule designed for peak
    demand: ``interval * Kc_mid * DESIGN_ET0_MM`` net, converted to gross
    with the method efficiency. This is common practice where no sensors exist.
    """

    def __init__(
        self,
        interval_days: int = 3,
        name: str = "fixed_schedule",
        skip_after_rain_mm: float | None = None,
        kc_adjusted: bool = False,
    ) -> None:
        if interval_days < 1:
            raise ValueError("interval_days must be >= 1")
        self.interval_days = interval_days
        self.name = name
        #: If set, skip a scheduled irrigation when at least this much rain fell
        #: in the previous 24 h (a common, fairer farmer heuristic).
        self.skip_after_rain_mm = skip_after_rain_mm
        #: If True, depth follows the crop-stage Kc curve (FAO calendar schedule)
        #: instead of always using the mid-season Kc - the strongest no-sensor baseline.
        self.kc_adjusted = kc_adjusted

    def depth_for(self, field_cfg: FieldConfig, day_index: int | None = None) -> float:
        crop = get_crop(field_cfg.crop_type)
        kc = crop_coefficient(crop, day_index) if self.kc_adjusted and day_index is not None else crop.kc_mid
        net = self.interval_days * kc * DESIGN_ET0_MM
        return net / get_method(field_cfg.irrigation_method).efficiency

    def decide(self, ctx: DecisionContext) -> Decision:
        if ctx.day_index % self.interval_days != 0:
            return Decision(0.0, 0.0, "schedule_off_day")
        if self.skip_after_rain_mm is not None and len(ctx.history):
            recent = ctx.history[ctx.history["timestamp"] >= pd.Timestamp(ctx.now) - pd.Timedelta(hours=24)]
            if float(recent["rainfall_mm"].fillna(0).sum()) >= self.skip_after_rain_mm:
                return Decision(0.0, 0.0, "schedule_skipped_rain")
        gross = self.depth_for(ctx.field_cfg, ctx.day_index)
        minutes = gross * ctx.field_cfg.area_m2 / ctx.field_cfg.flow_rate_lpm
        return Decision(gross, minutes, "schedule")


class AdvisorStrategy:
    """The Smart Irrigation Advisor engine (rule-only or ML hybrid)."""

    def __init__(self, engine: RecommendationEngine, name: str) -> None:
        self.engine = engine
        self.name = name

    def decide(self, ctx: DecisionContext) -> Decision:
        rec = self.engine.recommend(ctx.field_cfg, ctx.history, ctx.now, ctx.forecast_rain_24h_mm)
        if not rec.irrigation_needed:
            return Decision(0.0, 0.0, rec.decision_source)
        return Decision(rec.gross_depth_mm, rec.duration_min, rec.decision_source)
