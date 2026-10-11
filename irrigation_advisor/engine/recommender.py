"""Transparent irrigation recommendation engine.

Decision flow for one field (every step adds a human-readable reason):

1. **Forecast** - client-supplied 24 h rain forecast if given, otherwise the
   offline simulated weather provider (always labelled with its source).
2. **Sensor validation** - the latest reading is checked (range, spike,
   stuck, stale, future timestamp) and the history is cleaned with the same
   ``prepare_sensor_frame`` used for training.
3. **Features** - ``build_features`` (identical to training) gives FAO-56
   water status: depletion Dr, RAW, TAW, ETc, effective forecast rain.
4. **Decision path**
   * ``ml_hybrid``   - trained classifier/regressor + physics safety rules;
   * ``rule_based``  - FAO-56 rule only (no model loaded);
   * ``fallback_estimated``   - soil sensor unusable: depletion is projected
     from the last valid reading with a water balance;
   * ``fallback_conservative`` - no usable soil data in 24 h: at most one
     day of crop water use, skipped if meaningful rain is expected.
5. **Safety overrides** (applied to every path)
   * never irrigate at/above field capacity;
   * defer while it is raining unless stress is severe;
   * always irrigate under severe depletion (half-way between RAW and TAW).
6. **Plan** - net depth -> gross depth (method efficiency) -> litres and
   pump minutes, capped by the field's maximum run time.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..agronomy.crops import get_crop, get_method, get_soil
from ..agronomy.water_balance import (
    crop_coefficient,
    effective_rainfall,
    growth_stage,
    hargreaves_et0,
    net_irrigation_requirement,
    plan_irrigation,
    projected_depletion,
    root_depth,
)
from ..data.features import build_features
from ..data.preprocessing import prepare_sensor_frame
from ..data.validation import validate_reading
from ..data.weather_provider import SimulatedWeatherProvider, WeatherForecast, WeatherProvider
from ..ml.model_store import IrrigationModel
from ..utils import is_missing, round_or_none

logger = logging.getLogger(__name__)

MIN_NET_DEPTH_MM = 2.0
SEVERE_STRESS_POSITION = 0.5  # fraction of the way from RAW to TAW
RAINING_NOW_MM_PER_HOUR = 1.0
RAIN_LAST_6H_DEFER_MM = 5.0
ESTIMATION_MAX_AGE_HOURS = 24.0
DEFAULT_ET0_MM = 5.0  # used only if neither sensor nor forecast temperatures exist


@dataclass(frozen=True)
class FieldConfig:
    """Static configuration of a field needed for a recommendation."""

    field_id: str
    crop_type: str
    soil_type: str
    irrigation_method: str
    planting_date: date
    area_m2: float
    flow_rate_lpm: float
    latitude: float
    name: str = ""
    max_irrigation_minutes: float = 180.0
    longitude: float | None = None

    def __post_init__(self) -> None:
        get_crop(self.crop_type)
        get_soil(self.soil_type)
        get_method(self.irrigation_method)
        if self.area_m2 <= 0 or self.flow_rate_lpm <= 0:
            raise ValueError("area_m2 and flow_rate_lpm must be positive")

    def days_since_planting(self, ts: datetime) -> float:
        return (ts - datetime.combine(self.planting_date, datetime.min.time())).total_seconds() / 86400.0


@dataclass
class Recommendation:
    """Result returned to API clients and stored in the database."""

    field_id: str
    decision_time: datetime
    irrigation_needed: bool
    decision_source: str
    confidence: str
    summary: str
    probability: float | None = None
    net_depth_mm: float = 0.0
    gross_depth_mm: float = 0.0
    water_volume_l: float = 0.0
    duration_min: float = 0.0
    capped: bool = False
    soil_moisture_pct: float | None = None
    soil_moisture_estimated: bool = False
    depletion_mm: float | None = None
    raw_mm: float | None = None
    taw_mm: float | None = None
    depletion_ratio: float | None = None
    etc_24h_mm: float | None = None
    kc: float | None = None
    growth_stage: str | None = None
    forecast_rain_24h_mm: float = 0.0
    forecast_effective_rain_mm: float = 0.0
    forecast_source: str = ""
    rule_decision: bool | None = None
    ml_decision: bool | None = None
    model_version: str | None = None
    next_check_hours: int = 12
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sensor_issues: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["decision_time"] = self.decision_time.isoformat()
        for key, value in data.items():
            if isinstance(value, float):
                data[key] = round_or_none(value, 3)
        return data


@dataclass
class _WaterState:
    depletion_mm: float
    raw_mm: float
    taw_mm: float
    etc_24h_mm: float
    kc: float
    stage: str
    soil_moisture: float | None


class RecommendationEngine:
    """Combines ML predictions with transparent FAO-56 rules and fallbacks."""

    def __init__(
        self,
        model: IrrigationModel | None = None,
        weather: WeatherProvider | None = None,
        stale_hours: float = 6.0,
    ) -> None:
        self.model = model
        self.weather = weather or SimulatedWeatherProvider()
        self.stale_hours = stale_hours

    # ------------------------------------------------------------------ API
    def recommend(
        self,
        field_cfg: FieldConfig,
        readings: pd.DataFrame,
        now: datetime | None = None,
        forecast_rain_24h_mm: float | None = None,
    ) -> Recommendation:
        """Produce a recommendation from a field's recent readings.

        Args:
            field_cfg: Field configuration.
            readings: Recent raw readings (``timestamp``, ``soil_moisture``,
                ``temperature_c``, ``humidity_pct``, ``rainfall_mm``; optional
                ``battery_v``), any order, may be empty.
            now: Decision time; defaults to the latest reading time or the clock.
            forecast_rain_24h_mm: Optional client forecast overriding the
                simulated provider.
        """
        readings = self._normalise_readings(readings)
        if now is None:
            now = readings["timestamp"].max().to_pydatetime() if len(readings) else datetime.now()
        forecast = self.weather.forecast(field_cfg.latitude, now, 24, **self._location(field_cfg))
        if forecast_rain_24h_mm is not None:
            rain_fc, source = max(0.0, float(forecast_rain_24h_mm)), "client"
        else:
            rain_fc, source = forecast.total_rain_mm, forecast.source
        eff_rain = effective_rainfall(rain_fc)

        rec = Recommendation(
            field_id=field_cfg.field_id,
            decision_time=now,
            irrigation_needed=False,
            decision_source="rule_based",
            confidence="low",
            summary="",
            forecast_rain_24h_mm=round(rain_fc, 2),
            forecast_effective_rain_mm=round(eff_rain, 2),
            forecast_source=source,
            model_version=self.model.version if self.model else None,
        )
        if source != "client":
            rec.warnings.append("Rain forecast is SIMULATED (offline mode); supply forecast_rain_24h_mm for real forecasts")

        if readings.empty:
            rec.warnings.append("No sensor readings available for this field")
            return self._conservative(field_cfg, rec, now, forecast, rain_now=0.0)

        # --- validate the latest raw reading --------------------------------
        latest = readings.iloc[-1].to_dict()
        latest["timestamp"] = pd.Timestamp(latest["timestamp"]).to_pydatetime()
        history = [
            {**r, "timestamp": pd.Timestamp(r["timestamp"]).to_pydatetime()}
            for r in readings.iloc[:-1].tail(24).to_dict("records")
        ]
        validation = validate_reading(latest, history, now=now, stale_hours=self.stale_hours)
        rec.sensor_issues = [i.as_dict() for i in validation.issues]
        for issue in validation.issues:
            rec.warnings.append(f"Sensor check: {issue.message}")

        # --- identical preprocessing + features as training ------------------
        frame = readings.assign(
            field_id=field_cfg.field_id,
            crop_type=field_cfg.crop_type,
            soil_type=field_cfg.soil_type,
            irrigation_method=field_cfg.irrigation_method,
            planting_date=field_cfg.planting_date.isoformat(),
            latitude=field_cfg.latitude,
            forecast_rain_24h_mm=rain_fc,
        )
        frame = self._add_weather_service_columns(frame, field_cfg, now)
        prepared = prepare_sensor_frame(frame)
        feats = build_features(prepared)
        row = feats.iloc[-1]
        prep_row = prepared.iloc[-1]
        rain_now = float(row["rainfall_mm"]) if not is_missing(row["rainfall_mm"]) else 0.0
        rain_6h = float(row["rain_sum_6h"]) if not is_missing(row["rain_sum_6h"]) else 0.0

        hours_old = (now - pd.Timestamp(row["timestamp"]).to_pydatetime()).total_seconds() / 3600.0
        sm_usable = (
            validation.is_valid
            and not is_missing(row["soil_moisture"])
            and not bool(prep_row["spike"] or prep_row["stuck"])
            and hours_old <= self.stale_hours
        )
        etc = self._etc(field_cfg, row, forecast, now)

        if not sm_usable:
            return self._estimated(field_cfg, rec, prepared, now, forecast, etc, rain_now, rain_6h)

        if bool(prep_row["sm_imputed"]):
            rec.warnings.append("Latest hour had no valid soil reading; last valid value carried forward (<= 3 h)")
        state = _WaterState(
            depletion_mm=float(row["depletion_mm"]),
            raw_mm=float(row["raw_mm"]),
            taw_mm=float(row["taw_mm"]),
            etc_24h_mm=etc,
            kc=float(row["kc"]),
            stage=str(row["growth_stage"]),
            soil_moisture=float(row["soil_moisture"]),
        )
        self._fill_state(rec, state)

        rule_needed, rule_depth, margin = self._rule(state, eff_rain)
        rec.rule_decision = rule_needed
        decision, depth = rule_needed, rule_depth
        pred = self.model.predict(feats.iloc[[-1]])[0] if self.model is not None else None
        if pred is not None and getattr(self.model, "has_classifier", True):
            rec.probability = round(pred.probability, 4)
            rec.ml_decision = pred.irrigation_needed
            rec.decision_source = "ml_hybrid"
            decision = pred.irrigation_needed
            rec.reasons.append(
                f"ML model ({self.model.version}) irrigation probability {pred.probability:.2f} "
                f"(decision threshold {self.model.threshold:.2f})"
            )
            rec.confidence = self._prob_confidence(pred.probability, self.model.threshold)
            if pred.irrigation_needed != rule_needed:
                rec.warnings.append("ML model and FAO-56 rule disagree - confidence reduced")
                rec.confidence = _downgrade(rec.confidence)
        else:
            rec.decision_source = "rule_based"
            rec.confidence = "high" if margin >= 0.3 else "medium" if margin >= 0.1 else "low"
            rec.reasons.append(
                "Decision by the FAO-56 water-balance rule"
                + (" (no trained model loaded)" if self.model is None else " (selected over ML on validation data)")
            )
        if pred is not None and pred.net_depth_mm is not None:
            depth = pred.net_depth_mm
            depth_reason = f"Net depth from the ML regressor: {pred.net_depth_mm:.1f} mm (FAO-56 formula: {rule_depth:.1f} mm)"
        else:
            depth_reason = f"Net depth from the FAO-56 refill formula: {rule_depth:.1f} mm"
        if bool(prep_row["sm_imputed"]):
            rec.confidence = _downgrade(rec.confidence)

        decision = self._safety(rec, state, decision, rain_now, rain_6h)
        if decision:
            rec.reasons.append(depth_reason)
        return self._finalise(field_cfg, rec, decision, depth, state)

    # -------------------------------------------------------------- helpers
    def _location(self, field_cfg: FieldConfig) -> dict[str, float]:
        """Providers that need a longitude (Open-Meteo) declare ``uses_longitude``."""
        if getattr(self.weather, "uses_longitude", False) and field_cfg.longitude is not None:
            return {"longitude": field_cfg.longitude}
        return {}

    def _add_weather_service_columns(self, frame: pd.DataFrame, field_cfg: FieldConfig, now: datetime) -> pd.DataFrame:
        """Attach hourly ET0 / wind from providers that observe them (Open-Meteo).

        Training on Open-Meteo data uses Penman-Monteith ET0; supplying the same
        variable here keeps inference identical. Without it, features fall back
        to the Hargreaves estimate from the sensor temperatures.
        """
        observed = getattr(self.weather, "observed_hourly", None)
        if observed is None or frame.empty or "et0_mm" in frame.columns:
            return frame
        start = pd.Timestamp(frame["timestamp"].min()).floor("h").to_pydatetime()
        obs = observed(field_cfg.latitude, start, now, **self._location(field_cfg))
        if obs is None or obs.empty:
            return frame
        obs = obs.assign(hour=pd.to_datetime(obs["timestamp"]).dt.floor("h")).drop(columns="timestamp")
        out = frame.assign(hour=pd.to_datetime(frame["timestamp"]).dt.floor("h")).merge(obs, on="hour", how="left")
        # ET0 is a per-hour flux: keep it on one reading per hour so the hourly sum is not inflated.
        out.loc[out["hour"].duplicated(), "et0_mm"] = np.nan
        return out.drop(columns="hour")

    @staticmethod
    def _normalise_readings(readings: pd.DataFrame | None) -> pd.DataFrame:
        cols = ["timestamp", "soil_moisture", "temperature_c", "humidity_pct", "rainfall_mm"]
        if readings is None or len(readings) == 0:
            return pd.DataFrame(columns=cols)
        df = readings.copy()
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        if getattr(df["timestamp"].dt, "tz", None) is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert(None)
        return df.sort_values("timestamp").reset_index(drop=True)

    def _etc(self, field_cfg: FieldConfig, row: pd.Series, forecast: WeatherForecast, now: datetime) -> float:
        """Crop ET for the next 24 h: sensor-based Hargreaves, else forecast temperatures."""
        etc = row.get("etc_24h_mm")
        if not is_missing(etc):
            return float(etc)
        crop = get_crop(field_cfg.crop_type)
        kc = crop_coefficient(crop, field_cfg.days_since_planting(now))
        et0 = float(hargreaves_et0(forecast.temp_min_c, forecast.temp_max_c, field_cfg.latitude, now.timetuple().tm_yday))
        return kc * (et0 if et0 > 0 else DEFAULT_ET0_MM)

    @staticmethod
    def _rule(state: _WaterState, eff_rain: float) -> tuple[bool, float, float]:
        """FAO-56 rule: irrigate if depletion already or soon exceeds RAW."""
        projected = projected_depletion(state.depletion_mm, state.etc_24h_mm, eff_rain)
        needed = state.depletion_mm >= state.raw_mm or projected >= state.raw_mm
        depth = net_irrigation_requirement(state.depletion_mm, state.etc_24h_mm, eff_rain, state.taw_mm)
        margin = abs(projected / state.raw_mm - 1.0) if state.raw_mm > 0 else 0.0
        return needed, depth, margin

    @staticmethod
    def _prob_confidence(p: float, threshold: float) -> str:
        distance = abs(p - threshold)
        return "high" if distance >= 0.3 else "medium" if distance >= 0.15 else "low"

    @staticmethod
    def _fill_state(rec: Recommendation, state: _WaterState) -> None:
        rec.soil_moisture_pct = None if state.soil_moisture is None else round(state.soil_moisture, 2)
        rec.depletion_mm = round(state.depletion_mm, 2)
        rec.raw_mm = round(state.raw_mm, 2)
        rec.taw_mm = round(state.taw_mm, 2)
        rec.depletion_ratio = round(state.depletion_mm / state.raw_mm, 3) if state.raw_mm > 0 else None
        rec.etc_24h_mm = round(state.etc_24h_mm, 2)
        rec.kc = round(state.kc, 3)
        rec.growth_stage = state.stage
        rec.reasons.append(
            f"Root-zone depletion {max(state.depletion_mm, 0):.1f} mm = {100 * max(state.depletion_mm, 0) / state.raw_mm:.0f}% "
            f"of the stress threshold RAW {state.raw_mm:.1f} mm (TAW {state.taw_mm:.1f} mm)"
        )
        rec.reasons.append(
            f"Estimated crop water use ETc {state.etc_24h_mm:.1f} mm/day (Kc {state.kc:.2f}, stage {state.stage})"
        )
        if rec.forecast_rain_24h_mm > 0:
            rec.reasons.append(
                f"Rain forecast {rec.forecast_rain_24h_mm:.1f} mm in 24 h ({rec.forecast_source}), "
                f"effective {rec.forecast_effective_rain_mm:.1f} mm credited"
            )

    def _safety(self, rec: Recommendation, state: _WaterState, decision: bool, rain_now: float, rain_6h: float) -> bool:
        severe_level = state.raw_mm + SEVERE_STRESS_POSITION * (state.taw_mm - state.raw_mm)
        severe = state.depletion_mm >= severe_level
        if state.depletion_mm <= 0:
            if decision:
                rec.reasons.append("Safety rule: soil is at/above field capacity - irrigation would only drain away")
            rec.next_check_hours = 24
            return False
        if (rain_now >= RAINING_NOW_MM_PER_HOUR or rain_6h >= RAIN_LAST_6H_DEFER_MM) and not severe:
            if decision:
                rec.reasons.append(
                    f"Rain-aware deferral: {rain_6h:.1f} mm fell in the last 6 h - re-check after infiltration"
                )
            rec.next_check_hours = 3
            return False
        if severe and not decision:
            rec.reasons.append(
                f"Safety override: depletion {state.depletion_mm:.1f} mm exceeds the severe-stress level "
                f"{severe_level:.1f} mm - irrigating to protect the crop"
            )
            rec.warnings.append("Severe water stress detected")
            return True
        return decision

    def _finalise(
        self, field_cfg: FieldConfig, rec: Recommendation, decision: bool, depth: float, state: _WaterState
    ) -> Recommendation:
        rec.irrigation_needed = bool(decision)
        if decision:
            depth = float(min(max(depth, MIN_NET_DEPTH_MM), state.taw_mm))
            plan = plan_irrigation(
                depth, field_cfg.irrigation_method, field_cfg.area_m2, field_cfg.flow_rate_lpm,
                field_cfg.max_irrigation_minutes,
            )
            rec.net_depth_mm = round(plan.net_depth_mm, 2)
            rec.gross_depth_mm = round(plan.gross_depth_mm, 2)
            rec.water_volume_l = round(plan.water_volume_l, 1)
            rec.duration_min = round(plan.duration_min, 1)
            rec.capped = plan.capped
            rec.reasons.append(
                f"Apply {plan.net_depth_mm:.1f} mm net = {plan.gross_depth_mm:.1f} mm gross at "
                f"{plan.efficiency:.0%} {field_cfg.irrigation_method} efficiency over {field_cfg.area_m2:.0f} m2"
            )
            if plan.capped:
                rec.warnings.append(
                    f"Run time capped at {field_cfg.max_irrigation_minutes:.0f} min; remaining need will be "
                    "re-evaluated at the next check"
                )
            rec.next_check_hours = 24
            rec.summary = (
                f"Irrigate now: run {plan.duration_min:.0f} min (~{plan.water_volume_l:,.0f} L, "
                f"{plan.gross_depth_mm:.1f} mm)"
            )
        else:
            if rec.next_check_hours == 12 and rec.depletion_ratio is not None and rec.depletion_ratio > 0.7:
                rec.next_check_hours = 6
            rec.summary = f"No irrigation needed now - re-check in {rec.next_check_hours} h"
        return rec

    # ------------------------------------------------------------ fallbacks
    def _estimated(
        self,
        field_cfg: FieldConfig,
        rec: Recommendation,
        prepared: pd.DataFrame,
        now: datetime,
        forecast: WeatherForecast,
        etc: float,
        rain_now: float,
        rain_6h: float,
    ) -> Recommendation:
        """Soil sensor unusable now: project depletion from the last valid reading."""
        valid = prepared[~prepared["sm_invalid"] & prepared["soil_moisture"].notna()]
        if valid.empty:
            return self._conservative(field_cfg, rec, now, forecast, rain_now)
        last = valid.iloc[-1]
        last_ts = pd.Timestamp(last["timestamp"]).to_pydatetime()
        age_h = (now - last_ts).total_seconds() / 3600.0
        if age_h > ESTIMATION_MAX_AGE_HOURS:
            rec.warnings.append(f"Last valid soil reading is {age_h:.0f} h old - too old to extrapolate")
            return self._conservative(field_cfg, rec, now, forecast, rain_now)

        crop, soil = get_crop(field_cfg.crop_type), get_soil(field_cfg.soil_type)
        days = field_cfg.days_since_planting(now)
        zr = root_depth(crop, days)
        taw = 1000.0 * (soil.field_capacity - soil.wilting_point) * zr
        raw = crop.depletion_fraction * taw
        dr_last = 1000.0 * (soil.field_capacity - float(last["soil_moisture"]) / 100.0) * zr
        rain_since = float(prepared.loc[prepared["timestamp"] > last["timestamp"], "rainfall_mm"].sum())
        dr_now = max(0.0, dr_last) + etc * age_h / 24.0 - effective_rainfall(rain_since)
        est_sm = 100.0 * (soil.field_capacity - dr_now / (1000.0 * zr))
        state = _WaterState(dr_now, raw, taw, etc, crop_coefficient(crop, days), growth_stage(crop, days), est_sm)

        rec.decision_source = "fallback_estimated"
        rec.soil_moisture_estimated = True
        rec.reasons.append(
            f"Soil sensor reading unusable - depletion estimated from last valid reading "
            f"({last_ts:%Y-%m-%d %H:%M}, {age_h:.1f} h ago) with a water balance"
        )
        self._fill_state(rec, state)
        decision, depth, _ = self._rule(state, rec.forecast_effective_rain_mm)
        rec.rule_decision = decision
        rec.confidence = "low"
        rec.warnings.append("Decision based on estimated soil moisture - inspect the soil sensor")
        decision = self._safety(rec, state, decision, rain_now, rain_6h)
        return self._finalise(field_cfg, rec, decision, depth, state)

    def _conservative(
        self, field_cfg: FieldConfig, rec: Recommendation, now: datetime, forecast: WeatherForecast, rain_now: float
    ) -> Recommendation:
        """No usable soil data: replace at most one day of crop water use."""
        crop = get_crop(field_cfg.crop_type)
        days = field_cfg.days_since_planting(now)
        kc = crop_coefficient(crop, days)
        et0 = float(hargreaves_et0(forecast.temp_min_c, forecast.temp_max_c, field_cfg.latitude, now.timetuple().tm_yday))
        etc = kc * (et0 if et0 > 0 else DEFAULT_ET0_MM)
        rec.decision_source = "fallback_conservative"
        rec.confidence = "low"
        rec.etc_24h_mm = round(etc, 2)
        rec.kc = round(kc, 3)
        rec.growth_stage = growth_stage(crop, days)
        rec.soil_moisture_estimated = True
        rec.warnings.append("No usable soil-moisture data in the last 24 h - check sensor power and wiring")
        rec.reasons.append(
            f"Conservative fallback: without soil data, at most one day of crop water use "
            f"(ETc {etc:.1f} mm from forecast temperatures) is replaced"
        )
        if rec.forecast_effective_rain_mm >= etc or rain_now >= RAINING_NOW_MM_PER_HOUR:
            rec.reasons.append("Expected/ongoing rain covers the daily crop water use - skipping irrigation")
            rec.next_check_hours = 6
            rec.irrigation_needed = False
            rec.summary = f"No irrigation (rain expected) - fix sensor; re-check in {rec.next_check_hours} h"
            return rec
        net = max(MIN_NET_DEPTH_MM, etc - rec.forecast_effective_rain_mm)
        plan = plan_irrigation(net, field_cfg.irrigation_method, field_cfg.area_m2, field_cfg.flow_rate_lpm,
                               field_cfg.max_irrigation_minutes)
        rec.irrigation_needed = True
        rec.net_depth_mm = round(plan.net_depth_mm, 2)
        rec.gross_depth_mm = round(plan.gross_depth_mm, 2)
        rec.water_volume_l = round(plan.water_volume_l, 1)
        rec.duration_min = round(plan.duration_min, 1)
        rec.capped = plan.capped
        rec.next_check_hours = 6
        rec.summary = (
            f"Sensor problem - conservative irrigation {plan.duration_min:.0f} min (~{plan.water_volume_l:,.0f} L); "
            "inspect the sensor"
        )
        return rec


def _downgrade(confidence: str) -> str:
    return {"high": "medium", "medium": "low", "low": "low"}[confidence]


def field_config_from_row(row: dict[str, Any]) -> FieldConfig:
    """Build a :class:`FieldConfig` from a database row / dict."""
    planting = row["planting_date"]
    if isinstance(planting, str):
        planting = date.fromisoformat(planting[:10])
    return FieldConfig(
        field_id=str(row.get("id", row.get("field_id"))),
        name=str(row.get("name", "")),
        crop_type=row["crop_type"],
        soil_type=row["soil_type"],
        irrigation_method=row["irrigation_method"],
        planting_date=planting,
        area_m2=float(row["area_m2"]),
        flow_rate_lpm=float(row["flow_rate_lpm"]),
        latitude=float(row["latitude"]),
        max_irrigation_minutes=float(row.get("max_irrigation_minutes") or 180.0),
        longitude=None if row.get("longitude") is None else float(row["longitude"]),
    )


__all__ = ["FieldConfig", "Recommendation", "RecommendationEngine", "field_config_from_row"]
