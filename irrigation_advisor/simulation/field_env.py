"""Virtual field used for closed-loop strategy evaluation.

Unlike the training data generator (which replays a simulated farmer), here
the *strategy under test* controls irrigation and the field responds:

    weather + true soil bucket  ->  noisy sensors  ->  strategy decision
            ^                                               |
            +------------------ applied water --------------+

All strategies on the same field share identical weather, sensor noise and
sensor faults (common random numbers), so differences are caused by the
strategies alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..agronomy.crops import get_crop, get_method, get_soil
from ..agronomy.soil_model import SoilWaterModel
from ..data.climate import generate_hourly_weather, get_climate, simulate_rain_forecast
from ..data.simulator import FieldSpec, inject_sensor_faults
from ..engine.recommender import FieldConfig
from .strategies import DecisionContext, Strategy

DECISION_HOUR = 6
HISTORY_HOURS = 72
WARMUP_DAYS = 3


@dataclass
class SimulationResult:
    """Season totals and the hourly trace of one strategy on one field."""

    field_id: str
    strategy: str
    metrics: dict[str, float]
    trace: pd.DataFrame


class VirtualField:
    """Pre-generates weather and sensor noise for one field and season."""

    def __init__(self, spec: FieldSpec, climate_name: str = "semi_arid_monsoon", fault_rate: float = 1.0) -> None:
        self.spec = spec
        self.crop = get_crop(spec.crop_type)
        self.soil = get_soil(spec.soil_type)
        self.efficiency = get_method(spec.irrigation_method).efficiency
        rng = np.random.default_rng(spec.seed + 7)
        start = spec.planting_date - timedelta(days=WARMUP_DAYS)
        n_days = WARMUP_DAYS + self.crop.season_days + 1
        self.weather = generate_hourly_weather(get_climate(climate_name), spec.latitude, start, n_days, spec.seed + 7)
        n = len(self.weather)
        self.timestamps = pd.to_datetime(self.weather["timestamp"]).to_numpy()
        planting = datetime(spec.planting_date.year, spec.planting_date.month, spec.planting_date.day)
        self.days = (pd.to_datetime(self.timestamps) - planting).total_seconds().to_numpy() / 86400.0
        self.rain = self.weather["rainfall_mm"].to_numpy()
        self.et0 = self.weather["et0_true_mm"].to_numpy()
        # Sensor noise and faults are drawn once and shared by all strategies.
        self.sm_noise = spec.sensor_bias + rng.normal(0, 0.7, n)
        self.temp_obs = np.round(self.weather["temperature_c"].to_numpy() + rng.normal(0, 0.3, n), 1)
        self.hum_obs = np.round(np.clip(self.weather["humidity_pct"].to_numpy() + rng.normal(0, 2, n), 0, 100), 1)
        self.rain_obs = np.round(self.rain / 0.2) * 0.2
        fault_probe = np.zeros(n)
        self.fault_sm, self.temp_obs, self.fault = inject_sensor_faults(fault_probe, self.temp_obs, rng, fault_rate)
        next24 = pd.Series(self.rain[::-1]).rolling(24, min_periods=1).sum().to_numpy()[::-1]
        next24 = np.append(next24[1:], 0.0)
        self.forecast = simulate_rain_forecast(next24, rng)
        self.initial_theta = self.soil.field_capacity * 0.95
        self.field_cfg = FieldConfig(
            field_id=spec.field_id,
            crop_type=spec.crop_type,
            soil_type=spec.soil_type,
            irrigation_method=spec.irrigation_method,
            planting_date=spec.planting_date,
            area_m2=spec.area_m2,
            flow_rate_lpm=spec.flow_rate_lpm,
            latitude=spec.latitude,
            name=spec.field_id,
            max_irrigation_minutes=360.0,
        )

    def _sensor_value(self, i: int, theta: float) -> float:
        """Observed soil moisture: true value + bias/noise, then the shared fault pattern."""
        fault = self.fault[i]
        if fault == "dropout":
            return float("nan")
        if fault in ("spike", "out_of_range"):
            # Fault array was generated on a zero baseline; re-apply its offset/value.
            return float(self.fault_sm[i]) if fault == "out_of_range" else round(theta * 100 + self.fault_sm[i], 1)
        return round(theta * 100 + self.sm_noise[i], 1)

    def run(self, strategy: Strategy) -> SimulationResult:
        """Simulate the full season under ``strategy``."""
        n = len(self.timestamps)
        model = SoilWaterModel(self.soil, self.crop, self.initial_theta)
        sm_obs = np.full(n, np.nan)
        true_sm = np.empty(n)
        ks = np.empty(n)
        et_act = np.empty(n)
        et_pot = np.empty(n)
        drain = np.empty(n)
        raw_sm = np.empty(n)
        irrigation_gross = np.zeros(n)
        queue_net = 0.0
        events = 0
        pump_minutes = 0.0
        stuck_value: float | None = None

        for i in range(n):
            ts = pd.Timestamp(self.timestamps[i]).to_pydatetime()
            if ts.hour == DECISION_HOUR and self.days[i] >= 0:
                lo = max(0, i - HISTORY_HOURS)
                history = pd.DataFrame(
                    {
                        "timestamp": self.timestamps[lo:i],
                        "soil_moisture": sm_obs[lo:i],
                        "temperature_c": self.temp_obs[lo:i],
                        "humidity_pct": self.hum_obs[lo:i],
                        "rainfall_mm": self.rain_obs[lo:i],
                    }
                )
                ctx = DecisionContext(ts, self.field_cfg, int(self.days[i]), history, float(self.forecast[i - 1]))
                decision = strategy.decide(ctx)
                if decision.gross_depth_mm > 0:
                    events += 1
                    pump_minutes += decision.duration_min
                    irrigation_gross[i] = decision.gross_depth_mm
                    queue_net += decision.gross_depth_mm * self.efficiency

            # Water reaches the root zone at the method's application rate.
            rate = self.field_cfg.flow_rate_lpm * 60.0 / self.field_cfg.area_m2 * self.efficiency
            applied = min(queue_net, rate)
            queue_net -= applied
            res = model.step(self.rain[i], applied, self.et0[i], self.days[i])
            zr, _, raw, _ = model.status(self.days[i])
            true_sm[i] = model.theta * 100
            raw_sm[i] = 100 * (self.soil.field_capacity - raw / (1000 * zr))
            ks[i], et_act[i], et_pot[i], drain[i] = res.ks, res.et_actual_mm, res.et_potential_mm, res.drainage_mm

            obs = self._sensor_value(i, model.theta)
            if self.fault[i] == "stuck":
                stuck_value = stuck_value if stuck_value is not None else (sm_obs[i - 1] if i else obs)
                obs = stuck_value
            else:
                stuck_value = None
            sm_obs[i] = obs

        season = self.days >= 0
        gross_mm = float(irrigation_gross[season].sum())
        metrics = {
            "irrigation_gross_mm": round(gross_mm, 1),
            "water_m3_per_ha": round(gross_mm * 10.0, 1),  # 1 mm over 1 ha = 10 m3
            "water_volume_m3": round(gross_mm * self.field_cfg.area_m2 / 1000.0, 2),
            "irrigation_events": events,
            "pump_hours": round(pump_minutes / 60.0, 1),
            "stress_days": round(float((ks[season] < 0.999).sum()) / 24.0, 2),
            "severe_stress_days": round(float((ks[season] < 0.5).sum()) / 24.0, 2),
            "relative_transpiration": round(float(et_act[season].sum() / max(et_pot[season].sum(), 1e-9)), 4),
            "deep_drainage_mm": round(float(drain[season].sum()), 1),
            "rain_mm": round(float(self.rain[season].sum()), 1),
            "crop_et_potential_mm": round(float(et_pot[season].sum()), 1),
        }
        trace = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(self.timestamps),
                "true_soil_moisture": true_sm,
                "sensor_soil_moisture": sm_obs,
                "stress_threshold_sm": raw_sm,
                "ks": ks,
                "irrigation_gross_mm": irrigation_gross,
                "rain_mm": self.rain,
            }
        )[season].reset_index(drop=True)
        return SimulationResult(self.spec.field_id, strategy.name, metrics, trace)
