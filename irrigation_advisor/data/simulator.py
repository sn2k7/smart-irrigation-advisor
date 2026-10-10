"""Synthetic agricultural dataset generator (clearly labelled SYNTHETIC).

Each virtual field is simulated hour by hour:

1. stochastic weather (``data.climate``) for the whole crop season;
2. a 'true' soil water bucket (``agronomy.soil_model``) driven by rain,
   crop ET and the irrigation decisions of a simulated *farmer*;
3. low-cost sensors observing that state with calibration bias, noise,
   quantisation and injected faults (dropouts, spikes, stuck values,
   impossible values);
4. an imperfect 24 h rain forecast;
5. ground-truth labels from a counterfactual no-irrigation projection
   (see ``data.labels`` for the definitions).

The historical farmer behaviour is varied (fixed-interval vs. visual
inspection, different depths) so the dataset covers dry, wet and well
watered states. Nothing here is real-world data; results obtained on it
demonstrate the *method*, not field performance.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from ..agronomy.crops import CROPS, IRRIGATION_METHODS, SOILS, get_crop, get_method, get_soil
from ..agronomy.soil_model import SoilWaterModel, infiltrated_rain, project_max_depletion
from ..agronomy.water_balance import crop_coefficient, net_irrigation_requirement
from .climate import generate_hourly_weather, get_climate, simulate_rain_forecast
from .labels import HORIZON_HOURS, TARGET_CLASS, TARGET_REG

logger = logging.getLogger(__name__)

WARMUP_DAYS = 3
SENSOR_NOISE_SD = 0.7  # % VWC, typical capacitive probe after calibration
RAIN_GAUGE_RESOLUTION_MM = 0.2  # tipping-bucket resolution


@dataclass(frozen=True)
class FieldSpec:
    """Static description of one simulated field and its farmer."""

    field_id: str
    crop_type: str
    soil_type: str
    irrigation_method: str
    planting_date: date
    latitude: float
    area_m2: float
    flow_rate_lpm: float
    farmer_policy: str  # "fixed_interval" | "visual_check"
    farmer_interval_days: int
    farmer_depth_mm: float  # gross depth applied per irrigation by the farmer
    sensor_bias: float  # % VWC calibration offset of the probe
    seed: int


def random_field_specs(n_fields: int, rng: np.random.Generator, year: int = 2025) -> list[FieldSpec]:
    """Draw diverse field configurations."""
    crops, soils = sorted(CROPS), sorted(SOILS)
    methods = sorted(IRRIGATION_METHODS)
    # Typical application rates (mm/h); pump flow = rate * area / 60.
    rate_ranges = {"drip": (2.0, 5.0), "sprinkler": (6.0, 12.0), "furrow": (20.0, 40.0)}
    specs = []
    for i in range(n_fields):
        method = str(rng.choice(methods))
        area = round(float(rng.uniform(500, 5000)), 0)
        planting = date(year, 1, 1) + timedelta(days=int(rng.integers(0, 300)))
        specs.append(
            FieldSpec(
                field_id=f"F{i + 1:03d}",
                crop_type=crops[i % len(crops)] if i < len(crops) else str(rng.choice(crops)),
                soil_type=str(rng.choice(soils)),
                irrigation_method=method,
                planting_date=planting,
                latitude=round(float(rng.uniform(15.0, 22.0)), 2),
                area_m2=area,
                flow_rate_lpm=round(float(rng.uniform(*rate_ranges[method])) * area / 60.0, 1),
                farmer_policy=str(rng.choice(["fixed_interval", "visual_check"], p=[0.55, 0.45])),
                farmer_interval_days=int(rng.integers(3, 9)),
                farmer_depth_mm=round(float(rng.uniform(10, 35)), 1),
                sensor_bias=round(float(rng.normal(0.0, 1.0)), 2),
                seed=int(rng.integers(0, 2**31 - 1)),
            )
        )
    return specs


def _farmer_irrigation(
    spec: FieldSpec, model: SoilWaterModel, day_index: int, days: float, recent_rain_mm: float, rng: np.random.Generator
) -> float:
    """Net irrigation (mm) the simulated farmer applies at 07:00 on this day."""
    efficiency = get_method(spec.irrigation_method).efficiency
    if spec.farmer_policy == "fixed_interval":
        if day_index % spec.farmer_interval_days != 0:
            return 0.0
        if rng.random() < 0.15:  # labour / power not available that day
            return 0.0
        # Some farmers skip after a good rain, others do not.
        if recent_rain_mm > 10 and rng.random() < 0.6:
            return 0.0
        return spec.farmer_depth_mm * efficiency
    # visual_check: irrigate when the field 'looks' dry (noisy perception of stress)
    _, _, raw, dr = model.status(days)
    if dr > raw * rng.uniform(0.9, 1.8) and rng.random() > 0.15:
        return max(0.0, dr) * rng.uniform(0.5, 1.0)
    return 0.0


def inject_sensor_faults(sm: np.ndarray, temp: np.ndarray, rng: np.random.Generator, fault_rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply realistic sensor faults; returns (soil moisture, temperature, fault labels)."""
    n = len(sm)
    sm, temp = sm.copy(), temp.copy()
    fault = np.array(["none"] * n, dtype=object)
    if fault_rate <= 0:
        return sm, temp, fault

    # Random single-hour dropouts (radio loss) and a few multi-hour outages.
    drop = rng.random(n) < 0.015 * fault_rate
    for _ in range(rng.poisson(2 * fault_rate)):
        start = int(rng.integers(0, max(1, n - 10)))
        drop[start : start + int(rng.integers(2, 9))] = True
    sm[drop] = np.nan
    fault[drop] = "dropout"

    # Spikes (loose connector / EMI).
    spikes = (rng.random(n) < 0.004 * fault_rate) & ~drop
    sm[spikes] = sm[spikes] + rng.choice([-1, 1], spikes.sum()) * rng.uniform(15, 30, spikes.sum())
    fault[spikes] = "spike"

    # Stuck sensor episode.
    if rng.random() < 0.5 * fault_rate and n > 60:
        start = int(rng.integers(24, n - 40))
        length = int(rng.integers(12, 37))
        sm[start : start + length] = sm[start - 1] if not np.isnan(sm[start - 1]) else 20.0
        fault[start : start + length] = "stuck"

    # Physically impossible values (ADC errors).
    bad = (rng.random(n) < 0.002 * fault_rate) & (fault == "none")
    sm[bad] = rng.choice([-5.0, 99.9], bad.sum())
    fault[bad] = "out_of_range"
    bad_t = rng.random(n) < 0.001 * fault_rate
    temp[bad_t] = 85.0  # classic DHT22 error value
    return sm, temp, fault


def simulate_field(spec: FieldSpec, climate_name: str = "semi_arid_monsoon", fault_rate: float = 1.0) -> pd.DataFrame:
    """Simulate one complete crop season for ``spec`` (hourly rows)."""
    rng = np.random.default_rng(spec.seed)
    crop, soil = get_crop(spec.crop_type), get_soil(spec.soil_type)
    climate = get_climate(climate_name)
    start = spec.planting_date - timedelta(days=WARMUP_DAYS)
    n_days = WARMUP_DAYS + crop.season_days + 2  # +2 days so labels have a future window
    weather = generate_hourly_weather(climate, spec.latitude, start, n_days, spec.seed)

    ts = weather["timestamp"].to_numpy()
    rain = weather["rainfall_mm"].to_numpy()
    et0 = weather["et0_true_mm"].to_numpy()
    n = len(weather)
    planting_dt = datetime(spec.planting_date.year, spec.planting_date.month, spec.planting_date.day)
    days = (pd.to_datetime(ts) - planting_dt).total_seconds().to_numpy() / 86400.0

    model = SoilWaterModel(soil, crop, theta=soil.field_capacity * rng.uniform(0.85, 1.0))
    theta = np.empty(n)
    depletion = np.empty(n)
    raw_arr = np.empty(n)
    taw_arr = np.empty(n)
    irrigation = np.zeros(n)
    et_actual = np.empty(n)
    stress = np.empty(n)

    # Remaining farmer irrigation queue (net mm per hour).
    pending = 0.0
    for i in range(n):
        hour = pd.Timestamp(ts[i]).hour
        if hour == 7 and days[i] >= 0:
            day_index = int(days[i])
            recent_rain = float(rain[max(0, i - 24) : i].sum())
            pending += _farmer_irrigation(spec, model, day_index, days[i], recent_rain, rng)
        applied = min(pending, 25.0)  # pump capacity limits mm per hour
        pending -= applied
        irrigation[i] = applied
        res = model.step(rain[i], applied, et0[i], days[i])
        zr, taw, raw, dr = model.status(days[i])
        theta[i], depletion[i], raw_arr[i], taw_arr[i] = model.theta, dr, raw, taw
        et_actual[i], stress[i] = res.et_actual_mm, res.ks

    # --- ground-truth labels -------------------------------------------------
    needed = np.full(n, np.nan)
    target_mm = np.full(n, np.nan)
    for i in range(n - HORIZON_HOURS):
        fut = slice(i + 1, i + 1 + HORIZON_HOURS)
        max_dr, _ = project_max_depletion(theta[i], soil, crop, days[i], rain[fut], et0[fut])
        need = max_dr >= raw_arr[i]
        etc_next = crop_coefficient(crop, days[i]) * float(et0[fut].sum())
        eff_rain_next = float(sum(infiltrated_rain(r) for r in rain[fut]))
        needed[i] = float(need)
        target_mm[i] = net_irrigation_requirement(depletion[i], etc_next, eff_rain_next, taw_arr[i]) if need else 0.0

    # --- sensors -----------------------------------------------------------
    sm_true_pct = theta * 100.0
    sm_sensor = np.round(sm_true_pct + spec.sensor_bias + rng.normal(0, SENSOR_NOISE_SD, n), 1)
    temp_sensor = np.round(weather["temperature_c"].to_numpy() + rng.normal(0, 0.3, n), 1)
    hum_sensor = np.round(np.clip(weather["humidity_pct"].to_numpy() + rng.normal(0, 2.0, n), 0, 100), 1)
    rain_sensor = np.round(rain / RAIN_GAUGE_RESOLUTION_MM) * RAIN_GAUGE_RESOLUTION_MM
    sm_sensor, temp_sensor, fault = inject_sensor_faults(sm_sensor, temp_sensor, rng, fault_rate)

    actual_next24 = pd.Series(rain[::-1]).rolling(HORIZON_HOURS, min_periods=1).sum().to_numpy()[::-1]
    actual_next24 = np.append(actual_next24[1:], 0.0)  # rain in (t, t+24]
    forecast = simulate_rain_forecast(actual_next24, rng)

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "field_id": spec.field_id,
            "crop_type": spec.crop_type,
            "soil_type": spec.soil_type,
            "irrigation_method": spec.irrigation_method,
            "planting_date": spec.planting_date.isoformat(),
            "latitude": spec.latitude,
            "area_m2": spec.area_m2,
            "flow_rate_lpm": spec.flow_rate_lpm,
            "soil_moisture": sm_sensor,
            "temperature_c": temp_sensor,
            "humidity_pct": hum_sensor,
            "rainfall_mm": np.round(rain_sensor, 2),
            "forecast_rain_24h_mm": forecast,
            "irrigation_applied_mm": np.round(irrigation, 2),
            "sensor_fault": fault,
            "true_soil_moisture": np.round(sm_true_pct, 2),
            "true_depletion_mm": np.round(depletion, 2),
            "true_raw_mm": np.round(raw_arr, 2),
            "true_et_actual_mm": np.round(et_actual, 4),
            "true_ks": np.round(stress, 3),
            TARGET_CLASS: needed,
            TARGET_REG: np.round(target_mm, 2),
        }
    )
    # Keep the crop season only (warm-up hours are needed for the bucket state).
    mask = (days >= 0) & ~np.isnan(needed)
    return df.loc[mask].reset_index(drop=True)


def generate_dataset(
    n_fields: int = 16,
    seed: int = 42,
    climate_name: str = "semi_arid_monsoon",
    year: int = 2025,
    fault_rate: float = 1.0,
) -> tuple[pd.DataFrame, list[FieldSpec]]:
    """Generate the full synthetic dataset.

    Returns:
        (hourly dataset, list of field specs used)
    """
    if n_fields < 1:
        raise ValueError("n_fields must be >= 1")
    rng = np.random.default_rng(seed)
    specs = random_field_specs(n_fields, rng, year)
    frames = []
    for spec in specs:
        logger.info("Simulating field %s (%s on %s, %s)", spec.field_id, spec.crop_type, spec.soil_type, spec.farmer_policy)
        frames.append(simulate_field(spec, climate_name, fault_rate))
    data = pd.concat(frames, ignore_index=True)
    data[TARGET_CLASS] = data[TARGET_CLASS].astype(int)
    return data, specs


def specs_to_frame(specs: list[FieldSpec]) -> pd.DataFrame:
    """Field specifications as a table (saved alongside the dataset)."""
    rows = []
    for spec in specs:
        row = asdict(spec)
        row["planting_date"] = spec.planting_date.isoformat()
        rows.append(row)
    return pd.DataFrame(rows)
