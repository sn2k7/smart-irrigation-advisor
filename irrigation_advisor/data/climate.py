"""Stochastic weather generator used for synthetic data and simulated forecasts.

The generator is a classic two-state (wet/dry) first-order Markov chain for
rain occurrence with gamma-distributed daily rain amounts - the structure of
widely used weather generators such as WGEN/LARS-WG - combined with seasonal
temperature/humidity cycles and simple diurnal shapes.

Everything is seeded so results are reproducible. **All weather produced
here is synthetic** and only meant to exercise the system offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from ..agronomy.water_balance import hargreaves_et0, hourly_et0_fraction


@dataclass(frozen=True)
class ClimateProfile:
    """Parameters of the stochastic climate."""

    name: str
    tmax_annual_mean: float
    tmax_amplitude: float
    tmax_peak_doy: int
    diurnal_range_dry: float
    diurnal_range_wet: float
    rh_dry_season: float
    rh_wet_season: float
    wet_season_start_doy: int
    wet_season_end_doy: int
    p_wet_after_dry_wet_season: float
    p_wet_after_wet_wet_season: float
    p_wet_after_dry_dry_season: float
    p_wet_after_wet_dry_season: float
    rain_mean_mm_wet_season: float
    rain_mean_mm_dry_season: float
    rain_gamma_shape: float = 0.75


CLIMATES: dict[str, ClimateProfile] = {
    # Loosely representative of the Deccan plateau (India): hot pre-monsoon,
    # June-September monsoon, dry winter.
    "semi_arid_monsoon": ClimateProfile(
        name="semi_arid_monsoon",
        tmax_annual_mean=32.0,
        tmax_amplitude=5.5,
        tmax_peak_doy=130,
        diurnal_range_dry=14.0,
        diurnal_range_wet=8.0,
        rh_dry_season=42.0,
        rh_wet_season=78.0,
        wet_season_start_doy=158,
        wet_season_end_doy=275,
        p_wet_after_dry_wet_season=0.45,
        p_wet_after_wet_wet_season=0.70,
        p_wet_after_dry_dry_season=0.05,
        p_wet_after_wet_dry_season=0.25,
        rain_mean_mm_wet_season=11.0,
        rain_mean_mm_dry_season=5.0,
    ),
    # Wetter, milder sub-tropical climate for robustness experiments.
    "humid_subtropical": ClimateProfile(
        name="humid_subtropical",
        tmax_annual_mean=28.0,
        tmax_amplitude=6.0,
        tmax_peak_doy=170,
        diurnal_range_dry=11.0,
        diurnal_range_wet=7.0,
        rh_dry_season=60.0,
        rh_wet_season=82.0,
        wet_season_start_doy=120,
        wet_season_end_doy=290,
        p_wet_after_dry_wet_season=0.35,
        p_wet_after_wet_wet_season=0.60,
        p_wet_after_dry_dry_season=0.15,
        p_wet_after_wet_dry_season=0.40,
        rain_mean_mm_wet_season=9.0,
        rain_mean_mm_dry_season=6.0,
    ),
}


def get_climate(name: str) -> ClimateProfile:
    """Return a climate profile by name."""
    if name not in CLIMATES:
        raise KeyError(f"Unknown climate profile '{name}'. Valid: {sorted(CLIMATES)}")
    return CLIMATES[name]


def wet_season_weight(climate: ClimateProfile, doy: int) -> float:
    """Smooth 0..1 indicator of the rainy season (10-day cosine ramps)."""
    ramp = 10.0
    start, end = climate.wet_season_start_doy, climate.wet_season_end_doy
    if doy < start - ramp or doy > end + ramp:
        return 0.0
    if doy < start + ramp:
        return 0.5 - 0.5 * np.cos(np.pi * (doy - (start - ramp)) / (2 * ramp))
    if doy > end - ramp:
        return 0.5 + 0.5 * np.cos(np.pi * (doy - (end - ramp)) / (2 * ramp))
    return 1.0


def rain_probability(climate: ClimateProfile, doy: int, previous_wet: bool) -> float:
    """Markov-chain probability that the day is wet."""
    w = wet_season_weight(climate, doy)
    if previous_wet:
        return w * climate.p_wet_after_wet_wet_season + (1 - w) * climate.p_wet_after_wet_dry_season
    return w * climate.p_wet_after_dry_wet_season + (1 - w) * climate.p_wet_after_dry_dry_season


def generate_daily_weather(
    climate: ClimateProfile, latitude: float, start: date, n_days: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate daily Tmax/Tmin/RH/rain and the 'true' daily ET0 (mm)."""
    records: list[dict] = []
    wet = False
    t_anomaly = 0.0
    for i in range(n_days):
        day = start + timedelta(days=i)
        doy = day.timetuple().tm_yday
        w = wet_season_weight(climate, doy)
        wet = bool(rng.random() < rain_probability(climate, doy, wet))
        rain = 0.0
        if wet:
            mean = w * climate.rain_mean_mm_wet_season + (1 - w) * climate.rain_mean_mm_dry_season
            rain = float(rng.gamma(climate.rain_gamma_shape, mean / climate.rain_gamma_shape))
            rain = max(rain, 0.3)
        # AR(1) temperature anomaly gives realistic warm/cool spells.
        t_anomaly = 0.7 * t_anomaly + rng.normal(0.0, 1.2)
        seasonal = climate.tmax_amplitude * np.cos(2 * np.pi * (doy - climate.tmax_peak_doy) / 365.0)
        tmax = climate.tmax_annual_mean + seasonal - 2.5 * w + t_anomaly - (2.5 if wet else 0.0)
        drange = w * climate.diurnal_range_wet + (1 - w) * climate.diurnal_range_dry
        drange = drange - (2.0 if wet else 0.0) + rng.normal(0.0, 1.0)
        tmin = tmax - max(3.0, drange)
        rh = w * climate.rh_wet_season + (1 - w) * climate.rh_dry_season + (10.0 if wet else 0.0)
        rh = float(np.clip(rh + rng.normal(0.0, 5.0), 15.0, 98.0))
        et0 = float(hargreaves_et0(tmin, tmax, latitude, doy))
        # The 'true' ET0 differs from Hargreaves: humid/cloudy days evaporate
        # less. Models only ever see Hargreaves estimates, as in the field.
        humidity_factor = float(np.clip(1.0 + 0.3 * (0.55 - rh / 100.0), 0.8, 1.15))
        et0_true = et0 * humidity_factor * (0.65 if rain > 5 else 1.0) * rng.uniform(0.92, 1.08)
        records.append(
            {
                "date": day,
                "doy": doy,
                "tmax": tmax,
                "tmin": tmin,
                "rh_mean": rh,
                "rain_mm": rain,
                "et0_true_mm": et0_true,
            }
        )
    return pd.DataFrame.from_records(records)


def _temperature_shape(hour: np.ndarray) -> np.ndarray:
    """0..1 diurnal temperature shape: minimum at 06:00, maximum at 15:00."""
    h = hour.astype(float)
    rising = (h >= 6) & (h <= 15)
    shape = np.empty_like(h)
    shape[rising] = (1 - np.cos(np.pi * (h[rising] - 6) / 9.0)) / 2.0
    after = np.where(h > 15, h - 15, h + 24 - 15)
    shape[~rising] = (1 + np.cos(np.pi * after[~rising] / 15.0)) / 2.0
    return shape


def expand_to_hourly(daily: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Disaggregate daily weather to hourly temperature, RH, rain and true ET0."""
    hours = np.arange(24)
    t_shape = _temperature_shape(hours)
    et_fraction = np.array([hourly_et0_fraction(int(h)) for h in hours])
    frames: list[pd.DataFrame] = []
    for row in daily.itertuples(index=False):
        temp = row.tmin + (row.tmax - row.tmin) * t_shape + rng.normal(0, 0.3, 24)
        rh = np.clip(row.rh_mean + 30.0 * (0.5 - t_shape) + rng.normal(0, 2.0, 24), 10.0, 100.0)
        rain = np.zeros(24)
        if row.rain_mm > 0:
            duration = int(rng.integers(1, 6))
            if rng.random() < 0.7:  # convective afternoon / evening storms
                start_hour = int(rng.integers(13, 24 - duration + 1)) if 24 - duration >= 13 else 24 - duration
            else:
                start_hour = int(rng.integers(0, 24 - duration + 1))
            weights = rng.dirichlet(np.ones(duration))
            rain[start_hour : start_hour + duration] = row.rain_mm * weights
            rh[start_hour : start_hour + duration] = np.clip(rh[start_hour : start_hour + duration] + 15, 0, 100)
        base = datetime(row.date.year, row.date.month, row.date.day)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": [base + timedelta(hours=int(h)) for h in hours],
                    "temperature_c": temp,
                    "humidity_pct": rh,
                    "rainfall_mm": rain,
                    "et0_true_mm": row.et0_true_mm * et_fraction,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def generate_hourly_weather(
    climate: ClimateProfile, latitude: float, start: date, n_days: int, seed: int
) -> pd.DataFrame:
    """Convenience wrapper: daily generation followed by hourly disaggregation."""
    rng = np.random.default_rng(seed)
    daily = generate_daily_weather(climate, latitude, start, n_days, rng)
    return expand_to_hourly(daily, rng)


def simulate_rain_forecast(actual_next_24h_mm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Turn the *actual* next-24 h rain into an imperfect forecast.

    Error model (documented assumption, not calibrated to a real service):
    * 12 % of rain events are missed entirely (forecast 0 mm);
    * 8 % of dry periods get a false alarm (gamma-distributed amount);
    * otherwise a multiplicative log-normal error (sigma = 0.45).
    """
    actual = np.asarray(actual_next_24h_mm, dtype=float)
    u = rng.random(actual.shape)
    noise = rng.lognormal(0.0, 0.45, actual.shape)
    false_alarm = rng.gamma(1.0, 4.0, actual.shape)
    raining = actual >= 1.0
    forecast = np.where(raining, np.where(u < 0.12, 0.0, actual * noise), np.where(u < 0.08, false_alarm, actual))
    return np.round(np.clip(forecast, 0.0, None), 2)
