"""Offline weather provider.

The API must work without internet access or paid weather APIs, so forecasts
come from the deterministic stochastic weather generator in
:mod:`irrigation_advisor.data.climate`. For a given latitude and date the
"forecast" is always the same, which keeps demos reproducible.

The :class:`WeatherProvider` protocol makes it trivial to plug in a real
forecast service later (e.g. IMD / Open-Meteo) without touching the engine.
Clients can also send their own forecast (``forecast_rain_24h_mm``) in the
recommendation request, which always takes priority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Protocol

import numpy as np
import pandas as pd

from ..utils import stable_seed
from .climate import generate_hourly_weather, get_climate, rain_probability


@dataclass(frozen=True)
class WeatherForecast:
    """Forecast summary for the planning horizon."""

    start: datetime
    hours: int
    total_rain_mm: float
    rain_probability: float
    temp_max_c: float
    temp_min_c: float
    mean_humidity_pct: float
    source: str
    hourly: list[dict] = field(default_factory=list)


class WeatherProvider(Protocol):
    """Interface every weather source must implement."""

    def forecast(self, latitude: float, start: datetime, hours: int = 24) -> WeatherForecast: ...

    def hourly(self, latitude: float, start: datetime, hours: int) -> pd.DataFrame: ...


@lru_cache(maxsize=32)
def _year_of_weather(climate_name: str, latitude_key: float, year: int) -> pd.DataFrame:
    """Cached synthetic hourly weather for one calendar year at one location."""
    climate = get_climate(climate_name)
    n_days = (date(year + 1, 1, 1) - date(year, 1, 1)).days
    seed = stable_seed("weather", climate_name, latitude_key, year)
    df = generate_hourly_weather(climate, latitude_key, date(year, 1, 1), n_days, seed)
    return df.set_index("timestamp")


class SimulatedWeatherProvider:
    """Deterministic, offline weather source (clearly labelled *simulated*)."""

    source_name = "simulated-offline"

    def __init__(self, climate_name: str = "semi_arid_monsoon") -> None:
        get_climate(climate_name)  # validate early
        self.climate_name = climate_name

    def hourly(self, latitude: float, start: datetime, hours: int) -> pd.DataFrame:
        """Hourly simulated weather for ``[start, start + hours)``."""
        if hours <= 0:
            raise ValueError("hours must be positive")
        start = start.replace(minute=0, second=0, microsecond=0, tzinfo=None)
        end = start + timedelta(hours=hours)
        lat_key = round(float(latitude), 1)
        years = range(start.year, end.year + 1)
        frame = pd.concat([_year_of_weather(self.climate_name, lat_key, y) for y in years])
        window = frame.loc[(frame.index >= start) & (frame.index < end)]
        return window.reset_index()

    def forecast(self, latitude: float, start: datetime, hours: int = 24) -> WeatherForecast:
        """Summarise the next ``hours`` hours into a planning forecast."""
        window = self.hourly(latitude, start + timedelta(hours=1), hours)
        climate = get_climate(self.climate_name)
        doy = start.timetuple().tm_yday
        total_rain = float(window["rainfall_mm"].sum())
        # Probability blends the climatological Markov probability with the
        # simulated outcome so the number is informative but not certain.
        climatological = rain_probability(climate, doy, previous_wet=False)
        probability = float(np.clip(0.5 * climatological + (0.5 if total_rain >= 1.0 else 0.0), 0.0, 1.0))
        hourly = [
            {
                "timestamp": ts.isoformat(),
                "temperature_c": round(float(t), 1),
                "humidity_pct": round(float(h), 1),
                "rainfall_mm": round(float(r), 2),
            }
            for ts, t, h, r in zip(
                window["timestamp"], window["temperature_c"], window["humidity_pct"], window["rainfall_mm"]
            )
        ]
        return WeatherForecast(
            start=start,
            hours=hours,
            total_rain_mm=round(total_rain, 2),
            rain_probability=round(probability, 2),
            temp_max_c=round(float(window["temperature_c"].max()), 1),
            temp_min_c=round(float(window["temperature_c"].min()), 1),
            mean_humidity_pct=round(float(window["humidity_pct"].mean()), 1),
            source=self.source_name,
            hourly=hourly,
        )
