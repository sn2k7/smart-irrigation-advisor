"""Open-Meteo data source: real historical dataset + live weather provider.

Parameters taken from Open-Meteo (all hourly, local time zone):

==========================  ==========================================  ====================================
Project parameter           Open-Meteo variable                         Notes
==========================  ==========================================  ====================================
soil moisture (% VWC)       soil_moisture_0_to_7cm, _7_to_28cm          depth-weighted mean of the layers
                            (archive, ERA5-Land reanalysis)             in ``soil_layers`` (m3/m3 -> %)
temperature (°C)            temperature_2m
relative humidity (%)       relative_humidity_2m
rain in last 24 h (mm)      precipitation                               rolling 24 h sum (also a feature)
rain forecast 24 h (mm)     precipitation from the Historical Forecast  sum over (t, t+24 h]; real archived
                            API (archived model forecasts)              forecasts, not hindsight
ET0 (mm/h)                  et0_fao_evapotranspiration                  FAO-56 Penman-Monteith; replaces
                                                                        the Hargreaves estimate
wind speed (m/s)            wind_speed_10m (wind_speed_unit=ms)         stored/displayed; its effect enters
                                                                        through Penman-Monteith ET0
==========================  ==========================================  ====================================

Important: Open-Meteo soil moisture is a *model reanalysis* for a ~9 km grid
cell without irrigation, not a probe in your field. It is real, physically
consistent data that is well suited to developing and testing the pipeline,
but a deployment should use (or calibrate against) in-field sensors.

Open-Meteo is free for non-commercial use without an API key
(https://open-meteo.com/en/terms). Network access is needed only for
fetching; the fetched CSV is then used offline.

Search for ``PLACEHOLDER(real-dataset)`` / ``PLACEHOLDER(openmeteo)`` for
values to adapt.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from ..agronomy.crops import SOILS, get_crop, get_method, get_soil
from ..config import PROJECT_ROOT
from ..exceptions import DataValidationError, IrrigationAdvisorError
from ..utils import stable_seed
from .weather_provider import SimulatedWeatherProvider, WeatherForecast

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "et0_fao_evapotranspiration",
                "wind_speed_10m"]
#: Soil layers of the live forecast API (different from the archive layers).
FORECAST_SOIL_LAYERS = {"soil_moisture_0_to_1cm": 1, "soil_moisture_1_to_3cm": 2, "soil_moisture_3_to_9cm": 6,
                        "soil_moisture_9_to_27cm": 18}
RENAME = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "precipitation": "rainfall_mm",
    "et0_fao_evapotranspiration": "et0_mm",
    "wind_speed_10m": "wind_speed_ms",
}
#: Columns of the CSV written by :func:`build_dataset` (internal schema + extras).
OUTPUT_COLUMNS = [
    "timestamp", "field_id", "soil_moisture", "temperature_c", "humidity_pct", "rainfall_mm",
    "rain_last_24h_mm", "forecast_rain_24h_mm", "forecast_source", "et0_mm", "wind_speed_ms",
    "crop_type", "soil_type", "irrigation_method", "planting_date", "latitude",
]


class OpenMeteoError(IrrigationAdvisorError):
    """Open-Meteo could not be reached or returned an error."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Season:
    """One crop season repeated every year (planting month-day)."""

    crop_type: str
    planting_month_day: str  # "MM-DD"

    def planting_dates(self, start: date, end: date) -> list[date]:
        month, day = (int(x) for x in self.planting_month_day.split("-"))
        return [date(y, month, day) for y in range(start.year - 1, end.year + 1)]


@dataclass
class OpenMeteoConfig:
    """Mirrors ``config/openmeteo_dataset.json``."""

    name: str
    latitude: float
    longitude: float
    start_date: date
    end_date: date
    timezone: str = "Asia/Kolkata"
    #: archive variable -> layer thickness (cm) used as weight
    soil_layers: dict[str, float] = field(default_factory=lambda: {
        "soil_moisture_0_to_7cm": 7.0, "soil_moisture_7_to_28cm": 21.0})
    soil_type: str = "auto"
    irrigation_method: str = "drip"
    seasons: list[Season] = field(default_factory=list)
    use_historical_forecast: bool = True
    output_csv: Path = PROJECT_ROOT / "data" / "real" / "openmeteo_dataset.csv"
    dataset_config_out: Path = PROJECT_ROOT / "config" / "openmeteo_generated_dataset.json"
    cache_dir: Path = PROJECT_ROOT / "data" / "real" / "openmeteo_cache"

    def __post_init__(self) -> None:
        if not -90 <= self.latitude <= 90 or not -180 <= self.longitude <= 180:
            raise DataValidationError("latitude/longitude out of range")
        if self.end_date < self.start_date:
            raise DataValidationError("end_date must be after start_date")
        if self.soil_type != "auto":
            get_soil(self.soil_type)
        get_method(self.irrigation_method)
        if not self.seasons:
            raise DataValidationError("Add at least one crop season (crop_type + planting MM-DD)")
        for s in self.seasons:
            get_crop(s.crop_type)
            datetime.strptime(f"2001-{s.planting_month_day}", "%Y-%m-%d")
        if not self.soil_layers:
            raise DataValidationError("soil_layers must name at least one archive soil-moisture variable")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_openmeteo_config(path: str | Path) -> OpenMeteoConfig:
    """Read the JSON config (keys starting with '_' are comments)."""
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"Open-Meteo config not found: {path}")
    raw = {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    try:
        return OpenMeteoConfig(
            name=raw.get("name", path.stem),
            latitude=float(raw["latitude"]),
            longitude=float(raw["longitude"]),
            start_date=date.fromisoformat(raw["start_date"]),
            end_date=date.fromisoformat(raw["end_date"]),
            timezone=raw.get("timezone", "Asia/Kolkata"),
            soil_layers={k: float(v) for k, v in raw.get("soil_layers", {}).items() if not k.startswith("_")}
            or OpenMeteoConfig.__dataclass_fields__["soil_layers"].default_factory(),
            soil_type=raw.get("soil_type", "auto"),
            irrigation_method=raw.get("irrigation_method", "drip"),
            seasons=[Season(s["crop_type"], s["planting_month_day"]) for s in raw.get("seasons", [])],
            use_historical_forecast=bool(raw.get("use_historical_forecast", True)),
            output_csv=_resolve(raw.get("output_csv", "data/real/openmeteo_dataset.csv")),
            dataset_config_out=_resolve(raw.get("dataset_config_out", "config/openmeteo_generated_dataset.json")),
            cache_dir=_resolve(raw.get("cache_dir", "data/real/openmeteo_cache")),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise DataValidationError(f"Open-Meteo config problem: {exc}") from exc


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------
class OpenMeteoClient:
    """Small Open-Meteo client with retries, yearly chunking and a file cache."""

    def __init__(self, client: httpx.Client | None = None, timeout: float = 60.0, retries: int = 3,
                 cache_dir: Path | None = None) -> None:
        self._client = client or httpx.Client(timeout=timeout)
        self.retries = retries
        self.cache_dir = cache_dir

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        cache_file = None
        if self.cache_dir is not None:
            key = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
            cache_file = self.cache_dir / f"{url.split('//')[1].split('.')[0]}_{stable_seed(url, key)}.json"
            if cache_file.exists():
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if cached.get("_params") == params:
                    return cached
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self._client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if cache_file is not None:
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        cache_file.write_text(json.dumps({**data, "_params": params}), encoding="utf-8")
                    return data
                reason = response.json().get("reason", response.text[:200]) if response.content else ""
                if response.status_code in (400, 404):
                    raise OpenMeteoError(f"Open-Meteo rejected the request ({response.status_code}): {reason}")
                last = OpenMeteoError(f"HTTP {response.status_code}: {reason}")
            except httpx.HTTPError as exc:
                last = exc
            if attempt < self.retries:
                time.sleep(2 ** attempt)
        raise OpenMeteoError(f"Cannot reach Open-Meteo ({url}): {last}")

    @staticmethod
    def _frame(data: dict[str, Any]) -> pd.DataFrame:
        hourly = data.get("hourly")
        if not hourly or "time" not in hourly:
            raise OpenMeteoError("Open-Meteo response has no hourly data")
        df = pd.DataFrame(hourly)
        df["timestamp"] = pd.to_datetime(df.pop("time"))
        return df

    def _chunked(self, url: str, base: dict[str, Any], start: date, end: date) -> pd.DataFrame:
        frames = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(date(chunk_start.year, 12, 31), end)
            logger.info("Open-Meteo %s %s..%s", url.split("//")[1].split("/")[0], chunk_start, chunk_end)
            data = self._get(url, {**base, "start_date": chunk_start.isoformat(), "end_date": chunk_end.isoformat()})
            frames.append(self._frame(data))
            chunk_start = chunk_end + timedelta(days=1)
        return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")

    def archive(self, lat: float, lon: float, start: date, end: date, tz: str, soil_vars: list[str]) -> pd.DataFrame:
        """Hourly reanalysis: weather + soil moisture layers."""
        base = {"latitude": lat, "longitude": lon, "timezone": tz, "wind_speed_unit": "ms",
                "hourly": ",".join(WEATHER_VARS + soil_vars)}
        return self._chunked(ARCHIVE_URL, base, start, end)

    def historical_forecast_precipitation(self, lat: float, lon: float, start: date, end: date, tz: str) -> pd.DataFrame:
        """Archived model forecasts of precipitation (for an honest 'rain forecast')."""
        base = {"latitude": lat, "longitude": lon, "timezone": tz, "hourly": "precipitation"}
        df = self._chunked(HISTORICAL_FORECAST_URL, base, start, end + timedelta(days=1))
        return df.rename(columns={"precipitation": "forecast_precipitation"})

    def forecast(self, lat: float, lon: float, tz: str, past_days: int = 3, forecast_days: int = 2) -> pd.DataFrame:
        """Live forecast API: recent past + next days (weather + soil layers)."""
        params = {"latitude": lat, "longitude": lon, "timezone": tz, "wind_speed_unit": "ms",
                  "past_days": past_days, "forecast_days": forecast_days,
                  "hourly": ",".join(WEATHER_VARS + list(FORECAST_SOIL_LAYERS))}
        return self._frame(self._get(FORECAST_URL, params))

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def weighted_soil_moisture(df: pd.DataFrame, layers: dict[str, float]) -> pd.Series:
    """Depth-weighted mean of soil layers (m3/m3) -> % VWC."""
    missing = [c for c in layers if c not in df.columns]
    if missing:
        raise OpenMeteoError(f"Soil layers missing from Open-Meteo response: {missing}")
    weights = pd.Series(layers, dtype=float)
    values = df[list(layers)].astype(float)
    total = (values * weights).sum(axis=1, min_count=1)
    norm = values.notna().mul(weights).sum(axis=1).replace(0, np.nan)
    return 100.0 * total / norm


def next_24h_sum(series: pd.Series) -> pd.Series:
    """Sum over the following 24 hours (t, t+24h], NaN where incomplete."""
    rev = series[::-1].rolling(24, min_periods=24).sum()[::-1]
    return rev.shift(-1)


def estimate_field_capacity(sm_pct: pd.Series, rain_mm: pd.Series) -> tuple[float | None, float]:
    """Field-capacity estimate = median moisture 48 h after >= 10 mm/day rain; WP proxy = 1st percentile.

    Field capacity is classically the water content after free drainage
    (~2 days after a thorough wetting), which is what this measures.
    """
    daily_rain = rain_mm.rolling(24, min_periods=1).sum()
    soaked = daily_rain >= 10.0
    after = soaked.shift(48, fill_value=False) & ~(rain_mm.rolling(48, min_periods=1).sum() > 2.0)
    values = sm_pct[after].dropna()
    fc = float(values.median()) / 100.0 if len(values) >= 24 else None
    wp = float(np.nanpercentile(sm_pct.dropna(), 1)) / 100.0
    return fc, wp


def suggest_soil_type(fc: float | None, wp: float) -> str:
    """Soil class whose FC/WP are closest to the data-derived estimates."""
    def distance(name: str) -> float:
        s = SOILS[name]
        return (abs(s.field_capacity - fc) if fc is not None else 0.0) + abs(s.wilting_point - wp)
    return min(SOILS, key=distance)


def build_dataset(cfg: OpenMeteoConfig, client: OpenMeteoClient) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch Open-Meteo data and turn it into season-wise 'fields' in the internal schema."""
    raw = client.archive(cfg.latitude, cfg.longitude, cfg.start_date, cfg.end_date, cfg.timezone,
                         list(cfg.soil_layers))
    df = raw.rename(columns=RENAME)
    df["soil_moisture"] = weighted_soil_moisture(raw, cfg.soil_layers)
    df = df.set_index("timestamp").sort_index()
    df["rain_last_24h_mm"] = df["rainfall_mm"].rolling(24, min_periods=1).sum()

    df["forecast_source"] = "observed_next_24h_proxy"
    df["forecast_rain_24h_mm"] = next_24h_sum(df["rainfall_mm"])
    if cfg.use_historical_forecast:
        try:
            hf = client.historical_forecast_precipitation(cfg.latitude, cfg.longitude, cfg.start_date,
                                                          cfg.end_date, cfg.timezone).set_index("timestamp")
            forecast = next_24h_sum(hf["forecast_precipitation"]).reindex(df.index)
            has = forecast.notna()
            df.loc[has, "forecast_rain_24h_mm"] = forecast[has]
            df.loc[has, "forecast_source"] = "open_meteo_historical_forecast"
        except OpenMeteoError as exc:
            logger.warning("Historical forecasts unavailable (%s) - using observed next-24h rain as proxy", exc)
    proxy_share = float((df["forecast_source"] != "open_meteo_historical_forecast").mean())

    fc, wp = estimate_field_capacity(df["soil_moisture"], df["rainfall_mm"])
    soil_type = suggest_soil_type(fc, wp) if cfg.soil_type == "auto" else cfg.soil_type

    seasons = []
    for season in cfg.seasons:
        crop = get_crop(season.crop_type)
        for planting in season.planting_dates(cfg.start_date, cfg.end_date):
            start = pd.Timestamp(planting)
            end = start + pd.Timedelta(days=crop.season_days)
            block = df.loc[(df.index >= start) & (df.index < end)].copy()
            if len(block) < 24 * 14:
                continue  # season not covered by the downloaded period
            block["field_id"] = f"{crop.name}-{planting:%Y-%m-%d}"
            block["crop_type"] = crop.name
            block["planting_date"] = planting.isoformat()
            seasons.append(block)
    if not seasons:
        raise DataValidationError("No crop season falls inside start_date..end_date - adjust dates or seasons")
    out = pd.concat(seasons).reset_index()
    out["soil_type"] = soil_type
    out["irrigation_method"] = cfg.irrigation_method
    out["latitude"] = cfg.latitude
    out = out[OUTPUT_COLUMNS].sort_values(["field_id", "timestamp"]).reset_index(drop=True)

    report = {
        "source": "Open-Meteo (ERA5-Land reanalysis + archived forecasts)",
        "location": {"latitude": cfg.latitude, "longitude": cfg.longitude, "timezone": cfg.timezone},
        "period": [cfg.start_date.isoformat(), cfg.end_date.isoformat()],
        "hours_downloaded": int(len(df)),
        "rows_in_seasons": int(len(out)),
        "seasons": sorted(out["field_id"].unique().tolist()),
        "soil_layers": cfg.soil_layers,
        "soil_type": soil_type,
        "soil_type_mode": cfg.soil_type,
        "estimated_field_capacity_pct": None if fc is None else round(100 * fc, 1),
        "estimated_wilting_proxy_pct": round(100 * wp, 1),
        "soil_moisture_pct_range": [round(float(df["soil_moisture"].min()), 1),
                                    round(float(df["soil_moisture"].max()), 1)],
        "rain_total_mm": round(float(df["rainfall_mm"].sum()), 1),
        "et0_mean_mm_day": round(float(df["et0_mm"].sum() / max(len(df) / 24, 1)), 2),
        "wind_mean_ms": round(float(df["wind_speed_ms"].mean()), 2),
        "forecast_proxy_share": round(proxy_share, 3),
        "notes": [
            "Soil moisture is model reanalysis for the grid cell (no irrigation), not an in-field probe.",
            "forecast_rain_24h_mm comes from archived Open-Meteo forecasts where available; other hours "
            "use observed next-24h rain (a perfect-forecast proxy) - see forecast_source.",
        ],
    }
    return out, report


def write_dataset_config(cfg: OpenMeteoConfig, csv_path: Path) -> Path:
    """Write a dataset config for the custom-dataset adapter pointing at the CSV."""
    try:
        rel = csv_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel = str(csv_path)
    config = {
        "_README": "GENERATED by scripts/fetch_openmeteo.py from the Open-Meteo config - re-run the script "
                   "instead of editing by hand.",
        "name": cfg.name,
        "path": rel,
        "column_map": {c: c for c in ("timestamp", "field_id", "soil_moisture", "temperature_c", "humidity_pct",
                                      "rainfall_mm", "forecast_rain_24h_mm", "et0_mm", "wind_speed_ms",
                                      "crop_type", "soil_type", "irrigation_method", "planting_date",
                                      "latitude")},
        "timestamp_format": None,
        "units": {"soil_moisture": "percent", "temperature": "celsius", "rainfall": "mm", "rainfall_mode": "interval"},
        "fields": {"defaults": {}, "per_field": {}},
        "labels": {"mode": "derive"},
    }
    cfg.dataset_config_out.parent.mkdir(parents=True, exist_ok=True)
    cfg.dataset_config_out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return cfg.dataset_config_out


# ---------------------------------------------------------------------------
# Live weather provider (rain forecast + measured ET0/wind at decision time)
# ---------------------------------------------------------------------------
class OpenMeteoWeatherProvider:
    """Live forecasts from Open-Meteo, falling back to the offline simulator when unreachable.

    ``observed_hourly`` returns recent hourly ET0 and wind so the engine can use
    Penman-Monteith ET0 exactly as in a model trained on Open-Meteo data.
    """

    source_name = "open-meteo"
    uses_longitude = True

    def __init__(self, longitude: float, timezone: str = "Asia/Kolkata", client: OpenMeteoClient | None = None,
                 fallback: SimulatedWeatherProvider | None = None, cache_minutes: int = 30) -> None:
        self.longitude = longitude
        self.timezone = timezone
        self.client = client or OpenMeteoClient(timeout=15, retries=1)
        self.fallback = fallback or SimulatedWeatherProvider()
        self.cache_minutes = cache_minutes
        self._cache: dict[tuple[float, float], tuple[datetime, pd.DataFrame]] = {}

    def _data(self, latitude: float, longitude: float | None = None) -> pd.DataFrame:
        lon = self.longitude if longitude is None else longitude
        key = (round(latitude, 3), round(lon, 3))
        hit = self._cache.get(key)
        if hit and datetime.now() - hit[0] < timedelta(minutes=self.cache_minutes):
            return hit[1]
        df = self.client.forecast(latitude, lon, self.timezone, past_days=4, forecast_days=3)
        df = df.rename(columns=RENAME).set_index("timestamp").sort_index()
        self._cache[key] = (datetime.now(), df)
        return df

    def hourly(self, latitude: float, start: datetime, hours: int, longitude: float | None = None) -> pd.DataFrame:
        try:
            df = self._data(latitude, longitude)
            window = df.loc[(df.index >= start) & (df.index < start + timedelta(hours=hours))]
            if len(window) < hours:
                raise OpenMeteoError("requested window outside the forecast range")
            return window.reset_index()[["timestamp", "temperature_c", "humidity_pct", "rainfall_mm"]]
        except (OpenMeteoError, KeyError) as exc:
            logger.warning("Open-Meteo hourly unavailable (%s) - using simulated weather", exc)
            return self.fallback.hourly(latitude, start, hours)

    def forecast(self, latitude: float, start: datetime, hours: int = 24,
                 longitude: float | None = None) -> WeatherForecast:
        begin = start.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        try:
            df = self._data(latitude, longitude)
            window = df.loc[(df.index >= begin) & (df.index < begin + timedelta(hours=hours))]
            if len(window) < hours:
                raise OpenMeteoError("decision time is outside the live forecast range")
        except OpenMeteoError as exc:
            logger.warning("Open-Meteo forecast unavailable (%s) - using simulated weather", exc)
            fc = self.fallback.forecast(latitude, start, hours)
            return replace(fc, source=f"{fc.source} (Open-Meteo unreachable)")
        total = float(window["rainfall_mm"].sum())
        return WeatherForecast(
            start=start, hours=hours, total_rain_mm=round(total, 2),
            rain_probability=round(float((window["rainfall_mm"] >= 0.1).mean()), 2),
            temp_max_c=round(float(window["temperature_c"].max()), 1),
            temp_min_c=round(float(window["temperature_c"].min()), 1),
            mean_humidity_pct=round(float(window["humidity_pct"].mean()), 1),
            source=self.source_name,
            hourly=[{"timestamp": ts.isoformat(), "temperature_c": round(float(r.temperature_c), 1),
                     "humidity_pct": round(float(r.humidity_pct), 1), "rainfall_mm": round(float(r.rainfall_mm), 2),
                     "et0_mm": round(float(r.et0_mm), 3), "wind_speed_ms": round(float(r.wind_speed_ms), 2)}
                    for ts, r in window.iterrows()],
        )

    def observed_hourly(self, latitude: float, start: datetime, end: datetime,
                        longitude: float | None = None) -> pd.DataFrame:
        """Hourly ET0 (mm) and wind (m/s) for [start, end]; empty frame if unavailable."""
        try:
            df = self._data(latitude, longitude)
        except OpenMeteoError as exc:
            logger.warning("Open-Meteo ET0 unavailable (%s) - engine falls back to Hargreaves", exc)
            return pd.DataFrame(columns=["timestamp", "et0_mm", "wind_speed_ms"])
        window = df.loc[(df.index >= start) & (df.index <= end), ["et0_mm", "wind_speed_ms"]]
        return window.reset_index()

    def recent_readings(self, latitude: float, hours: int = 72, longitude: float | None = None) -> pd.DataFrame:
        """Latest past hours as readings (virtual sensor at the grid cell) incl. soil moisture."""
        df = self._data(latitude, longitude)
        now = pd.Timestamp(datetime.now()).floor("h")
        past = df.loc[(df.index > now - pd.Timedelta(hours=hours)) & (df.index <= now)].copy()
        past["soil_moisture"] = weighted_soil_moisture(past, FORECAST_SOIL_LAYERS)
        return past.reset_index()[["timestamp", "soil_moisture", "temperature_c", "humidity_pct", "rainfall_mm",
                                   "et0_mm", "wind_speed_ms"]]
