"""Application configuration loaded from environment variables / ``.env``.

All settings have safe defaults so the project runs offline on a fresh
Windows or Linux machine without creating a ``.env`` file. Every variable is
prefixed with ``SIA_`` (Smart Irrigation Advisor).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:  # python-dotenv is optional at runtime; settings still work without it.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is listed in requirements
    load_dotenv = None  # type: ignore[assignment]

#: Repository root (the directory that contains ``irrigation_advisor/``).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, float(default)))


def _resolve_path(value: str) -> Path:
    """Resolve a path relative to the project root (keeps Windows paths intact)."""
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    database_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "irrigation.db")
    model_path: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "irrigation_model.joblib")
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    log_level: str = "INFO"
    #: Optional shared secret for IoT devices (header ``X-Device-Key``). Empty = disabled.
    iot_api_key: str = ""
    #: Default latitude used by the simulated weather provider when a field has none.
    default_latitude: float = 18.5
    #: Default longitude (Open-Meteo provider) when a field has none.
    default_longitude: float = 79.49244
    #: "simulated" (offline, default) or "openmeteo" (live forecasts; falls back to simulated when offline).
    weather_provider: str = "simulated"
    #: IANA time zone for Open-Meteo requests (field-local timestamps).
    timezone: str = "Asia/Kolkata"
    #: Offset (hours) used to convert timezone-aware timestamps to field-local time.
    utc_offset_hours: float = 5.5
    #: Hours of history loaded from the database for each recommendation.
    history_hours: int = 72
    #: Readings older than this (relative to "now") are treated as stale.
    stale_reading_hours: float = 6.0
    #: Climate profile used for simulated weather.
    climate_profile: str = "semi_arid_monsoon"
    api_title: str = "Smart Irrigation Advisor API"


def _build_settings() -> Settings:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    defaults = Settings()
    return Settings(
        database_path=_resolve_path(_env_str("SIA_DATABASE_PATH", str(defaults.database_path))),
        model_path=_resolve_path(_env_str("SIA_MODEL_PATH", str(defaults.model_path))),
        data_dir=_resolve_path(_env_str("SIA_DATA_DIR", str(defaults.data_dir))),
        reports_dir=_resolve_path(_env_str("SIA_REPORTS_DIR", str(defaults.reports_dir))),
        log_dir=_resolve_path(_env_str("SIA_LOG_DIR", str(defaults.log_dir))),
        log_level=_env_str("SIA_LOG_LEVEL", defaults.log_level).upper(),
        iot_api_key=_env_str("SIA_IOT_API_KEY", defaults.iot_api_key),
        default_latitude=_env_float("SIA_DEFAULT_LATITUDE", defaults.default_latitude),
        default_longitude=_env_float("SIA_DEFAULT_LONGITUDE", defaults.default_longitude),
        weather_provider=_env_str("SIA_WEATHER_PROVIDER", defaults.weather_provider).lower(),
        timezone=_env_str("SIA_TIMEZONE", defaults.timezone),
        utc_offset_hours=_env_float("SIA_UTC_OFFSET_HOURS", defaults.utc_offset_hours),
        history_hours=_env_int("SIA_HISTORY_HOURS", defaults.history_hours),
        stale_reading_hours=_env_float("SIA_STALE_READING_HOURS", defaults.stale_reading_hours),
        climate_profile=_env_str("SIA_CLIMATE_PROFILE", defaults.climate_profile),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance (call ``get_settings.cache_clear()`` in tests)."""
    return _build_settings()
