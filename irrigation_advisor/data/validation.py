"""Sensor validation rules shared by the API, preprocessing and the engine.

Two levels of validation exist:

* :func:`validate_reading` - one incoming reading (optionally with recent
  history) checked for physical plausibility, spikes, stuck sensors and bad
  timestamps. Produces human-readable issues for API responses.
* :func:`flag_frame` - vectorised, *backward-looking* checks over a whole
  time series. Used in preprocessing for both training and inference, so the
  model never sees values the online validator would have rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

import pandas as pd

from ..utils import is_missing

#: Physically plausible ranges for each measured variable.
SENSOR_LIMITS: dict[str, tuple[float, float]] = {
    "soil_moisture": (0.0, 60.0),  # % volumetric water content
    "temperature_c": (-10.0, 55.0),
    "humidity_pct": (0.0, 100.0),
    "rainfall_mm": (0.0, 150.0),  # per reading interval (<= 1 h)
}
#: Max plausible soil-moisture jump (percentage points per hour) without rain.
MAX_MOISTURE_JUMP_PER_HOUR = 15.0
#: Rain (mm) in the same reading that can explain a large moisture rise.
RAIN_EXPLAINS_JUMP_MM = 1.0
#: A sensor reporting an identical value this many times in a row is "stuck".
STUCK_WINDOW = 12
STUCK_TOLERANCE = 0.01
#: Rolling-median spike filter (hours / percentage points).
SPIKE_WINDOW = 6
SPIKE_THRESHOLD = 10.0
LOW_BATTERY_V = 3.3
FUTURE_TOLERANCE = timedelta(minutes=10)


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding."""

    code: str
    field: str
    message: str
    severity: str  # "error" -> value unusable, "warning" -> usable but suspicious

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "field": self.field, "message": self.message, "severity": self.severity}


@dataclass
class ValidationResult:
    """Outcome of validating one reading."""

    issues: list[ValidationIssue] = field(default_factory=list)
    invalid_fields: set[str] = field(default_factory=set)

    @property
    def soil_moisture_valid(self) -> bool:
        return "soil_moisture" not in self.invalid_fields

    @property
    def is_valid(self) -> bool:
        """A reading is valid for decision making when soil moisture is usable."""
        return self.soil_moisture_valid and "timestamp" not in self.invalid_fields

    @property
    def codes(self) -> list[str]:
        return [i.code for i in self.issues]

    def add(self, code: str, field_name: str, message: str, severity: str = "error") -> None:
        self.issues.append(ValidationIssue(code, field_name, message, severity))
        if severity == "error":
            self.invalid_fields.add(field_name)


def validate_reading(
    reading: dict[str, Any],
    history: Sequence[dict[str, Any]] | None = None,
    now: datetime | None = None,
    stale_hours: float = 6.0,
) -> ValidationResult:
    """Validate a single sensor reading.

    Args:
        reading: Mapping with ``timestamp`` (datetime) and sensor values.
            Missing optional values are allowed (``None``).
        history: Previous readings of the same field, oldest first.
        now: Reference time for stale/future checks (defaults to ``datetime.now()``).
        stale_hours: Age beyond which a reading is flagged stale.
    """
    result = ValidationResult()
    now = now or datetime.now()

    ts = reading.get("timestamp")
    if not isinstance(ts, datetime):
        result.add("invalid_timestamp", "timestamp", "Timestamp is missing or not a datetime")
    else:
        if ts > now + FUTURE_TOLERANCE:
            result.add("future_timestamp", "timestamp", f"Timestamp {ts.isoformat()} is in the future")
        elif now - ts > timedelta(hours=stale_hours):
            result.add(
                "stale_reading",
                "timestamp",
                f"Reading is {(now - ts).total_seconds() / 3600:.1f} h old (> {stale_hours} h)",
                severity="warning",
            )

    for name, (low, high) in SENSOR_LIMITS.items():
        value = reading.get(name)
        if is_missing(value):
            if name == "soil_moisture":
                result.add("missing_value", name, "Soil moisture value is missing")
            elif name != "rainfall_mm":
                result.add("missing_value", name, f"{name} is missing", severity="warning")
            continue
        if not low <= float(value) <= high:
            result.add("out_of_range", name, f"{name}={value} outside plausible range [{low}, {high}]")

    battery = reading.get("battery_v")
    if not is_missing(battery) and float(battery) < LOW_BATTERY_V:
        result.add("low_battery", "battery_v", f"Battery at {battery} V - readings may degrade", "warning")

    sm = reading.get("soil_moisture")
    if history and result.soil_moisture_valid and not is_missing(sm):
        _check_against_history(result, reading, list(history))
    return result


def _check_against_history(result: ValidationResult, reading: dict[str, Any], history: list[dict[str, Any]]) -> None:
    """Spike and stuck-sensor checks against previous valid readings."""
    sm = float(reading["soil_moisture"])
    previous = [h for h in history if not is_missing(h.get("soil_moisture"))]
    if not previous:
        return
    last = previous[-1]
    ts, last_ts = reading.get("timestamp"), last.get("timestamp")
    if isinstance(ts, datetime) and isinstance(last_ts, datetime):
        hours = max((ts - last_ts).total_seconds() / 3600.0, 1.0)
        jump = sm - float(last["soil_moisture"])
        rain = float(reading.get("rainfall_mm") or 0.0)
        # A rise can be explained by rain/irrigation, a sudden drop cannot.
        explained = jump > 0 and rain >= RAIN_EXPLAINS_JUMP_MM
        if abs(jump) / hours > MAX_MOISTURE_JUMP_PER_HOUR and not explained and hours <= 3:
            result.add(
                "spike",
                "soil_moisture",
                f"Soil moisture changed {jump:+.1f} points in {hours:.1f} h without rain",
            )
    recent = [float(h["soil_moisture"]) for h in previous[-(STUCK_WINDOW - 1) :]] + [sm]
    if len(recent) >= STUCK_WINDOW and (max(recent) - min(recent)) <= STUCK_TOLERANCE:
        result.add("stuck_sensor", "soil_moisture", f"Soil moisture identical for {STUCK_WINDOW} readings")


# ---------------------------------------------------------------------------
# Vectorised frame-level checks (backward-looking only)
# ---------------------------------------------------------------------------
def apply_range_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Replace physically impossible values by NaN and count them.

    Adds ``<col>_out_of_range`` boolean columns. Operates on a copy.
    """
    out = df.copy()
    for name, (low, high) in SENSOR_LIMITS.items():
        if name not in out.columns:
            continue
        values = pd.to_numeric(out[name], errors="coerce")
        bad = values.notna() & ((values < low) | (values > high))
        out[f"{name}_out_of_range"] = bad
        out[name] = values.mask(bad)
    return out


def flag_frame(sm: pd.Series, rain: pd.Series | None = None) -> pd.DataFrame:
    """Backward-looking spike and stuck detection on an hourly soil-moisture series.

    Args:
        sm: Hourly soil moisture for ONE field (may contain NaN).
        rain: Hourly rain for the same index (explains upward jumps).

    Returns:
        DataFrame with boolean columns ``spike`` and ``stuck``.
    """
    rain = rain if rain is not None else pd.Series(0.0, index=sm.index)
    # Median of the *previous* SPIKE_WINDOW hours (shift(1) => no look-ahead).
    ref = sm.shift(1).rolling(SPIKE_WINDOW, min_periods=2).median()
    deviation = sm - ref
    rain_recent = rain.fillna(0.0).rolling(2, min_periods=1).sum()
    spike = (deviation.abs() > SPIKE_THRESHOLD) & ~((deviation > 0) & (rain_recent >= RAIN_EXPLAINS_JUMP_MM))
    rng = sm.rolling(STUCK_WINDOW, min_periods=STUCK_WINDOW).max() - sm.rolling(
        STUCK_WINDOW, min_periods=STUCK_WINDOW
    ).min()
    stuck = rng <= STUCK_TOLERANCE
    return pd.DataFrame({"spike": spike.fillna(False).astype(bool), "stuck": stuck.fillna(False).astype(bool)})


def summarize_issues(results: Iterable[ValidationResult]) -> dict[str, int]:
    """Count issue codes over many validation results."""
    counts: dict[str, int] = {}
    for res in results:
        for code in res.codes:
            counts[code] = counts.get(code, 0) + 1
    return counts


def nan_fraction(values: pd.Series) -> float:
    """Share of missing values in a series (0 for empty series)."""
    return float(values.isna().mean()) if len(values) else 0.0


__all__ = [
    "SENSOR_LIMITS",
    "ValidationIssue",
    "ValidationResult",
    "validate_reading",
    "apply_range_checks",
    "flag_frame",
    "summarize_issues",
    "nan_fraction",
]
