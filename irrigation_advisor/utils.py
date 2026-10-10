"""Reusable, dependency-light helper functions used across the project."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the closed interval ``[low, high]``."""
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")
    return max(low, min(high, value))


def is_missing(value: Any) -> bool:
    """True for ``None``, NaN and +/-inf."""
    if value is None:
        return True
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def safe_float(value: Any, default: float | None = None) -> float | None:
    """Convert to ``float`` or return ``default`` when missing / non-numeric."""
    return default if is_missing(value) else float(value)


def round_or_none(value: Any, digits: int = 2) -> float | None:
    """Round finite numbers, map missing values to ``None`` (JSON friendly)."""
    return None if is_missing(value) else round(float(value), digits)


def stable_seed(*parts: Any) -> int:
    """Deterministic 32-bit seed from arbitrary parts.

    Python's built-in ``hash`` is randomised per process, so we use SHA-256 to
    obtain reproducible seeds (e.g. simulated weather for a given date and
    location is identical across runs and machines).
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def to_local_naive(ts: datetime, utc_offset_hours: float) -> datetime:
    """Convert a timestamp to naive field-local time.

    Naive timestamps are assumed to already be field-local. Timezone-aware ones
    are converted to UTC and shifted by ``utc_offset_hours``.
    """
    if ts.tzinfo is None:
        return ts
    utc = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return utc + timedelta(hours=utc_offset_hours)


def parse_date(value: str | date | datetime) -> date:
    """Parse ``YYYY-MM-DD`` strings / datetimes into a ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars/arrays, dates and paths."""

    def default(self, o: Any) -> Any:  # noqa: D401 - json API
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def _sanitise(value: Any) -> Any:
    """Recursively replace non-finite floats (incl. numpy floats) by ``None``.

    ``np.float64`` subclasses ``float``, so the encoder's ``default`` hook is
    never called for it and NaN would be written as invalid JSON (``NaN``).
    """
    if isinstance(value, dict):
        return {k: _sanitise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def to_json(data: Any, indent: int | None = None) -> str:
    """Serialise ``data`` as strict JSON (numpy types, dates, NaN -> null)."""
    return json.dumps(_sanitise(data), cls=NumpyJSONEncoder, indent=indent, allow_nan=False)


def write_json(data: Any, path: Path) -> Path:
    """Write ``data`` to ``path`` as pretty JSON (UTF-8)."""
    ensure_dir(path.parent)
    path.write_text(to_json(data, indent=2), encoding="utf-8")
    return path


def mean_or_none(values: Iterable[float]) -> float | None:
    """Mean of finite values, ``None`` if there are none."""
    finite = [float(v) for v in values if not is_missing(v)]
    return sum(finite) / len(finite) if finite else None
