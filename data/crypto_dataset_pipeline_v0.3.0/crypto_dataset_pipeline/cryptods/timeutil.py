"""Time helpers.

Convention used everywhere in the package: timestamps are ``int64`` microseconds since the
Unix epoch, UTC (column suffix ``_us``).  Durations are ``int64`` microseconds as well.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

US = 1
MS = 1_000
SEC = 1_000_000
MIN = 60 * SEC
HOUR = 60 * MIN
DAY = 24 * HOUR

_UNITS = {
    "us": US, "ms": MS, "s": SEC, "sec": SEC, "min": MIN, "m": MIN,
    "h": HOUR, "d": DAY, "w": 7 * DAY,
}
_DUR_RE = re.compile(r"^\s*(\d+)\s*([a-z]+)\s*$")


def parse_duration(text) -> int:
    """Parse '5min', '1h', '30d', '250ms' (or an int number of microseconds) to microseconds."""
    if isinstance(text, (int, np.integer)):
        return int(text)
    m = _DUR_RE.match(str(text).lower())
    if not m:
        raise ValueError(f"cannot parse duration {text!r}; use e.g. '5min', '1h', '30d'")
    n, unit = int(m.group(1)), m.group(2)
    if unit not in _UNITS:
        raise ValueError(f"unknown duration unit {unit!r} in {text!r}")
    return n * _UNITS[unit]


def duration_label(us: int) -> str:
    for unit in ("d", "h", "min", "s", "ms"):
        size = _UNITS[unit]
        if us % size == 0 and us >= size:
            return f"{us // size}{unit}"
    return f"{us}us"


def parse_time(text) -> int:
    """Parse an ISO-8601 timestamp (must be UTC or carry an offset) to epoch microseconds."""
    if isinstance(text, (int, np.integer)):
        return int(text)
    if isinstance(text, datetime):
        dt = text
    else:
        s = str(text).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        raise ValueError(f"timestamp {text!r} has no timezone; use an explicit UTC offset or 'Z'")
    dt = dt.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = dt - epoch
    return (delta.days * 86400 + delta.seconds) * SEC + delta.microseconds


def to_iso(us) -> str | None:
    if us is None or (isinstance(us, float) and np.isnan(us)):
        return None
    us = int(us)
    dt = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=us)
    return dt.isoformat().replace("+00:00", "Z")


def now_us() -> int:
    """Wall clock (used only for received_at / ingested_at metadata, never for market time)."""
    return int(datetime.now(timezone.utc).timestamp() * SEC)


def floor_to(us, step: int, origin: int = 0):
    """Floor timestamps to a grid of ``step`` anchored at ``origin`` (works on scalars and arrays)."""
    return origin + ((np.asarray(us, dtype=np.int64) - origin) // step) * step


def day_start(us) -> np.ndarray:
    return floor_to(us, DAY, 0)


def us_to_datetime_index(us) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(us, dtype="int64"), unit="us", utc=True)


def days_between(start_us: int, end_us: int):
    """UTC calendar dates d with [d, d+1day) intersecting [start, end)."""
    d = int(floor_to(start_us, DAY))
    out = []
    while d < end_us:
        out.append(datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=d))
        d += DAY
    return [x.date() for x in out]


def date_to_us(d) -> int:
    return parse_time(datetime(d.year, d.month, d.day, tzinfo=timezone.utc))


def normalize_exchange_ts(values, unit: str) -> np.ndarray:
    """Convert exchange timestamps given in ``unit`` ('ms' or 'us' or 'ns') to microseconds."""
    v = np.asarray(values, dtype=np.int64)
    if unit == "ms":
        return v * 1000
    if unit == "us":
        return v
    if unit == "ns":
        return v // 1000
    raise ValueError(unit)


def infer_epoch_unit(values) -> str:
    """Infer ms/us/ns for epoch integers by magnitude (only valid for dates 2001..2286).

    Used as a *check* against the documented unit, never silently: callers compare it with the
    documented convention and raise if they disagree.
    """
    v = np.asarray(values, dtype=np.int64)
    if v.size == 0:
        return "unknown"
    m = int(np.median(v))
    if 1e12 <= m < 1e13:
        return "ms"
    if 1e15 <= m < 1e16:
        return "us"
    if 1e18 <= m < 1e19:
        return "ns"
    return "unknown"
