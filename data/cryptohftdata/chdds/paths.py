"""Filesystem layout. Raw data is read-only; everything we produce goes under DERIVED."""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = Path(os.environ.get("CHDDS_RAW_ROOT", PROJECT_ROOT / "cryptohftdata_raw"))
EXCHANGE = "binance_futures"
SYMBOL = "BTCUSDT"
DERIVED = Path(os.environ.get("CHDDS_DERIVED", PROJECT_ROOT / "derived"))

STREAMS = ("trades", "orderbook", "ticker", "liquidations", "open_interest", "mark_price")

# Declared inventory of the download (inclusive dates, UTC).
HISTORY_START = date(2025, 6, 28)
HISTORY_END = date(2026, 10, 31)

NS = 1_000_000_000
MS_NS = 1_000_000
HOUR_NS = 3600 * NS


def raw_key(day: date, hour: int, stream: str) -> str:
    return f"{EXCHANGE}/{day.isoformat()}/{hour:02d}/{SYMBOL}_{stream}.parquet"


def raw_path(day: date, hour: int, stream: str) -> Path:
    """Raw file path. Older S3 downloads are `.parquet.zst` (outer zstd); REST downloads from
    2026-08-19 10:00 are plain `.parquet`. Returns the existing variant (zst preferred), else the zst name."""
    base = RAW_ROOT / raw_key(day, hour, stream)
    z = base.with_name(base.name + ".zst")
    if z.exists():
        return z
    if base.exists():
        return base
    return z


def hour_start_ns(day: date, hour: int) -> int:
    dt = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
    return int(dt.timestamp()) * NS


def iter_hours(start: date = HISTORY_START, end: date = HISTORY_END):
    d = start
    while d <= end:
        for h in range(24):
            yield d, h
        d += timedelta(days=1)


def ns_to_iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / NS, tz=timezone.utc).isoformat(timespec="milliseconds")


def hour_index(ns: int) -> int:
    """Hours since HISTORY_START 00:00 UTC (grid index)."""
    return (ns - hour_start_ns(HISTORY_START, 0)) // HOUR_NS
