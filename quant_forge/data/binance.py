"""Binance spot klines: REST download and a local parquet cache.

The public endpoint ``GET /api/v3/klines`` needs no API key.  :func:`load_klines` keeps one parquet file per
symbol and interval (``<cache_dir>/<SYMBOL>.parquet``) and only downloads what is missing: on an update it
re-downloads from the last cached kline onwards (that kline may have been incomplete when it was stored) and
replaces the overlap.  The file format is the one used by the earlier research notebooks
(``~/.cache/quantforge/binance_klines_4h``), so existing caches are reused as they are.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from ..utils.cache import DEFAULT_CACHE_ROOT

__all__ = ["BINANCE_SPOT_KLINES_URL", "KLINE_COLUMNS", "INTERVALS", "interval_timedelta", "fetch_klines",
           "fetch_klines_hybrid", "FETCHERS", "load_klines", "default_klines_dir"]

BINANCE_SPOT_KLINES_URL = "https://api.binance.com/api/v3/klines"
KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "n_trades",
                 "taker_buy_base", "taker_buy_quote", "ignore"]
_FLOAT_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
INTERVALS = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "2h": "2h",
             "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1D", "3d": "3D", "1w": "7D"}


def interval_timedelta(interval: str) -> pd.Timedelta:
    """Bar length of a Binance interval code such as ``"4h"`` or ``"1d"``."""
    try:
        return pd.Timedelta(INTERVALS[interval])
    except KeyError:
        raise ValueError(f"unsupported interval {interval!r}; choose one of {list(INTERVALS)}") from None


def default_klines_dir(interval: str) -> Path:
    return DEFAULT_CACHE_ROOT / f"binance_klines_{interval}"


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def fetch_klines(symbol: str, interval: str = "4h", start="2017-08-01", end=None, *, limit: int = 1000,
                 pause: float = 0.15, timeout: float = 30, url: str = BINANCE_SPOT_KLINES_URL) -> pd.DataFrame:
    """Download klines of ``symbol`` (e.g. ``"BTCUSDT"``) with open time in ``[start, end)``.

    Pages of ``limit`` klines are requested until the end is reached or a short page arrives.  Numeric price and
    volume columns are floats; ``open_time``/``close_time`` stay integer milliseconds.  The last kline can be
    the one still forming when ``end`` is in the future.
    """
    interval_timedelta(interval)
    t = int(_utc(start).timestamp() * 1000)
    end_ms = int((_utc(end) if end is not None else pd.Timestamp.now(tz="UTC")).timestamp() * 1000)
    rows: list = []
    while t < end_ms:
        query = f"{url}?symbol={symbol}&interval={interval}&startTime={t}&limit={limit}"
        with urllib.request.urlopen(query, timeout=timeout) as response:
            batch = json.load(response)
        if not batch:
            break
        rows += batch
        t = batch[-1][0] + 1
        if len(batch) < limit:
            break
        time.sleep(pause)
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS).drop(columns="ignore")
    for c in _FLOAT_COLUMNS:
        df[c] = df[c].astype(float)
    return df


def fetch_klines_hybrid(symbol: str, interval: str = "1m", start="2017-08-01", end=None, **archive_kwargs) -> pd.DataFrame:
    """Archive files (``data.binance.vision``) for the bulk of the history, the REST endpoint for the rest.

    The archive publishes monthly files and daily files up to the day before today; anything after its last
    kline up to ``end`` is fetched from REST.  Output format = :func:`fetch_klines`.
    """
    from .archive import fetch_klines_archive

    end_ts = _utc(end) if end is not None else pd.Timestamp.now(tz="UTC")
    bar = interval_timedelta(interval)
    df = fetch_klines_archive(symbol, interval, start=start, end=end_ts, **archive_kwargs)
    tail_start = pd.to_datetime(int(df["open_time"].max()), unit="ms", utc=True) + bar if len(df) else _utc(start)
    if tail_start < end_ts:
        tail = fetch_klines(symbol, interval, start=tail_start, end=end_ts)
        if len(tail):
            tail["n_trades"] = tail["n_trades"].astype("int64")
            df = pd.concat([df, tail], ignore_index=True) if len(df) else tail
    return df.drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)


FETCHERS = {"rest": fetch_klines, "archive": fetch_klines_hybrid}


def load_klines(symbol: str, interval: str = "4h", *, end, start="2017-08-01", cache_dir=None,
                update: bool = True, fetcher=fetch_klines, source: str | None = None) -> pd.DataFrame:
    """Klines of ``symbol`` from the parquet cache, downloading only what is missing up to ``end``.

    The cache is considered complete when its last kline closes at or after ``end`` (open time + bar length).
    Otherwise (and if ``update``) the history from the last cached kline onwards is fetched and merged.  The
    returned frame contains every cached kline (also those after ``end``); trimming is the caller's job
    (:func:`quant_forge.data.klines_to_bars` does it).

    source  ``"rest"`` (default, :func:`fetch_klines`) or ``"archive"`` (:func:`fetch_klines_hybrid`: the public
            archive for the bulk, REST for the last days; the right choice for 1-minute data).  ``fetcher``
            overrides it.
    """
    if source is not None:
        fetcher = FETCHERS[source]
    bar = interval_timedelta(interval)
    end = _utc(end)
    cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else default_klines_dir(interval)
    path = cache_dir / f"{symbol}.parquet"
    cached = pd.read_parquet(path) if path.exists() else None
    if cached is not None and len(cached):
        last_open = pd.to_datetime(int(cached["open_time"].max()), unit="ms", utc=True)
        if last_open + bar >= end or not update:
            return cached
        fresh = fetcher(symbol, interval, start=last_open, end=end)
        merged = pd.concat([cached.loc[cached["open_time"] < fresh["open_time"].min()] if len(fresh) else cached,
                            fresh], ignore_index=True)
        merged = merged.drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    else:
        merged = fetcher(symbol, interval, start=start, end=end)
    cache_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return merged
