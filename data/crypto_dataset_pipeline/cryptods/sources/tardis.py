"""Tardis.dev downloadable CSV datasets (optional source for historical L2 / quotes / option chains /
liquidations).

Verified 2026-10-20 (docs.tardis.dev/downloadable-csv-files):
* URL: https://datasets.tardis.dev/v1/{exchange}/{data_type}/{YYYY}/{MM}/{DD}/{symbol}.csv.gz
* API key optional; without a key only the FIRST DAY OF EACH MONTH is downloadable.  Free days are
  discontinuous samples, never a continuous history: every feature built from them is null on
  other days (reason not_available_in_period).
* All timestamps are microseconds since epoch UTC.  ``timestamp`` = exchange time (falls back to
  local_timestamp when the exchange gives none); ``local_timestamp`` = message arrival time at
  Tardis' collector (an observed reception log, usable for availability replay).
* incremental_book_L2: amount is the ABSOLUTE level size (0 = remove); rows before the first
  is_snapshot=true must be skipped; a snapshot after non-snapshot rows resets the book.
* liquidations.side is the "liquidation side": "buy - short position was liquidated", "sell - long
  position was liquidated", "unknown"; amount is "liquidation amount as provided by exchange"
  (units therefore follow the exchange: contracts for inverse venues).
* options_chain: IV unit and delta convention are not documented by Tardis (values are passed
  through from the exchange, e.g. Deribit).
The key is read from the environment variable named in config ``api_keys.tardis_env``.
"""
from __future__ import annotations

import gzip
import io
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from ..http import Http
from ..storage import RawStore

BASE = "https://datasets.tardis.dev/v1"
PROVIDER = "tardis"


def url(exchange: str, data_type: str, day, symbol: str) -> str:
    return f"{BASE}/{exchange}/{data_type}/{day.year:04d}/{day.month:02d}/{day.day:02d}/{symbol}.csv.gz"


def free_without_key(day) -> bool:
    return day.day == 1


class Tardis:
    def __init__(self, http: Http, raw: RawStore, key_env: str = "TARDIS_API_KEY"):
        self.http = http
        self.raw = raw
        self.key = os.environ.get(key_env) or None

    def fetch(self, exchange: str, data_type: str, day, symbol: str) -> bytes:
        rel = f"{exchange}/{data_type}/{day.isoformat()}/{symbol}.csv.gz"
        latest = self.raw.latest(PROVIDER, rel)
        if latest is not None:
            return latest.read_bytes()
        if self.key is None and not free_without_key(day):
            raise PermissionError(f"Tardis {day} requires an API key (only the first day of each month is free)")
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else None
        r = self.http.get(url(exchange, data_type, day, symbol), PROVIDER, headers=headers)
        self.raw.put(PROVIDER, rel, r.content, {"url": r.url, "api_key_used": bool(self.key)})
        return r.content


def read_csv_gz(data: bytes, str_cols=()) -> pd.DataFrame:
    raw = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
    co = pacsv.ConvertOptions(column_types={c: pa.string() for c in str_cols}) if str_cols else None
    return pacsv.read_csv(io.BytesIO(raw), convert_options=co).to_pandas()


def iter_csv_gz_batches(data: bytes, block_size: int = 1 << 24):
    raw = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
    reader = pacsv.open_csv(io.BytesIO(raw), read_options=pacsv.ReadOptions(block_size=block_size))
    for batch in reader:
        yield batch.to_pandas()


def normalize_options_chain(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "ts_us": df["timestamp"].astype(np.int64), "received_at_us": df["local_timestamp"].astype(np.int64),
        "instrument": df["symbol"], "expiry_us": df["expiration"].astype(np.int64),
        "strike": df["strike_price"].astype(float), "type": df["type"].str.lower(),
        "bid_iv": pd.to_numeric(df["bid_iv"], errors="coerce"), "ask_iv": pd.to_numeric(df["ask_iv"], errors="coerce"),
        "mark_iv": pd.to_numeric(df["mark_iv"], errors="coerce"), "delta": pd.to_numeric(df["delta"], errors="coerce"),
        "underlying_price": pd.to_numeric(df["underlying_price"], errors="coerce"),
        "bid_price": pd.to_numeric(df["bid_price"], errors="coerce"),
        "ask_price": pd.to_numeric(df["ask_price"], errors="coerce"),
        "bid_amount": pd.to_numeric(df["bid_amount"], errors="coerce"),
        "ask_amount": pd.to_numeric(df["ask_amount"], errors="coerce"),
        "state": "open", "source": "tardis_options_chain",
    })
    return out


def normalize_quotes(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "ts_us": df["timestamp"].astype(np.int64), "received_at_us": df["local_timestamp"].astype(np.int64),
        "bid": pd.to_numeric(df["bid_price"], errors="coerce"), "bid_qty": pd.to_numeric(df["bid_amount"], errors="coerce"),
        "ask": pd.to_numeric(df["ask_price"], errors="coerce"), "ask_qty": pd.to_numeric(df["ask_amount"], errors="coerce"),
        "source": "tardis_quotes"})


def normalize_liquidations(df: pd.DataFrame, exchange: str) -> pd.DataFrame:
    side = df["side"].str.lower()
    pos = np.where(side == "sell", "long", np.where(side == "buy", "short", "unknown"))
    return pd.DataFrame({
        "ts_us": df["timestamp"].astype(np.int64), "received_at_us": df["local_timestamp"].astype(np.int64),
        "exchange": exchange, "symbol": df["symbol"], "id": df.get("id"),
        "liquidated_position": pos, "qty": pd.to_numeric(df["amount"], errors="coerce"),
        "price": pd.to_numeric(df["price"], errors="coerce"),
        "price_kind": "as_reported_by_tardis", "source": "tardis_liquidations"})


def book_events_from_incremental(df: pd.DataFrame) -> pd.DataFrame:
    """Tardis incremental_book_L2 rows -> generic book events (one row per level change)."""
    return pd.DataFrame({
        "ts_us": df["timestamp"].astype(np.int64), "received_at_us": df["local_timestamp"].astype(np.int64),
        "is_snapshot": df["is_snapshot"].astype(str).str.lower().eq("true"),
        "side": np.where(df["side"].str.lower().str.startswith("b"), 1, -1).astype(np.int8),
        "price": df["price"].astype(str), "qty": df["amount"].astype(str)})
