"""Deribit public API: DVOL index candles and option tickers.

Verified 2026-10-20:
* public/get_volatility_index_data(currency, start_timestamp, end_timestamp, resolution) with
  resolution in {1, 60, 3600, 43200, 1D}; result.data = [[ts_ms, open, high, low, close], ...];
  result.continuation -> next end_timestamp (null when done).  The docs do NOT state the value unit.
  Deribit's DVOL methodology article (insights.deribit.com) describes DVOL as the 30-day annualised
  implied volatility and its example (DVOL=90 -> ~4.5%/day) implies annualised percentage points.
  The pipeline stores values as published with unit "annualised vol, percentage points (per
  Deribit methodology article)" and flags any observation < 2 as unit-suspicious (not rescaled).
* Whether ``ts`` labels the candle start or end is not documented; the pipeline treats it as the
  START (conservative: available at ts + resolution).
* public/ticker: option fields bid_iv, ask_iv, mark_iv (unit not stated in docs), greeks.delta
  ("Black Scholes Delta"), underlying_price, underlying_index, best_bid_price, best_ask_price
  (option prices are quoted in the underlying currency on Deribit inverse options), timestamp (ms).
* Historical option chains are not available from the free API (only the current state); history
  comes from live collection or Tardis free first-of-month files.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .. import timeutil as tu
from ..http import Http
from ..storage import RawStore

BASE = "https://www.deribit.com/api/v2"
PROVIDER = "deribit"
RES_US = {"1": tu.SEC, "60": tu.MIN, "3600": tu.HOUR, "43200": 12 * tu.HOUR, "1D": tu.DAY}
_MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
_OPT_RE = re.compile(r"^(?P<cur>[A-Z]+)(?:_[A-Z]+)?-(?P<d>\d{1,2})(?P<m>[A-Z]{3})(?P<y>\d{2})-(?P<k>[\dd.]+)-(?P<t>[CP])$")


def parse_option_name(name: str):
    """'BTC-27DEC24-100000-C' -> (currency, expiry_us at 08:00 UTC, strike, 'C'|'P').  Deribit options
    expire at 08:00 UTC (contract specification); strikes may use 'd' as decimal separator."""
    m = _OPT_RE.match(name)
    if not m:
        return None
    y = 2000 + int(m.group("y"))
    mon = _MONTHS[m.group("m")]
    d = int(m.group("d"))
    from datetime import datetime, timezone
    exp = tu.parse_time(datetime(y, mon, d, 8, 0, tzinfo=timezone.utc))
    strike = float(m.group("k").replace("d", "."))
    return m.group("cur"), exp, strike, m.group("t")


def normalize_dvol(rows, currency: str, resolution: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["currency", "period_start_us", "period_end_us", "open", "high", "low", "close"])
    a = np.asarray(rows, dtype=np.float64)
    ps = a[:, 0].astype(np.int64) * 1000
    df = pd.DataFrame({"currency": currency, "period_start_us": ps, "period_end_us": ps + RES_US[resolution],
                       "open": a[:, 1], "high": a[:, 2], "low": a[:, 3], "close": a[:, 4]})
    df["unit_suspicious"] = df["close"] < 2.0
    return df.drop_duplicates("period_start_us").sort_values("period_start_us").reset_index(drop=True)


class Deribit:
    def __init__(self, http: Http, raw: RawStore):
        self.http = http
        self.raw = raw

    def dvol(self, currency: str, start_us: int, end_us: int, resolution: str = "60") -> pd.DataFrame:
        rows = []
        end_ms = end_us // 1000
        start_ms = start_us // 1000
        while True:
            r = self.http.get(f"{BASE}/public/get_volatility_index_data", PROVIDER, params={
                "currency": currency, "start_timestamp": start_ms, "end_timestamp": end_ms, "resolution": resolution})
            self.raw.put(PROVIDER, f"dvol/{currency}/{resolution}/{start_ms}_{end_ms}.json", r.content,
                         {"url": r.url, "received_at_us": tu.now_us()})
            res = json.loads(r.content)["result"]
            rows.extend(res.get("data") or [])
            cont = res.get("continuation")
            if not cont or cont <= start_ms or not res.get("data"):
                break
            end_ms = int(cont)
        df = normalize_dvol(rows, currency, resolution)
        return df[(df.period_start_us >= start_us) & (df.period_start_us < end_us)].reset_index(drop=True)

    def option_instruments(self, currency: str) -> list[dict]:
        r = self.http.get(f"{BASE}/public/get_instruments", PROVIDER,
                          params={"currency": currency, "kind": "option", "expired": "false"})
        return json.loads(r.content)["result"]

    def ticker(self, instrument: str) -> dict:
        r = self.http.get(f"{BASE}/public/ticker", PROVIDER, params={"instrument_name": instrument})
        return json.loads(r.content)["result"]


def normalize_ticker(msg: dict, received_at_us: int | None = None) -> dict | None:
    """Deribit ticker (REST result or WS ticker.* data) -> option quote row."""
    name = msg.get("instrument_name")
    p = parse_option_name(name or "")
    if p is None:
        return None
    cur, exp, k, typ = p
    g = msg.get("greeks") or {}
    return {"ts_us": int(msg["timestamp"]) * 1000, "received_at_us": received_at_us, "instrument": name,
            "currency": cur, "expiry_us": exp, "strike": k, "type": "call" if typ == "C" else "put",
            "bid_iv": _f(msg.get("bid_iv")), "ask_iv": _f(msg.get("ask_iv")), "mark_iv": _f(msg.get("mark_iv")),
            "delta": _f(g.get("delta")), "underlying_price": _f(msg.get("underlying_price")),
            "bid_price": _f(msg.get("best_bid_price")), "ask_price": _f(msg.get("best_ask_price")),
            "bid_amount": _f(msg.get("best_bid_amount")), "ask_amount": _f(msg.get("best_ask_amount")),
            "state": msg.get("state"), "source": "deribit_ticker"}


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan
