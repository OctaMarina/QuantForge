"""Coinbase Exchange public candles (USD spot reference and the USDT-USD FX rate).

Verified 2026-10-20 (docs.cdp.coinbase.com ... get-product-candles):
* GET /products/{product_id}/candles?granularity=60&start=&end= ; granularity in
  {60, 300, 900, 3600, 21600, 86400}; max 300 candles per request.
* Bucket = [time, low, high, open, close, volume]; ``time`` is the bucket START.
* "No data is published for intervals where there are no ticks" -> a missing minute is unknown
  (no trade OR missing data) and is never treated as a valid price.
* Volume unit is not stated by the docs (assumed base units; not used by any feature).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import timeutil as tu
from ..http import Http
from ..storage import RawStore

BASE = "https://api.exchange.coinbase.com"
PROVIDER = "coinbase"


def normalize_candles(rows, granularity_s: int) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["period_start_us", "period_end_us", "open", "high", "low", "close", "volume"])
    a = np.asarray(rows, dtype=np.float64)
    df = pd.DataFrame({
        "period_start_us": (a[:, 0].astype(np.int64)) * tu.SEC,
        "low": a[:, 1], "high": a[:, 2], "open": a[:, 3], "close": a[:, 4], "volume": a[:, 5],
    })
    df["period_end_us"] = df["period_start_us"] + granularity_s * tu.SEC
    return df.drop_duplicates("period_start_us").sort_values("period_start_us").reset_index(drop=True)


class Coinbase:
    def __init__(self, http: Http, raw: RawStore):
        self.http = http
        self.raw = raw

    def candles(self, product: str, start_us: int, end_us: int, granularity_s: int = 60) -> pd.DataFrame:
        step = 300 * granularity_s * tu.SEC
        rows = []
        cur = start_us
        while cur < end_us:
            nxt = min(cur + step, end_us)
            rel = f"candles/{product}/{granularity_s}/{tu.to_iso(cur)}_{tu.to_iso(nxt)}.json".replace(":", "")
            cached = self.raw.latest(PROVIDER, rel)
            if cached is not None:
                data = cached.read_bytes()
            else:
                # end is inclusive on Coinbase; request [cur, nxt - 1 bucket] to avoid overlap
                r = self.http.get(f"{BASE}/products/{product}/candles", PROVIDER, params={
                    "granularity": granularity_s, "start": tu.to_iso(cur),
                    "end": tu.to_iso(nxt - granularity_s * tu.SEC)})
                data = r.content
                self.raw.put(PROVIDER, rel, data, {"url": r.url, "received_at_us": tu.now_us(),
                                                   "http_date": r.headers.get("Date")})
            page = json.loads(data)
            if isinstance(page, dict):
                raise RuntimeError(f"coinbase error for {product}: {page}")
            rows.extend(page)
            cur = nxt
        df = normalize_candles(rows, granularity_s)
        return df[(df.period_start_us >= start_us) & (df.period_start_us < end_us)].reset_index(drop=True)
