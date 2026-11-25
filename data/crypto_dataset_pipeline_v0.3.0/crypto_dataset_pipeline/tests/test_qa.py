"""Trades->1m bars vs native klines: exact agreement where definitions coincide; differences reported."""
import numpy as np
import pandas as pd

from conftest import random_trades
from helpers import build_bars
from cryptods.qa import compare_with_klines


def _klines_from_trades(t, spec):
    """Reference klines computed independently with pandas groupby (Binance kline semantics)."""
    g = t.assign(m=(t.ts_us // 60_000_000) * 60_000_000, p=t.price_i / spec.price_scale, q=t.qty_i / spec.qty_scale,
                 n=t.notional_i / spec.notional_scale, b=np.where(t.aggressor > 0, t.qty_i / spec.qty_scale, 0.0))
    k = g.sort_values("trade_id").groupby("m").agg(open=("p", "first"), high=("p", "max"), low=("p", "min"),
                                                     close=("p", "last"), volume=("q", "sum"), quote_volume=("n", "sum"),
                                                     n_trades=("p", "size"), taker_buy_base=("b", "sum")).reset_index()
    return k.rename(columns={"m": "period_start_us"})


def test_kline_crosscheck_exact_and_detects_differences(spec, tmp_path):
    t = random_trades(spec, 20000, seed=8, mean_dt=100_000)
    bars, _ = build_bars(spec, t, tmp_path, "time", interval_us=60_000_000, watermark=int(t.ts_us.max()))
    k = _klines_from_trades(t, spec)
    r = compare_with_klines(bars, k)
    assert r["minutes_compared"] > 5
    assert all(v == 0 for kk, v in r.items() if kk.startswith("mismatch_"))
    k2 = k.copy()
    k2.loc[2, "n_trades"] += 1      # e.g. a source that counts aggregated messages differently
    k2.loc[3, "high"] += 0.01
    r2 = compare_with_klines(bars, k2)
    assert r2["mismatch_n_trades"] == 1 and r2["mismatch_high"] == 1
