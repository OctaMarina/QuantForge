"""Synthetic market data shared by the tests of the data / features / meta-labeling packages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_forge.data import klines_to_bars

H4_MS = 4 * 3600 * 1000


def make_klines(n: int = 400, start: str = "2020-01-01", seed: int = 0, interval_ms: int = H4_MS,
                vol: float = 0.01) -> pd.DataFrame:
    """Binance-like klines of a geometric random walk (open_time / close_time in ms)."""
    rng = np.random.default_rng(seed)
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    close = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    open_ = np.r_[100.0, close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, vol, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, vol, n))
    volume = rng.lognormal(3, 0.5, n)
    open_time = t0 + interval_ms * np.arange(n)
    return pd.DataFrame({"open_time": open_time, "open": open_, "high": high, "low": low, "close": close,
                         "volume": volume, "close_time": open_time + interval_ms - 1, "quote_volume": volume * close,
                         "n_trades": rng.integers(50, 500, n), "taker_buy_base": volume * rng.uniform(0.3, 0.7, n),
                         "taker_buy_quote": volume * close * 0.5})


def make_bars_panel(assets=("BTC", "ETH", "SOL"), n: int = 700, seed: int = 0, starts=None) -> dict:
    """``{asset: bars}`` of synthetic 4h klines (later assets may start later via ``starts`` offsets)."""
    out = {}
    for i, a in enumerate(assets):
        offset = 0 if starts is None else starts[i]
        k = make_klines(n - offset, start=str(pd.Timestamp("2020-01-01") + pd.Timedelta(hours=4 * offset)), seed=seed + i)
        out[a] = klines_to_bars(k, bar="4h", warmup_days=0)
    return out


def make_trade_panel(n: int = 900, assets=("A1", "A2", "A3", "A4"), seed: int = 0, signal: float = 1.0) -> pd.DataFrame:
    """Trades of one strategy with decision / entry / exit times, a label driven by ``f_inf`` and noise features."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2019-01-01", tz="UTC")
    dec = start + pd.to_timedelta(np.sort(rng.uniform(0, 4 * 365, n)), unit="D")
    hold = pd.to_timedelta(rng.uniform(0.5, 10, n), unit="D")
    f_inf = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(signal * f_inf - 0.4)))
    y = (rng.uniform(size=n) < p).astype(int)
    ret = np.where(y == 1, rng.uniform(0.01, 0.08, n), -rng.uniform(0.005, 0.04, n))
    df = pd.DataFrame({"strategy": "STRAT", "asset": rng.choice(list(assets), n), "side": 1,
                       "decision_time": dec, "entry_time": dec + pd.Timedelta(hours=4), "exit_time": dec + hold,
                       "trade_return": ret, "y": y, "v_rvol_30d": rng.uniform(0.2, 1.0, n),
                       "f_inf": f_inf, "f_inf_copy": f_inf + rng.normal(0, 0.05, n)})
    for j in range(6):
        df[f"t_noise_{j}"] = rng.normal(size=n)
    df["m_n_assets"] = float(len(assets))
    return df.sort_values("decision_time").reset_index(drop=True)
