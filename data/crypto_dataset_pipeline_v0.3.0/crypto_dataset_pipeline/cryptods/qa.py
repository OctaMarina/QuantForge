"""Quality checks against venue-native aggregates.

Binance klines (labelled by open time) are defined on the same executions as the trade archive, so for
1-minute intervals the following must match exactly when both sources are complete:
open/high/low/close, volume, quote volume, number of trades and taker-buy base volume (= our
observed-aggressor buy volume).  Differences are reported, never patched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compare_with_klines(bars_1m: pd.DataFrame, klines: pd.DataFrame, tol_rel: float = 1e-12) -> dict:
    """``bars_1m``: float view of 1-minute time bars built from trades; ``klines``: normalised klines."""
    b = bars_1m[(bars_1m["complete"] == 1) & (bars_1m["n_trades"] > 0)]
    k = klines.rename(columns={"period_start_us": "bar_start_us"})
    m = b.merge(k, on="bar_start_us", suffixes=("_tr", "_kl"))
    out = {"minutes_compared": int(len(m)), "minutes_only_in_trades": int(len(b) - len(m)),
           "minutes_only_in_klines": int(len(k[k["bar_start_us"].between(b["bar_start_us"].min(), b["bar_start_us"].max())]) - len(m))
           if len(b) else 0}
    if not len(m):
        return out
    checks = {"open": ("open_tr", "open_kl"), "high": ("high_tr", "high_kl"), "low": ("low_tr", "low_kl"),
              "close": ("close_tr", "close_kl"), "volume": ("volume_tr", "volume_kl"),
              "quote_volume": ("notional", "quote_volume"), "taker_buy_base": ("buy_volume", "taker_buy_base")}
    for name, (a, c) in checks.items():
        x, y = m[a].to_numpy(float), m[c].to_numpy(float)
        bad = ~np.isclose(x, y, rtol=tol_rel, atol=1e-9)
        out[f"mismatch_{name}"] = int(bad.sum())
    nt = m["n_trades_tr"].to_numpy() != m["n_trades_kl"].to_numpy(dtype=float)
    out["mismatch_n_trades"] = int(nt.sum())
    out["examples"] = m.loc[nt, ["bar_start_us", "n_trades_tr", "n_trades_kl"]].head(5).to_dict("records")
    return out
