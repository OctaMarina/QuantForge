"""D1: per-hour market conditions used to stratify the experiment. Trades only, so the same measures exist in the months
without anchors and the reference periods can be compared with them.
  rv_1h        realised volatility of the hour, sum of squared 1-minute log returns of the time_1m close (bps)
  max_abs_1m   largest absolute 1-minute return in the hour (bps)
  notional     traded notional (USDT), n_trades, volume
  depth5, spread: median 5 bps depth and spread from the strict dataset (only hours with a valid book)
Output: derived/d1/hour_table.parquet
"""
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402


def main():
    files = sorted(glob.glob(str(P.DERIVED / "datasets" / "btcusdt_initial" / "time_1m-*" / "*.parquet")))
    cols = ["t_end_ns", "bar_close", "bar_n_trades", "bar_volume_btc", "bar_notional_usdt", "spread_bps",
            "depth_bid_5bps_btc", "depth_ask_5bps_btc"]
    df = pl.concat([pl.read_parquet(f, columns=cols) for f in files]).sort("t_end_ns")
    # a minute without trades has no close: the last known close carries over, so it contributes a zero return
    df = df.with_columns(pl.col("bar_close").fill_nan(None).forward_fill())
    df = df.with_columns(r=(pl.col("bar_close").log().diff() * 1e4).fill_null(0.0),
                         hour_ns=(pl.col("t_end_ns") - 1) // P.HOUR_NS * P.HOUR_NS)
    h = df.group_by("hour_ns").agg(
        rv_1h=(pl.col("r") ** 2).sum().sqrt(), max_abs_1m=pl.col("r").abs().max(), notional=pl.col("bar_notional_usdt").sum(),
        n_trades=pl.col("bar_n_trades").sum(), volume=pl.col("bar_volume_btc").sum(), n_min=pl.len(),
        close=pl.col("bar_close").last(), spread=pl.col("spread_bps").fill_nan(None).median(),
        depth5=(pl.col("depth_bid_5bps_btc") + pl.col("depth_ask_5bps_btc")).fill_nan(None).median()).sort("hour_ns")
    h.write_parquet(P.DERIVED / "d1" / "hour_table.parquet")
    print(h.height, h.select(pl.col("rv_1h", "notional", "depth5", "spread").quantile(0.5)).rows())


if __name__ == "__main__":
    main()
