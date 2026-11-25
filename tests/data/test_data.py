import numpy as np
import pandas as pd
import pytest

from quant_forge.data import ALL_ASSETS, NEW_ASSETS, SEEN_ASSETS, TICK_SIZES, interval_timedelta, klines_to_bars, load_klines
from tests.synthetic import H4_MS, make_klines


def test_klines_to_bars_columns_and_trimming():
    k = make_klines(60)
    k.loc[10, "n_trades"] = 0
    bars = klines_to_bars(k, bar="4h", warmup_days=1, end="2020-01-08 00:00")
    assert list(bars.columns) == ["bar_start", "bar_end", "open", "high", "low", "close", "volume", "quote_volume",
                                  "n_trades", "taker_buy_base", "bar_start_us", "bar_end_us", "bar_seq", "complete"]
    assert bars.index.name == "t" and (bars.index == bars["bar_end"]).all()
    assert bars["bar_start"].iloc[0] == pd.Timestamp("2020-01-02", tz="UTC")          # one warm-up day dropped
    assert bars["bar_end"].max() <= pd.Timestamp("2020-01-08", tz="UTC")
    assert (np.diff(bars["bar_seq"]) == 1).all()
    assert (bars["bar_end_us"] - bars["bar_start_us"] == 4 * 3600 * 10 ** 6).all()
    assert bars["complete"].loc[bars["n_trades"] == 0].eq(0).all()
    assert bars["n_trades"].dtype == float


def test_bar_seq_jumps_over_missing_klines():
    k = make_klines(30).drop(index=[5, 6]).reset_index(drop=True)
    bars = klines_to_bars(k, bar="4h", warmup_days=0)
    assert (np.diff(bars["bar_seq"]) == 1).sum() == len(bars) - 2 and np.diff(bars["bar_seq"]).max() == 3


def test_interval_timedelta():
    assert interval_timedelta("4h") == pd.Timedelta(hours=4)
    assert interval_timedelta("1w") == pd.Timedelta(days=7)
    with pytest.raises(ValueError):
        interval_timedelta("7h")


def test_load_klines_incremental_cache(tmp_path):
    full = make_klines(100)
    calls = []

    def fetcher(symbol, interval, start, end):
        calls.append(pd.Timestamp(start))
        s, e = int(pd.Timestamp(start).timestamp() * 1000), int(pd.Timestamp(end).timestamp() * 1000)
        part = full.loc[(full["open_time"] >= s) & (full["open_time"] < e)].copy()
        if len(part):          # the last kline of a page is still forming: different close
            part.loc[part.index[-1], "close"] = -1.0
        return part.reset_index(drop=True)

    end1 = pd.to_datetime(full["open_time"].iloc[49], unit="ms", utc=True)
    first = load_klines("XUSDT", "4h", end=end1, start="2020-01-01", cache_dir=tmp_path, fetcher=fetcher)
    assert len(first) == 49 and first["close"].iloc[-1] == -1.0
    # complete up to end1 -> no new request
    load_klines("XUSDT", "4h", end=end1, cache_dir=tmp_path, fetcher=fetcher)
    assert len(calls) == 1
    end2 = pd.to_datetime(full["open_time"].iloc[-1], unit="ms", utc=True) + pd.Timedelta(hours=4)
    second = load_klines("XUSDT", "4h", end=end2, cache_dir=tmp_path, fetcher=fetcher)
    assert calls[1] == pd.to_datetime(full["open_time"].iloc[48], unit="ms", utc=True)     # restart at last kline
    assert len(second) == 100 and second["open_time"].is_unique
    assert second["close"].iloc[48] == full["close"].iloc[48]                              # stale kline replaced
    assert (tmp_path / "XUSDT.parquet").exists()


def test_universe():
    assert len(SEEN_ASSETS) == len(NEW_ASSETS) == 20 and not set(SEEN_ASSETS) & set(NEW_ASSETS)
    assert set(TICK_SIZES) == set(ALL_ASSETS)
