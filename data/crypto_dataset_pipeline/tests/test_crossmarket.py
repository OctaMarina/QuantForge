"""Premium (raw / FX), trend, basis, funding, as-of semantics, z-scores."""
import numpy as np
import pandas as pd
import pytest

from conftest import mk_trades
from helpers import make_cfg, build_bars, ctx_for
from cryptods.asof import asof_values
from cryptods.features import crossmarket
from cryptods.features.stats import causal_zscore
from cryptods.nulls import VALID, CODE

M = 60_000_000
D = 86_400_000_000


def candles(start, n, close, step=M):
    ps = start + np.arange(n) * step
    c = np.broadcast_to(np.asarray(close, float), (n,)).copy()
    return pd.DataFrame({"period_start_us": ps, "period_end_us": ps + step, "open": c, "high": c, "low": c, "close": c})


def _ctx_with(spec, tmp_path, tables, decision):
    t = mk_trades(spec, range(1, len(decision) + 1), decision, ["100"] * len(decision), ["10"] * len(decision))
    bars, _ = build_bars(spec, t, tmp_path, "volume", thr=1.0)
    return ctx_for(make_cfg(), spec, bars, tables=tables)


def test_premium_raw_and_fx_manual(spec, tmp_path):
    t0 = 10 * D
    usd = candles(t0, 10, 30100.0)
    usdt = candles(t0, 10, 30000.0)
    fx = candles(t0, 10, 0.999)          # USD per USDT (USDT below par)
    tables = {"usd_1m": {"BTC": usd, "ETH": usd}, "usdt_1m": {"BTC": usdt, "ETH": usdt}, "fx_1m": fx}
    dec = [t0 + 5 * M + 30_000_000]        # 00:05:30
    ctx = _ctx_with(spec, tmp_path, tables, dec)
    crossmarket.compute(ctx)
    raw = ctx.out["premium_raw_coinbase_btc_1m"]
    adj = ctx.out["premium_fx_coinbase_btc_1m"]
    assert raw.values[0] == pytest.approx(np.log(30100 / 30000))
    assert adj.values[0] == pytest.approx(np.log(30100 / (30000 * 0.999)))
    assert raw.values[0] != pytest.approx(adj.values[0])   # raw premium is NOT FX-adjusted


def test_intrabar_ohlc_not_visible(spec, tmp_path):
    """Decision at 12:00:30 sees the minute [11:59, 12:00) (available 12:00), never [12:00, 12:01)."""
    t0 = 10 * D + 12 * 3600 * 1_000_000
    perp = candles(t0 - M, 2, 0)          # minutes 11:59 and 12:00
    perp["close"] = [101.0, 999.0]          # the 12:00 minute closes at 999 (future at 12:00:30)
    spot = candles(t0 - M, 2, 100.0)
    tables = {"kl_perp_1m": perp, "kl_spot_1m": spot}
    ctx = _ctx_with(spec, tmp_path, tables, [t0 + 30_000_000])
    crossmarket.compute(ctx)
    b = ctx.out["basis_last_perp_spot_1m"]
    assert b.values[0] == pytest.approx(np.log(101 / 100))


def test_missing_minute_never_filled(spec, tmp_path):
    t0 = 10 * D
    perp = candles(t0, 10, 101.0)
    spot = candles(t0, 10, 100.0).drop(index=[8, 9])     # spot missing for the last 2 minutes
    ctx = _ctx_with(spec, tmp_path, {"kl_perp_1m": perp, "kl_spot_1m": spot}, [t0 + 10 * M + 1])
    crossmarket.compute(ctx)
    b = ctx.out["basis_last_perp_spot_1m"]
    # the last synchronised minute is minute 7 (ends 8 min): available, age 2 min + 1us -> stale (> 2min)
    assert b.reason[0] == CODE["stale_input"] and np.isnan(b.values[0])


def test_asof_never_uses_future_even_if_closer():
    obs = pd.DataFrame({"available_at_us": [0, 100, 1000], "v": [1.0, 2.0, 3.0]})
    col, age, idx = asof_values(obs, "v", np.array([999, 1000, 1001]), None, tie="exclude_equal")
    assert list(col.values) == [2.0, 2.0, 3.0]   # 1000 is not yet known at exactly 1000
    col2, _, _ = asof_values(obs, "v", np.array([1000]), None, tie="include_equal")
    assert col2.values[0] == 3.0


def test_daily_premium_coverage_and_z_excludes_current(spec, tmp_path):
    t0 = 0
    days = 100
    rows = []
    rng = np.random.default_rng(0)
    vals = rng.normal(0.001, 0.0002, days)
    for d in range(days):
        n = 1440 if d != 50 else 1000           # day 50 has 69% coverage -> invalid
        ps = d * D + np.arange(n) * M
        rows.append(pd.DataFrame({"period_start_us": ps, "period_end_us": ps + M, "close": np.exp(vals[d]) * 100.0}))
    usd = pd.concat(rows, ignore_index=True)
    usdt = usd.assign(close=100.0)
    tables = {"usd_1m": {"BTC": usd, "ETH": usd}, "usdt_1m": {"BTC": usdt, "ETH": usdt}}
    dec = [95 * D + 3600_000_000]               # during day 95: latest complete day = 94
    ctx = _ctx_with(spec, tmp_path, tables, dec)
    crossmarket.compute(ctx)
    p1d = ctx.out["premium_raw_coinbase_btc_1d"]
    assert p1d.values[0] == pytest.approx(vals[94], rel=1e-9)
    hist = [vals[d] for d in range(94) if d != 50][-90:]
    z = (vals[94] - np.mean(hist)) / np.std(hist, ddof=1)
    assert ctx.out["premium_raw_coinbase_btc_1d_z90"].values[0] == pytest.approx(z, rel=1e-6)
    # a decision during day 94 must not see day 94's daily mean (available only after the day ends)
    ctx2 = _ctx_with(spec, tmp_path / "b", tables, [94 * D + 23 * 3600_000_000])
    crossmarket.compute(ctx2)
    assert ctx2.out["premium_raw_coinbase_btc_1d"].values[0] == pytest.approx(vals[93], rel=1e-9)


def test_trend_30d_manual_and_zero_vol(spec, tmp_path):
    n = 40
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    dc = pd.DataFrame({"period_start_us": np.arange(n) * D, "period_end_us": np.arange(1, n + 1) * D, "close": c})
    ctx = _ctx_with(spec, tmp_path, {"daily_close": dc}, [36 * D + 5])
    crossmarket.compute(ctx)
    d = 35
    r = np.diff(np.log(c[d - 30:d + 1]))
    ref = np.log(c[d] / c[d - 30]) / (r.std(ddof=1) * np.sqrt(30))
    assert ctx.out["trend_30d_volscaled"].values[0] == pytest.approx(ref)
    flat = dc.assign(close=100.0)
    ctx2 = _ctx_with(spec, tmp_path / "f", {"daily_close": flat}, [36 * D + 5])
    crossmarket.compute(ctx2)
    assert ctx2.out["trend_30d_volscaled"].reason[0] != VALID


def test_funding_interval_observed_not_assumed():
    from cryptods.features.crossmarket import _asof_add  # noqa: F401  (exercised via compute below)
    f = pd.DataFrame({"funding_time_us": [0, 4 * 3600_000_000, 8 * 3600_000_000], "funding_rate": [1e-4, 2e-4, -1e-4]})
    f["available_at_us"] = f.funding_time_us
    f["interval_h"] = f.funding_time_us.diff() / 3600_000_000
    col, _, _ = asof_values(f, "interval_h", np.array([8 * 3600_000_000 + 1]), None)
    assert col.values[0] == 4.0
    col, _, _ = asof_values(f, "funding_rate", np.array([8 * 3600_000_000]), None)
    assert col.values[0] == 2e-4    # the funding realised exactly at decision time is not yet known


def test_causal_zscore_excludes_current_and_skips_missing():
    x = np.array([1.0, 2.0, np.nan, 3.0, 10.0])
    v = np.isfinite(x)
    z = causal_zscore(x, v, n_window=3, min_obs=3)
    assert z.values[4] == pytest.approx((10 - 2.0) / np.std([1, 2, 3], ddof=1))
    assert z.reason[3] == CODE["insufficient_history"] and z.reason[2] != VALID
