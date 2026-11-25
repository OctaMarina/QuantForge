import warnings

import numpy as np
import pandas as pd

from quant_forge import features as qf
from quant_forge.features.elapsed import asynchronous_panel_features, elapsed_asset_features, elapsed_bar_features
from tests.synthetic import make_bars_panel


def irregular_bars(n=4000, seed=0, start="2020-01-01", mean_minutes=90.0):
    """Event-like bars: exponential durations, a geometric random walk, volumes and taker flow."""
    rng = np.random.default_rng(seed)
    dur = rng.exponential(mean_minutes * 60e6, n).astype(np.int64) + 1_000_000          # microseconds
    end_us = int(pd.Timestamp(start, tz="UTC").value // 1000) + np.cumsum(dur)
    start_us = np.r_[end_us[0] - dur[0], end_us[:-1] + 1000]
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[100.0, close[:-1]]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.lognormal(3, 0.3, n)
    idx = pd.DatetimeIndex(pd.to_datetime(end_us, unit="us", utc=True), name="t")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol,
                         "quote_volume": vol * close, "n_trades": rng.integers(10, 100, n).astype(float),
                         "taker_buy_base": vol * rng.uniform(0.3, 0.7, n), "bar_start_us": start_us,
                         "bar_end_us": end_us}, index=idx)


def test_time_bars_match_bar_count_definitions():
    bars = make_bars_panel(("BTC", "ETH"), n=2200)
    el = elapsed_asset_features(bars["BTC"])
    ref = qf.asset_features(bars["BTC"], 6)
    for col in ["v_rvol_30d", "t_ret_7d", "t_ret_30d_z", "t_dd_90d", "t_up_30d", "cal_hour_sin", "cal_dow_cos"]:
        both = el[col].notna() & ref[col].notna()
        assert both.mean() > 0.4
        np.testing.assert_allclose(el.loc[both, col], ref.loc[both, col], rtol=1e-9, atol=1e-12)


def test_point_in_time_per_asset_and_panel():
    bars = {"BTC": irregular_bars(seed=0), "A": irregular_bars(seed=1, mean_minutes=150),
            "B": irregular_bars(seed=2, start="2020-03-01", mean_minutes=60)}
    full = elapsed_bar_features(bars)
    tau = pd.Timestamp("2020-06-10 07:13", tz="UTC")
    part = elapsed_bar_features({a: b.loc[b.index <= tau] for a, b in bars.items()})
    for a in bars:
        lhs = full[a].loc[full[a].index <= tau]
        pd.testing.assert_frame_equal(lhs, part[a][lhs.columns], check_exact=False, rtol=1e-12, atol=1e-12)


def test_asof_alignment_uses_only_bars_ended_at_or_before_t():
    a = irregular_bars(seed=4)
    b = a.copy()
    b.index = b.index + pd.Timedelta(seconds=1)          # B's bars end one second after A's
    b["bar_end_us"] = b["bar_end_us"] + 1_000_000
    own = {"BTC": elapsed_asset_features(a), "B": elapsed_asset_features(b)}
    panel = asynchronous_panel_features(own)
    # at A's bar ends, B's latest completed bar is the previous one: its 1-day return is B's value one bar earlier
    m = panel["BTC"]["m_ret_1d"].to_numpy()
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expect = np.nanmean(np.vstack([own["BTC"]["t_ret_1d"].to_numpy(), np.r_[np.nan, own["B"]["t_ret_1d"].to_numpy()[:-1]]]), axis=0)
    ok = np.isfinite(expect)
    np.testing.assert_allclose(m[ok], expect[ok])
    assert (panel["B"]["m_n_assets"].iloc[200:] == 2).all()


def test_staleness_drops_inactive_assets():
    a = irregular_bars(seed=5)
    b = irregular_bars(seed=6).iloc[:200]                # B stops trading early
    own = {"BTC": elapsed_asset_features(a), "B": elapsed_asset_features(b)}
    p = asynchronous_panel_features(own, max_staleness="2D")["BTC"]
    late = p.index > b.index[-1] + pd.Timedelta("2D")
    assert late.any() and (p.loc[late, "m_n_assets"] == 1).all()
