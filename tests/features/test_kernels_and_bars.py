import numpy as np
import pandas as pd
import pytest

from quant_forge import features as qf
from quant_forge.features import kernels
from tests.synthetic import make_bars_panel


def _wilder_reference(a, n):
    out = np.full(len(a), np.nan)
    fin = [i for i, v in enumerate(a) if np.isfinite(v)]
    if not fin or len(a) < fin[0] + n:
        return out
    k = fin[0] + n - 1
    out[k] = np.mean(a[fin[0]:k + 1])
    for j in range(k + 1, len(a)):
        out[j] = out[j - 1] + (a[j] - out[j - 1]) / n if np.isfinite(a[j]) else out[j - 1]
    return out


def test_wilder_mean_matches_reference_and_carries_nans():
    rng = np.random.default_rng(1)
    a = rng.normal(size=300)
    a[:3] = np.nan
    a[[50, 51, 200]] = np.nan
    ref = _wilder_reference(a, 14)
    np.testing.assert_array_equal(qf.wilder_mean(a, 14, use_numba=False), ref)
    np.testing.assert_array_equal(qf.wilder_mean(a, 14), ref)                 # numba path when available
    assert qf.wilder_mean(a, 14)[50] == qf.wilder_mean(a, 14)[49]               # NaN carries the last value
    assert np.isnan(qf.wilder_mean(a[:10], 14)).all()


def test_rsi_flat_and_trending():
    flat = pd.Series(np.full(40, 5.0))
    assert qf.rsi(flat, 14).iloc[-1] == 50.0
    up = pd.Series(np.arange(1, 41, dtype=float))
    assert qf.rsi(up, 14).iloc[-1] == 100.0


@pytest.mark.parametrize("window", [1, 5, 37])
def test_bars_since_rolling_max_equals_pandas_apply(window):
    rng = np.random.default_rng(window)
    x = pd.Series(rng.integers(0, 20, 400).astype(float))          # ties: first occurrence must win
    x.iloc[[30, 31, 250]] = np.nan
    ref = x.rolling(window).apply(lambda v: len(v) - 1 - np.argmax(v), raw=True)
    pd.testing.assert_series_equal(qf.bars_since_rolling_max(x, window), ref)


def test_asset_features_layout():
    bars = make_bars_panel(("BTC",), n=300)["BTC"]
    f = qf.asset_features(bars, bars_per_day=1)
    assert len(f.columns) == 72 + len(qf.AUX_COLUMNS)
    assert [c for c in f.columns if c.startswith("_")] == list(qf.AUX_COLUMNS)
    assert qf.model_columns(f) == [c for c in f.columns if not c.startswith("_")]
    fams = {c.split("_")[0] for c in qf.model_columns(f)}
    assert fams == {"v", "t", "f", "cal"}
    pd.testing.assert_frame_equal(qf.volatility_features(bars, 1), f[[c for c in f if c.startswith("v_")]])


def test_bar_features_are_point_in_time():
    """Features at time t must not change when the data after t is removed (asset, market and rank families)."""
    bars = make_bars_panel(("BTC", "ETH", "SOL", "ADA"), n=500, starts=[0, 0, 60, 120])
    full = qf.bar_features(bars, bars_per_day=1)
    cut = pd.Timestamp("2020-02-20", tz="UTC")
    trunc = qf.bar_features({a: b.loc[b.index <= cut] for a, b in bars.items()}, bars_per_day=1)
    for a in bars:
        A = full[a].loc[full[a].index <= cut]
        B = trunc[a]
        pd.testing.assert_frame_equal(A, B, check_exact=False, rtol=1e-12, atol=1e-12)
    assert {"m_avg_corr_30d", "x_beta_30d", "xr_rvol_30d", "b_dd_90d"} <= set(full["ETH"].columns)


def test_bar_features_parallel_identical():
    bars = make_bars_panel(("BTC", "ETH"), n=250)
    a = qf.bar_features(bars, bars_per_day=1)
    b = qf.bar_features(bars, bars_per_day=1, n_jobs=2)
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])


def test_add_raw_levels():
    bars = make_bars_panel(("BTC",), n=120)
    f = qf.add_raw_levels(qf.bar_features(bars, bars_per_day=1), bars)["BTC"]
    np.testing.assert_array_equal(f["raw_close"], bars["BTC"]["close"])
    assert f.columns[-5:].tolist() == ["raw_open", "raw_close", "raw_volume", "raw_quote_volume", "raw_n_trades"]
