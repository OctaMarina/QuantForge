import numpy as np
import pandas as pd
import pytest

from quant_forge import features as qf


def test_ffd_weights_closed_forms():
    np.testing.assert_array_equal(qf.ffd_weights(0.0), [1.0])
    np.testing.assert_array_equal(qf.ffd_weights(1.0), [-1.0, 1.0])
    w = qf.ffd_weights(0.4, 1e-4)
    assert w[-1] == 1.0 and abs(w[0]) >= 1e-4 and w[-2] == pytest.approx(-0.4)
    assert len(qf.ffd_weights(0.4, 1e-4, max_size=10)) == 10


def test_ffd_weights_match_mlfinpy():
    mf = pytest.importorskip("mlfinpy.util.frac_diff")
    for d in (0.1, 0.3, 0.55, 0.9):
        for lim in (5, 10_000):
            np.testing.assert_array_equal(qf.ffd_weights(d, 1e-4, lim), mf.get_weights_ffd(d, 1e-4, lim).ravel())


def test_frac_diff_ffd_equals_explicit_sum():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=500))
    y, width = qf.frac_diff_ffd(x, 0.35, 1e-3)
    w = qf.ffd_weights(0.35, 1e-3)
    assert width == len(w) and np.isnan(y[:width - 1]).all()
    ref = np.array([np.dot(w, x[i - width + 1:i + 1]) for i in range(width - 1, len(x))])
    np.testing.assert_allclose(y[width - 1:], ref, rtol=1e-12, atol=1e-12)
    d1, _ = qf.frac_diff_ffd(x, 1.0)
    np.testing.assert_allclose(d1[1:], np.diff(x))


def test_select_ffd_order_and_features():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(3)
    idx = pd.date_range("2018-01-01", periods=1500, freq="4h", tz="UTC")
    walk = pd.Series(np.cumsum(rng.normal(0, 0.02, len(idx))), index=idx)        # unit root
    noise = pd.Series(rng.normal(0, 0.02, len(idx)), index=idx)                    # stationary
    short = walk.iloc[-100:]                                                         # no history before train_end
    sel = qf.select_ffd_order({"W": walk, "N": noise, "S": short}, train_end=idx[1200], thresh=1e-3)
    assert sel.per_asset["N"] == 0.0 and sel.per_asset["W"] > 0.0
    assert not sel.has_history["S"] and sel.per_asset["S"] == sel.fallback
    assert sel.pooled in [round(0.1 * i, 1) for i in range(11)]
    assert list(sel.tradeoff.columns) == ["share_rejecting", "median_adf", "median_corr", "window_bars", "window_days", "crit5"]
    feats = {"W": pd.DataFrame({"_log_close": walk, "_log_open": walk, "_log_volume": noise})}
    out = qf.add_ffd_features(feats, 0.4, thresh=1e-3, bars_per_day=6, z_window_days=30, z_min_days=10)["W"]
    assert out.columns[-6:].tolist() == ["ffd_log_close", "ffd_log_close_z", "ffd_log_open", "ffd_log_open_z",
                                         "ffd_log_volume", "ffd_log_volume_z"]
