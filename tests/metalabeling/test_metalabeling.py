import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

import quant_forge.strategies as qs
from quant_forge import features as qf
from quant_forge import metalabeling as ml
from quant_forge.utils import DiskCache
from quant_forge.validation import calendar_windows
from tests.synthetic import make_bars_panel, make_trade_panel

FAST = ml.MetaLabelingConfig(min_train=150, n_placebo=2, n_perm=1, train_assets=("A1", "A2", "A3"))
FAST_LR = ml.MetaLabelingConfig(min_train=150, n_placebo=2, n_perm=1, train_assets=("A1", "A2", "A3"), model_names=("LR",))


def _panel():
    tr = make_trade_panel(900, seed=11, signal=1.5)
    ds = ml.prepare_meta_dataset(tr, track_record=True)
    return ds


def test_prepare_meta_dataset():
    ds = _panel()
    assert {"s_swr_30", "s_open_positions"} <= set(ds.trades.columns)
    assert ds.family["f_inf"] == "F" and "m_n_assets" in ds.candidates
    assert ds.candidates_for("STRAT") == ds.candidates


def test_fit_window_outputs_and_determinism():
    ds = _panel()
    w0, w1 = pd.Timestamp("2021-06-01", tz="UTC"), pd.Timestamp("2022-06-01", tz="UTC")
    a = ml.fit_window(ds.trades, w0, w1, ds.candidates, FAST, name="STRAT")
    b = ml.fit_window(ds.trades, w0, w1, ds.candidates, FAST, name="STRAT")
    pred, diag = a
    pd.testing.assert_frame_equal(pred, b.predictions)
    assert {"row", "p", "prior", "window", "meta", "p_LR-FULL", "p_LR-FULL_raw", "p_LR-PCA"} <= set(pred.columns)
    test = ds.trades.loc[(ds.trades["decision_time"] >= w0) & (ds.trades["decision_time"] < w1)]
    assert pred["row"].tolist() == test.index.tolist()
    assert diag["meta"] in diag["tuning"] and diag["constants"].norm["U"] == 1.0
    assert any("f_inf" in diag["groups"][g] for g in diag["selected"])
    assert diag["n_train"] == int(((ds.trades["exit_time"] < w0) & ds.trades["asset"].isin(FAST.train_assets)).sum())
    assert ml.fit_window(ds.trades, pd.Timestamp("2019-02-01", tz="UTC"), w1, ds.candidates, FAST) is None


def test_fit_window_without_feature_selection():
    ds = _panel()
    w0, w1 = pd.Timestamp("2021-06-01", tz="UTC"), pd.Timestamp("2022-06-01", tz="UTC")
    cfg = ml.MetaLabelingConfig(min_train=150, train_assets=("A1", "A2", "A3"), model_names=("LR",), pca_models=(),
                                feature_selection=False)
    pred, diag = ml.fit_window(ds.trades, w0, w1, ds.candidates, cfg, name="STRAT")
    assert list(diag["tuning"]) == ["LR-FULL"] and diag["meta"] == "LR-FULL"
    assert diag["selected"] == [] and diag["groups"] == {} and diag["mda"] is None
    assert np.allclose(pred["p"], pred["p_LR-FULL"]) and diag["constants"].norm["U"] == 1.0
    full = ml.fit_window(ds.trades, w0, w1, ds.candidates, FAST_LR, name="STRAT")
    np.testing.assert_allclose(pred["p_LR-FULL"], full.predictions["p_LR-FULL"])      # same LR-FULL fit
    assert "feature_selection" not in FAST.cache_key() and cfg.cache_key()["feature_selection"] is False
    assert len(ml.window_summary({("STRAT", w0): diag})) == 1


def test_run_walk_forward_cache_and_reports(tmp_path):
    ds = _panel()
    windows = calendar_windows("2021-01-01", end="2023-01-01", years=1)
    cache = DiskCache(tmp_path, key=FAST_LR.cache_key())
    res = ml.run_walk_forward(ds.trades, ["STRAT"], windows, {"STRAT": ds.candidates}, FAST_LR, cache=cache, n_jobs=1,
                              inner_n_jobs=1, verbose=False)
    assert len(res) == 2 and len(list(cache.root.glob("*.pkl"))) == 2
    again = ml.run_walk_forward(ds.trades, ["STRAT"], windows, {"STRAT": ds.candidates}, FAST_LR, cache=cache, n_jobs=2,
                                verbose=False)
    for k in res:
        pd.testing.assert_frame_equal(res[k].predictions, again[k].predictions)
    pred = ml.collect_predictions(res, ds.trades, ["strategy", "asset", "y", "trade_return", "v_rvol_30d"])
    assert len(pred) == sum(len(r.predictions) for r in res.values())
    diags = {k: r.diagnostics for k, r in res.items()}
    assert len(ml.window_summary(diags)) == 2
    rec = ml.feature_selection_records(diags, ds.family)
    assert {"PLACEBO", "F"} <= set(rec["family"])
    assert ml.probability_columns(pred)["META"] == "p"
    assert len(ml.sizing_constants_table(diags)) == 2


def test_build_trade_panel_on_synthetic_bars():
    bars = make_bars_panel(("BTC", "ETH"), n=600)
    feats = qf.add_raw_levels(qf.bar_features(bars, bars_per_day=1), bars)
    cols = qf.model_columns(feats["BTC"])
    S = qs.Strategies
    strategies = {S.RSI_CROSS_ABOVE: "RSI", S.DONCHIAN_BREAKOUT_CONTINUATION: "DONCHIAN"}
    t1 = ml.build_trade_panel(bars, feats, strategies, feature_columns=cols, tick_sizes={"BTC": 0.01, "ETH": 0.01})
    t2 = ml.build_trade_panel(bars, feats, strategies, feature_columns=cols, tick_sizes={"BTC": 0.01, "ETH": 0.01},
                              n_jobs=2)
    pd.testing.assert_frame_equal(t1, t2)
    assert len(t1) and t1["decision_time"].is_monotonic_increasing
    assert (t1["y"] == (t1["trade_return"] > 0)).all()
    assert any(c.startswith("lib_") and c.endswith("_rel") for c in t1.columns)
    assert set(t1["strategy"]) <= {"RSI", "DONCHIAN"}
