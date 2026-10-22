"""Flow features: aggressor mapping, tick rule, imbalance, Kyle/Hasbrouck vs reference OLS."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from conftest import mk_trades, random_trades
from helpers import make_cfg, build_bars, ctx_for
from cryptods.features import flow
from cryptods.features.stats import ols_no_intercept_from_sums
from cryptods.nulls import VALID, CODE


def test_tick_rule_initialisation_and_carry(spec, tmp_path):
    # prices: 100, 100, 101, 101, 100, 100 -> signs: ?, ?, +, +, -, -   (first two unknown: no
    # previous price change has been observed yet -> never back-filled)
    t = mk_trades(spec, range(1, 7), range(1, 7), ["100", "100", "101", "101", "100", "100"], ["1"] * 6)
    bars, b = build_bars(spec, t, tmp_path, "volume", thr=100)  # one open bar
    part = b.partial()
    assert part.tick_unknown_volume_i.iloc[0] == 2 * spec.qty_scale
    assert part.tick_buy_volume_i.iloc[0] == 2 * spec.qty_scale
    assert part.tick_sell_volume_i.iloc[0] == 2 * spec.qty_scale
    # state carried across batches: second batch starting with an unchanged price inherits "-"
    t2 = mk_trades(spec, [7], [7], ["100"], ["1"])
    b.process(t2)
    assert b.partial().tick_sell_volume_i.iloc[0] == 3 * spec.qty_scale


def test_aggressor_identity_and_imbalance_bounds(spec, tmp_path):
    t = random_trades(spec, 5000, seed=2)
    bars, _ = build_bars(spec, t, tmp_path, "volume", thr=2.0)
    assert (bars.buy_volume_i + bars.sell_volume_i == bars.volume_i).all()
    cfg = make_cfg()
    ctx = ctx_for(cfg, spec, bars)
    flow.compute(ctx)
    imb = ctx.out["aggr_imbalance_volume"]
    ok = imb.reason == VALID
    assert ok.all() and (np.abs(imb.values) <= 1).all()


def test_imbalance_zero_denominator_and_partial_classification(spec, tmp_path):
    t = mk_trades(spec, [1, 2, 3, 4], [1, 2, 3, 4], ["1"] * 4, ["1", "1", "1", "1"], side=[1, 0, -1, 1])
    bars, _ = build_bars(spec, t, tmp_path, "volume", thr=2.0)
    ctx = ctx_for(make_cfg(), spec, bars)
    flow.compute(ctx)
    imb = ctx.out["aggr_imbalance_volume"]
    # bar0 contains an unclassified trade -> imbalance null (incomplete_coverage), never imputed
    assert imb.reason[0] == CODE["incomplete_coverage"] and np.isnan(imb.values[0])
    assert ctx.out["aggr_classified_frac_volume"].values[0] == pytest.approx(0.5)
    from cryptods.features.stats import imbalance
    v, r = imbalance(np.array([0.0]), np.array([0.0]))
    assert np.isnan(v[0]) and r[0] == CODE["zero_denominator"]


def _trade_level_reference(t, spec, sign_col):
    p = t.price_i.to_numpy() / spec.price_scale
    q = t.qty_i.to_numpy() / spec.qty_scale
    s = t[sign_col].to_numpy()
    dp = np.diff(p)
    r = np.diff(np.log(p))
    x_k = (s * q)[1:]
    x_h = (s * np.sqrt(p * q))[1:]
    m = s[1:] != 0
    k = sm.OLS(dp[m], x_k[m]).fit()
    h = sm.OLS(r[m], x_h[m]).fit()
    return k.params[0], k.tvalues[0], h.params[0], h.tvalues[0]


def test_kyle_hasbrouck_trade_level_vs_statsmodels(spec, tmp_path):
    t = random_trades(spec, 400, seed=11)
    bars, b = build_bars(spec, t, tmp_path, "volume", thr=10_000)  # everything in the open bar
    part = b.partial()
    lam, tst, rb, rt = ols_no_intercept_from_sums(part.kyle_aggr_n, part.kyle_aggr_sxx, part.kyle_aggr_sxy, part.kyle_aggr_syy, 5)
    lh, th, _, _ = ols_no_intercept_from_sums(part.hasb_aggr_n, part.hasb_aggr_sxx, part.hasb_aggr_sxy, part.hasb_aggr_syy, 5)
    kl, kt, hl, ht = _trade_level_reference(t, spec, "aggressor")
    assert lam[0] == pytest.approx(kl, rel=1e-9) and tst[0] == pytest.approx(kt, rel=1e-7)
    assert lh[0] == pytest.approx(hl, rel=1e-9) and th[0] == pytest.approx(ht, rel=1e-7)
    # t-stat uses coef / sqrt(var), not coef / var
    se = np.sqrt((part.kyle_aggr_syy[0] - lam[0] * part.kyle_aggr_sxy[0]) / (part.kyle_aggr_n[0] - 1) / part.kyle_aggr_sxx[0])
    assert tst[0] == pytest.approx(lam[0] / se) and tst[0] != pytest.approx(lam[0] / se ** 2)


def test_hasbrouck_bar_regressor_is_sum_of_roots(spec, tmp_path):
    t = mk_trades(spec, [1, 2, 3], [1, 2, 3], ["100", "101", "102"], ["1", "4", "9"], side=[1, 1, -1])
    _, b = build_bars(spec, t, tmp_path, "volume", thr=1000)
    xb = b.partial().hasb_aggr_xb.iloc[0]
    manual = np.sqrt(100 * 1) + np.sqrt(101 * 4) - np.sqrt(102 * 9)
    wrong = np.sign(102 - 100) * np.sqrt(100 * 1 + 101 * 4 + 102 * 9)
    assert xb == pytest.approx(manual)
    assert abs(xb - wrong) > 1


def test_bar_level_rolling_lambda_vs_statsmodels(spec, tmp_path):
    t = random_trades(spec, 20000, seed=4)
    bars, _ = build_bars(spec, t, tmp_path, "volume", thr=3.0)
    W = 50
    cfg = make_cfg(features={"flow": {"bar_windows": [W]}})
    ctx = ctx_for(cfg, spec, bars)
    flow.compute(ctx)
    lam = ctx.out[f"kyle_lambda_bars{W}_aggr"]
    tt = ctx.out[f"kyle_tstat_bars{W}_aggr"]
    i = len(bars) - 1
    c = bars.close.to_numpy()
    x = (bars.buy_volume - bars.sell_volume).to_numpy()
    y = np.diff(c)[i - W:i]
    xx = x[i - W + 1:i + 1]
    ref = sm.OLS(y, xx).fit()
    assert lam.values[i] == pytest.approx(ref.params[0], rel=1e-9)
    assert tt.values[i] == pytest.approx(ref.tvalues[0], rel=1e-7)
    assert np.isnan(lam.values[W - 1]) and lam.reason[W - 1] != VALID  # first bar has no previous close
