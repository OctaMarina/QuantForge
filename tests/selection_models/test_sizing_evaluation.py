import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from quant_forge.evaluation import (annualized_sharpe, delta_log_loss_ci, evaluate_sizing, exposure_path,
                                    information_coefficient, max_drawdown, monthly_pnl, paired_sharpe_table,
                                    reliability_table, verdict)
from quant_forge.sizing import (SIZING_RULES, afml_bet_size, break_even_win_rate, capital_constrained_sizes,
                                discretize_size, ecdf, fit_sizing_constants, normalized_sizes, rule_sizes)
from tests.synthetic import make_trade_panel


def test_afml_bet_size():
    p = np.array([0.2, 0.5, 0.6, 0.9])
    s = afml_bet_size(p, 0.5)
    assert s[0] == 0 and s[1] == 0 and 0 < s[2] < s[3] < 1
    assert afml_bet_size([0.3], 0.25)[0] > 0                       # centred at break-even: trades below 1/2
    np.testing.assert_allclose(discretize_size([0.14, 0.26]), [0.1, 0.3])
    np.testing.assert_allclose(ecdf(np.array([1.0, 2.0, 3.0]), [0.5, 2.0, 9.0]), [0, 2 / 3, 1])


def test_break_even_and_constants():
    r = np.array([0.1, -0.05, 0.2, -0.05])
    p_be, W, L = break_even_win_rate(r)
    assert (W, L) == pytest.approx((0.15, 0.05)) and p_be == pytest.approx(0.25)
    rng = np.random.default_rng(0)
    n = 500
    p = rng.uniform(0.1, 0.7, n)
    y = (rng.uniform(size=n) < p).astype(int)
    ret = np.where(y == 1, 0.05, -0.02)
    vol = rng.uniform(0.2, 1.0, n)
    k = fit_sizing_constants(p, y, ret, vol)
    raw = rule_sizes(p, vol, k)
    assert set(raw) == set(SIZING_RULES)
    for rule in SIZING_RULES:
        assert np.mean(raw[rule] / k.norm[rule]) == pytest.approx(1.0)          # equal expected exposure
    pred = pd.DataFrame({"strategy": "S", "window": 1, "p": p, "v_rvol_30d": vol})
    sizes, unc = normalized_sizes(pred, {("S", 1): k}, cap=2.0, return_uncapped=True)
    assert sizes.max().max() <= 2.0 and (unc >= sizes - 1e-15).all().all()


def test_capital_constraint_never_exceeds_budget():
    tr = make_trade_panel(400, seed=1)
    w = np.random.default_rng(0).uniform(0, 3, len(tr))
    K = 50_000.0
    out = capital_constrained_sizes(tr["entry_time"], tr["exit_time"], w, K, 10_000.0)
    assert (out <= w + 1e-12).all()
    en, ex = tr["entry_time"].to_numpy(), tr["exit_time"].to_numpy()
    for t in en:
        open_ = ((en <= t) & (ex > t)) * out * 10_000.0
        assert open_.sum() <= K + 1e-6
    np.testing.assert_array_equal(capital_constrained_sizes(tr["entry_time"], tr["exit_time"], w, 1e12), w)


def test_economics_primitives():
    t = pd.Series(pd.to_datetime(["2020-01-15", "2020-03-02", "2020-03-20"], utc=True))
    m, mi = monthly_pnl(t, np.array([0.01, -0.02, 0.03]), np.ones(3), 100.0)
    np.testing.assert_allclose(m, [1.0, 0.0, 1.0]) and list(mi) == [0, 2, 2]
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert annualized_sharpe(x) == pytest.approx(x.mean() / x.std(ddof=1) * np.sqrt(12))
    assert max_drawdown([1, -3, 1, 5]) == -3
    e = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True))
    x_ = pd.Series(pd.to_datetime(["2020-01-03", "2020-01-04"], utc=True))
    ex = exposure_path(e, x_, np.array([1.0, 1.0]), 1.0)
    assert ex["max"] == 2 and ex["max_pos"] == 2 and ex["mean"] == pytest.approx(4 / 3)


def test_evaluate_sizing_and_metrics():
    tr = make_trade_panel(800, seed=2)
    tr["window"] = tr["decision_time"].dt.year
    tr["month"] = tr["exit_time"].dt.strftime("%Y-%m")
    sizes = pd.DataFrame({"U": np.ones(len(tr)), "D": 0.5 / tr["v_rvol_30d"], "P": np.where(tr["f_inf"] > 0, 1.5, 0.5)})
    ev = evaluate_sizing(tr, sizes, perm_rules=("P",), n_boot=200, n_perm=50)
    tab, M, B = ev
    assert tab.loc["U", "Δ vs U"] == 0 and np.isnan(tab.loc["U", "p perm"]) and 0 <= tab.loc["P", "p perm"] <= 1
    assert tab.loc["P", "p perm"] < 0.1                            # informative sizing beats shuffles
    ev2 = evaluate_sizing(tr, sizes, perm_rules=("P",), n_boot=200, n_perm=50)
    pd.testing.assert_frame_equal(tab, ev2.table)
    pt = paired_sharpe_table(ev, [("P", "U")])
    assert verdict(pt.iloc[0]) in {"yes", "no (worse)", "not distinguishable from 0"}
    tr["prior"] = tr["y"].mean()
    tr["p"] = 1 / (1 + np.exp(-tr["f_inf"]))
    m, lo, hi = delta_log_loss_ci(tr, "prior", n_boot=100)
    assert m == lo == hi == 0
    assert information_coefficient(tr, "trade_return", "trade_return") == pytest.approx(1.0)
    t, ece = reliability_table(tr, "p", bins=5)
    assert len(t) == 5 and 0 <= ece < 0.2
