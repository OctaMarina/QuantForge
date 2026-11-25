import numpy as np
import pandas as pd

from quant_forge import evaluation as qe


def setup():
    cal = pd.date_range(pd.Timestamp("2021-01-01", tz="UTC"), periods=120, freq="D")
    rng = np.random.default_rng(0)
    closes = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(cal)))),
                           "B": 50 * np.exp(np.cumsum(rng.normal(0, 0.03, len(cal))))}, index=cal)
    n = 300
    entry = cal[0] + pd.to_timedelta(rng.uniform(0, 100, n), unit="D")
    exit_ = entry + pd.to_timedelta(rng.exponential(3, n), unit="D")
    exit_ = exit_.where(exit_ < cal[-1] + pd.Timedelta(hours=23), cal[-1] + pd.Timedelta(hours=23))
    gross = rng.normal(0, 0.05, n)
    side = rng.choice([1, -1], n)
    fee = 5 / 1e4
    tr = pd.DataFrame({"asset": rng.choice(["A", "B"], n), "side": side, "entry_time": entry, "exit_time": exit_,
                       "entry_price": rng.uniform(40, 120, n), "trade_return_gross": gross})
    tr["trade_return"] = gross - fee * (2 + side * gross)
    return cal, closes, tr


def test_mark_to_market_sums_to_net_returns():
    cal, closes, tr = setup()
    td = qe.expand_trade_days(tr, closes, fee_bps=5)
    np.testing.assert_allclose(td.unit_return(), tr["trade_return"].to_numpy(), atol=1e-12)
    w = np.random.default_rng(1).uniform(0, 2, len(tr))
    d = qe.daily_pnl(td, w, notional=1000)
    assert d.shape == (len(cal),) and np.isclose(d.sum(), 1000 * (w * tr["trade_return"]).sum())
    g = (tr["asset"] == "B").astype(int).to_numpy()
    G = qe.daily_pnl(td, np.vstack([w, 2 * w]), notional=1000, group=g, n_groups=2)
    assert G.shape == (2, 2, len(cal)) and np.allclose(G.sum(0)[0], d) and np.allclose(G.sum(0)[1], 2 * d)


def test_paired_bootstrap_and_tables():
    cal, closes, tr = setup()
    td = qe.expand_trade_days(tr, closes, fee_bps=5)
    D = qe.daily_pnl(td, np.vstack([np.ones(len(tr)), np.full(len(tr), 3.0)]))
    months = qe.month_labels(cal)
    B = qe.bootstrap_sharpe(D, months, n_boot=300, seed=7)
    assert np.allclose(B[0], B[1])                         # Sharpe is scale invariant, draws are paired
    assert np.allclose(qe.bootstrap_sharpe(D[1:], months, n_boot=300, seed=7)[0], B[1])
    pdiff = qe.paired_differences({"a": D[0], "b": D[1]}, {"a": B[0], "b": B[1]}, [("b", "a")])
    assert abs(pdiff.loc["b − a", "Δ Sharpe"]) < 1e-12
    tab = qe.portfolio_table({"a": D[0]}, budget=20_000, trades={"a": tr}, sizes={"a": np.ones(len(tr))})
    assert tab.loc["a", "trades"] == len(tr) and tab.loc["a", "max DD / K"] <= 0
    assert qe.max_drawdown_daily([1, -3, 1, -1]) == -3
