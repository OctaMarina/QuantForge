"""The BTCUSDT volume-bar dataset used by the reference notebook (skipped if absent)."""

from __future__ import annotations

import numpy as np
import pytest

import quant_forge.strategies as qs

from .helpers import compare_diagnostics, compare_trades
from .test_equivalence import REF_KWARGS, _signal_level

REAL_CASES = [
    ("donchian_breakout_continuation", dict(fee_bps=5)),     # the experiment run in the notebook
    ("donchian_breakout_continuation", dict(fee_bps=5, allow_shorts=True)),
    ("rsi_cross_above", {}),
    ("tis_btc_momentum_confluence", {}),
    ("supertrend_regime_confluence", {}),
    ("supertrend_regime_confluence", dict(trailing_stop=True, sl_mode="SuperTrend", execution="signal_close")),
]


@pytest.mark.parametrize("strategy,kw", REAL_CASES, ids=[f"{s}-{i}" for i, (s, _) in enumerate(REAL_CASES)])
def test_real_data_matches_reference(reference, btc_bars, strategy, kw):
    m = REF_KWARGS.get(strategy, {})
    ref, cap = reference.run(btc_bars, strategy=strategy, **{m.get(k, k): v for k, v in kw.items()})
    res = qs.run_strategy(btc_bars, strategy, **kw)
    _signal_level(strategy, btc_bars, res.config, cap)
    compare_trades(strategy, ref, res)
    compare_diagnostics(ref, res)
    assert len(ref) > 20


def test_availability_equals_bar_end_in_the_dataset(btc_bars):
    """Precondition of the strict no-lookahead property of the TIS HTF feature (see its spec)."""
    assert (btc_bars.bar_available_us == btc_bars.bar_end_us).all()
    assert (btc_bars.decision_time_us == btc_bars.bar_end_us).all()


def test_notebook_experiment_numbers(btc_bars):
    """Numbers saved in the outputs of the reference notebook: strategy_432_btc, fee 5 bps, all bars."""
    res = qs.run_strategy(btc_bars, "strategy_432_btc", fee_bps=5)
    t = res.closed_trades_frame()
    assert len(t) == 254                                               # df.trade_id.count()
    pnl = t["trade_return"] * 10_000                                   # plot_strategy_equity(position_size=10_000)
    assert pnl.sum() == pytest.approx(3026.39765, abs=1e-4)
    assert 100 * (t["trade_return"] > 0).mean() == pytest.approx(50.393701, abs=1e-5)
    assert 100 * t["trade_return"].mean() == pytest.approx(0.11915, abs=1e-5)
    test = t.sort_values("decision_time").iloc[-int(np.ceil(0.30 * len(t))):]   # train_test_split(test_size=0.30)
    assert len(test) == 77
    assert (test["trade_return"] * 10_000).sum() == pytest.approx(1605.191596, abs=1e-4)
    assert res.diagnostics["n_candidate_events"] > len(t)
