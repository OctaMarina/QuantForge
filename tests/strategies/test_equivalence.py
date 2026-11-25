"""The library must reproduce the reference notebook implementation.

For identical inputs and parameters, compare (1) the indicator/feature arrays, (2) the
rule-level signals, (3) every closed trade: entry/exit times and prices, exit reasons,
returns, labels, sizing and equity, and (4) the censoring/discard counters.
"""

from __future__ import annotations

import numpy as np
import pytest

import quant_forge.strategies as qs
from quant_forge.strategies.bars import prepare_bars

from .conftest import drop_calendar_gap, inject_breaks, make_bars
from .helpers import FEATURE_MAP, assert_same, compare_diagnostics, compare_trades

REF_KWARGS = {  # library parameter -> reference parameter
    "donchian_breakout_continuation": {"adx_min": "dc_adx_min", "equity_pct": "dc_equity_pct",
                                       "ema_length": "ema200_length"},
    "supertrend_regime_confluence": {"risk_pct": "st_risk_pct"},
}

LOOSE_DONCHIAN = dict(dc_length=10, adx_min=10.0, min_break_atr=0.2, hold_bars=8, ema_length=50)
LOOSE_TIS = dict(ma_length=10, streak_length=2, atr_length=7)
LOOSE_ST = dict(min_signal_score=45, st_atr_length=7, regime_lookback=20, trend_ema_length=30, volume_ma_length=10)

CASES = {
    "rsi_cross_above": [
        dict(rsi_length=7, rsi_level=55.0),
        dict(rsi_length=7, rsi_level=45.0, execution="signal_close", slippage_ticks=2, fee_bps=8.0),
        dict(),
    ],
    "donchian_breakout_continuation": [
        dict(**LOOSE_DONCHIAN),
        dict(**LOOSE_DONCHIAN, allow_shorts=True, fee_bps=5, slippage_ticks=3),
        dict(**LOOSE_DONCHIAN, allow_shorts=True, allow_longs=False, execution="signal_close", equity_pct=60.0),
        dict(**{**LOOSE_DONCHIAN, "hold_bars": 3}, allow_shorts=True, execution="signal_close", fee_bps=5),
        dict(fee_bps=5),
    ],
    "tis_btc_momentum_confluence": [
        dict(**LOOSE_TIS, htf="4h"),
        dict(**LOOSE_TIS, htf="4h", execution="signal_close", fee_bps=5.0),
        dict(**LOOSE_TIS, htf="1D"),
        dict(**LOOSE_TIS, htf="2h", tail_mult=0.8, slippage_ticks=3, cap_pct=100.0, risk_pct=5.0),
        dict(**LOOSE_TIS, htf="6h", htf_boundary_tolerance_hours=0.1),
    ],
    "supertrend_regime_confluence": [
        dict(**LOOSE_ST),
        dict(**LOOSE_ST, execution="signal_close"),
        dict(**LOOSE_ST, sl_mode="SuperTrend", tp_mode="None"),
        dict(**LOOSE_ST, sl_mode="Percent", sl_pct=1.0, tp_mode="Percent", tp_pct=1.5, trailing_stop=True),
        dict(**LOOSE_ST, trailing_stop=True, trail_atr_mult=1.0, sl_atr_mult=2.0, tp_rr=4.0),
        dict(**LOOSE_ST, trailing_stop=True, trail_atr_mult=1.0, sl_atr_mult=2.0, execution="signal_close"),
        dict(**LOOSE_ST, size_mode="Equity %", equity_pct=50.0, cooldown_bars=0, allow_shorts=False),
        dict(**LOOSE_ST, size_mode="Fixed Units", fixed_units=0.5, no_leverage=False, st_source="close"),
        dict(**LOOSE_ST, use_trend_filter=False, skip_ranging=False, use_volume_filter=False,
             adaptive_multiplier=False, st_source="ohlc4", cooldown_bars=2),
        dict(**LOOSE_ST, sl_mode="SuperTrend", cooldown_bars=20, execution="signal_close", st_source="hlc3"),
        dict(min_signal_score=65),
    ],
}
def _late(df):
    """Bars become available hours after they end: decisions and HTF confirmations are delayed."""
    rng = np.random.default_rng(5)
    df = df.copy()
    df["bar_available_us"] = df["bar_end_us"] + rng.integers(0, 3 * 3_600_000_000, len(df))
    df["decision_time_us"] = df["bar_available_us"]
    return df


DATA = {
    "clean": lambda: make_bars(4000, seed=11),
    "breaks": lambda: inject_breaks(make_bars(4000, seed=12)),
    "calendar_gap": lambda: drop_calendar_gap(make_bars(5000, seed=13), 0.5, 40.0),
    "minimal_columns": lambda: make_bars(3000, seed=14, with_extras=False),
    "late_availability": lambda: _late(make_bars(4000, seed=15)),
}
_cache: dict = {}


def data(name):
    if name not in _cache:
        _cache[name] = DATA[name]()
    return _cache[name]


def ref_kwargs(strategy, kw):
    m = REF_KWARGS.get(strategy, {})
    return {m.get(k, k): v for k, v in kw.items()}


def case_params():
    for strategy, cases in CASES.items():
        for i, kw in enumerate(cases):
            for dname in DATA:
                yield pytest.param(strategy, kw, dname, id=f"{strategy}-{i}-{dname}")


def _signal_level(strategy, df, cfg, cap):
    """Compare library stages against the arrays captured inside the reference."""
    st = qs.get_strategy(strategy)
    bars = prepare_bars(df, volume_column=st.volume_column(cfg))
    feats = st.compute_features(bars, cfg)
    sigs = st.generate_signals(bars, feats, cfg)
    if strategy == "rsi_cross_above":
        assert_same(sigs.entry_side > 0, cap["enter"], "rsi enter", exact=True)
        assert_same(sigs.extras["exit_reason"] != None, cap["exit_ma"], "rsi exit", exact=True)  # noqa: E711
        assert_same(feats["rsi_value"], cap["rsi_value"], "rsi_value")
    elif strategy == "tis_btc_momentum_confluence":
        assert_same(sigs.entry_side > 0, cap["enter"], "tis enter", exact=True)
        has_exit = sigs.extras["exit_reason"] != None  # noqa: E711
        assert_same(has_exit, cap["exit_ma"] | cap["exit_tail"], "tis exit", exact=True)
        assert_same(sigs.extras["exit_reason"] == "BELOW_SMA", cap["exit_ma"], "tis below_sma", exact=True)
        for k in FEATURE_MAP[strategy].values():
            assert_same(feats[k], cap[k], f"tis feature {k}")
    elif strategy == "donchian_breakout_continuation":
        assert_same(sigs.entry_side == 1, cap["longs"], "donchian longs", exact=True)
        assert_same(sigs.entry_side == -1, cap["shorts"], "donchian shorts", exact=True)
        for r, n in FEATURE_MAP[strategy].items():
            assert_same(feats[n], cap[r if r != "ema200" else "ema200"], f"donchian feature {r}")
    else:
        start = bars.start_us
        in_window = sigs.extras["in_window"]
        assert_same(sigs.entry_side == 1, cap["longs"] & in_window, "st longs", exact=True)
        assert_same(sigs.entry_side == -1, cap["shorts"] & in_window, "st shorts", exact=True)
        assert_same(sigs.extras["flip"], cap["st_flip"], "st flips", exact=True)
        for k in FEATURE_MAP[strategy].values():
            assert_same(feats[k], cap[k], f"st feature {k}")
        assert start.shape == in_window.shape
    return sigs


@pytest.mark.parametrize("strategy,kw,dname", list(case_params()))
def test_matches_reference(reference, reference_monotone_htf, strategy, kw, dname):
    df = data(dname)
    snapshot = df.copy()
    # Out-of-order availability is the one situation where the library deliberately differs from the
    # reference (HTF confirmation correction); there the reference with the same one-line fix is the oracle.
    oracle = reference_monotone_htf if (dname == "late_availability" and strategy == "tis_btc_momentum_confluence") else reference
    ref, cap = oracle.run(df, strategy=strategy, **ref_kwargs(strategy, kw))
    res = qs.run_strategy(df, strategy, **kw)
    _signal_level(strategy, df, res.config, cap)
    compare_trades(strategy, ref, res)
    compare_diagnostics(ref, res)
    assert res.diagnostics["n_closed_trades"] == len(ref)
    assert df.equals(snapshot) and list(df.columns) == list(snapshot.columns), "input DataFrame was mutated"


def test_cases_are_not_vacuous(reference):
    """Every strategy produces closed trades, and the data breaks really censor positions."""
    counts, censored = {}, 0
    for strategy, cases in CASES.items():
        for kw in cases:
            for dname in ("clean", "breaks", "calendar_gap"):
                res = qs.run_strategy(data(dname), strategy, **kw)
                counts.setdefault(strategy, 0)
                counts[strategy] += res.diagnostics["n_closed_trades"]
                censored += res.diagnostics["n_censored_data_break"]
    assert all(v >= 40 for v in counts.values()), counts
    assert censored > 0
