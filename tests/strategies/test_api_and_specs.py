"""Public interface, specifications, configuration and the 'no precomputed indicator reuse' rule."""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pandas as pd
import pytest

import quant_forge.strategies as qs
from quant_forge.strategies.bars import prepare_bars

from .conftest import make_bars
from .test_equivalence import CASES, LOOSE_DONCHIAN, LOOSE_ST, data

NAMES = qs.list_strategies()


def test_registry():
    assert set(NAMES) == {"rsi_cross_above", "donchian_breakout_continuation",
                          "tis_btc_momentum_confluence", "supertrend_regime_confluence"}
    assert qs.get_strategy("strategy_432_btc") is qs.get_strategy("donchian_breakout_continuation")
    with pytest.raises(ValueError, match="Unknown strategy"):
        qs.run_strategy(make_bars(50), "nope")


@pytest.mark.parametrize("name", NAMES)
def test_spec_is_inspectable_and_consistent(name):
    st = qs.get_strategy(name)
    spec = qs.describe_strategy(name)
    assert spec.name == name and spec.description and spec.warmup
    assert {"bar_start_us", "bar_end_us", "open", "high", "low", "close"} <= set(spec.required_columns)
    assert spec.timestamp_rules and spec.indicators and spec.entry_rules and spec.exit_rules
    assert spec.invalid_data and spec.stateful_behavior and spec.execution
    # parameters mirror the config dataclass exactly
    fields = {f.name: f.default for f in dataclasses.fields(st.config_class)}
    assert {p.name: p.default for p in spec.parameters} == fields
    assert all(p.description for p in spec.parameters)
    # frames / text rendering work
    for frame in (spec.inputs_frame(), spec.indicators_frame(), spec.features_frame(), spec.parameters_frame()):
        assert len(frame) > 0
    assert name in spec.describe()
    # required vs optional features are disjoint and cover all
    assert set(spec.required_features).isdisjoint(spec.optional_features)
    assert set(spec.required_features) | set(spec.optional_features) == set(spec.feature_names)


@pytest.mark.parametrize("name", NAMES)
def test_features_match_spec_and_selection(name):
    df = data("clean")
    kw = CASES[name][0]
    spec = qs.describe_strategy(name)
    allf = qs.run_strategy(df, name, **kw)
    assert list(allf.features.columns) == ["event_id", *spec.feature_names]
    req = qs.run_strategy(df, name, features="required", **kw)
    assert list(req.features.columns) == ["event_id", *spec.required_features]
    some = qs.run_strategy(df, name, features=[spec.feature_names[-1]], **kw)
    assert list(some.features.columns) == ["event_id", spec.feature_names[-1]]
    pd.testing.assert_frame_equal(allf.events, req.events)
    with pytest.raises(ValueError, match="unknown features"):
        qs.run_strategy(df, name, features=["nope"], **kw)
    assert len(allf.features) == len(allf.events) > 0
    assert not allf.features[list(spec.required_features)].isna().all().any()


@pytest.mark.parametrize("name", NAMES)
def test_runs_with_only_required_columns(name):
    df = data("clean")
    kw = CASES[name][0]
    st = qs.get_strategy(name)
    cols = list(st.required_columns(st.make_config(**kw)))
    full = qs.run_strategy(df, name, **kw)
    minimal = qs.run_strategy(df[cols], name, **kw)
    # On clean data the optional columns are inert, so candidates must be identical.
    key = ["bar_start_us", "side"]
    pd.testing.assert_frame_equal(full.events[key], minimal.events[key])
    assert minimal.events["bar_id"].isna().all() and full.events["bar_id"].notna().all()
    assert full.diagnostics["n_bars"] == minimal.diagnostics["n_bars"]
    missing = cols[0] if cols[0] != "bar_start_us" else cols[1]
    with pytest.raises(ValueError, match="Missing columns"):
        qs.run_strategy(df.drop(columns=[missing]), name, **kw)


def test_supertrend_volume_column_is_configurable():
    df = data("clean").rename(columns={"volume": "vol_x"})
    with pytest.raises(ValueError, match="Missing columns"):
        qs.run_strategy(df, "supertrend_regime_confluence", **LOOSE_ST)
    a = qs.run_strategy(df, "supertrend_regime_confluence", volume_column="vol_x", **LOOSE_ST)
    b = qs.run_strategy(data("clean"), "supertrend_regime_confluence", **LOOSE_ST)
    pd.testing.assert_frame_equal(a.trades, b.trades)
    assert "vol_x" in qs.get_strategy("supertrend_regime_confluence").required_columns(a.config)


# ---------------------------------------------------------------------------- config
BAD = [
    ("rsi_cross_above", dict(rsi_length=1)), ("rsi_cross_above", dict(rsi_level=100.0)),
    ("rsi_cross_above", dict(execution="market")), ("rsi_cross_above", dict(fee_bps=-1)),
    ("rsi_cross_above", dict(slippage_ticks=1.5)), ("rsi_cross_above", dict(initial_capital=0)),
    ("donchian_breakout_continuation", dict(dc_length=0)), ("donchian_breakout_continuation", dict(equity_pct=101)),
    ("donchian_breakout_continuation", dict(hold_bars=True)), ("donchian_breakout_continuation", dict(adx_min=-1)),
    ("tis_btc_momentum_confluence", dict(htf="1M")), ("tis_btc_momentum_confluence", dict(htf="banana")),
    ("tis_btc_momentum_confluence", dict(cap_pct=150.0)), ("tis_btc_momentum_confluence", dict(streak_length=1)),
    ("supertrend_regime_confluence", dict(sl_mode="x")), ("supertrend_regime_confluence", dict(tp_mode="x")),
    ("supertrend_regime_confluence", dict(size_mode="x")), ("supertrend_regime_confluence", dict(st_source="x")),
    ("supertrend_regime_confluence", dict(min_signal_score=101)),
    ("supertrend_regime_confluence", dict(bt_from="2030-01-01", bt_to="2020-01-01")),
    ("supertrend_regime_confluence", dict(risk_pct=150.0)),
]


@pytest.mark.parametrize("name,kw", BAD)
def test_invalid_configuration_is_rejected(name, kw):
    with pytest.raises(ValueError):
        qs.get_strategy(name).make_config(**kw)


def test_config_is_immutable_and_overrides_validated():
    cfg = qs.DonchianBreakoutConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.dc_length = 5
    with pytest.raises(TypeError, match="unknown parameters"):
        qs.run_strategy(make_bars(100), "donchian_breakout_continuation", dc_lenght=5)
    with pytest.raises(TypeError, match="expects a"):
        qs.run_strategy(make_bars(100), "rsi_cross_above", cfg)
    assert cfg.replace(fee_bps=1.0).fee_bps == 1.0 and cfg.to_dict()["dc_length"] == 30
    res = qs.run_strategy(data("clean"), "donchian_breakout_continuation", cfg.replace(**LOOSE_DONCHIAN))
    assert res.config.dc_length == 10 and res.metadata["config"]["dc_length"] == 10


def test_defaults_match_reference_signature(reference):
    sig = inspect.signature(reference._ns["strategy_events"]).parameters
    d = lambda k: sig[k].default  # noqa: E731
    don = qs.DonchianBreakoutConfig()
    assert (don.dc_length, don.atr_length, don.adx_length, don.ema_length) == (d("dc_length"), d("atr_length"), d("adx_length"), d("ema200_length"))
    assert (don.min_break_atr, don.adx_min, don.hold_bars, don.equity_pct) == (d("min_break_atr"), d("dc_adx_min"), d("hold_bars"), d("dc_equity_pct"))
    assert (don.allow_longs, don.allow_shorts, don.tick_size, don.execution) == (True, False, d("tick_size"), d("execution"))
    # per-strategy fallbacks written inside the reference (fee_bps/slippage/capital None handling)
    assert (don.fee_bps, don.slippage_ticks, don.initial_capital) == (10.0, 0, 10_000.0)
    rsi = qs.RsiCrossAboveConfig()
    assert (rsi.rsi_length, rsi.rsi_level, rsi.fee_bps, rsi.slippage_ticks, rsi.initial_capital) == (d("rsi_length"), d("rsi_level"), 5.0, 0, 100_000.0)
    tis = qs.TisMomentumConfig()
    for k in ("ma_length", "streak_length", "atr_length", "atr_mult", "risk_pct", "tail_mult", "conf_mult", "htf",
              "cap_pct", "htf_boundary_tolerance_hours"):
        assert getattr(tis, k) == d(k), k
    assert (tis.fee_bps, tis.slippage_ticks, tis.initial_capital) == (0.0, 5, 100_000.0)
    st = qs.SupertrendRegimeConfig()
    skip = {"fee_bps", "slippage_ticks", "initial_capital", "allow_shorts", "risk_pct"}
    for f in dataclasses.fields(st):
        if f.name in skip:
            continue
        assert getattr(st, f.name) == d(f.name), f.name
    assert (st.risk_pct, st.allow_shorts, st.fee_bps, st.slippage_ticks, st.initial_capital) == (d("st_risk_pct"), True, 6.0, 2, 10_000.0)


# ------------------------------------------------------ never reuse precomputed indicators
DECOYS = ["rsi_value", "rsi_14", "sma_trend", "atr_value", "atr_14", "adx_value", "plus_di", "ema200",
          "ema_trend", "st_band", "st_direction", "dc_high", "dc_low", "signal_score", "htf_momentum"]


@pytest.mark.parametrize("name", NAMES)
def test_precomputed_columns_with_matching_names_are_ignored(name):
    df = data("clean")
    kw = CASES[name][0]
    decoyed = df.copy()
    rng = np.random.default_rng(0)
    for c in DECOYS:
        decoyed[c] = rng.normal(50, 20, len(df))
    a, b = qs.run_strategy(df, name, **kw), qs.run_strategy(decoyed, name, **kw)
    pd.testing.assert_frame_equal(a.trades, b.trades)
    pd.testing.assert_frame_equal(a.features, b.features)
    assert "never reused" in a.metadata["indicator_policy"]
    # ...and a context column may not shadow a strategy feature
    clash = qs.describe_strategy(name).feature_names[0]
    with pytest.raises(ValueError, match="collide"):
        qs.run_strategy(decoyed, name, context_columns=[clash], **kw)


def test_donchian_and_supertrend_adx_are_different_variants():
    """Same column name idea, different calculation: they must not be interchangeable."""
    df = data("clean")
    bars = prepare_bars(df, volume_column="volume")
    don = qs.get_strategy("donchian_breakout_continuation")
    st = qs.get_strategy("supertrend_regime_confluence")
    adx_d = don.compute_features(bars, don.default_config())["adx_value"]
    adx_s = st.compute_features(bars, st.default_config())["st_adx"]
    # Donchian: undefined DM/TR delay the seed -> first ADX at offset 2*14-1; SuperTrend: offset 14-1
    assert np.isnan(adx_d[:27]).all() and np.isfinite(adx_d[27])
    assert np.isnan(adx_s[:13]).all() and np.isfinite(adx_s[13])
    # the zero-filled seed is a fraction of a single DX (<= 100/14): clearly not the Donchian value
    assert adx_s[13] <= 100 / 14 + 1e-9
    assert np.nanmax(np.abs(adx_d[30:300] - adx_s[30:300])) > 1e-3


# ------------------------------------------------------------------------------ small inputs
@pytest.mark.parametrize("name", NAMES)
def test_empty_tiny_and_all_invalid_inputs(name):
    base = make_bars(200, seed=3)
    for df in (base.iloc[:0], base.iloc[:1], base.iloc[:5],
               base.assign(complete=0)):
        res = qs.run_strategy(df, name)
        assert len(res.events) == 0 and len(res.trades) == 0
        assert res.closed_trades_frame().empty
        assert res.diagnostics["n_bars"] == len(df)
    ev = qs.run_strategy(base.iloc[:0], name, evaluate=False)
    assert ev.trades is None and ev.events.empty


# ----------------------------------------------------------------------- enum and package layout
def test_strategies_enum_covers_the_registry():
    canonical = {m.value for m in qs.Strategies}
    assert canonical == set(qs.list_strategies()), "every registered strategy needs a Strategies member (and vice versa)"
    assert qs.Strategies.STRATEGY_432_BTC is qs.Strategies.DONCHIAN_BREAKOUT_CONTINUATION      # alias member
    for member in qs.Strategies:
        assert member == member.value and isinstance(member, str)
        assert qs.get_strategy(member) is qs.get_strategy(member.value) is member.strategy
        assert member.spec is qs.describe_strategy(member) and member.config_class is member.strategy.config_class
        assert member.spec.name == member.value


def test_enum_members_can_be_used_instead_of_strings():
    df = data("clean")
    kw = CASES["donchian_breakout_continuation"][0]
    by_enum = qs.run_strategy(df, qs.Strategies.DONCHIAN_BREAKOUT_CONTINUATION, **kw)
    by_alias = qs.run_strategy(df, qs.Strategies.STRATEGY_432_BTC, **kw)
    by_method = qs.Strategies.DONCHIAN_BREAKOUT_CONTINUATION.run(df, **kw)
    by_string = qs.run_strategy(df, "donchian_breakout_continuation", **kw)
    for other in (by_alias, by_method, by_string):
        pd.testing.assert_frame_equal(by_enum.trades, other.trades)
        pd.testing.assert_frame_equal(by_enum.events, other.events)
    assert by_enum.strategy == "donchian_breakout_continuation" and type(by_enum.strategy) is str
    assert by_enum.events.event_id.iloc[0].startswith("donchian_breakout_continuation:")     # no enum repr leaks
    ev = qs.Strategies.RSI_CROSS_ABOVE.run(df, evaluate=False, rsi_length=7, rsi_level=55.0)
    assert ev.trades is None


def test_strategy_implementations_live_in_their_own_subpackage():
    import importlib
    import pkgutil

    impl = importlib.import_module("quant_forge.strategies.implementations")
    modules = {m.name for m in pkgutil.iter_modules(impl.__path__)}
    assert modules == {"rsi_cross_above", "donchian_breakout", "tis_momentum", "supertrend_regime"}
    registered = {type(qs.get_strategy(n)).__module__.rsplit(".", 1)[0] for n in qs.list_strategies()}
    assert registered == {"quant_forge.strategies.implementations"}
    core = {m.name for m in pkgutil.iter_modules(importlib.import_module("quant_forge.strategies").__path__)}
    assert not core & modules, "strategy modules must not be mixed with the framework modules"
