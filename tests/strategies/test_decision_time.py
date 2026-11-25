"""Feature availability at decision time, causality and timing separation.

Decision, entry and exit times are different quantities and must keep their order:
decision (signal bar available) <= entry (next_open) <= exit; features of an event may use
nothing that happened after the decision bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quant_forge.strategies as qs
from quant_forge.strategies.bars import prepare_bars

from .conftest import make_bars
from .test_equivalence import CASES, data

NAMES = qs.list_strategies()
FLAT = [(s, i, kw) for s, cs in CASES.items() for i, kw in enumerate(cs)]


def _stages(name, df, kw):
    st = qs.get_strategy(name)
    cfg = st.make_config(**kw)
    bars = prepare_bars(df, volume_column=st.volume_column(cfg))
    feats = st.compute_features(bars, cfg)
    return st, cfg, bars, feats, st.generate_signals(bars, feats, cfg)


def _scramble_future(df: pd.DataFrame, k: int, seed: int = 0) -> pd.DataFrame:
    """Replace every bar from row k on by unrelated but internally valid bars."""
    fut = make_bars(len(df) - k, seed=seed + 99, start="2024-02-01")
    out = df.copy()
    for c in ("open", "high", "low", "close", "volume"):
        out.loc[out.index[k:], c] = fut[c].to_numpy() * (out["close"].iloc[k - 1] / fut["close"].iloc[0])
    return out


@pytest.mark.parametrize("name,idx,kw", FLAT, ids=[f"{s}-{i}" for s, i, _ in FLAT])
@pytest.mark.parametrize("dname", ["clean", "breaks", "late_availability"])
def test_features_and_signals_use_no_future_information(name, idx, kw, dname):
    df = data(dname)
    n = len(df)
    if name == "tis_btc_momentum_confluence" and dname == "late_availability":
        # With out-of-order availability "the past" is defined by availability rather than row order, so row
        # truncation is not the right invariant; see test_htf_correction.py for the availability-based test.
        pytest.skip("covered by test_features_use_only_bars_available_at_the_decision_time")
    _, _, _, full_f, full_s = _stages(name, df, kw)
    for k in (n // 3, n // 2 + 7, n - 50):
        for variant in (df.iloc[:k], _scramble_future(df, k)):
            _, _, _, f, s = _stages(name, variant, kw)
            for key, arr in full_f.items():
                np.testing.assert_array_equal(arr[:k], f[key][:k], err_msg=f"{name}.{key} changed with future data (k={k})")
            np.testing.assert_array_equal(full_s.entry_side[:k], s.entry_side[:k])


@pytest.mark.parametrize("name,idx,kw", FLAT, ids=[f"{s}-{i}" for s, i, _ in FLAT])
def test_trades_known_by_k_do_not_depend_on_later_bars(name, idx, kw):
    df = data("breaks")
    full = qs.run_strategy(df, name, **kw).trades
    for k in (len(df) // 2, len(df) - 300):
        part = qs.run_strategy(df.iloc[:k], name, **kw).trades
        cols = ["event_id", "side", "exit_bar_index", "entry_price", "exit_price", "exit_reason", "pnl_usd"]
        a = part[part["trade_status"].eq("closed")][cols].reset_index(drop=True)
        b = full[full["trade_status"].eq("closed") & (full["exit_bar_index"] < k)][cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("name,idx,kw", FLAT, ids=[f"{s}-{i}" for s, i, _ in FLAT])
@pytest.mark.parametrize("dname", ["clean", "late_availability"])
def test_decision_entry_exit_times_are_ordered(name, idx, kw, dname):
    df = data(dname)
    res = qs.run_strategy(df, name, **kw)
    ev, tr = res.events, res.trades
    assert (ev["decision_time_us"] >= ev["bar_end_us"]).all()
    assert (ev["decision_time_us"] >= df.set_index("bar_start_us").loc[ev["bar_start_us"], "decision_time_us"].to_numpy()).all()
    closed = tr[tr["trade_status"].eq("closed")]
    assert (closed["exit_time"] >= closed["entry_time"]).all()
    if res.config.execution == "next_open":
        assert (tr["entry_time"] >= tr["decision_time"]).all(), "filled before the decision was available"
        has = closed["exit_decision_time"].notna()
        assert (closed.loc[has, "exit_time"] >= closed.loc[has, "exit_decision_time"]).all()
        assert (closed["exit_bar_index"] > closed["entry_bar_index"]).all() or \
               (closed["exit_bar_index"] >= closed["entry_bar_index"]).all()
        assert (tr["entry_bar_index"] > tr["signal_bar_index"]).all()
    else:
        assert (tr["entry_bar_index"] == tr["signal_bar_index"]).all()
        if dname == "late_availability" and name == "donchian_breakout_continuation":
            # optimistic fill precedes availability, and that is flagged explicitly
            assert closed["signal_close_before_available"].astype(bool).any()
            before = closed["entry_time"] < closed["decision_time"]
            assert closed.loc[before, "signal_close_before_available"].astype(bool).all()


@pytest.mark.parametrize("name", NAMES)
def test_event_references_point_to_the_decision_bar(name):
    df = data("breaks").sample(frac=1.0, random_state=3).reset_index(drop=True)   # unsorted input
    res = qs.run_strategy(df, name, **CASES[name][0])
    ev = res.events
    assert ev["event_id"].is_unique and len(ev) > 0
    src = df.iloc[ev["source_row"].to_numpy()]
    np.testing.assert_array_equal(src["bar_start_us"].to_numpy(), ev["bar_start_us"].to_numpy())
    np.testing.assert_array_equal(src["bar_end_us"].to_numpy(), ev["bar_end_us"].to_numpy())
    assert (src["bar_id"].to_numpy() == ev["bar_id"].to_numpy()).all()
    expected_decision = np.maximum.reduce([src["bar_end_us"], src["bar_available_us"], src["decision_time_us"]])
    np.testing.assert_array_equal(expected_decision, ev["decision_time_us"].to_numpy())
    # ids are independent of the row order and of evaluation
    sorted_res = qs.run_strategy(data("breaks"), name, **CASES[name][0])
    assert set(sorted_res.events["event_id"]) == set(ev["event_id"])
    assert set(qs.run_strategy(df, name, evaluate=False, **CASES[name][0]).events["event_id"]) == set(ev["event_id"])
    # features are those of the decision bar
    feat = res.features.set_index("event_id")
    assert feat.index.is_unique and set(feat.index) == set(ev["event_id"])


@pytest.mark.parametrize("name", NAMES)
def test_segment_features_do_not_depend_on_other_segments(name):
    """Indicators restart at every break: a segment alone gives the same features as inside the full series."""
    df = data("breaks")
    kw = CASES[name][0]
    _, _, bars, feats, _ = _stages(name, df, kw)
    checked = 0
    for idx in bars.segments():
        if len(idx) < 150 or not bars.valid[idx].all():
            continue
        a, b = idx[0], idx[-1] + 1
        _, _, _, alone, _ = _stages(name, df.iloc[a:b], kw)
        for key, arr in feats.items():
            np.testing.assert_array_equal(arr[a:b], alone[key], err_msg=f"{name}.{key} segment {a}:{b}")
        checked += 1
    assert checked >= 3


@pytest.mark.parametrize("name,idx,kw", FLAT, ids=[f"{s}-{i}" for s, i, _ in FLAT])
def test_warmup_bound_in_spec_holds(name, idx, kw):
    for dname in ("clean", "breaks"):
        res = qs.run_strategy(data(dname), name, evaluate=False, **kw)
        bound = res.diagnostics["min_segment_bars_before_first_candidate"]
        assert (res.events["segment_offset"] + 1 >= bound).all()


def test_context_columns_are_taken_from_the_signal_bar_only():
    df = data("clean")
    res = qs.run_strategy(df, "donchian_breakout_continuation", context_columns=["aggr_imbalance_notional"],
                          **CASES["donchian_breakout_continuation"][0])
    tr = res.closed_trades_frame()
    src = df.set_index("bar_start_us")["aggr_imbalance_notional"]
    ev = res.events.set_index("event_id").loc[tr["event_id"]]
    np.testing.assert_array_equal(tr["aggr_imbalance_notional"].to_numpy(), src.loc[ev["bar_start_us"]].to_numpy())
    assert "aggr_imbalance_notional" in tr.attrs["model_input_columns"]
    with pytest.raises(ValueError, match="Missing context columns"):
        qs.run_strategy(df, "rsi_cross_above", context_columns=["nope"])
    with pytest.raises(ValueError, match="collide"):
        qs.run_strategy(df, "rsi_cross_above", context_columns=["entry_price"])
