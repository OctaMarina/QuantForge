"""Behaviour at data gaps and at the end of the dataset.

Positions/orders that cannot be followed to a conclusion must be *explained*, not scored:
non-closed trades have no label and no exit; orders lost to a break or to the end of the data
are reported on the event, never turned into trades or losses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quant_forge.strategies as qs

from .test_equivalence import CASES, data

NAMES = qs.list_strategies()


def pick_trade(name, kw, df, min_hold=3):
    res = qs.run_strategy(df, name, **kw)
    t = res.trades
    ok = t[(t.trade_status == "closed") & (t.entry_bar_index > 400) & (t.signal_bar_index == t.entry_bar_index - 1)
           & (t.exit_bar_index >= t.entry_bar_index + min_hold)]
    assert len(ok), f"no suitable trade for {name}"
    return res, ok.iloc[0]


def seq_break_after(df, pos):
    out = df.copy()
    out.loc[out.index[pos + 1:], "bar_seq"] += 100
    return out


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("how", ["bar_seq_gap", "invalid_bar", "segment_id"])
def test_open_position_is_censored_by_a_data_break(name, how):
    df = data("clean")
    kw = CASES[name][0]
    base, t = pick_trade(name, kw, df)
    e = int(t.entry_bar_index)
    if how == "bar_seq_gap":
        broken = seq_break_after(df, e)
    elif how == "segment_id":
        broken = df.copy()
        broken.loc[broken.index[e + 1:], "segment_id"] = 1
    else:
        broken = df.copy()
        broken.loc[broken.index[e + 2], "complete"] = 0
    res = qs.run_strategy(broken, name, **kw)
    tr = res.trades.set_index("event_id")
    row = tr.loc[t.event_id]
    assert row.trade_status == "censored_data_break"
    assert pd.isna(row.label) and pd.isna(row.exit_time) and pd.isna(row.exit_price) and pd.isna(row.trade_return)
    assert pd.isna(row.exit_reason) and pd.isna(row.pnl_usd)
    last = e if how != "invalid_bar" else e + 1
    assert row.last_mark_price == df["close"].iloc[last]          # information for diagnostics only
    assert res.events.set_index("event_id").loc[t.event_id, "status"] == "executed"
    assert t.event_id not in set(res.closed_trades_frame()["event_id"])
    assert res.diagnostics["n_censored_data_break"] >= 1 and res.diagnostics["equity_path_complete"] is False
    assert res.diagnostics["censored_positions"] >= 1
    # trades completed before the break are untouched
    before = base.trades[base.trades.trade_id < t.trade_id].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, res.trades[res.trades.trade_id < t.trade_id].reset_index(drop=True))
    # a censored trade is not counted as a loss
    assert (res.trades["label"].isna() == (res.trades["trade_status"] != "closed")).all()


@pytest.mark.parametrize("name", NAMES)
def test_pending_entry_is_discarded_by_a_break_before_its_fill(name):
    df = data("clean")
    kw = CASES[name][0]
    assert kw.get("execution", "next_open") == "next_open"
    _, t = pick_trade(name, kw, df)
    s = int(t.signal_bar_index)
    res = qs.run_strategy(seq_break_after(df, s), name, **kw)
    ev = res.events.set_index("event_id").loc[t.event_id]
    assert ev.status == "order_discarded_data_break" and pd.isna(ev.trade_id)
    assert t.event_id not in set(res.trades.event_id)
    assert res.diagnostics["discarded_pending_data_break"] >= 1


@pytest.mark.parametrize("name", NAMES)
def test_end_of_dataset_leaves_orders_unfilled_and_positions_open(name):
    df = data("clean")
    kw = CASES[name][0]
    _, t = pick_trade(name, kw, df)
    s, e = int(t.signal_bar_index), int(t.entry_bar_index)

    # (1) the signal bar is the last bar: the order never had a chance to fill
    res = qs.run_strategy(df.iloc[:s + 1], name, **kw)
    ev = res.events.set_index("event_id").loc[t.event_id]
    assert ev.status == "order_unfilled_end_of_data" and pd.isna(ev.trade_id)
    assert res.diagnostics["pending_unfilled_at_end"] == 1 and t.event_id not in set(res.trades.event_id)

    # (2) the entry bar is the last bar: open position, no exit, no label, not a loss
    res = qs.run_strategy(df.iloc[:e + 1], name, **kw)
    row = res.trades.set_index("event_id").loc[t.event_id]
    assert row.trade_status == "open_at_end" and pd.isna(row.label) and pd.isna(row.exit_time)
    assert pd.isna(row.pnl_usd) and row.last_mark_price == df["close"].iloc[e]
    assert res.events.set_index("event_id").loc[t.event_id, "status"] == "executed"
    assert res.diagnostics["n_open_at_end"] == 1 and t.event_id not in set(res.closed_trades_frame().event_id)
    assert res.diagnostics["equity_path_complete"] is False
    # the trades that closed earlier are identical to the full run
    full = qs.run_strategy(df, name, **kw).trades
    pd.testing.assert_frame_equal(
        res.trades[res.trades.trade_status == "closed"].reset_index(drop=True),
        full[(full.trade_status == "closed") & (full.trade_id < t.trade_id)].reset_index(drop=True))


@pytest.mark.parametrize("name", ["rsi_cross_above", "donchian_breakout_continuation", "tis_btc_momentum_confluence"])
def test_pending_exit_at_end_keeps_trade_open(name):
    """next_open: the exit decision is on the last bar, its fill would need the next bar."""
    df = data("clean")
    kw = CASES[name][0]
    _, t = pick_trade(name, kw, df)
    x = int(t.exit_signal_bar_index)
    res = qs.run_strategy(df.iloc[:x + 1], name, **kw)
    row = res.trades.set_index("event_id").loc[t.event_id]
    assert row.trade_status == "open_at_end" and pd.isna(row.label)
    assert res.diagnostics["pending_unfilled_at_end"] == 1


@pytest.mark.parametrize("name,idx,kw", [(s, i, kw) for s, cs in CASES.items() for i, kw in enumerate(cs)
                                         if kw.get("execution") == "signal_close"])
def test_signal_close_end_of_data(name, idx, kw):
    df = data("clean")
    res = qs.run_strategy(df, name, **kw)
    t = res.trades[(res.trades.trade_status == "closed") & (res.trades.exit_bar_index > res.trades.entry_bar_index + 2)]
    t = t.iloc[2]
    s = int(t.signal_bar_index)
    cut = qs.run_strategy(df.iloc[:s + 1], name, **kw)
    row = cut.trades.set_index("event_id").loc[t.event_id]
    assert row.trade_status == "open_at_end" and pd.isna(row.label)       # filled on the signal bar itself
    assert cut.diagnostics["pending_unfilled_at_end"] == 0


@pytest.mark.parametrize("name", NAMES)
def test_candidate_on_last_bar_is_reported_not_traded(name):
    df = data("clean")
    kw = CASES[name][0]
    ev = qs.run_strategy(df, name, evaluate=False, **kw).events
    s = int(ev.bar_index.iloc[len(ev) // 2])
    res = qs.run_strategy(df.iloc[:s + 1], name, **kw)
    last = res.events[res.events.bar_index == s].iloc[0]
    assert last.status in {"order_unfilled_end_of_data", "executed", "skipped_position_open",
                           "skipped_pending_order", "skipped_cooldown", "skipped_same_side_position"}
    assert last.status != "order_unfilled_end_of_data" or pd.isna(last.trade_id)


def test_calendar_gap_resets_htf_streak():
    from .conftest import drop_calendar_gap, make_bars
    df = drop_calendar_gap(make_bars(5000, seed=13), 0.5, 40.0)
    kw = dict(ma_length=10, streak_length=2, atr_length=7, htf="4h")
    st = qs.get_strategy("tis_btc_momentum_confluence")
    cfg = st.make_config(**kw)
    bars = qs.prepare_bars(df)
    feats = st.compute_features(bars, cfg)
    gap = int(np.argmax(np.diff(bars.start_us) > 30 * 3_600_000_000)) + 1
    assert bars.adjacent[gap], "the gap must look adjacent to the segmentation (bar_seq renumbered)"
    hours = (bars.end_us - bars.end_us[gap]) / 3.6e9
    window = (hours >= 0) & (hours < 4.0)           # first bars after the hole, before a 2nd full 4h candle
    assert feats["htf_green_streak"][window].max() <= 1
    before = (bars.end_us[gap - 1] - bars.end_us) / 3.6e9
    assert feats["htf_green_streak"][(before >= 0) & (before < 400)].max() >= 2   # streaks exist normally
