"""Structure of the result: events vs trades vs outcomes, diagnostics, metadata, roles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quant_forge.strategies as qs
from quant_forge.strategies.execution import STATUS_DETAIL
from quant_forge.strategies.results import ROLE_EXECUTION, ROLE_FEATURE, ROLE_OUTCOME, ROLE_REFERENCE

from .test_equivalence import CASES, data

NAMES = qs.list_strategies()


@pytest.mark.parametrize("name", NAMES)
def test_candidate_executed_and_outcome_are_distinct_layers(name):
    df = data("breaks")
    res = qs.run_strategy(df, name, **CASES[name][-1 if name == "supertrend_regime_confluence" else 0])
    ev, tr, d = res.events, res.trades, res.diagnostics
    assert set(ev.status) <= set(STATUS_DETAIL) and (ev.status_detail == ev.status.map(STATUS_DETAIL)).all()
    assert sum(d["event_status_counts"].values()) == d["n_candidate_events"] == len(ev)
    executed = ev[ev.status == "executed"]
    assert len(executed) == len(tr) == d["n_executed"]
    assert set(executed.event_id) == set(tr.event_id) and executed.trade_id.notna().all()
    assert (ev[ev.status != "executed"].trade_id.isna()).all()
    assert len(ev) > len(tr) or name == "rsi_cross_above"           # candidates are a superset of trades
    assert d["n_executed"] == d["n_closed_trades"] + d["n_censored_data_break"] + d["n_open_at_end"]
    assert set(tr.trade_status) <= {"closed", "censored_data_break", "open_at_end"}
    assert tr.trade_id.is_unique and (tr.trade_id.to_numpy() == np.arange(1, len(tr) + 1)).all()
    # outcome fields exist only for closed trades
    nc = tr[tr.trade_status != "closed"]
    assert nc[["exit_price", "trade_return", "pnl_usd"]].isna().all().all() and nc.label.isna().all()
    assert tr[tr.trade_status == "closed"].label.notna().all()
    # separate decision / entry / exit times
    assert {"decision_time", "entry_time", "exit_decision_time", "exit_time"} <= set(tr.columns)
    assert d["n_data_breaks"] > 0 and d["n_invalid_bars"] > 0


@pytest.mark.parametrize("name", NAMES)
def test_evaluate_false_gives_events_without_outcomes(name):
    df = data("clean")
    kw = CASES[name][0]
    ev_only = qs.run_strategy(df, name, evaluate=False, **kw)
    full = qs.run_strategy(df, name, **kw)
    assert ev_only.trades is None and not ev_only.evaluated and full.evaluated
    assert (ev_only.events.status == "not_evaluated").all() and ev_only.events.trade_id.isna().all()
    assert ev_only.diagnostics["evaluated"] is False and "n_executed" not in ev_only.diagnostics
    pd.testing.assert_frame_equal(ev_only.events.drop(columns=["status", "status_detail", "trade_id"]),
                                  full.events.drop(columns=["status", "status_detail", "trade_id"]))
    pd.testing.assert_frame_equal(ev_only.features, full.features)
    cand = ev_only.candidates_frame()
    forbidden = set(ev_only.post_decision_columns) | {"label", "trade_return", "exit_price", "entry_price"}
    assert not (set(cand.columns) & forbidden)
    with pytest.raises(ValueError, match="evaluate=False"):
        ev_only.closed_trades_frame()
    assert ev_only.metadata["evaluate"] is False


@pytest.mark.parametrize("name", NAMES)
def test_closed_trades_frame_and_column_roles(name):
    df = data("clean")
    res = qs.run_strategy(df, name, context_columns=["aggr_imbalance_notional"], **CASES[name][0])
    frame = res.closed_trades_frame()
    closed = res.trades[res.trades.trade_status == "closed"]
    assert len(frame) == len(closed) > 0
    assert frame["label"].dtype == np.int8 and set(frame.label) <= {0, 1}
    np.testing.assert_array_equal(frame["label"].to_numpy(), (frame["pnl_usd"] > 0).astype(int).to_numpy())
    assert {"trade_return", "trade_return_gross", "label", "entry_time", "exit_time", "decision_time"} <= set(frame.columns)
    for f in res.spec.feature_names:
        assert f in frame.columns
    # nothing from the input leaks in unless asked for
    assert not ({"open", "high", "low", "close", "volume", "bar_id", "bar_seq", "segment_id", "series_id"} & set(frame.columns))
    assert "aggr_imbalance_notional" in frame.columns

    roles = frame.attrs["column_roles"]
    assert set(roles) == set(frame.columns)
    model_cols, post = frame.attrs["model_input_columns"], frame.attrs["post_decision_columns"]
    assert set(model_cols) == set(res.spec.feature_names) | {"aggr_imbalance_notional"}
    assert not (set(model_cols) & set(post))
    assert {"label", "trade_return", "trade_return_gross", "pnl_usd", "exit_time", "exit_price", "exit_reason",
            "entry_price", "entry_time", "fees_usd", "position_qty"} <= set(post)
    assert all(roles[c] == ROLE_FEATURE for c in model_cols)
    assert roles["label"] == roles["trade_return"] == ROLE_OUTCOME
    assert roles["entry_price"] == roles["position_qty"] == ROLE_EXECUTION
    assert roles["event_id"] == roles["decision_time"] == ROLE_REFERENCE
    assert res.model_input_columns == model_cols and set(post) <= set(res.post_decision_columns)
    assert not ({"last_mark_price", "last_mark_time"} & set(frame.columns))   # only for non-closed trades
    assert frame[model_cols].shape == (len(frame), len(model_cols))


@pytest.mark.parametrize("name", NAMES)
def test_diagnostics_and_metadata_document_the_run(name):
    df = data("breaks")
    kw = CASES[name][0]
    res = qs.run_strategy(df, name, **kw)
    d, m = res.diagnostics, res.metadata
    for k in ("n_bars", "n_valid_bars", "n_invalid_bars", "n_segments", "n_data_breaks", "n_candidate_events",
              "event_status_counts", "n_executed", "n_closed_trades", "n_censored_data_break", "n_open_at_end",
              "n_discarded_pending_orders", "remaining_realized_equity_usd", "equity_path_complete",
              "min_segment_bars_before_first_candidate"):
        assert k in d, k
    assert d["n_bars"] == len(df) and d["n_valid_bars"] + d["n_invalid_bars"] == len(df)
    assert m["strategy"] == name and m["config"] == res.config.to_dict() and m["evaluate"] is True
    assert m["assumptions"] and "indicator_policy" in m and m["n_input_rows"] == len(df)
    assert m["config"]["execution"] == kw.get("execution", "next_open")
    assert isinstance(res.summary(), pd.Series)
    # the result holds copies: mutating them leaves the caller's frame untouched
    res.events.loc[:, "bar_start_us"] = 0
    assert (df["bar_start_us"] > 0).all()


def test_supertrend_extras_and_donchian_extras_are_in_the_trade_table():
    df = data("clean")
    st = qs.run_strategy(df, "supertrend_regime_confluence", **CASES["supertrend_regime_confluence"][4]).trades
    for c in ("sl_price", "tp_price", "ambiguous_intrabar", "exit_time_is_bar_end_proxy", "trailing_activated",
              "position_sizing_mode"):
        assert c in st.columns
    assert st.loc[st.exit_time_is_bar_end_proxy.astype(bool), "exit_decision_time"].isna().all()   # stop/TP: no decision bar
    don = qs.run_strategy(df, "donchian_breakout_continuation", **CASES["donchian_breakout_continuation"][0]).trades
    assert {"holding_bars", "equity_path_complete", "signal_close_before_available"} <= set(don.columns)
