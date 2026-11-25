"""Structured strategy results and their assembly from simulation output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .bars import PreparedBars
from .execution import CLOSED, NOT_EVALUATED, STATUS_DETAIL, TradeRecord
from .spec import StrategySpec

# Column roles used to separate what a model may see from what only exists afterwards.
ROLE_REFERENCE = "reference"   # identifiers and facts known at the decision bar (not model inputs)
ROLE_FEATURE = "feature"       # decision-time values eligible as ML inputs
ROLE_EXECUTION = "execution"   # known only once the order is placed/filled
ROLE_OUTCOME = "outcome"       # known only after the trade ends (never ML inputs)

EVENT_COLUMNS = (
    "event_id", "strategy", "side", "bar_index", "source_row", "bar_id",
    "bar_start_us", "bar_end_us", "decision_time_us", "decision_time",
    "segment", "segment_offset", "status", "status_detail", "trade_id",
)

CORE_TRADE_ROLES: dict[str, str] = {
    "trade_id": ROLE_REFERENCE, "event_id": ROLE_REFERENCE, "strategy": ROLE_REFERENCE,
    "side": ROLE_REFERENCE, "signal_bar_index": ROLE_REFERENCE, "decision_time": ROLE_REFERENCE,
    "entry_time": ROLE_EXECUTION, "entry_price": ROLE_EXECUTION, "entry_bar_index": ROLE_EXECUTION,
    "position_qty": ROLE_EXECUTION, "position_notional_usd": ROLE_EXECUTION,
    "position_exposure_pct": ROLE_EXECUTION, "entry_equity_usd": ROLE_EXECUTION,
    "execution": ROLE_EXECUTION,
    "trade_status": ROLE_OUTCOME, "exit_decision_time": ROLE_OUTCOME, "exit_time": ROLE_OUTCOME,
    "exit_signal_bar_index": ROLE_OUTCOME, "exit_bar_index": ROLE_OUTCOME,
    "exit_price": ROLE_OUTCOME, "exit_reason": ROLE_OUTCOME,
    "trade_return_gross": ROLE_OUTCOME, "trade_return": ROLE_OUTCOME,
    "trade_return_pct": ROLE_OUTCOME, "label": ROLE_OUTCOME, "pnl_usd": ROLE_OUTCOME,
    "fees_usd": ROLE_OUTCOME, "exit_equity_usd": ROLE_OUTCOME, "holding_hours": ROLE_OUTCOME,
    "last_mark_price": ROLE_OUTCOME, "last_mark_time": ROLE_OUTCOME,
}


CORE_TRADE_COLUMNS = tuple(CORE_TRADE_ROLES)


def to_utc(us) -> pd.Series:
    """Microsecond epochs (NaN allowed) -> tz-aware UTC timestamps (NaT for NaN)."""
    us = np.asarray(us, dtype=float)
    ok = np.isfinite(us)
    stamps = pd.Series(pd.to_datetime(np.where(ok, us, 0).astype(np.int64), unit="us", utc=True))
    return stamps.where(ok)


def event_id(strategy: str, bar_start_us: float, side: int) -> str:
    """Stable id: strategy, signal-bar start (µs) and side. Independent of row order."""
    return f"{strategy}:{int(bar_start_us)}:{'L' if side > 0 else 'S'}"


def build_events(strategy: str, bars: PreparedBars, cand_idx: np.ndarray,
                 sides: np.ndarray, status: dict[int, str] | None,
                 trade_of_signal: dict[int, int] | None) -> pd.DataFrame:
    idx = np.asarray(cand_idx, dtype=int)
    st = [(NOT_EVALUATED if status is None else status.get(int(i), NOT_EVALUATED)) for i in idx]
    tid = [None if not trade_of_signal else trade_of_signal.get(int(i)) for i in idx]
    frame = pd.DataFrame({
        "event_id": [event_id(strategy, bars.start_us[i], s) for i, s in zip(idx, sides)],
        "strategy": strategy,
        "side": np.asarray(sides, dtype=np.int8),
        "bar_index": idx.astype(np.int64),
        "source_row": bars.source_row[idx],
        "bar_id": (bars.bar_id[idx] if bars.bar_id is not None else np.full(len(idx), None, dtype=object)),
        "bar_start_us": bars.start_us[idx].astype(np.int64),
        "bar_end_us": bars.end_us[idx].astype(np.int64),
        "decision_time_us": bars.ready_us[idx].astype(np.int64),
        "decision_time": to_utc(bars.ready_us[idx]),
        "segment": bars.segment[idx].astype(np.int64),
        "segment_offset": bars.segment_offset[idx].astype(np.int64),
        "status": st,
        "status_detail": [STATUS_DETAIL[s] for s in st],
        "trade_id": pd.array(tid, dtype="Int64"),
    })
    frame = frame.astype({"event_id": object, "strategy": object, "bar_id": object,
                          "status": object, "status_detail": object, "side": "int8"})
    return frame.loc[:, list(EVENT_COLUMNS)]


def build_trades_frame(records: list[TradeRecord], bars: PreparedBars, strategy: str,
                       execution: str, ids: dict[int, str],
                       extra_roles: dict[str, str]) -> pd.DataFrame:
    extra_cols = list(extra_roles)
    rows = []
    for t in records:
        qty_notional = t.qty * t.entry_price
        closed = t.status == CLOSED
        row = {
            "trade_id": t.trade_id, "event_id": ids[t.signal_idx], "strategy": strategy,
            "side": t.side, "trade_status": t.status,
            "signal_bar_index": t.signal_idx, "entry_bar_index": t.entry_idx,
            "exit_signal_bar_index": t.exit_signal_idx, "exit_bar_index": t.exit_idx,
            "decision_time_us": bars.ready_us[t.signal_idx], "entry_time_us": t.entry_us,
            "exit_decision_time_us": t.exit_decision_us if t.exit_decision_us is not None else np.nan,
            "exit_time_us": t.exit_us if t.exit_us is not None else np.nan,
            "entry_price": t.entry_price, "exit_price": t.exit_price, "exit_reason": t.exit_reason,
            "trade_return_gross": t.gross_return, "trade_return": t.net_return,
            "trade_return_pct": 100.0 * t.net_return,
            "label": (int(t.pnl > 0) if closed else None),
            "pnl_usd": t.pnl, "fees_usd": t.fees if closed else np.nan,
            "position_qty": t.qty, "position_notional_usd": qty_notional,
            "position_exposure_pct": 100.0 * qty_notional / t.entry_equity,
            "entry_equity_usd": t.entry_equity, "exit_equity_usd": t.exit_equity,
            "holding_hours": ((t.exit_us - t.entry_us) / 3_600_000_000.0) if closed else np.nan,
            "execution": execution,
            "last_mark_price": t.last_mark_price,
            "last_mark_time_us": (bars.end_us[t.last_mark_idx] if t.last_mark_idx is not None else np.nan),
        }
        for k in extra_cols:
            row[k] = t.extras.get(k)
        rows.append(row)
    cols = ["trade_id", "event_id", "strategy", "side", "trade_status", "signal_bar_index",
            "entry_bar_index", "exit_signal_bar_index", "exit_bar_index",
            "decision_time_us", "entry_time_us", "exit_decision_time_us", "exit_time_us",
            "entry_price", "exit_price", "exit_reason", "trade_return_gross", "trade_return",
            "trade_return_pct", "label", "pnl_usd", "fees_usd", "position_qty",
            "position_notional_usd", "position_exposure_pct", "entry_equity_usd",
            "exit_equity_usd", "holding_hours", "execution", "last_mark_price",
            "last_mark_time_us", *extra_cols]
    frame = pd.DataFrame(rows, columns=cols).astype({
        "event_id": object, "strategy": object, "trade_status": object, "exit_reason": object,
        "execution": object, "side": "int8"})
    for col in ("entry_price", "exit_price", "trade_return_gross", "trade_return", "trade_return_pct",
                "pnl_usd", "fees_usd", "position_qty", "position_notional_usd", "position_exposure_pct",
                "entry_equity_usd", "exit_equity_usd", "holding_hours", "last_mark_price"):
        frame[col] = frame[col].astype("float64")
    for src, dst in (("decision_time_us", "decision_time"), ("entry_time_us", "entry_time"),
                     ("exit_decision_time_us", "exit_decision_time"), ("exit_time_us", "exit_time"),
                     ("last_mark_time_us", "last_mark_time")):
        frame[dst] = to_utc(frame[src].to_numpy(dtype=float))
    frame = frame.drop(columns=["decision_time_us", "entry_time_us", "exit_decision_time_us",
                                "exit_time_us", "last_mark_time_us"])
    frame["label"] = frame["label"].astype("Int8")
    for col in ("trade_id", "signal_bar_index", "entry_bar_index"):
        frame[col] = frame[col].astype("int64")
    for col in ("exit_signal_bar_index", "exit_bar_index"):
        frame[col] = frame[col].astype("Int64")
    order = ["trade_id", "event_id", "strategy", "side", "trade_status",
             "decision_time", "entry_time", "exit_decision_time", "exit_time",
             "signal_bar_index", "entry_bar_index", "exit_signal_bar_index", "exit_bar_index"]
    rest = [c for c in frame.columns if c not in order]
    return frame.loc[:, order + rest]


@dataclass
class StrategyResult:
    """Everything produced by one strategy run.

    * ``events``      one row per *candidate event* (rule signal on a decision bar);
                      ``status`` says what became of it once execution is simulated.
    * ``features``    decision-time features of those events (``event_id`` keyed).
    * ``trades``      one row per *executed entry* with its outcome, or ``None`` when
                      the run was made with ``evaluate=False``.  Non-closed trades
                      (``censored_data_break``/``open_at_end``) carry no label.
    * ``diagnostics`` counters about data quality, skipped orders and censoring.
    * ``metadata``    configuration and assumptions documenting the run.
    """

    strategy: str
    config: Any
    spec: StrategySpec
    events: pd.DataFrame
    features: pd.DataFrame
    trades: pd.DataFrame | None
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]
    feature_columns: tuple[str, ...] = ()
    context_columns: tuple[str, ...] = ()
    trade_extra_roles: dict[str, str] = field(default_factory=dict)

    # -- state -----------------------------------------------------------------
    @property
    def evaluated(self) -> bool:
        return self.trades is not None

    # -- roles -------------------------------------------------------------------
    @property
    def model_input_columns(self) -> list[str]:
        """Columns that were known at decision time and may be used as ML inputs."""
        return list(self.feature_columns) + list(self.context_columns)

    @property
    def post_decision_columns(self) -> list[str]:
        """Columns of the trade table that exist only after the decision (never ML inputs)."""
        return [c for c, r in self.column_roles().items() if r in (ROLE_EXECUTION, ROLE_OUTCOME)]

    def column_roles(self) -> dict[str, str]:
        roles = {c: ROLE_FEATURE for c in self.model_input_columns}
        for c, r in {**CORE_TRADE_ROLES, **self.trade_extra_roles}.items():
            roles[c] = r
        return roles

    # -- frames --------------------------------------------------------------------
    def candidates_frame(self) -> pd.DataFrame:
        """Candidate events joined with their decision-time features (no outcomes)."""
        out = self.events.merge(self.features, on="event_id", how="left", validate="1:1")
        out.attrs["column_roles"] = self.column_roles()
        return out

    def closed_trades_frame(self) -> pd.DataFrame:
        """Closed trades with decision-time features, returns and labels.

        Columns are tagged in ``frame.attrs["column_roles"]`` and summarised in
        ``frame.attrs["model_input_columns"]`` (usable as ML inputs) and
        ``frame.attrs["post_decision_columns"]`` (execution/outcome information that
        is only known afterwards and must never be fed to a model).
        """
        if self.trades is None:
            raise ValueError("this run was made with evaluate=False: there are no trades")
        closed = self.trades.loc[self.trades["trade_status"].eq(CLOSED)]
        out = closed.merge(self.features, on="event_id", how="left", validate="1:1")
        out = out.drop(columns=["last_mark_price", "last_mark_time"])   # only meaningful for non-closed trades
        out["label"] = out["label"].astype("int8")
        out = out.sort_values("trade_id", kind="stable").reset_index(drop=True)
        roles = self.column_roles()
        out.attrs.update(
            column_roles={c: roles.get(c, ROLE_OUTCOME) for c in out.columns},
            model_input_columns=[c for c in self.model_input_columns if c in out.columns],
            post_decision_columns=[c for c in out.columns if roles.get(c) in (ROLE_EXECUTION, ROLE_OUTCOME)],
        )
        return out

    def summary(self) -> pd.Series:
        """Compact numbers for a quick look (counts only; no performance claims)."""
        d = self.diagnostics
        keys = ("n_bars", "n_valid_bars", "n_candidate_events", "n_executed", "n_closed_trades",
                "n_censored_data_break", "n_open_at_end", "n_discarded_pending_orders")
        return pd.Series({k: d.get(k) for k in keys})
