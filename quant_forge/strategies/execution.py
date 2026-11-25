"""Shared execution accounting: costs, trade settlement and the trade ledger.

Position/order *logic* (when to enter, exit, reverse, apply stops) is strategy
specific and lives in each strategy module.  What is shared here is the pure
bookkeeping that must be identical everywhere: fill prices with slippage, fees,
PnL, return definitions and the records of what happened to every event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Event statuses (candidate events after simulation).
NOT_EVALUATED = "not_evaluated"
EXECUTED = "executed"
SKIPPED_PENDING_ORDER = "skipped_pending_order"
SKIPPED_POSITION_OPEN = "skipped_position_open"
SKIPPED_SAME_SIDE_POSITION = "skipped_same_side_position"
SKIPPED_COOLDOWN = "skipped_cooldown"
SKIPPED_ZERO_QUANTITY = "skipped_zero_quantity"
REJECTED_ORDER = "rejected_order"
ORDER_DISCARDED_DATA_BREAK = "order_discarded_data_break"
ORDER_UNFILLED_END_OF_DATA = "order_unfilled_end_of_data"

STATUS_DETAIL = {
    NOT_EVALUATED: "events generated without simulating execution or outcomes",
    EXECUTED: "an entry fill happened; see the trade row for its outcome",
    SKIPPED_PENDING_ORDER: "another order was still waiting for its fill bar",
    SKIPPED_POSITION_OPEN: "a position was already open (no pyramiding/reversal for this strategy)",
    SKIPPED_SAME_SIDE_POSITION: "a position on the same side was already open",
    SKIPPED_COOLDOWN: "cooldown since the previous entry had not elapsed",
    SKIPPED_ZERO_QUANTITY: "position sizing returned a non-positive quantity",
    REJECTED_ORDER: "fill rejected (non-positive price or non-positive equity)",
    ORDER_DISCARDED_DATA_BREAK: "the pending entry order was discarded by a data break before its fill",
    ORDER_UNFILLED_END_OF_DATA: "the dataset ended before the pending entry order could fill",
}

# Trade statuses.
CLOSED = "closed"
CENSORED_DATA_BREAK = "censored_data_break"
OPEN_AT_END = "open_at_end"


@dataclass(frozen=True)
class CostModel:
    """Proportional fee per side and fixed slippage in price units per fill."""

    fee_bps: float
    slippage_ticks: int
    tick_size: float

    @property
    def fee_rate(self) -> float:
        return self.fee_bps / 10_000.0

    @property
    def slippage(self) -> float:
        return self.slippage_ticks * self.tick_size

    def entry_fill(self, raw: float, side: int) -> float:
        """Buy fills higher, sell fills lower."""
        return float(raw + side * self.slippage)

    def exit_fill(self, raw: float, side: int) -> float:
        """Closing a long sells (lower fill); closing a short buys (higher fill)."""
        return float(raw - side * self.slippage)


@dataclass(frozen=True)
class Settlement:
    gross_return: float   # price return in the trade direction, before costs
    exit_fee: float
    fees: float           # entry fee + exit fee in quote currency
    pnl: float            # quote-currency PnL after fees
    net_return: float     # pnl / entry notional
    notional: float


def settle(side: int, qty: float, entry_price: float, exit_price: float,
           entry_fee: float, fee_rate: float) -> Settlement:
    notional = qty * entry_price
    exit_fee = qty * exit_price * fee_rate
    fees = entry_fee + exit_fee
    pnl = qty * side * (exit_price - entry_price) - fees
    return Settlement(
        gross_return=side * (exit_price / entry_price - 1.0),
        exit_fee=exit_fee, fees=fees, pnl=pnl,
        net_return=pnl / notional, notional=notional,
    )


@dataclass
class TradeRecord:
    """What happened to one executed entry.

    ``status`` is ``closed`` (outcome known), ``censored_data_break`` (a data
    break ended observation) or ``open_at_end`` (the dataset ended).  Outcome
    fields of non-closed trades stay NaN/None: they are *not* losses.
    """

    trade_id: int
    signal_idx: int
    side: int
    entry_idx: int
    entry_us: float
    entry_price: float
    qty: float
    entry_fee: float
    entry_equity: float
    status: str = CLOSED
    exit_signal_idx: int | None = None
    exit_idx: int | None = None
    exit_us: float | None = None
    exit_decision_us: float | None = None
    exit_price: float = np.nan
    exit_reason: str | None = None
    gross_return: float = np.nan
    net_return: float = np.nan
    pnl: float = np.nan
    fees: float = np.nan
    exit_equity: float = np.nan
    last_mark_idx: int | None = None
    last_mark_price: float = np.nan
    extras: dict[str, Any] = field(default_factory=dict)


class Ledger:
    """Collects trades, event statuses and the counters used for diagnostics."""

    def __init__(self, initial_capital: float):
        self.trades: list[TradeRecord] = []
        self.status: dict[int, str] = {}      # signal bar index -> status
        self.trade_of_signal: dict[int, int] = {}
        self.equity = float(initial_capital)
        self.opened = 0
        self.censored_data_break = 0
        self.open_at_end = 0
        self.discarded_pending_data_break = 0
        self.pending_unfilled_at_end = 0
        self.unrealized_censored_pnl = 0.0
        self.unrealized_open_at_end_pnl = 0.0
        self.equity_path_complete = True
        self.counters: dict[str, int] = {}

    def mark(self, signal_idx: int, status: str) -> None:
        self.status[int(signal_idx)] = status

    def bump(self, key: str, by: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + by

    # -- the two ways observation of a position can end without an exit -------
    def censor(self, trade: TradeRecord, last_idx: int | None, last_close: float | None) -> None:
        trade.status = CENSORED_DATA_BREAK
        self.censored_data_break += 1
        self.equity_path_complete = False
        if last_idx is not None:
            trade.last_mark_idx, trade.last_mark_price = last_idx, float(last_close)
            self.unrealized_censored_pnl += trade.side * trade.qty * (last_close - trade.entry_price)

    def end_open(self, trade: TradeRecord, last_idx: int | None, last_close: float | None) -> None:
        trade.status = OPEN_AT_END
        self.open_at_end += 1
        if last_idx is not None:
            trade.last_mark_idx, trade.last_mark_price = last_idx, float(last_close)
            self.unrealized_open_at_end_pnl += trade.side * trade.qty * (last_close - trade.entry_price)

    @property
    def discarded_pending_orders(self) -> int:
        return self.discarded_pending_data_break + self.pending_unfilled_at_end

    @property
    def censored_positions(self) -> int:
        """Reference-compatible: data-break censoring plus positions open at the end."""
        return self.censored_data_break + self.open_at_end
