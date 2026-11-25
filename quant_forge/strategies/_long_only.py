"""Position/execution handling shared by the two long-only, one-position-at-a-time
strategies (RSI cross above, TIS momentum confluence).

State machine (identical to the reference notebook):

* at most one position and one pending order; while an order is pending no new
  order is created;
* ``next_open``: an order created at a bar's close fills at the open of the next
  bar whose first trade is at/after the signal's ``ready`` time;
* ``signal_close``: orders fill at the signal bar's close (optimistic, Pine-like);
* exits are evaluated on every bar while a position exists, *including the bar on
  which the entry filled*;
* a data break censors the open position (no exit, no label) and discards the
  pending order.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .bars import PreparedBars
from .execution import (EXECUTED, ORDER_DISCARDED_DATA_BREAK, ORDER_UNFILLED_END_OF_DATA,
                        REJECTED_ORDER, SKIPPED_PENDING_ORDER, SKIPPED_POSITION_OPEN,
                        SKIPPED_ZERO_QUANTITY, CostModel, Ledger, TradeRecord, settle)


def simulate_long_only(
    bars: PreparedBars,
    entry: np.ndarray,              # bool: candidate entry on this bar
    exit_reason: np.ndarray,        # object: reason string where an exit condition holds, else None
    *,
    execution: str,
    cost: CostModel,
    initial_capital: float,
    quantity_for: Callable[[int, float], float],   # (signal_idx, equity) -> quantity
) -> Ledger:
    n = bars.n
    o, c, first, end, ready = bars.o, bars.c, bars.first_us, bars.end_us, bars.ready_us
    valid, adjacent = bars.valid, bars.adjacent
    fee = cost.fee_rate
    ledger = Ledger(initial_capital)
    position: TradeRecord | None = None
    pending: dict | None = None

    def open_position(signal_idx: int, fill_idx: int, ts: float, raw: float, quantity: float) -> None:
        nonlocal position
        if quantity <= 0:
            ledger.mark(signal_idx, SKIPPED_ZERO_QUANTITY)
            return
        px = cost.entry_fill(raw, 1)
        if px <= 0:
            ledger.mark(signal_idx, REJECTED_ORDER)
            return
        entry_fee = quantity * px * fee
        ledger.equity -= entry_fee
        ledger.opened += 1
        position = TradeRecord(
            trade_id=ledger.opened, signal_idx=signal_idx, side=1, entry_idx=fill_idx,
            entry_us=float(ts), entry_price=px, qty=float(quantity), entry_fee=float(entry_fee),
            entry_equity=float(ledger.equity + entry_fee))
        ledger.trades.append(position)
        ledger.mark(signal_idx, EXECUTED)
        ledger.trade_of_signal[signal_idx] = position.trade_id

    def close_position(fill_idx: int, ts: float, raw: float, reason: str, exit_signal_idx: int) -> None:
        nonlocal position
        px = cost.exit_fill(raw, 1)
        if px <= 0:
            raise ValueError("Exit price became nonpositive after slippage")
        pos = position
        s = settle(1, pos.qty, pos.entry_price, px, pos.entry_fee, fee)
        ledger.equity += pos.qty * (px - pos.entry_price) - s.exit_fee
        pos.exit_signal_idx, pos.exit_idx = exit_signal_idx, fill_idx
        pos.exit_us, pos.exit_decision_us = float(ts), float(ready[exit_signal_idx])
        pos.exit_price, pos.exit_reason = px, reason
        pos.gross_return, pos.net_return, pos.pnl, pos.fees = s.gross_return, s.net_return, s.pnl, s.fees
        pos.exit_equity = ledger.equity
        position = None

    for i in range(n):
        if not valid[i] or (i > 0 and not adjacent[i]):
            if position is not None:
                had_mark = i > 0 and valid[i - 1]
                ledger.censor(position, i - 1 if had_mark else None, c[i - 1] if had_mark else None)
            if pending is not None:
                ledger.discarded_pending_data_break += 1
                if pending["action"] == "entry":
                    ledger.mark(pending["signal_idx"], ORDER_DISCARDED_DATA_BREAK)
            position, pending = None, None
        if not valid[i]:
            continue

        # next_open orders need the signal to be available BEFORE the first trade of the fill bar.
        if (execution == "next_open" and pending is not None
                and i > pending["signal_idx"] and first[i] >= pending["ready_us"]):
            order, pending = pending, None
            if order["action"] == "entry":
                open_position(order["signal_idx"], i, first[i], o[i], order["qty"])
            else:
                close_position(i, first[i], o[i], order["reason"], order["signal_idx"])

        if pending is not None:
            if entry[i]:
                ledger.mark(i, SKIPPED_PENDING_ORDER)
            continue
        if position is not None:
            if entry[i]:
                ledger.mark(i, SKIPPED_POSITION_OPEN)
            reason = exit_reason[i]
            if reason is not None:
                if execution == "signal_close":
                    close_position(i, end[i], c[i], reason, i)
                else:
                    pending = {"action": "exit", "signal_idx": i, "ready_us": ready[i], "reason": reason}
        elif entry[i]:
            qty = quantity_for(i, ledger.equity)
            if qty <= 0:
                ledger.mark(i, SKIPPED_ZERO_QUANTITY)
            elif execution == "signal_close":
                open_position(i, i, end[i], c[i], qty)
            else:
                pending = {"action": "entry", "signal_idx": i, "ready_us": ready[i], "qty": qty}

    if position is not None:
        last_ok = n > 0 and valid[n - 1]
        ledger.end_open(position, n - 1 if last_ok else None, c[n - 1] if last_ok else None)
    if pending is not None:
        ledger.pending_unfilled_at_end += 1
        if pending["action"] == "entry":
            ledger.mark(pending["signal_idx"], ORDER_UNFILLED_END_OF_DATA)
    return ledger
