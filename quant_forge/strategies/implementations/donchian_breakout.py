"""Strategy 432 BTC - Donchian breakout continuation (Pine v6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..base import (SignalSet, Strategy, StrategyConfig, check_execution_settings, check_int,
                   check_nonnegative, check_positive)
from ..bars import PreparedBars
from ..execution import (EXECUTED, ORDER_DISCARDED_DATA_BREAK, ORDER_UNFILLED_END_OF_DATA,
                        REJECTED_ORDER, SKIPPED_PENDING_ORDER, SKIPPED_POSITION_OPEN,
                        CostModel, Ledger, TradeRecord, settle)
from ..indicators import atr_wilder, dmi_warmup_undefined, ema_first_value_seed
from ..results import ROLE_EXECUTION, ROLE_OUTCOME
from ..inputs import BAR_INPUTS
from ..spec import FeatureSpec, IndicatorSpec, StrategySpec, param, parameters_from_config


@dataclass(frozen=True)
class DonchianBreakoutConfig(StrategyConfig):
    dc_length: int = param(30, "Donchian channel length over PRIOR bars (>= 1).")
    atr_length: int = param(14, "Wilder ATR length (>= 1).")
    adx_length: int = param(14, "Wilder DMI/ADX length (>= 1).")
    ema_length: int = param(200, "Trend EMA length (>= 1).")
    min_break_atr: float = param(0.5, "Minimum breakout distance beyond the channel, in ATRs.")
    adx_min: float = param(22.0, "Minimum ADX to enter.")
    hold_bars: int = param(20, "Max holding in bars, counted from the entry SIGNAL bar (>= 1).")
    equity_pct: float = param(95.0, "Percent of equity invested per entry, in (0, 100].")
    allow_longs: bool = param(True, "Enable long entries.")
    allow_shorts: bool = param(False, "Enable short entries.")
    initial_capital: float = param(10_000.0, "Starting equity in quote currency.")
    fee_bps: float = param(10.0, "Proportional fee per fill, in basis points.")
    slippage_ticks: int = param(0, "Adverse slippage per fill, in ticks.")
    tick_size: float = param(0.1, "Price value of one tick.")
    execution: str = param("next_open", "'next_open' (default) or 'signal_close' (Pine process_orders_on_close).")

    def __post_init__(self):
        for name in ("dc_length", "atr_length", "adx_length", "ema_length", "hold_bars"):
            check_int(name, getattr(self, name), 1)
        check_positive("equity_pct", self.equity_pct)
        if self.equity_pct > 100:
            raise ValueError("equity_pct must be <= 100")
        check_nonnegative("min_break_atr", self.min_break_atr)
        check_nonnegative("adx_min", self.adx_min)
        check_execution_settings(execution=self.execution, initial_capital=self.initial_capital,
                                 fee_bps=self.fee_bps, slippage_ticks=self.slippage_ticks,
                                 tick_size=self.tick_size)


SPEC = StrategySpec(
    name="donchian_breakout_continuation",
    title="Strategy 432 BTC - Donchian breakout continuation",
    description="Enter on a close beyond the prior-bars Donchian channel confirmed by ADX, trend EMA and a minimum "
                "ATR-sized breakout; exit on a close back through the opposite channel edge or after hold_bars.",
    source="reference notebook primary_strategies.ipynb, strategy 'donchian_breakout_continuation' / 'strategy_432_btc' (Pine v6)",
    aliases=("strategy_432_btc",),
    inputs=BAR_INPUTS,
    timestamp_rules=(
        "bar_start_us must be finite and strictly increasing (input is stably sorted by it)",
        "decision time of a bar = max(bar_end_us, bar_available_us, decision_time_us)",
        "next_open orders fill only on a bar whose first trade is at/after the signal's decision time",
        "signal_close fills at bar_end_us, which can precede the decision time (flagged per trade)",
    ),
    indicators=(
        IndicatorSpec("donchian", "highest high / lowest low of the PRIOR dc_length bars (current bar excluded)",
                      "partial windows allowed from the 2nd bar of a segment (Pine ignores leading na)",
                      "restarted at every segment", ("dc_length",), ("dc_high", "dc_low")),
        IndicatorSpec("atr", "Wilder ATR (first true range = high-low)",
                      "RMA seeded with the SMA of the first atr_length true ranges",
                      "restarted at every segment", ("atr_length",), ("atr_value",)),
        IndicatorSpec("dmi_adx", "Wilder DMI/ADX, variant with UNDEFINED warm-up (first DM/TR are na)",
                      "RMA seeds delay ADX to segment offset 2*adx_length-1; DI is forward-filled when the "
                      "smoothed TR is 0; warm-up DX is never zero-filled",
                      "restarted at every segment", ("adx_length",), ("adx_value", "plus_di", "minus_di")),
        IndicatorSpec("ema", "EMA, alpha = 2/(ema_length+1), recursive (adjust=False)",
                      "seeded with the first close of the segment; NOT masked during warm-up",
                      "restarted at every segment", ("ema_length",), ("ema_trend",)),
    ),
    features=(
        FeatureSpec("dc_high", True, "highest high of the prior dc_length bars", "float64"),
        FeatureSpec("dc_low", True, "lowest low of the prior dc_length bars", "float64"),
        FeatureSpec("atr_value", True, "Wilder ATR on the signal bar", "float64"),
        FeatureSpec("adx_value", True, "ADX on the signal bar", "float64"),
        FeatureSpec("ema_trend", True, "trend EMA on the signal bar (reference column 'ema200')", "float64"),
        FeatureSpec("long_breakout_atr", True, "(close - dc_high)/ATR when close > dc_high else 0", "float64"),
        FeatureSpec("short_breakout_atr", True, "(dc_low - close)/ATR when close < dc_low else 0", "float64"),
        FeatureSpec("plus_di", False, "+DI on the signal bar", "float64"),
        FeatureSpec("minus_di", False, "-DI on the signal bar", "float64"),
        FeatureSpec("di_diff", False, "plus_di - minus_di", "float64", "plus_di - minus_di"),
        FeatureSpec("atr_rel", False, "ATR relative to the signal close", "float64", "atr_value / close"),
        FeatureSpec("ema_distance", False, "signal close relative to the trend EMA", "float64", "close / ema_trend - 1"),
    ),
    parameters=parameters_from_config(DonchianBreakoutConfig),
    warmup="First candidate at segment offset max(2*adx_length - 1, atr_length - 1, 1), i.e. needs "
           "max(2*adx_length, atr_length, 2) bars; the EMA is not masked, so it is only meaningful after ~ema_length bars.",
    invalid_data=(
        "invalid bars never signal; any break splits the segment and restarts every indicator",
        "an open position is censored (no exit, no label), the pending order is discarded and "
        "later sizing equity is flagged incomplete (equity_path_complete=False)",
    ),
    entry_rules=(
        "long: close > dc_high, ADX >= adx_min, close > EMA, (close - dc_high)/ATR >= min_break_atr",
        "short (if allowed): close < dc_low, ADX >= adx_min, close < EMA, (dc_low - close)/ATR >= min_break_atr",
        "all indicators finite",
    ),
    exit_rules=("long: close < dc_low; short: close > dc_high  -> DONCHIAN_EXIT",
                "bars since the entry SIGNAL bar >= hold_bars -> MAX_HOLD_BARS (both -> DONCHIAN_EXIT_AND_MAX_HOLD)",
                "close-based only: there is no intrabar stop or take profit"),
    stateful_behavior=("one position at a time, no reversal in a single step; no new order while an order is pending",
                       "no re-entry on the bar of a signal_close exit; with next_open a new entry may be "
                       "signalled on the bar where the exit filled",
                       "the hold counter starts at the entry SIGNAL bar, even with next_open"),
    execution=("quantity = equity*equity_pct/100/entry_price, fractional, sized at the fill",
               "next_open fills at the next eligible bar's open (default); signal_close at the signal close",
               "fee and slippage applied on both fills"),
    limitations=("no lot rounding, funding or margin liquidation; not a broker emulator",),
)


class DonchianBreakoutContinuation(Strategy):
    name = "donchian_breakout_continuation"
    aliases = ("strategy_432_btc",)
    config_class = DonchianBreakoutConfig
    spec = SPEC
    trade_extra_roles = {
        "holding_bars": ROLE_OUTCOME,
        "equity_path_complete": ROLE_EXECUTION,
        "signal_close_before_available": ROLE_OUTCOME,
    }

    def cost_model(self, config):
        return CostModel(config.fee_bps, config.slippage_ticks, config.tick_size)

    def min_segment_bars(self, config):
        return max(2 * config.adx_length, config.atr_length, 2)

    def compute_features(self, bars: PreparedBars, config):
        n = bars.n
        feats = {k: np.full(n, np.nan) for k in (
            "dc_high", "dc_low", "atr_value", "adx_value", "plus_di", "minus_di",
            "ema_trend", "long_breakout_atr", "short_breakout_atr")}
        for idx in bars.segments():
            if not bars.valid[idx].all():
                continue
            hg, lg, cg = bars.h[idx], bars.l[idx], bars.c[idx]
            upper = pd.Series(hg).shift(1).rolling(config.dc_length, min_periods=1).max().to_numpy()
            lower = pd.Series(lg).shift(1).rolling(config.dc_length, min_periods=1).min().to_numpy()
            atr = atr_wilder(hg, lg, cg, config.atr_length)
            pdi, mdi, adx = dmi_warmup_undefined(hg, lg, cg, config.adx_length)
            ema = ema_first_value_seed(cg, config.ema_length)
            lb = np.divide(cg - upper, atr, out=np.zeros(len(idx)), where=(cg > upper) & (atr > 0))
            sb = np.divide(lower - cg, atr, out=np.zeros(len(idx)), where=(cg < lower) & (atr > 0))
            for key, vals in zip(
                    ("dc_high", "dc_low", "atr_value", "adx_value", "plus_di", "minus_di",
                     "ema_trend", "long_breakout_atr", "short_breakout_atr"),
                    (upper, lower, atr, adx, pdi, mdi, ema, lb, sb)):
                feats[key][idx] = vals
        return feats

    def optional_features(self, bars, feats, config):
        return {
            "di_diff": feats["plus_di"] - feats["minus_di"],
            "atr_rel": feats["atr_value"] / bars.c,
            "ema_distance": bars.c / feats["ema_trend"] - 1,
        }

    def generate_signals(self, bars, feats, config):
        c = bars.c
        upper, lower = feats["dc_high"], feats["dc_low"]
        atr, adx, ema = feats["atr_value"], feats["adx_value"], feats["ema_trend"]
        ok = np.isfinite(upper) & np.isfinite(lower) & np.isfinite(atr) & np.isfinite(adx) & np.isfinite(ema)
        longs = (ok & config.allow_longs & (c > upper) & (adx >= config.adx_min) & (c > ema)
                 & (feats["long_breakout_atr"] >= config.min_break_atr))
        shorts = (ok & config.allow_shorts & (c < lower) & (adx >= config.adx_min) & (c < ema)
                  & (feats["short_breakout_atr"] >= config.min_break_atr))
        side = np.where(longs, 1, np.where(shorts, -1, 0)).astype(np.int8)
        return SignalSet(side, {"exit_long_channel": c < lower, "exit_short_channel": c > upper})

    def simulate(self, bars, feats, signals, config):
        n = bars.n
        o, c, first, end, ready = bars.o, bars.c, bars.first_us, bars.end_us, bars.ready_us
        valid, adjacent = bars.valid, bars.adjacent
        cost, fee = self.cost_model(config), self.cost_model(config).fee_rate
        side_at = signals.entry_side
        exit_long, exit_short = signals.extras["exit_long_channel"], signals.extras["exit_short_channel"]
        signal_close = config.execution == "signal_close"
        ledger = Ledger(config.initial_capital)
        position: TradeRecord | None = None
        pending: dict | None = None

        def open_pos(sig: int, i: int, side: int) -> None:
            nonlocal position
            raw = c[i] if signal_close else o[i]
            px = cost.entry_fill(raw, side)
            if px <= 0:
                raise ValueError("Nonpositive entry price after slippage")
            if ledger.equity <= 0:
                ledger.mark(sig, REJECTED_ORDER)
                return
            qty = ledger.equity * config.equity_pct / 100 / px
            entry_fee = qty * px * fee
            ledger.opened += 1
            position = TradeRecord(
                trade_id=ledger.opened, signal_idx=sig, side=side, entry_idx=i,
                entry_us=float(end[i] if signal_close else first[i]), entry_price=px, qty=qty,
                entry_fee=entry_fee, entry_equity=ledger.equity,
                extras={"equity_path_complete": ledger.equity_path_complete})
            ledger.equity -= entry_fee
            ledger.trades.append(position)
            ledger.mark(sig, EXECUTED)
            ledger.trade_of_signal[sig] = position.trade_id

        def close_pos(sig: int, i: int, reason: str) -> None:
            nonlocal position
            p = position
            px = cost.exit_fill(c[i] if signal_close else o[i], p.side)
            if px <= 0:
                raise ValueError("Nonpositive exit price after slippage")
            ts = end[i] if signal_close else first[i]
            s = settle(p.side, p.qty, p.entry_price, px, p.entry_fee, fee)
            ledger.equity += p.side * p.qty * (px - p.entry_price) - s.exit_fee
            p.exit_signal_idx, p.exit_idx = sig, i
            p.exit_us, p.exit_decision_us = float(ts), float(ready[sig])
            p.exit_price, p.exit_reason = px, reason
            p.gross_return, p.net_return, p.pnl, p.fees = s.gross_return, s.net_return, s.pnl, s.fees
            p.exit_equity = ledger.equity
            p.extras["holding_bars"] = sig - p.signal_idx
            p.extras["signal_close_before_available"] = bool(
                signal_close and (ready[p.signal_idx] > p.entry_us or ready[sig] > ts))
            position = None

        for i in range(n):
            if not valid[i] or (i > 0 and not adjacent[i]):
                if position is not None:
                    had_mark = i > 0 and valid[i - 1]
                    ledger.censor(position, i - 1 if had_mark else None, c[i - 1] if had_mark else None)
                if pending is not None:
                    ledger.discarded_pending_data_break += 1
                    if pending["action"] == "entry":
                        ledger.mark(pending["sig"], ORDER_DISCARDED_DATA_BREAK)
                position = pending = None
            if not valid[i]:
                continue
            if pending is not None and i > pending["sig"] and first[i] >= ready[pending["sig"]]:
                order, pending = pending, None
                if order["action"] == "entry":
                    open_pos(order["sig"], i, order["side"])
                else:
                    close_pos(order["sig"], i, order["reason"])
            if pending is not None:
                if side_at[i]:
                    ledger.mark(i, SKIPPED_PENDING_ORDER)
                continue
            if position is not None:
                if side_at[i]:
                    ledger.mark(i, SKIPPED_POSITION_OPEN)
                channel_exit = exit_long[i] if position.side == 1 else exit_short[i]
                timeout = i - position.signal_idx >= config.hold_bars
                if channel_exit or timeout:
                    reason = ("DONCHIAN_EXIT_AND_MAX_HOLD" if channel_exit and timeout
                              else "DONCHIAN_EXIT" if channel_exit else "MAX_HOLD_BARS")
                    if signal_close:
                        close_pos(i, i, reason)
                    else:
                        pending = dict(action="exit", sig=i, reason=reason)
                # no same-close re-entry: the entry branch below is an `elif`
            elif side_at[i]:
                side = int(side_at[i])
                if signal_close:
                    open_pos(i, i, side)
                else:
                    pending = dict(action="entry", sig=i, side=side)
        if position is not None:
            last_ok = n > 0 and valid[n - 1]
            ledger.end_open(position, n - 1 if last_ok else None, c[n - 1] if last_ok else None)
        if pending is not None:
            ledger.pending_unfilled_at_end += 1
            if pending["action"] == "entry":
                ledger.mark(pending["sig"], ORDER_UNFILLED_END_OF_DATA)
        return ledger
