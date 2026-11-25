"""RSI cross-above strategy: long when RSI crosses above a level, exit when it crosses below."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._long_only import simulate_long_only
from ..base import (SignalSet, Strategy, StrategyConfig, check_execution_settings, check_int)
from ..bars import PreparedBars
from ..execution import CostModel
from ..indicators import rsi_wilder
from ..inputs import BAR_INPUTS
from ..spec import FeatureSpec, IndicatorSpec, StrategySpec, param, parameters_from_config

RSI_EXIT_REASON = "RSI_CROSS_BELOW_LEVEL"


@dataclass(frozen=True)
class RsiCrossAboveConfig(StrategyConfig):
    rsi_length: int = param(14, "Wilder RSI length (>= 2).")
    rsi_level: float = param(70.0, "Level crossed upward to enter and downward to exit, in (0, 100).")
    initial_capital: float = param(100_000.0, "Starting equity in quote currency (the position size is fixed at 1 unit).")
    fee_bps: float = param(5.0, "Proportional fee per fill, in basis points.")
    slippage_ticks: int = param(0, "Adverse slippage per fill, in ticks.")
    tick_size: float = param(0.1, "Price value of one tick.")
    execution: str = param("next_open", "'next_open' (fill at the next bar's open) or 'signal_close' (optimistic).")

    def __post_init__(self):
        check_int("rsi_length", self.rsi_length, 2)
        if isinstance(self.rsi_level, bool) or not (0 < self.rsi_level < 100):
            raise ValueError("rsi_level must be strictly between 0 and 100")
        check_execution_settings(execution=self.execution, initial_capital=self.initial_capital,
                                 fee_bps=self.fee_bps, slippage_ticks=self.slippage_ticks,
                                 tick_size=self.tick_size)



SPEC = StrategySpec(
    name="rsi_cross_above",
    title="RSI cross above level",
    description="Long-only. Enter when the Wilder RSI crosses above a level; exit when it crosses back below.",
    source="reference notebook primary_strategies.ipynb, strategy 'rsi_cross_above'",
    aliases=(),
    inputs=BAR_INPUTS,
    timestamp_rules=(
        "bar_start_us must be finite and strictly increasing (input is stably sorted by it)",
        "decision time of a bar = max(bar_end_us, bar_available_us, decision_time_us)",
        "next_open orders fill only on a bar whose first trade is at/after the signal's decision time",
    ),
    indicators=(IndicatorSpec(
        "rsi", "RSI from Wilder RMAs of gains and losses of close-to-close changes",
        "RMA seeded with the SMA of the first `rsi_length` finite changes (first change is undefined)",
        "restarted at every segment; first value at segment offset rsi_length",
        ("rsi_length",), ("rsi_value",)),),
    features=(FeatureSpec("rsi_value", True, "RSI on the signal bar", "float64",
                          "100 - 100/(1 + avgGain/avgLoss); 100 if no losses, 50 if flat"),),
    parameters=parameters_from_config(RsiCrossAboveConfig),
    warmup="RSI needs rsi_length price changes (segment offset rsi_length) and a defined previous "
           "RSI to detect a cross, so the first candidate is at segment offset rsi_length + 1.",
    invalid_data=(
        "invalid bars (non-finite/inconsistent OHLC or timestamps, complete falsy) never signal",
        "any invalid bar, bar_seq gap, overlapping bar or segment_id change breaks the segment",
        "indicators restart in each segment; an open position is censored (not closed, no label) and "
        "a pending order is discarded at a break",
    ),
    entry_rules=("rsi[t] > level and rsi[t-1] <= level (within the segment)",),
    exit_rules=("rsi[t] < level and rsi[t-1] >= level -> exit reason RSI_CROSS_BELOW_LEVEL",),
    stateful_behavior=("one long position at a time", "no new order while an order is pending",
                       "entries are ignored while a position is open"),
    execution=("quantity is 1 unit; equity only tracks realised PnL and fees",
               "next_open fills at the next eligible bar's open (default); signal_close at the signal close",
               "fee and slippage applied on both fills"),
    limitations=("no lot rounding, funding or margin",),
)


class RsiCrossAbove(Strategy):
    name = "rsi_cross_above"
    config_class = RsiCrossAboveConfig
    spec = SPEC

    def cost_model(self, config):
        return CostModel(config.fee_bps, config.slippage_ticks, config.tick_size)

    def min_segment_bars(self, config):
        return config.rsi_length + 2

    def compute_features(self, bars: PreparedBars, config):
        rsi = np.full(bars.n, np.nan)
        for idx in bars.segments():
            if bars.valid[idx].all():
                rsi[idx] = rsi_wilder(bars.c[idx], config.rsi_length)
        return {"rsi_value": rsi}

    def generate_signals(self, bars, feats, config):
        rsi = feats["rsi_value"]
        previous = bars.lag(rsi)
        level = config.rsi_level
        enter = (rsi > level) & (previous <= level)
        leave = (rsi < level) & (previous >= level)
        reason = np.where(leave, RSI_EXIT_REASON, None).astype(object)
        return SignalSet(enter.astype(np.int8), {"exit_reason": reason})

    def simulate(self, bars, feats, signals, config):
        return simulate_long_only(
            bars, signals.entry_side > 0, signals.extras["exit_reason"],
            execution=config.execution, cost=self.cost_model(config),
            initial_capital=config.initial_capital, quantity_for=lambda idx, equity: 1.0)
