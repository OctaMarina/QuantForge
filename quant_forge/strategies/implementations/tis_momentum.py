"""TIS - BTC Momentum + Confluence (Pine v6): SMA trend + green-bar streaks on two timeframes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._long_only import simulate_long_only
from ..base import (SignalSet, Strategy, StrategyConfig, check_execution_settings, check_int,
                   check_positive)
from ..bars import PreparedBars
from ..execution import CostModel
from ..htf import higher_timeframe_momentum, htf_duration_us
from ..indicators import atr_wilder, sma
from ..inputs import BAR_INPUTS
from ..spec import FeatureSpec, IndicatorSpec, StrategySpec, param, parameters_from_config


@dataclass(frozen=True)
class TisMomentumConfig(StrategyConfig):
    ma_length: int = param(50, "SMA length of the trend filter (>= 2).")
    streak_length: int = param(3, "Consecutive green bars required on the bar series and on the HTF (>= 2).")
    atr_length: int = param(14, "Wilder ATR length (>= 1); ATR is used for sizing and the tail-cut exit.")
    atr_mult: float = param(3.0, "ATR multiple defining the risk distance used ONLY for position sizing.")
    risk_pct: float = param(1.0, "Percent of equity risked per trade at the sizing distance.")
    tail_mult: float = param(2.0, "Exit when (open - close) > tail_mult * ATR.")
    conf_mult: float = param(1.5, "Multiplier applied to the risk-based quantity.")
    htf: str = param("1D", "Fixed higher timeframe (pandas offset alias), e.g. '1D' or '4h'.")
    cap_pct: float = param(40.0, "Notional cap as percent of equity, in (0, 100].")
    htf_boundary_tolerance_hours: float = param(1.5, "Max distance of source bars from an HTF bucket boundary.")
    initial_capital: float = param(100_000.0, "Starting equity in quote currency.")
    fee_bps: float = param(0.0, "Proportional fee per fill, in basis points.")
    slippage_ticks: int = param(5, "Adverse slippage per fill, in ticks.")
    tick_size: float = param(0.1, "Price value of one tick.")
    execution: str = param("next_open", "'next_open' or 'signal_close' (optimistic).")

    def __post_init__(self):
        check_int("ma_length", self.ma_length, 2)
        check_int("streak_length", self.streak_length, 2)
        check_int("atr_length", self.atr_length, 1)
        for name in ("atr_mult", "risk_pct", "tail_mult", "conf_mult", "cap_pct",
                     "htf_boundary_tolerance_hours"):
            check_positive(name, getattr(self, name))
        if not (0 < self.cap_pct <= 100):
            raise ValueError("cap_pct must be <= 100")
        try:
            htf_duration_us(self.htf)
        except (TypeError, ValueError) as exc:
            raise ValueError("htf must be a fixed timeframe such as '1D' or '4h'") from exc
        check_execution_settings(execution=self.execution, initial_capital=self.initial_capital,
                                 fee_bps=self.fee_bps, slippage_ticks=self.slippage_ticks,
                                 tick_size=self.tick_size)


SPEC = StrategySpec(
    name="tis_btc_momentum_confluence",
    title="TIS - BTC Momentum + Confluence",
    description="Long-only. Close above an SMA with a green-bar streak on the bar series AND on "
                "fully confirmed higher-timeframe candles; exit below the SMA or on a long upper-tail bar.",
    source="reference notebook primary_strategies.ipynb, strategy 'tis_btc_momentum_confluence' (Pine v6)",
    aliases=(),
    inputs=BAR_INPUTS,
    timestamp_rules=(
        "bar_start_us must be finite and strictly increasing (input is stably sorted by it)",
        "decision time of a bar = max(bar_end_us, bar_available_us, decision_time_us)",
        "HTF candles are UTC epoch-aligned buckets built from bars whose END falls in the bucket; their "
        "close is usable only at/after max(bucket end, decision time of every contributing bar)",
        "next_open orders fill only on a bar whose first trade is at/after the signal's decision time",
    ),
    indicators=(
        IndicatorSpec("sma", "simple moving average of close", "undefined until ma_length bars",
                      "restarted at every segment", ("ma_length",), ("sma_trend",)),
        IndicatorSpec("atr", "Wilder ATR (first true range = high-low)",
                      "RMA seeded with the SMA of the first atr_length true ranges",
                      "restarted at every segment", ("atr_length",), ("atr_value",)),
        IndicatorSpec("green_streak", "consecutive bars with close > open", "starts at 0",
                      "restarted at every segment", ("streak_length",), ("green_streak",)),
        IndicatorSpec("htf_green_streak", "consecutive green completed HTF candles rebuilt from the bars",
                      "0; candles that miss a bucket boundary by more than the tolerance or a missing "
                      "calendar bucket reset the streak",
                      "rebuilt inside every segment; a bucket cut by a break is incomplete",
                      ("htf", "streak_length", "htf_boundary_tolerance_hours"),
                      ("htf_green_streak", "htf_momentum")),
    ),
    features=(
        FeatureSpec("sma_trend", True, "SMA(close, ma_length) on the signal bar", "float64"),
        FeatureSpec("atr_value", True, "Wilder ATR on the signal bar (sizing and tail-cut)", "float64"),
        FeatureSpec("green_streak", True, "consecutive green bars ending at the signal bar", "int32"),
        FeatureSpec("htf_green_streak", True, "consecutive confirmed green HTF candles known at decision time", "int32"),
        FeatureSpec("htf_momentum", True, "htf_green_streak >= streak_length", "bool"),
    ),
    parameters=parameters_from_config(TisMomentumConfig),
    warmup="First candidate needs max(ma_length, atr_length, streak_length) bars in the segment plus "
           "streak_length fully confirmed HTF candles (data dependent, at least streak_length x htf of history).",
    invalid_data=(
        "invalid bars never signal; any break splits the segment and restarts every indicator",
        "an HTF bucket not covered within the boundary tolerance counts as incomplete and resets its streak",
        "an open position is censored (not closed, no label) and a pending order is discarded at a break",
    ),
    entry_rules=("close > SMA(ma_length)", "green_streak >= streak_length", "htf_green_streak >= streak_length",
                 "ATR finite and > 0"),
    exit_rules=("close < SMA -> BELOW_SMA", "(open - close) > tail_mult * ATR -> TAIL_CUT"),
    stateful_behavior=("one long position at a time; no new order while an order is pending",
                       "entries are ignored while a position is open",
                       "sizing uses equity at the signal bar; equity is updated on exits"),
    execution=(
        "quantity = min(equity*risk_pct/100/(ATR*atr_mult)*conf_mult, equity*cap_pct/100/signal_close)",
        "ATR*atr_mult only sizes the position: the strategy has NO protective stop",
        "next_open fills at the next eligible bar's open (default); signal_close at the signal close",
        "fee and slippage applied on both fills",
    ),
    limitations=("with volume bars the reconstructed HTF candles are approximate (bars span boundaries)",
                 "an HTF candle is published at max(bucket end, latest decision time of ALL its bars): if bars become "
                 "available long after they end, the value at the last bars of a bucket depends on the availability of "
                 "later bars of the same bucket (conservative: it can only delay confirmation; no effect when "
                 "bar_available_us == bar_end_us). Preserved from the reference",
                 "CORRECTION vs the reference: HTF confirmation times are made monotone (running maximum) before the "
                 "lookup. The reference used a binary search on a possibly unsorted array when bars become available "
                 "out of order, which made its result depend on the array length and the numpy version. Identical to "
                 "the reference whenever confirmations are already monotone (e.g. bar_available_us == bar_end_us)",
                 "no lot rounding, funding or margin"),
)


class TisMomentumConfluence(Strategy):
    name = "tis_btc_momentum_confluence"
    config_class = TisMomentumConfig
    spec = SPEC

    def cost_model(self, config):
        return CostModel(config.fee_bps, config.slippage_ticks, config.tick_size)

    def min_segment_bars(self, config):
        return max(config.ma_length, config.atr_length, config.streak_length)

    def compute_features(self, bars: PreparedBars, config):
        n = bars.n
        feats = {
            "sma_trend": np.full(n, np.nan), "atr_value": np.full(n, np.nan),
            "green_streak": np.zeros(n, dtype=np.int32),
            "htf_green_streak": np.zeros(n, dtype=np.int32),
            "htf_momentum": np.zeros(n, dtype=bool),
        }
        for idx in bars.segments():
            if not bars.valid[idx].all():
                continue
            gc = bars.c[idx]
            feats["sma_trend"][idx] = sma(gc, config.ma_length)
            feats["atr_value"][idx] = atr_wilder(bars.h[idx], bars.l[idx], gc, config.atr_length)
            run = 0
            for j, a in enumerate(idx):
                run = run + 1 if gc[j] > bars.o[a] else 0
                feats["green_streak"][a] = run
            ok, counts = higher_timeframe_momentum(
                idx, bars.o, bars.c, bars.start_us, bars.end_us, bars.ready_us,
                config.streak_length, config.htf, config.htf_boundary_tolerance_hours)
            feats["htf_green_streak"][idx] = counts
            feats["htf_momentum"][idx] = ok
        return feats

    def generate_signals(self, bars, feats, config):
        sm, atr, c, o = feats["sma_trend"], feats["atr_value"], bars.c, bars.o
        enter = ((c > sm) & (feats["green_streak"] >= config.streak_length)
                 & feats["htf_momentum"] & np.isfinite(atr) & (atr > 0))
        exit_ma = (c < sm) & np.isfinite(sm)
        exit_tail = ((o - c) > config.tail_mult * atr) & np.isfinite(atr)
        reason = np.where(exit_ma, "BELOW_SMA", np.where(exit_tail, "TAIL_CUT", None)).astype(object)
        return SignalSet(enter.astype(np.int8), {"exit_reason": reason})

    def simulate(self, bars, feats, signals, config):
        atr, c = feats["atr_value"], bars.c

        def quantity_for(idx: int, equity: float) -> float:
            dist = atr[idx] * config.atr_mult
            if not np.isfinite(dist) or dist <= 0 or equity <= 0:
                return 0.0
            raw = (equity * config.risk_pct / 100.0 / dist) * config.conf_mult
            cap = (equity * config.cap_pct / 100.0) / c[idx]
            return float(max(0, min(raw, cap)))

        return simulate_long_only(
            bars, signals.entry_side > 0, signals.extras["exit_reason"],
            execution=config.execution, cost=self.cost_model(config),
            initial_capital=config.initial_capital, quantity_for=quantity_for)

    def assumptions(self, config):
        return super().assumptions(config) + [
            f"HTF confirmation: last completed UTC {config.htf} candle, never the developing one"]
