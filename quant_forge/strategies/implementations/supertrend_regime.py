"""DefinedEdge SuperTrend Regime Confluence (Pine v6).

Adaptive-multiplier SuperTrend flips, scored 0-100 and filtered by trend, regime
and volume, traded with ATR/percent/SuperTrend stops, RR/percent take profit and an
optional trailing stop, with cooldown and long/short reversals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..base import (SignalSet, Strategy, StrategyConfig, check_choice, check_execution_settings,
                   check_int, check_nonnegative, check_positive)
from ..bars import PreparedBars
from ..execution import (EXECUTED, ORDER_DISCARDED_DATA_BREAK, ORDER_UNFILLED_END_OF_DATA,
                        REJECTED_ORDER, SKIPPED_COOLDOWN, SKIPPED_PENDING_ORDER,
                        SKIPPED_SAME_SIDE_POSITION, SKIPPED_ZERO_QUANTITY, CostModel, Ledger,
                        TradeRecord, settle)
from ..indicators import atr_wilder, dmi_zero_filled, ema_first_value_seed, sma
from ..results import ROLE_EXECUTION, ROLE_OUTCOME
from ..inputs import BAR_INPUTS
from ..spec import (ColumnRequirement, FeatureSpec, IndicatorSpec, StrategySpec, param,
                   parameters_from_config)

_SOURCES = ("open", "high", "low", "close", "hl2", "hlc3", "ohlc4")


@dataclass(frozen=True)
class SupertrendRegimeConfig(StrategyConfig):
    st_atr_length: int = param(10, "ATR length of the SuperTrend (>= 1).")
    st_base_multiplier: float = param(3.0, "Base ATR multiplier of the SuperTrend bands.")
    st_source: str = param("hl2", "Band source: open/high/low/close/hl2/hlc3/ohlc4.")
    regime_lookback: int = param(40, "SMA length of the ATR used for the ATR-ratio regime (>= 2).")
    adx_length: int = param(14, "DMI/ADX length (>= 1).")
    adx_threshold: float = param(20.0, "ADX below this (with a low ATR ratio) marks the ranging regime.")
    adaptive_multiplier: bool = param(True, "Scale the multiplier by regime (clipped to 0.5x-2x of base).")
    trend_ema_length: int = param(50, "Trend EMA length (>= 2).")
    volume_ma_length: int = param(20, "Volume SMA length (>= 2).")
    volume_column: str = param("volume", "Input column holding bar volume.")
    min_signal_score: int = param(65, "Minimum flip score, 0-100.")
    size_mode: str = param("Risk %", "'Risk %', 'Equity %' or 'Fixed Units'.")
    risk_pct: float = param(5.0, "Percent of equity risked at the stop distance (size_mode 'Risk %').")
    equity_pct: float = param(15.0, "Percent of equity as notional (size_mode 'Equity %').")
    fixed_units: float = param(1.0, "Units per trade (size_mode 'Fixed Units').")
    no_leverage: bool = param(True, "Cap the quantity at 0.90 * equity / signal close.")
    sl_mode: str = param("ATR", "Stop distance: 'ATR', 'Percent' or 'SuperTrend'.")
    sl_atr_mult: float = param(6.0, "ATR multiple of the stop (sl_mode 'ATR').")
    sl_pct: float = param(3.0, "Percent of signal close of the stop (sl_mode 'Percent').")
    tp_mode: str = param("RR", "Take profit: 'RR', 'Percent' or 'None'.")
    tp_rr: float = param(2.5, "Reward/risk multiple (tp_mode 'RR').")
    tp_pct: float = param(6.0, "Percent of signal close (tp_mode 'Percent').")
    trailing_stop: bool = param(False, "Enable the ATR trailing stop.")
    trail_atr_mult: float = param(2.5, "ATR multiple of the trailing offset (ATR of the signal bar).")
    use_trend_filter: bool = param(True, "Require the flip direction to agree with the trend EMA.")
    skip_ranging: bool = param(True, "Skip flips in the ranging regime.")
    use_volume_filter: bool = param(True, "Require volume above its SMA on the flip bar.")
    cooldown_bars: int = param(5, "Bars after the last entry's signal bar during which entries are skipped (>= 0).")
    allow_longs: bool = param(True, "Enable long entries.")
    allow_shorts: bool = param(True, "Enable short entries.")
    bt_from: str = param("2015-01-01", "Entries allowed from this bar START time (UTC).")
    bt_to: str = param("2035-01-01", "Entries allowed up to this bar START time (UTC).")
    initial_capital: float = param(10_000.0, "Starting equity in quote currency.")
    fee_bps: float = param(6.0, "Proportional fee per fill, in basis points.")
    slippage_ticks: int = param(2, "Adverse slippage per fill, in ticks.")
    tick_size: float = param(0.1, "Price value of one tick.")
    execution: str = param("next_open", "'next_open' (default) or 'signal_close' (optimistic).")

    def __post_init__(self):
        for name, minimum in (("st_atr_length", 1), ("regime_lookback", 2), ("adx_length", 1),
                              ("trend_ema_length", 2), ("volume_ma_length", 2),
                              ("min_signal_score", 0), ("cooldown_bars", 0)):
            check_int(name, getattr(self, name), minimum)
        if self.min_signal_score > 100:
            raise ValueError("min_signal_score must be <= 100")
        check_choice("size_mode", self.size_mode, ("Risk %", "Equity %", "Fixed Units"))
        check_choice("sl_mode", self.sl_mode, ("ATR", "Percent", "SuperTrend"))
        check_choice("tp_mode", self.tp_mode, ("RR", "Percent", "None"))
        check_choice("st_source", self.st_source, _SOURCES)
        for name in ("st_base_multiplier", "adx_threshold", "risk_pct", "equity_pct", "sl_atr_mult",
                     "sl_pct", "tp_rr", "tp_pct", "trail_atr_mult"):
            check_positive(name, getattr(self, name))
        check_nonnegative("fixed_units", self.fixed_units)
        if self.risk_pct > 100 or self.equity_pct > 100:
            raise ValueError("risk_pct and equity_pct must be <= 100")
        check_execution_settings(execution=self.execution, initial_capital=self.initial_capital,
                                 fee_bps=self.fee_bps, slippage_ticks=self.slippage_ticks,
                                 tick_size=self.tick_size)
        start, stop = (_utc(self.bt_from), _utc(self.bt_to))
        if not start < stop:
            raise ValueError("bt_from must be earlier than bt_to")


def _utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _epoch_us(ts: pd.Timestamp) -> int:
    return int((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(1, "us"))


SPEC = StrategySpec(
    name="supertrend_regime_confluence",
    title="DefinedEdge SuperTrend Regime Confluence",
    description="SuperTrend flips with regime-adaptive multiplier, scored and filtered; stops/targets are anchored "
                "to the signal close; longs and shorts with reversals and a cooldown.",
    source="reference notebook primary_strategies.ipynb, strategy 'supertrend_regime_confluence' (Pine v6)",
    aliases=(),
    inputs=BAR_INPUTS + (
        ColumnRequirement("volume", True, "volume", "bar volume (name configurable via `volume_column`); finite and >= 0"),),
    timestamp_rules=(
        "bar_start_us must be finite and strictly increasing (input is stably sorted by it)",
        "decision time of a bar = max(bar_end_us, bar_available_us, decision_time_us)",
        "the entry window bt_from..bt_to compares against bar START times (Pine `time`) and gates entries only",
        "next_open orders fill only on a bar whose first trade is at/after the signal's decision time",
        "intrabar stop/TP exits get exit_time = bar end (a proxy, flagged exit_time_is_bar_end_proxy)",
    ),
    indicators=(
        IndicatorSpec("atr", "Wilder ATR (first true range = high-low)",
                      "RMA seeded with the SMA of the first st_atr_length true ranges",
                      "restarted at every segment", ("st_atr_length",), ("atr_value",)),
        IndicatorSpec("atr_ratio", "ATR / SMA(ATR, regime_lookback)", "undefined until the SMA exists",
                      "restarted at every segment", ("regime_lookback",), ("st_atr_ratio",)),
        IndicatorSpec("adx_zero_filled", "Wilder DMI/ADX, variant with ZERO-FILLED warm-up (differs from the "
                                         "Donchian ADX)",
                      "first DM = 0, DI = 0 until the smoothed TR exists, DX = 0 if DI sum is 0; ADX is "
                      "available from segment offset adx_length-1 with a seed made mostly of zero-filled DX, "
                      "so it is biased low at the start of every segment",
                      "restarted at every segment", ("adx_length",), ("st_adx",)),
        IndicatorSpec("regime", "2 = volatile (ratio > 1.4); 0 = ranging (ADX < threshold and ratio < 0.9); else 1",
                      "regime 1 while the ratio is undefined", "per bar", ("adx_threshold",), ("st_regime",)),
        IndicatorSpec("supertrend", "SuperTrend bands on `st_source` with the adaptive multiplier",
                      "first direction = 1; first band is the lower/upper base; no value while ATR is undefined",
                      "restarted at every segment", ("st_atr_length", "st_base_multiplier", "st_source",
                                                      "adaptive_multiplier"),
                      ("st_band", "st_direction", "st_adapt_multiplier")),
        IndicatorSpec("trend_ema", "EMA seeded with the first close, not masked during warm-up",
                      "first close of the segment", "restarted at every segment", ("trend_ema_length",), ("st_ema",)),
        IndicatorSpec("volume_ma", "SMA of volume", "undefined until volume_ma_length bars",
                      "restarted at every segment", ("volume_ma_length",), ("st_vol_ma",)),
        IndicatorSpec("signal_score", "0-100 score of a flip: volume, displacement, EMA alignment, regime, prior distance",
                      "computed on flips only (0 elsewhere)", "flip needs a previous band in the same segment",
                      ("min_signal_score",), ("signal_score",)),
    ),
    features=(
        FeatureSpec("atr_value", True, "ATR(st_atr_length) on the signal bar (stop, trailing offset)", "float64"),
        FeatureSpec("st_band", True, "SuperTrend band", "float64"),
        FeatureSpec("st_direction", True, "SuperTrend direction (+1/-1)", "int8"),
        FeatureSpec("st_regime", True, "regime code 0/1/2", "int8"),
        FeatureSpec("st_atr_ratio", True, "ATR / SMA(ATR)", "float64"),
        FeatureSpec("st_adx", True, "zero-filled-warm-up ADX", "float64"),
        FeatureSpec("st_ema", True, "trend EMA", "float64"),
        FeatureSpec("st_vol_ma", True, "volume SMA", "float64"),
        FeatureSpec("st_adapt_multiplier", True, "adaptive SuperTrend multiplier", "float64"),
        FeatureSpec("signal_score", True, "flip score 0-100", "int16"),
        FeatureSpec("sl_distance", True, "stop distance from the signal close", "float64"),
        FeatureSpec("tp_distance", True, "take-profit distance (NaN if tp_mode None)", "float64"),
    ),
    parameters=parameters_from_config(SupertrendRegimeConfig),
    warmup="First flip at segment offset st_atr_length (needs a previous band); with the volume filter also "
           "volume_ma_length - 1 bars. The regime ratio needs st_atr_length + regime_lookback - 1 bars, "
           "regime 1 is assumed before.",
    invalid_data=(
        "invalid bars (including non-finite or negative volume) never signal; breaks split the segment and "
        "restart every indicator",
        "an open position is censored (no exit, no label), the pending order is discarded",
    ),
    entry_rules=(
        "SuperTrend direction flip with signal_score >= min_signal_score",
        "trend filter: direction agrees with the EMA (optional)",
        "regime filter: skip ranging (optional)",
        "volume filter: volume > volume SMA (optional)",
        "inside the bt_from..bt_to window; outside the cooldown (bars since the previous entry's signal bar > cooldown_bars)",
    ),
    exit_rules=(
        "stop = signal close -/+ stop distance; take profit anchored to the signal close",
        "gaps through stop/TP at the OPEN fill at the open; when both are touched in one bar the stop wins",
        "trailing stop (optional) activates after a 1R move and may be flagged ambiguous within the bar",
        "sl_mode 'SuperTrend': a flip against the position exits it (SUPERTREND_FLIP) even if the new entry is filtered",
        "an opposite entry signal reverses the position (OPPOSITE_SIGNAL)",
    ),
    stateful_behavior=("one position at a time; reversal = close then open on the same fill",
                       "cooldown measured from the last entry's signal bar", "no new order while one is pending",
                       "a same-side signal while in position is ignored"),
    execution=("sizing 'Risk %': equity*risk_pct/100/stop_distance; 'Equity %'; 'Fixed Units'; optional no-leverage cap",
               "next_open fills at the next eligible bar's open (default); signal_close at the signal close",
               "stops/targets use OHLC only: tick order inside a bar is unknown (stop-first, ambiguity flagged)",
               "fee and slippage applied on both fills"),
    limitations=("OHLC-only intrabar assumptions", "no lot rounding, funding or margin",
                 "the zero-filled-warm-up ADX is preserved from the reference although it differs from the Donchian ADX"),
)


class SupertrendRegimeConfluence(Strategy):
    name = "supertrend_regime_confluence"
    config_class = SupertrendRegimeConfig
    spec = SPEC
    trade_extra_roles = {
        "sl_price": ROLE_EXECUTION, "tp_price": ROLE_EXECUTION,
        "position_sizing_mode": ROLE_EXECUTION,
        "ambiguous_intrabar": ROLE_OUTCOME, "exit_time_is_bar_end_proxy": ROLE_OUTCOME,
        "trailing_activated": ROLE_OUTCOME,
    }

    def volume_column(self, config):
        return config.volume_column

    def cost_model(self, config):
        return CostModel(config.fee_bps, config.slippage_ticks, config.tick_size)

    def min_segment_bars(self, config):
        n = config.st_atr_length + 1
        return max(n, config.volume_ma_length) if config.use_volume_filter else n

    # ------------------------------------------------------------------ features
    def compute_features(self, bars: PreparedBars, config):
        n = bars.n
        f = {
            "atr_value": np.full(n, np.nan), "st_atr_ratio": np.full(n, np.nan),
            "st_adx": np.full(n, np.nan), "st_regime": np.full(n, -1, dtype=np.int8),
            "st_adapt_multiplier": np.full(n, np.nan), "st_band": np.full(n, np.nan),
            "st_direction": np.zeros(n, dtype=np.int8), "signal_score": np.zeros(n, dtype=np.int16),
            "st_ema": np.full(n, np.nan), "st_vol_ma": np.full(n, np.nan),
            "sl_distance": np.full(n, np.nan), "tp_distance": np.full(n, np.nan),
            "st_flip": np.zeros(n, dtype=bool),
        }
        base = config.st_base_multiplier
        for idx in bars.segments():
            if not bars.valid[idx].all():
                continue
            m = len(idx)
            og, hg, lg, cg, vg = bars.o[idx], bars.h[idx], bars.l[idx], bars.c[idx], bars.v[idx]
            atr = atr_wilder(hg, lg, cg, config.st_atr_length)
            atrma = sma(atr, config.regime_lookback)
            ratio = np.divide(atr, atrma, out=np.full(m, np.nan), where=np.isfinite(atrma) & (atrma > 0))
            _, _, adx = dmi_zero_filled(hg, lg, cg, config.adx_length)
            ema = ema_first_value_seed(cg, config.trend_ema_length)
            vm = sma(vg, config.volume_ma_length)
            src = {"open": og, "high": hg, "low": lg, "close": cg, "hl2": (hg + lg) / 2,
                   "hlc3": (hg + lg + cg) / 3, "ohlc4": (og + hg + lg + cg) / 4}[config.st_source]

            ratio_ok = np.isfinite(ratio)
            with np.errstate(invalid="ignore"):
                reg = np.where(ratio_ok & (ratio > 1.4), 2,
                               np.where(np.isfinite(adx) & (adx < config.adx_threshold) & ratio_ok & (ratio < 0.9), 0, 1))
                if config.adaptive_multiplier:
                    mult = np.where(reg == 2, base * (1.0 + (ratio - 1.0) * 0.4),
                                    np.where(reg == 0, base * 0.85, base))
                else:
                    mult = np.full(m, base)
            mult = np.clip(mult, base * 0.5, base * 2)

            band = np.full(m, np.nan)
            direction = np.zeros(m, dtype=np.int8)
            flips = np.zeros(m, dtype=bool)
            band_prev, direction_prev = np.nan, 1
            for j in range(m):
                if not np.isfinite(atr[j]):
                    continue
                upper_base = src[j] + mult[j] * atr[j]
                lower_base = src[j] - mult[j] * atr[j]
                old_band = band_prev if np.isfinite(band_prev) else (
                    lower_base if direction_prev == 1 else upper_base)
                if direction_prev == 1:
                    b = max(lower_base, old_band)
                    d = -1 if cg[j] < b else 1
                    if d == -1:
                        b = upper_base
                else:
                    b = min(upper_base, old_band)
                    d = 1 if cg[j] > b else -1
                    if d == 1:
                        b = lower_base
                band[j], direction[j] = b, d
                flips[j] = j > 0 and np.isfinite(band_prev) and d != direction_prev
                band_prev, direction_prev = b, d

            score = np.zeros(m, dtype=np.int16)
            for j in np.flatnonzero(flips):
                d, b, a = direction[j], band[j], atr[j]
                # Pine scoreSignal is only evaluated on a SuperTrend flip.
                vr = vg[j] / vm[j] if np.isfinite(vm[j]) and vm[j] > 0 else 1.0
                vol_score = 20 if vr >= 2.5 else 14 if vr >= 1.5 else 8 if vr >= 1.0 else 3
                disp = cg[j] - b if d == 1 else b - cg[j]
                disp_atr = disp / a if a > 0 else 0.0
                disp_score = (25 if disp_atr >= 1.5 else 18 if disp_atr >= 0.8 else 12 if disp_atr >= 0.3
                              else 5 if disp_atr > 0 else 0)
                aligned = (d == 1 and cg[j] > ema[j]) or (d == -1 and cg[j] < ema[j])
                ema_dist = abs(cg[j] - ema[j]) / a if a > 0 else 0
                ema_score = 20 if aligned and ema_dist > 0.5 else 14 if aligned else 8 if ema_dist < 0.3 else 2
                regime_score = 15 if reg[j] == 1 else 8 if reg[j] == 2 else 3
                prev_dist = abs(cg[j - 1] - band[j - 1]) / a if j > 0 and np.isfinite(band[j - 1]) else 0
                prior_score = 20 if prev_dist >= 2 else 14 if prev_dist >= 1 else 8 if prev_dist >= 0.5 else 3
                score[j] = int(min(vol_score + disp_score + ema_score + regime_score + prior_score, 100))

            finite = np.isfinite(atr)
            with np.errstate(invalid="ignore"):
                sl = (atr * config.sl_atr_mult if config.sl_mode == "ATR" else
                      cg * config.sl_pct / 100.0 if config.sl_mode == "Percent" else np.abs(cg - band))
                tp = (sl * config.tp_rr if config.tp_mode == "RR" else
                      cg * config.tp_pct / 100.0 if config.tp_mode == "Percent" else np.full(m, np.nan))
            sl, tp = np.where(finite, sl, np.nan), np.where(finite, tp, np.nan)

            for key, vals in (("atr_value", atr), ("st_atr_ratio", ratio), ("st_adx", adx), ("st_regime", reg),
                              ("st_adapt_multiplier", mult), ("st_band", band), ("st_direction", direction),
                              ("signal_score", score), ("st_ema", ema), ("st_vol_ma", vm),
                              ("sl_distance", sl), ("tp_distance", tp), ("st_flip", flips)):
                f[key][idx] = vals
        return f

    # ------------------------------------------------------------------- signals
    def generate_signals(self, bars, feats, config):
        c, v = bars.c, bars.v
        direction, ema, vm = feats["st_direction"], feats["st_ema"], feats["st_vol_ma"]
        flip = feats["st_flip"]
        aligned = ((direction == 1) & (c > ema)) | ((direction == -1) & (c < ema))
        filters = (
            (feats["signal_score"] >= config.min_signal_score)
            & (aligned if config.use_trend_filter else True)
            & ((feats["st_regime"] != 0) if config.skip_ranging else True)
            & ((np.isfinite(vm) & (v > vm)) if config.use_volume_filter else True)
        )
        # Pine's `time` is the bar START; the backtest window gates entries only.
        lo, hi = _epoch_us(_utc(config.bt_from)), _epoch_us(_utc(config.bt_to))
        in_window = (bars.start_us >= lo) & (bars.start_us <= hi)
        longs = flip & (direction == 1) & filters & config.allow_longs & in_window
        shorts = flip & (direction == -1) & filters & config.allow_shorts & in_window
        side = np.where(longs, 1, np.where(shorts, -1, 0)).astype(np.int8)
        return SignalSet(side, {"flip": flip, "in_window": in_window})

    # ---------------------------------------------------------------- simulation
    def simulate(self, bars, feats, signals, config):
        n = bars.n
        o, h, l, c = bars.o, bars.h, bars.l, bars.c
        first, end, ready = bars.first_us, bars.end_us, bars.ready_us
        valid, adjacent = bars.valid, bars.adjacent
        cost = self.cost_model(config)
        fee = cost.fee_rate
        side_at, flips = signals.entry_side, signals.extras["flip"]
        st_bands, st_dirs = feats["st_band"], feats["st_direction"]
        atr_vals, sl_distances, tp_distances = feats["atr_value"], feats["sl_distance"], feats["tp_distance"]
        next_open = config.execution == "next_open"
        ledger = Ledger(config.initial_capital)
        position: dict | None = None     # {"rec": TradeRecord, "sl", "tp", trailing state...}
        pending: dict | None = None
        last_entry_bar = 0               # same initial value as the Pine script

        def quantity_for(signal_idx: int) -> float:
            dist, price, equity = sl_distances[signal_idx], c[signal_idx], ledger.equity
            if not (np.isfinite(dist) and dist > 0 and price > 0 and equity > 0):
                return 0.0
            if config.size_mode == "Risk %":
                q = equity * config.risk_pct / 100.0 / dist
            elif config.size_mode == "Equity %":
                q = equity * config.equity_pct / 100.0 / price
            else:
                q = config.fixed_units
            if config.no_leverage:
                q = min(q, 0.90 * equity / price)
            return float(max(q, 0.0)) if np.isfinite(q) else 0.0

        def open_pos(signal_idx: int, fill_idx: int, raw: float, quantity: float) -> bool:
            nonlocal position, last_entry_bar
            side = int(side_at[signal_idx])
            if quantity <= 0:
                ledger.mark(signal_idx, SKIPPED_ZERO_QUANTITY)
                return False
            px = cost.entry_fill(raw, side)
            if px <= 0:
                ledger.mark(signal_idx, REJECTED_ORDER)
                return False
            ledger.opened += 1
            fee_entry = quantity * px * fee
            equity_before = ledger.equity
            ledger.equity -= fee_entry
            signal_close = c[signal_idx]
            stop_distance, target_distance = sl_distances[signal_idx], tp_distances[signal_idx]
            rec = TradeRecord(
                trade_id=ledger.opened, signal_idx=signal_idx, side=side, entry_idx=fill_idx,
                entry_us=float(first[fill_idx] if next_open else end[fill_idx]), entry_price=px,
                qty=quantity, entry_fee=fee_entry, entry_equity=equity_before)
            rec.extras.update(
                sl_price=signal_close - side * stop_distance,
                tp_price=(signal_close + side * target_distance) if np.isfinite(target_distance) else np.nan,
                position_sizing_mode=config.size_mode, ambiguous_intrabar=False,
                exit_time_is_bar_end_proxy=False, trailing_activated=False)
            position = dict(
                rec=rec, side=side, sl=rec.extras["sl_price"], tp=rec.extras["tp_price"],
                sl_dist=stop_distance, fill_idx=fill_idx, trail_active=False, trail_level=np.nan,
                trail_offset=atr_vals[signal_idx] * config.trail_atr_mult, ambiguous=False)
            ledger.trades.append(rec)
            ledger.mark(signal_idx, EXECUTED)
            ledger.trade_of_signal[signal_idx] = rec.trade_id
            last_entry_bar = signal_idx
            return True

        def close_pos(fill_idx: int, raw: float, reason: str, *, when_us=None, intrabar=False,
                      exit_signal_idx: int | None = None) -> None:
            nonlocal position
            if position is None:
                return
            pos = position
            rec: TradeRecord = pos["rec"]
            px = cost.exit_fill(raw, rec.side)
            if px <= 0:
                raise ValueError("Non-positive exit price after slippage")
            s = settle(rec.side, rec.qty, rec.entry_price, px, rec.entry_fee, fee)
            ledger.equity += rec.qty * rec.side * (px - rec.entry_price) - s.exit_fee
            at_us = end[fill_idx] if when_us is None else when_us
            rec.exit_signal_idx, rec.exit_idx = exit_signal_idx, fill_idx
            rec.exit_us = float(at_us)
            rec.exit_decision_us = float(ready[exit_signal_idx]) if exit_signal_idx is not None else None
            rec.exit_price, rec.exit_reason = px, reason
            rec.gross_return, rec.net_return, rec.pnl, rec.fees = s.gross_return, s.net_return, s.pnl, s.fees
            rec.exit_equity = ledger.equity
            rec.extras.update(ambiguous_intrabar=bool(pos["ambiguous"]),
                              exit_time_is_bar_end_proxy=bool(intrabar),
                              trailing_activated=bool(pos["trail_active"]))
            ledger.bump("ambiguous_intrabar_exits", int(bool(pos["ambiguous"])))
            position = None

        def check_stop_or_target(i: int) -> None:
            """Bracket simulator with conservative OHLC-only exit priorities.

            Opening gaps are handled BEFORE high/low-dependent trailing updates.  A
            previously active stop is checked before new trailing highs/lows.  A newly
            updated trailing stop that is also crossed in the same candle is flagged
            ambiguous because the tick path is unknown.
            """
            if position is None:
                return
            pos = position
            side = pos["side"]
            fixed_stop, target = pos["sl"], pos["tp"]
            old_stop = fixed_stop
            if pos["trail_active"] and np.isfinite(pos["trail_level"]):
                old_stop = (max(old_stop, pos["trail_level"]) if side == 1
                            else min(old_stop, pos["trail_level"]))
            if side == 1:
                open_stop = o[i] <= old_stop
                open_target = np.isfinite(target) and o[i] >= target
                old_stop_hit = l[i] <= old_stop
                target_hit = np.isfinite(target) and h[i] >= target
            else:
                open_stop = o[i] >= old_stop
                open_target = np.isfinite(target) and o[i] <= target
                old_stop_hit = h[i] >= old_stop
                target_hit = np.isfinite(target) and l[i] <= target

            if open_stop or open_target:
                if open_stop:
                    reason = "TRAILING_STOP" if old_stop != fixed_stop else "STOP_LOSS"
                else:
                    reason = "TAKE_PROFIT"
                close_pos(i, o[i], reason, intrabar=False, when_us=first[i])
                return
            if old_stop_hit:
                pos["ambiguous"] |= bool(target_hit)
                reason = "TRAILING_STOP" if old_stop != fixed_stop else "STOP_LOSS"
                close_pos(i, old_stop, reason, intrabar=True)
                return

            new_stop = old_stop
            if config.trailing_stop and np.isfinite(pos["trail_offset"]) and pos["trail_offset"] > 0:
                previously_active = pos["trail_active"]
                favorable = h[i] if side == 1 else l[i]
                entry_px = pos["rec"].entry_price
                activated = previously_active or (
                    favorable >= entry_px + pos["sl_dist"] if side == 1
                    else favorable <= entry_px - pos["sl_dist"])
                if activated:
                    pos["trail_active"] = True
                    candidate = favorable - pos["trail_offset"] * side
                    pos["trail_level"] = (
                        max(pos["trail_level"], candidate) if side == 1 else min(pos["trail_level"], candidate)
                    ) if np.isfinite(pos["trail_level"]) else candidate
                    new_stop = (max(old_stop, pos["trail_level"]) if side == 1
                                else min(old_stop, pos["trail_level"]))
            new_hit = (l[i] <= new_stop if side == 1 else h[i] >= new_stop)
            if new_hit and new_stop != old_stop:
                pos["ambiguous"] = True
                close_pos(i, new_stop, "TRAILING_STOP", intrabar=True)
                return
            if target_hit:
                close_pos(i, target, "TAKE_PROFIT", intrabar=True)

        for i in range(n):
            # Data breaks censor the position instead of silently closing it.
            if not valid[i] or (i > 0 and not adjacent[i]):
                if position is not None:
                    had_mark = i > 0 and valid[i - 1]
                    ledger.censor(position["rec"], i - 1 if had_mark else None, c[i - 1] if had_mark else None)
                    position = None
                if pending is not None:
                    ledger.discarded_pending_data_break += 1
                    if pending["action"] in ("entry", "reverse"):
                        ledger.mark(pending["signal_idx"], ORDER_DISCARDED_DATA_BREAK)
                    pending = None
            if not valid[i]:
                continue

            if next_open and pending is not None and i > pending["signal_idx"] and first[i] >= pending["ready_us"]:
                order, pending = pending, None
                if order["action"] == "reverse":
                    if position is not None:
                        close_pos(i, o[i], "OPPOSITE_SIGNAL", when_us=first[i], exit_signal_idx=order["signal_idx"])
                    open_pos(order["signal_idx"], i, o[i], order["qty"])
                elif order["action"] == "exit":
                    if position is not None:
                        close_pos(i, o[i], "SUPERTREND_FLIP", when_us=first[i], exit_signal_idx=order["signal_idx"])
                else:
                    open_pos(order["signal_idx"], i, o[i], order["qty"])

            # Bracket orders can fill intrabar, including the entry bar for next_open;
            # with signal_close they cannot fill earlier than the bar after the signal.
            if position is not None and (next_open or position["fill_idx"] < i):
                check_stop_or_target(i)

            desired_side = int(side_at[i])
            if pending is not None:
                if desired_side:
                    ledger.mark(i, SKIPPED_PENDING_ORDER)
                continue
            if not np.isfinite(st_bands[i]):
                continue
            cooldown_ok = (i - last_entry_bar > config.cooldown_bars)

            # A flip in SuperTrend-stop mode exits even when entry filters/cooldown reject a new trade.
            flip_exit = (config.sl_mode == "SuperTrend" and flips[i] and position is not None
                         and st_dirs[i] != position["side"])
            if flip_exit:
                if config.execution == "signal_close":
                    close_pos(i, c[i], "SUPERTREND_FLIP", when_us=end[i], exit_signal_idx=i)
                else:
                    pending = {"action": "exit", "signal_idx": i, "ready_us": ready[i]}
            if not (cooldown_ok and desired_side):
                if desired_side:
                    ledger.mark(i, SKIPPED_COOLDOWN)
                continue
            if position is not None and position["side"] == desired_side:
                ledger.mark(i, SKIPPED_SAME_SIDE_POSITION)
                continue
            planned = quantity_for(i)
            if planned <= 0:
                ledger.mark(i, SKIPPED_ZERO_QUANTITY)
                continue
            if config.execution == "signal_close":
                if position is not None:
                    close_pos(i, c[i], "OPPOSITE_SIGNAL", when_us=end[i], exit_signal_idx=i)
                open_pos(i, i, c[i], planned)
            else:
                pending = {"action": "reverse" if position is not None else "entry",
                           "signal_idx": i, "ready_us": ready[i], "qty": planned}

        if position is not None:
            last_ok = n > 0 and valid[n - 1]
            ledger.end_open(position["rec"], n - 1 if last_ok else None, c[n - 1] if last_ok else None)
        if pending is not None:
            ledger.pending_unfilled_at_end += 1
            if pending["action"] in ("entry", "reverse"):
                ledger.mark(pending["signal_idx"], ORDER_UNFILLED_END_OF_DATA)
        return ledger
