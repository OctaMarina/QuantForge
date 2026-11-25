"""Strategy base class: the four-stage pipeline shared by every strategy.

    prepare_bars -> compute_features -> generate_signals -> simulate -> assemble

* ``compute_features``  indicators and decision-time features, per segment;
* ``generate_signals``  rule-level *candidate events* (no state, no outcomes);
* ``simulate``          stateful position/order handling (cooldowns, open
                        positions, reversals, stops) producing trades and statuses;
* assembly              :class:`~quant_forge.strategies.results.StrategyResult`.

``run(..., evaluate=False)`` stops after signal generation, giving candidate
events and their features without any information about their future.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .. import __version__
from .bars import PreparedBars, prepare_bars
from .execution import CLOSED, Ledger
from .results import (CORE_TRADE_COLUMNS, EVENT_COLUMNS, StrategyResult, build_events,
                      build_trades_frame, event_id)
from .spec import StrategySpec


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class StrategyConfig:
    """Base class of the per-strategy configuration dataclasses (immutable)."""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **changes: Any):
        return dataclasses.replace(self, **changes)


def check_int(name: str, value: Any, minimum: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def check_positive(name: str, value: Any) -> None:
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0")


def check_nonnegative(name: str, value: Any) -> None:
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0")


def check_choice(name: str, value: Any, choices: Sequence[str]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}")


def check_execution_settings(*, execution, initial_capital, fee_bps, slippage_ticks, tick_size) -> None:
    check_choice("execution", execution, ("next_open", "signal_close"))
    check_positive("initial_capital", initial_capital)
    check_nonnegative("fee_bps", fee_bps)
    check_int("slippage_ticks", slippage_ticks, 0)
    check_positive("tick_size", tick_size)


# --------------------------------------------------------------------------- stages
@dataclass
class SignalSet:
    """Rule-level output of ``generate_signals``.

    ``entry_side`` marks the candidate events: +1 long, -1 short, 0 none.
    ``extras`` holds strategy-specific arrays used by the simulation (exit
    conditions, flips...).  Nothing here depends on position state or outcomes.
    """

    entry_side: np.ndarray
    extras: dict[str, np.ndarray] = field(default_factory=dict)


def _select_features(spec: StrategySpec, features) -> tuple[str, ...]:
    if features in (None, "all"):
        return spec.feature_names
    if features == "required":
        return spec.required_features
    if isinstance(features, str):
        raise ValueError("features must be 'all', 'required' or a sequence of feature names")
    names = tuple(features)
    unknown = [f for f in names if f not in spec.feature_names]
    if unknown:
        raise ValueError(f"unknown features {unknown}; available: {list(spec.feature_names)}")
    return tuple(dict.fromkeys(names))


class Strategy(ABC):
    name: str
    aliases: tuple[str, ...] = ()
    config_class: type[StrategyConfig]
    spec: StrategySpec
    # extra trade-table columns this strategy adds, with their roles
    trade_extra_roles: dict[str, str] = {}

    # ---- configuration --------------------------------------------------------
    def default_config(self) -> StrategyConfig:
        return self.config_class()

    def make_config(self, config: StrategyConfig | None = None, **overrides: Any) -> StrategyConfig:
        if config is None:
            config = self.config_class()
        elif not isinstance(config, self.config_class):
            raise TypeError(f"{self.name} expects a {self.config_class.__name__}, got {type(config).__name__}")
        if overrides:
            valid = {f.name for f in dataclasses.fields(self.config_class)}
            unknown = sorted(set(overrides) - valid)
            if unknown:
                raise TypeError(f"unknown parameters for {self.name}: {unknown}; valid: {sorted(valid)}")
            config = dataclasses.replace(config, **overrides)
        return config

    def volume_column(self, config) -> str | None:
        return None

    def required_columns(self, config=None) -> tuple[str, ...]:
        config = self.make_config(config)
        extra = (self.volume_column(config),) if self.volume_column(config) else ()
        return tuple(dict.fromkeys([*self.spec.required_columns, *extra]))

    @abstractmethod
    def cost_model(self, config): ...

    @abstractmethod
    def min_segment_bars(self, config) -> int:
        """Lower bound of consecutive valid bars before the first candidate can occur."""

    # ---- stages ----------------------------------------------------------------
    @abstractmethod
    def compute_features(self, bars: PreparedBars, config) -> dict[str, np.ndarray]: ...

    def optional_features(self, bars: PreparedBars, feats: dict[str, np.ndarray], config) -> dict[str, np.ndarray]:
        return {}

    @abstractmethod
    def generate_signals(self, bars: PreparedBars, feats: dict[str, np.ndarray], config) -> SignalSet: ...

    @abstractmethod
    def simulate(self, bars: PreparedBars, feats: dict[str, np.ndarray], signals: SignalSet, config) -> Ledger: ...

    def assumptions(self, config) -> list[str]:
        return list(self.spec.execution) + list(self.spec.limitations)

    # ---- orchestration -----------------------------------------------------------
    def run(self, df: pd.DataFrame, config: StrategyConfig | None = None, *,
            evaluate: bool = True, features="all",
            context_columns: Sequence[str] = (), **overrides: Any) -> StrategyResult:
        """Run the strategy on a bar DataFrame (never mutated).

        evaluate         False -> only candidate events and features (no execution,
                         no outcomes).  True -> also simulate trades.
        features         'all' | 'required' | sequence of names from ``spec``.
        context_columns  extra input columns copied *from the signal bar* as model
                         inputs.  The caller asserts they are known at decision time.
        """
        cfg = self.make_config(config, **overrides)
        selected = _select_features(self.spec, features)
        context_columns = tuple(dict.fromkeys(context_columns))
        reserved = (set(self.spec.feature_names) | set(EVENT_COLUMNS) | set(CORE_TRADE_COLUMNS)
                    | set(self.trade_extra_roles))
        clash = sorted(set(context_columns) & reserved)
        if clash:
            raise ValueError(f"context_columns collide with strategy/result columns: {clash}")
        bars = prepare_bars(df, volume_column=self.volume_column(cfg), context_columns=context_columns)

        feats = self.compute_features(bars, cfg) if bars.n else {}
        if bars.n and any(f in self.spec.optional_features for f in selected):
            feats = {**feats, **self.optional_features(bars, feats, cfg)}
        signals = (self.generate_signals(bars, feats, cfg) if bars.n
                   else SignalSet(np.zeros(0, dtype=np.int8)))
        cand_idx = np.flatnonzero(signals.entry_side != 0)
        sides = signals.entry_side[cand_idx]

        ledger = self.simulate(bars, feats, signals, cfg) if (evaluate and bars.n) else None
        if evaluate and ledger is None:
            ledger = Ledger(getattr(cfg, "initial_capital", 0.0))

        events = build_events(self.name, bars, cand_idx, sides,
                              None if ledger is None else ledger.status,
                              None if ledger is None else ledger.trade_of_signal)
        feature_frame = pd.DataFrame({"event_id": events["event_id"].to_numpy()})
        for f in selected:
            feature_frame[f] = feats[f][cand_idx] if bars.n else np.array([])
        for c in context_columns:
            feature_frame[c] = bars.context[c].iloc[cand_idx].to_numpy()

        trades = None
        if ledger is not None:
            ids = {int(i): event_id(self.name, bars.start_us[i], s) for i, s in zip(cand_idx, sides)}
            trades = build_trades_frame(ledger.trades, bars, self.name, getattr(cfg, "execution", ""),
                                        ids, self.trade_extra_roles)

        diagnostics = self._diagnostics(bars, events, ledger, cfg)
        metadata = self._metadata(bars, cfg, evaluate, selected, context_columns)
        return StrategyResult(
            strategy=self.name, config=cfg, spec=self.spec, events=events, features=feature_frame,
            trades=trades, diagnostics=diagnostics, metadata=metadata,
            feature_columns=tuple(selected), context_columns=context_columns,
            trade_extra_roles=dict(self.trade_extra_roles))

    # ---- documentation of the run -------------------------------------------------
    def _diagnostics(self, bars: PreparedBars, events: pd.DataFrame, ledger: Ledger | None, cfg) -> dict[str, Any]:
        n = bars.n
        d: dict[str, Any] = {
            "evaluated": ledger is not None,
            "n_bars": n,
            "n_valid_bars": int(bars.valid.sum()),
            "n_invalid_bars": int(n - bars.valid.sum()),
            "n_segments": int(len(np.unique(bars.segment))) if n else 0,
            "n_data_breaks": int((~bars.adjacent[1:]).sum()) if n > 1 else 0,
            "min_segment_bars_before_first_candidate": int(self.min_segment_bars(cfg)),
            "n_candidate_events": int(len(events)),
            "n_candidate_long": int((events["side"] > 0).sum()),
            "n_candidate_short": int((events["side"] < 0).sum()),
            "first_bar_start_us": int(bars.start_us[0]) if n else None,
            "last_bar_end_us": int(np.nanmax(bars.end_us)) if n else None,
            "event_status_counts": events["status"].value_counts().to_dict(),
        }
        if ledger is not None:
            d.update({
                "n_executed": int(ledger.opened),
                "n_closed_trades": int(sum(t.status == CLOSED for t in ledger.trades)),
                "n_censored_data_break": int(ledger.censored_data_break),
                "n_open_at_end": int(ledger.open_at_end),
                "censored_positions": int(ledger.censored_positions),
                "total_opened_positions": int(ledger.opened),
                "n_discarded_pending_orders": int(ledger.discarded_pending_orders),
                "discarded_pending_data_break": int(ledger.discarded_pending_data_break),
                "pending_unfilled_at_end": int(ledger.pending_unfilled_at_end),
                "remaining_realized_equity_usd": float(ledger.equity),
                "unrealized_censored_pnl_usd": float(ledger.unrealized_censored_pnl),
                "unrealized_open_at_end_pnl_usd": float(ledger.unrealized_open_at_end_pnl),
                "equity_path_complete": bool(ledger.equity_path_complete and ledger.open_at_end == 0),
                **ledger.counters,
            })
        return d

    def _metadata(self, bars: PreparedBars, cfg, evaluate: bool, selected, context_columns) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "library_version": __version__,
            "config": cfg.to_dict(),
            "config_class": type(cfg).__name__,
            "evaluate": evaluate,
            "features": list(selected),
            "context_columns": list(context_columns),
            "columns_used": list(bars.columns_used),
            "n_input_rows": bars.n,
            "indicator_policy": "indicators are always recomputed from OHLC(V) per segment; "
                                "precomputed columns of the input frame are never reused",
            "decision_time": "max(bar_end_us, bar_available_us, decision_time_us) of the signal bar",
            "assumptions": self.assumptions(cfg),
        }
