"""Strategy registry and the public entry points."""

from __future__ import annotations

from typing import Any

import pandas as pd

from enum import Enum

from .base import Strategy
from .implementations import (DonchianBreakoutContinuation, RsiCrossAbove, SupertrendRegimeConfluence,
                              TisMomentumConfluence)
from .results import StrategyResult
from .spec import StrategySpec

_STRATEGIES: dict[str, Strategy] = {}
_ALIASES: dict[str, str] = {}


def register(strategy: Strategy) -> None:
    if strategy.name in _STRATEGIES or strategy.name in _ALIASES:
        raise ValueError(f"strategy {strategy.name!r} is already registered")
    _STRATEGIES[strategy.name] = strategy
    for alias in strategy.aliases:
        if alias in _STRATEGIES or alias in _ALIASES:
            raise ValueError(f"alias {alias!r} is already registered")
        _ALIASES[alias] = strategy.name


for _cls in (RsiCrossAbove, DonchianBreakoutContinuation, TisMomentumConfluence, SupertrendRegimeConfluence):
    register(_cls())


def list_strategies() -> list[str]:
    """Canonical names of the registered strategies."""
    return list(_STRATEGIES)


def get_strategy(name: str | Enum) -> Strategy:
    """Look a strategy up by ``Strategies`` member, canonical name or alias (e.g. ``strategy_432_btc``)."""
    if isinstance(name, Enum):
        name = name.value
    canonical = _ALIASES.get(name, name)
    try:
        return _STRATEGIES[canonical]
    except KeyError:
        raise ValueError(f"Unknown strategy {name!r}; choose one of {list_strategies()}") from None


def describe_strategy(name: str | Enum) -> StrategySpec:
    """The inspectable :class:`StrategySpec` of a strategy."""
    return get_strategy(name).spec


def run_strategy(df: pd.DataFrame, strategy: str | Enum, config=None, *, evaluate: bool = True,
                 features="all", context_columns=(), **overrides: Any) -> StrategyResult:
    """Run ``strategy`` on the bar DataFrame ``df``; ``df`` is never modified.

    strategy         a ``Strategies`` member (preferred) or a strategy name.
    config           a strategy's config dataclass (see ``get_strategy(name).config_class``)
                     or ``None`` for the defaults.
    overrides        keyword overrides of config fields, e.g. ``fee_bps=5``.
    evaluate         ``False`` returns candidate events and decision-time features only.
    features         ``'all'``, ``'required'`` or a list of feature names.
    context_columns  input columns copied from the signal bar as extra model inputs;
                     the caller asserts they are known at decision time.
    """
    return get_strategy(strategy).run(df, config, evaluate=evaluate, features=features,
                                      context_columns=context_columns, **overrides)
