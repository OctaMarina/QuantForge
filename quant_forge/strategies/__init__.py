"""Primary trading strategies: specification, event generation, execution and results.

Typical use::

    import quant_forge.strategies as qs

    qs.list_strategies()
    print(qs.Strategies.DONCHIAN_BREAKOUT_CONTINUATION.spec.describe())

    result = qs.run_strategy(bars, qs.Strategies.DONCHIAN_BREAKOUT_CONTINUATION, fee_bps=5)
    trades = result.closed_trades_frame()          # features + returns + labels
    result.model_input_columns                      # known at decision time
    result.post_decision_columns                    # only known afterwards

    events_only = qs.Strategies.RSI_CROSS_ABOVE.run(bars, evaluate=False)

Pipeline per strategy: ``prepare_bars -> compute_features -> generate_signals ->
simulate -> assemble``.  See :class:`~quant_forge.strategies.base.Strategy`.
"""

from .base import SignalSet, Strategy, StrategyConfig
from .bars import PreparedBars, prepare_bars
from .execution import CostModel
from .implementations import (DonchianBreakoutConfig, DonchianBreakoutContinuation, RsiCrossAbove,
                              RsiCrossAboveConfig, SupertrendRegimeConfig, SupertrendRegimeConfluence,
                              TisMomentumConfig, TisMomentumConfluence)
from .names import Strategies
from .registry import describe_strategy, get_strategy, list_strategies, register, run_strategy
from .results import StrategyResult
from .spec import (ColumnRequirement, FeatureSpec, IndicatorSpec, ParameterSpec, StrategySpec)

__all__ = [
    "Strategies", "run_strategy", "list_strategies", "get_strategy", "describe_strategy", "register",
    "Strategy", "StrategyConfig", "StrategyResult", "StrategySpec", "SignalSet",
    "ColumnRequirement", "FeatureSpec", "IndicatorSpec", "ParameterSpec",
    "PreparedBars", "prepare_bars", "CostModel",
    "RsiCrossAbove", "RsiCrossAboveConfig",
    "DonchianBreakoutContinuation", "DonchianBreakoutConfig",
    "TisMomentumConfluence", "TisMomentumConfig",
    "SupertrendRegimeConfluence", "SupertrendRegimeConfig",
]
