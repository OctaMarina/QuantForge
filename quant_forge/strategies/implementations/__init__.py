"""The strategies themselves: one module per strategy, each holding its configuration, specification and
implementation (features, signals, position handling).  Framework code (pipeline, results, indicators, execution
accounting) lives one level up in :mod:`quant_forge.strategies`.
"""

from .donchian_breakout import DonchianBreakoutConfig, DonchianBreakoutContinuation
from .rsi_cross_above import RsiCrossAbove, RsiCrossAboveConfig
from .supertrend_regime import SupertrendRegimeConfig, SupertrendRegimeConfluence
from .tis_momentum import TisMomentumConfig, TisMomentumConfluence

__all__ = [
    "RsiCrossAbove", "RsiCrossAboveConfig",
    "DonchianBreakoutContinuation", "DonchianBreakoutConfig",
    "TisMomentumConfluence", "TisMomentumConfig",
    "SupertrendRegimeConfluence", "SupertrendRegimeConfig",
]
