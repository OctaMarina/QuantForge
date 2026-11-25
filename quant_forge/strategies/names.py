"""Enumeration of the available strategies, so callers never type a strategy string."""

from __future__ import annotations

from enum import Enum


class Strategies(str, Enum):
    """Available strategies.  Members are ``str`` subclasses equal to the canonical strategy name, so they work
    anywhere a name is accepted::

        qs.run_strategy(bars, qs.Strategies.DONCHIAN_BREAKOUT_CONTINUATION, fee_bps=5)
        qs.Strategies.RSI_CROSS_ABOVE.run(bars)
        qs.Strategies.SUPERTREND_REGIME_CONFLUENCE.spec.describe()
    """

    RSI_CROSS_ABOVE = "rsi_cross_above"
    DONCHIAN_BREAKOUT_CONTINUATION = "donchian_breakout_continuation"
    STRATEGY_432_BTC = "donchian_breakout_continuation"   # alias of the Donchian strategy (name used in Pine/notebook)
    TIS_BTC_MOMENTUM_CONFLUENCE = "tis_btc_momentum_confluence"
    SUPERTREND_REGIME_CONFLUENCE = "supertrend_regime_confluence"

    @property
    def strategy(self):
        """The registered :class:`~quant_forge.strategies.base.Strategy` object."""
        from .registry import get_strategy
        return get_strategy(self.value)

    @property
    def spec(self):
        return self.strategy.spec

    @property
    def config_class(self):
        return self.strategy.config_class

    def run(self, df, config=None, **kwargs):
        """Shortcut for ``run_strategy(df, self, config, **kwargs)``."""
        return self.strategy.run(df, config, **kwargs)
