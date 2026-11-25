"""Market data acquisition and bar formatting.

    from quant_forge.data import load_bars, BarConfig, SEEN_ASSETS, NEW_ASSETS

    bars = load_bars(SEEN_ASSETS + NEW_ASSETS, BarConfig("time", 6), end="2026-11-19")     # 4h klines
    bars = load_bars(SEEN_ASSETS + NEW_ASSETS, BarConfig("volume", 6), end="2026-11-19")   # volume bars, 6/day
    bars = load_binance_bars(SEEN_ASSETS, end="2026-11-19", interval="4h")                 # time bars only

* :mod:`~quant_forge.data.binance`     REST download of spot klines + incremental parquet cache;
* :mod:`~quant_forge.data.archive`     Binance public archive (data.binance.vision): klines and aggTrades, streamed;
* :mod:`~quant_forge.data.bars`        klines -> bar frame; :class:`BarConfig` / :func:`load_bars` for any bar type;
* :mod:`~quant_forge.data.event_bars`  volume and dollar bars from aggTrades with point-in-time thresholds;
* :mod:`~quant_forge.data.diagnostics` achieved bars per day and bar durations;
* :mod:`~quant_forge.data.universe`    the asset lists and tick sizes of the cross-asset research.
"""

from .bars import TIME_INTERVAL_OF_BARS_PER_DAY, BarConfig, klines_to_bars, load_bars, load_binance_bars
from .binance import KLINE_COLUMNS, fetch_klines, fetch_klines_hybrid, interval_timedelta, load_klines
from .diagnostics import bar_durations, sampling_by_period, sampling_summary
from .event_bars import (DEFAULT_EVENT_SPECS, EventBarBuilder, EventBarSpec, build_event_bars, build_event_bars_many,
                         daily_thresholds, event_bars_from_trades, event_bars_to_frame, event_cache_status,
                         load_event_bars)
from .universe import ALL_ASSETS, NEW_ASSETS, SEEN_ASSETS, TICK_SIZES

__all__ = ["load_bars", "BarConfig", "TIME_INTERVAL_OF_BARS_PER_DAY", "load_binance_bars", "klines_to_bars",
           "fetch_klines", "fetch_klines_hybrid", "load_klines", "interval_timedelta", "KLINE_COLUMNS",
           "EventBarSpec", "DEFAULT_EVENT_SPECS", "EventBarBuilder", "event_bars_from_trades", "build_event_bars",
           "build_event_bars_many", "event_bars_to_frame", "load_event_bars", "event_cache_status", "daily_thresholds",
           "bar_durations", "sampling_summary", "sampling_by_period",
           "SEEN_ASSETS", "NEW_ASSETS", "ALL_ASSETS", "TICK_SIZES"]
