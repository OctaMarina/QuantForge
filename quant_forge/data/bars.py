"""Bar frames used by :mod:`quant_forge.strategies` and :mod:`quant_forge.features`: time bars from exchange klines
and, through :func:`load_bars`, volume / dollar bars from trades (:mod:`quant_forge.data.event_bars`).

    from quant_forge.data import BarConfig, load_bars
    bars = load_bars(ALL_ASSETS, BarConfig("time", 6), end="2026-11-19")      # 4h klines
    bars = load_bars(ALL_ASSETS, BarConfig("dollar", 6), end="2026-11-19")    # dollar bars, 6 per day on average

Output columns of the time bars (index ``t`` = bar end, tz-aware UTC):

=================  ==========================================================================================
``bar_start``      bar open time (UTC)
``bar_end``        bar close time = start + bar length (also the index)
OHLC, ``volume``   base-asset volume; ``quote_volume``, ``n_trades`` (float), ``taker_buy_base``
``bar_start_us``   integer microseconds (strategy-library convention); ``bar_end_us`` likewise
``bar_seq``        position on the regular grid counted from ``seq_origin``; a jump means missing bars, which
                   splits indicator segments inside the strategy library
``complete``       1 when the bar had at least one trade
=================  ==========================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from .binance import INTERVALS, interval_timedelta, load_klines

__all__ = ["klines_to_bars", "load_binance_bars", "BarConfig", "load_bars", "TIME_INTERVAL_OF_BARS_PER_DAY"]

#: Binance interval whose bars match a number of bars per day (the time-bar counterpart of an event-bar target).
TIME_INTERVAL_OF_BARS_PER_DAY = {1440: "1m", 480: "3m", 288: "5m", 96: "15m", 48: "30m", 24: "1h", 12: "2h", 6: "4h",
                                 4: "6h", 3: "8h", 2: "12h", 1: "1d"}


def klines_to_bars(klines: pd.DataFrame, *, bar="4h", end=None, warmup_days: float = 30,
                   seq_origin="2017-01-01") -> pd.DataFrame:
    """Bar frame from a klines table (columns of :data:`quant_forge.data.binance.KLINE_COLUMNS`).

    bar          bar length (Timedelta or Binance interval code such as ``"4h"``).
    end          drop bars that end after this time (the fixed data end of a study).
    warmup_days  drop the first days after the first kline (listing effects: thin books, wild prints).
    seq_origin   origin of ``bar_seq``; only differences of ``bar_seq`` matter.
    """
    bar = interval_timedelta(bar) if isinstance(bar, str) and bar in INTERVALS else pd.Timedelta(bar)
    start = pd.to_datetime(klines["open_time"], unit="ms", utc=True)
    df = pd.DataFrame({"bar_start": start, "bar_end": start + bar, "open": klines["open"], "high": klines["high"],
                       "low": klines["low"], "close": klines["close"], "volume": klines["volume"],
                       "quote_volume": klines["quote_volume"], "n_trades": klines["n_trades"].astype(float),
                       "taker_buy_base": klines["taker_buy_base"]})
    keep = df["bar_start"] >= start.iloc[0] + pd.Timedelta(days=warmup_days)
    if end is not None:
        end = pd.Timestamp(end)
        keep &= df["bar_end"] <= (end.tz_localize("UTC") if end.tzinfo is None else end)
    df = df.loc[keep]
    df = df.set_index("bar_end", drop=False).rename_axis("t")
    # explicit unit: pandas >= 3 infers millisecond resolution from unit="ms"
    df["bar_start_us"] = df["bar_start"].astype("datetime64[us, UTC]").astype("int64")
    df["bar_end_us"] = df["bar_end"].astype("datetime64[us, UTC]").astype("int64")
    df["bar_seq"] = ((df["bar_start"] - pd.Timestamp(seq_origin, tz="UTC")) // bar).astype("int64")
    df["complete"] = (df["n_trades"] > 0).astype(int)
    return df


def load_binance_bars(assets: Sequence[str], *, end, interval: str = "4h", quote: str = "USDT",
                      warmup_days: float = 30, start="2017-08-01", cache_dir=None,
                      update: bool = True) -> Mapping[str, pd.DataFrame]:
    """``{asset: bars}`` for Binance spot ``<asset><quote>`` klines, cached locally, trimmed at ``end``.

    The order of the returned dict is the order of ``assets`` (cross-asset aggregates depend on it at
    floating-point rounding level).
    """
    out = {}
    for asset in assets:
        k = load_klines(f"{asset}{quote}", interval, end=end, start=start, cache_dir=cache_dir, update=update)
        out[asset] = klines_to_bars(k, bar=interval_timedelta(interval), end=end, warmup_days=warmup_days)
    return out


@dataclass(frozen=True)
class BarConfig:
    """How bars are sampled: ``bar_type`` ``"time"``, ``"volume"`` or ``"dollar"`` and the (target) frequency.

    bars_per_day   time bars: 24 h / bar length, which must be a Binance interval (48 -> 30m, 24 -> 1h, 6 -> 4h,
                   1 -> 1d); event bars: the target average of the trailing-volume threshold (realised frequency
                   varies with activity);
    window_days, max_gap_hours  event bars only (see :class:`~quant_forge.data.event_bars.EventBarSpec`).
    """

    bar_type: str
    bars_per_day: float
    window_days: int = 30
    max_gap_hours: float = 4.0

    def __post_init__(self):
        if self.bar_type not in ("time", "volume", "dollar"):
            raise ValueError("bar_type must be 'time', 'volume' or 'dollar'")
        if self.bar_type == "time" and self.bars_per_day not in TIME_INTERVAL_OF_BARS_PER_DAY:
            raise ValueError(f"time bars need bars_per_day in {sorted(TIME_INTERVAL_OF_BARS_PER_DAY)}")
        if not self.bars_per_day > 0:
            raise ValueError("bars_per_day must be > 0")

    @property
    def interval(self) -> str | None:
        """Binance interval of a time-bar config (``None`` for event bars)."""
        return TIME_INTERVAL_OF_BARS_PER_DAY[self.bars_per_day] if self.bar_type == "time" else None

    @property
    def event_spec(self):
        from .event_bars import EventBarSpec

        if self.bar_type == "time":
            return None
        return EventBarSpec(self.bar_type, self.bars_per_day, self.window_days, self.max_gap_hours)

    @property
    def key(self) -> str:
        """Short identifier, e.g. ``time_4h``, ``volume_6``, ``dollar_48``."""
        return f"time_{self.interval}" if self.bar_type == "time" else f"{self.bar_type}_{self.bars_per_day:g}"

    @property
    def label(self) -> str:
        return f"time {self.interval}" if self.bar_type == "time" else f"{self.bar_type} {self.bars_per_day:g}/d"

    @property
    def nominal_hours(self) -> float:
        """Average bar length implied by ``bars_per_day``."""
        return 24.0 / self.bars_per_day


def load_bars(assets: Sequence[str], config: BarConfig, *, end, quote: str = "USDT", warmup_days: float = 30,
              start="2017-08-01", update: bool = True, klines_cache_dir=None,
              event_cache_dir=None) -> Mapping[str, pd.DataFrame]:
    """``{asset: bar frame}`` for any :class:`BarConfig`, trimmed at ``end``, in the order of ``assets``.

    Time bars come from the kline cache (:func:`load_binance_bars`, ``warmup_days`` after the first kline are
    dropped); volume / dollar bars from the event-bar cache built from aggTrades
    (:func:`quant_forge.data.event_bars.load_event_bars`; their threshold warm-up is ``config.window_days`` days
    after the first trade, so ``warmup_days`` is not applied again).  With ``update`` missing data is downloaded /
    built first.
    """
    if config.bar_type == "time":
        return load_binance_bars(assets, end=end, interval=config.interval, quote=quote, warmup_days=warmup_days,
                                 start=start, cache_dir=klines_cache_dir, update=update)
    from .event_bars import load_event_bars

    return load_event_bars(assets, config.event_spec, end=end, quote=quote, cache_dir=event_cache_dir, update=update)
