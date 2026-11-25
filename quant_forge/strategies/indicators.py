"""Indicator kernels shared by the strategies.

Every function is a faithful port of the reference notebook.  Each is meant to
be called on ONE segment (a run of adjacent valid bars): indicators therefore
restart at every data break.  A strategy never reuses an indicator column that
happens to exist in the caller's DataFrame; it always recomputes from OHLC(V).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_rma(values, length: int) -> np.ndarray:
    """Wilder RMA seeded with the SMA of the first ``length`` finite values.

    A non-finite input resets the accumulator, so undefined warm-up values
    (e.g. the first price change) delay the first output instead of being
    treated as zero.
    """
    a = np.asarray(values, dtype=float)
    out = np.full(len(a), np.nan, dtype=float)
    total, count, prev = 0.0, 0, np.nan
    for i, val in enumerate(a):
        if not np.isfinite(val):
            total, count, prev = 0.0, 0, np.nan
            continue
        if count < length:
            total += val
            count += 1
            if count == length:
                prev = total / length
                out[i] = prev
        else:
            prev = (prev * (length - 1) + val) / length
            out[i] = prev
    return out


def rsi_wilder(close, length: int) -> np.ndarray:
    """RSI from Wilder-smoothed gains/losses (flat market -> 50, no losses -> 100)."""
    change = np.diff(close, prepend=np.nan)
    avg_gain = wilder_rma(np.maximum(change, 0.0), length)
    avg_loss = wilder_rma(np.maximum(-change, 0.0), length)
    out = np.full(len(close), np.nan, dtype=float)
    ready = np.isfinite(avg_gain) & np.isfinite(avg_loss)
    positive = ready & (avg_loss > 0)
    out[positive] = 100.0 - 100.0 / (1.0 + avg_gain[positive] / avg_loss[positive])
    out[ready & (avg_loss == 0) & (avg_gain > 0)] = 100.0
    out[ready & (avg_loss == 0) & (avg_gain == 0)] = 50.0
    return out


def true_range_first_defined(high, low, close) -> np.ndarray:
    """True range; the first bar uses ``high - low`` (Pine ``ta.tr(true)``)."""
    previous_close = np.r_[np.nan, close[:-1]]
    true_range = np.maximum.reduce([
        high - low,
        np.abs(high - previous_close),
        np.abs(low - previous_close),
    ])
    true_range[0] = high[0] - low[0]
    return true_range


def atr_wilder(high, low, close, length: int) -> np.ndarray:
    """ATR = Wilder RMA of the true range (first bar's TR is ``high - low``)."""
    return wilder_rma(true_range_first_defined(high, low, close), length)


def dmi_warmup_undefined(high, low, close, length: int):
    """Wilder DMI/ADX used by the Donchian strategy.

    The first DM/TR is undefined and delays seeding; DI values are forward
    filled when the smoothed true range is zero.  Returns ``(+DI, -DI, ADX)``.
    """
    up = np.r_[np.nan, np.diff(high)]
    down = np.r_[np.nan, -np.diff(low)]
    plus = np.where((up > down) & (up > 0), up, 0.0)
    minus = np.where((down > up) & (down > 0), down, 0.0)
    plus[0] = minus[0] = np.nan
    prev = np.r_[np.nan, close[:-1]]
    tr = np.maximum.reduce([high - low, abs(high - prev), abs(low - prev)])
    tr = wilder_rma(tr, length)

    def di(dm):
        smoothed = wilder_rma(dm, length)
        values = np.divide(100 * smoothed, tr, out=np.full(len(tr), np.nan),
                           where=np.isfinite(tr) & (tr > 0))
        return pd.Series(values).ffill().to_numpy()

    pdi, mdi = di(plus), di(minus)
    total = pdi + mdi
    dx = 100 * abs(pdi - mdi) / np.where(total == 0, 1.0, total)
    return pdi, mdi, wilder_rma(dx, length)


def dmi_zero_filled(high, low, close, length: int):
    """Wilder DMI/ADX used by the SuperTrend strategy (a *different* variant).

    The first DM is treated as 0 (not undefined), DI is 0 while the smoothed true
    range is unavailable, and DX is 0 whenever +DI + -DI is 0.  ADX is therefore
    available from segment offset ``length - 1`` but its SMA seed averages mostly
    zero-filled DX values, which biases ADX low for the first bars of every segment
    (compare :func:`dmi_warmup_undefined`).  Returns ``(+DI, -DI, ADX)``.
    """
    m = len(high)
    upmove = np.r_[np.nan, np.diff(high)]
    downmove = np.r_[np.nan, -np.diff(low)]
    plusdm = np.where((upmove > downmove) & (upmove > 0), upmove, 0.0)
    minusdm = np.where((downmove > upmove) & (downmove > 0), downmove, 0.0)
    smoothtr = atr_wilder(high, low, close, length)
    smoothpd = wilder_rma(plusdm, length)
    smoothnd = wilder_rma(minusdm, length)
    ok = np.isfinite(smoothtr) & (smoothtr > 0)
    plusdi = np.divide(100 * smoothpd, smoothtr, out=np.zeros(m), where=ok)
    minusdi = np.divide(100 * smoothnd, smoothtr, out=np.zeros(m), where=ok)
    denom = plusdi + minusdi
    dx = np.divide(100 * np.abs(plusdi - minusdi), denom, out=np.zeros(m), where=denom > 0)
    return plusdi, minusdi, wilder_rma(dx, length)


def sma(values, length: int) -> np.ndarray:
    return pd.Series(values).rolling(length, min_periods=length).mean().to_numpy()


def ema_first_value_seed(values, length: int) -> np.ndarray:
    """EMA with ``alpha = 2/(length+1)`` seeded by the first value (no warm-up mask)."""
    return pd.Series(values).ewm(span=length, adjust=False).mean().to_numpy()
