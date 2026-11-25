"""Numerical kernels of the feature library (causal: the value at row ``i`` uses rows ``<= i`` only).

These are the feature-engineering versions of the smoothers.  They deliberately differ from the
strategy-library kernels in :mod:`quant_forge.strategies.indicators`, which must reproduce TradingView rules:

* :func:`wilder_mean` starts at the first finite value and *carries the last value* through later NaNs
  (the strategy kernel ``wilder_rma`` restarts after a NaN instead);
* :func:`rsi` works on a pandas Series and returns 50 for a flat market.

The Wilder recursion runs in numba when it is installed (identical arithmetic, about 100x faster) and in
plain Python otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

__all__ = ["wilder_mean", "rsi", "bars_since_rolling_max", "HAVE_NUMBA"]

try:  # optional accelerator
    from numba import njit as _njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - depends on the environment
    _njit = None
    HAVE_NUMBA = False


def _wilder_recursion_py(a: np.ndarray, out: np.ndarray, k: int, n: int) -> None:
    for j in range(k + 1, len(a)):
        out[j] = out[j - 1] + (a[j] - out[j - 1]) / n if np.isfinite(a[j]) else out[j - 1]


_wilder_recursion_nb = _njit(cache=True)(_wilder_recursion_py) if HAVE_NUMBA else None


def wilder_mean(values, n: int, *, use_numba: bool | None = None) -> np.ndarray:
    """Wilder's smoothing ``m_j = m_{j-1} + (x_j - m_{j-1}) / n``.

    Seeded with the arithmetic mean of the first ``n`` values starting at the first finite one; NaN before the
    seed.  A non-finite input after the seed leaves the previous value unchanged.
    """
    a = np.asarray(values, dtype=float)
    out = np.full(len(a), np.nan)
    fin = np.flatnonzero(np.isfinite(a))
    if len(fin) and len(a) >= fin[0] + n:
        k = fin[0] + n - 1
        out[k] = a[fin[0]:k + 1].mean()
        fast = HAVE_NUMBA if use_numba is None else (use_numba and HAVE_NUMBA)
        (_wilder_recursion_nb if fast else _wilder_recursion_py)(a, out, k, n)
    return out


def rsi(close: pd.Series, n: int) -> pd.Series:
    """Relative strength index (0-100) with Wilder smoothing of gains and losses; 50 when both are zero."""
    ch = close.diff()
    gain = pd.Series(wilder_mean(ch.clip(lower=0).to_numpy(), n), index=close.index)
    loss = pd.Series(wilder_mean((-ch).clip(lower=0).to_numpy(), n), index=close.index)
    return (100 * gain / (gain + loss)).mask((gain == 0) & (loss == 0), 50.0)


def bars_since_rolling_max(x: pd.Series, window: int) -> pd.Series:
    """Number of bars since the maximum of the trailing ``window`` bars (0 = the current bar is the maximum).

    Equal to ``x.rolling(window).apply(lambda v: len(v) - 1 - np.argmax(v), raw=True)`` (first occurrence of the
    maximum; NaN when the window is not full of observations) but vectorised.
    """
    a = x.to_numpy(dtype=float)
    out = np.full(len(a), np.nan)
    if window >= 1 and len(a) >= window:
        view = sliding_window_view(a, window)
        arg = np.argmax(view, axis=1)
        nan_count = np.convolve(np.isnan(a).astype(np.int64), np.ones(window, dtype=np.int64), mode="valid")
        out[window - 1:] = np.where(nan_count == 0, window - 1 - arg, np.nan)
    return pd.Series(out, index=x.index)
