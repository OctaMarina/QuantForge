"""Fractional differentiation with a fixed-width window (FFD, AFML ch. 5) and the choice of its order.

* :func:`ffd_weights` - the FFD weights ``w_k = -w_{k-1} (d - k + 1) / k`` truncated at ``|w_k| < thresh``
  (AFML Snippet 5.3; numerically identical to ``mlfinpy.util.frac_diff.get_weights_ffd``);
* :func:`frac_diff_ffd` - the causal FFD of a 1-D array by convolution (NaN until the window is full);
* :func:`adf_statistic` - ADF statistic and 5 % critical value (constant, fixed lag; AFML Snippet 5.4);
* :func:`select_ffd_order` - per-asset minimum order that rejects a unit root on a training span, a fallback
  for assets without enough history and one pooled order for a pooled model, plus the
  stationarity / memory / window trade-off table;
* :func:`add_ffd_features` - FFD of selected auxiliary log series, raw and trailing-standardised.

``adf_statistic`` and ``select_ffd_order`` need ``statsmodels``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["ffd_weights", "frac_diff_ffd", "adf_statistic", "FFDOrderSelection", "select_ffd_order",
           "add_ffd_features", "DEFAULT_FFD_SOURCES"]

#: (auxiliary source column, output name) pairs used by :func:`add_ffd_features`.
DEFAULT_FFD_SOURCES = (("_log_close", "ffd_log_close"), ("_log_open", "ffd_log_open"),
                       ("_log_volume", "ffd_log_volume"))


def ffd_weights(d: float, thresh: float = 1e-4, max_size: int | None = None) -> np.ndarray:
    """FFD weights ordered oldest ... newest (the last weight, applied to the current value, is 1).

    max_size  cap on the number of weights (``lim`` in mlfinpy); ``None`` = no cap.
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < thresh:
            break
        weights.append(w)
        k += 1
        if max_size is not None and k - 1 == max_size - 1:
            break
    return np.array(weights[::-1])


def frac_diff_ffd(x, d: float, thresh: float = 1e-4) -> tuple[np.ndarray, int]:
    """Causal fixed-width fractional difference of ``x``; returns ``(values, window width)``.

    ``values[i] = sum_k w_k x[i - k]`` for ``i >= width - 1``, NaN before.  NaNs inside the window propagate.
    """
    x = np.asarray(x, dtype=float)
    w = ffd_weights(d, thresh, len(x) if len(x) else None)
    out = np.full(len(x), np.nan)
    if len(w) <= len(x):
        out[len(w) - 1:] = np.convolve(x, w[::-1], mode="valid")
    return out, len(w)


def adf_statistic(x, *, min_obs: int = 200, maxlag: int = 1, regression: str = "c") -> tuple[float, float]:
    """``(ADF statistic, 5 % critical value)`` of the finite values of ``x``; NaNs when fewer than ``min_obs``."""
    import warnings

    from statsmodels.tsa.stattools import adfuller

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < min_obs:
        return np.nan, np.nan
    with warnings.catch_warnings():           # statsmodels >= 0.15 announces a future result object
        warnings.simplefilter("ignore", FutureWarning)
        res = adfuller(x, maxlag=maxlag, regression=regression, autolag=None)
    return res[0], res[4]["5%"]


@dataclass
class FFDOrderSelection:
    """Result of :func:`select_ffd_order`.

    per_asset    smallest grid order whose ADF statistic is below the 5 % critical value (1.0 if none), or the
                 fallback for assets without ``min_obs`` training observations;
    fallback     median order of the assets with history;
    pooled       grid order closest to the ``pooled_quantile`` of the per-asset orders (assets with history);
    has_history  bool Series: asset had ``min_obs`` finite FFD values for some order on the training span;
    table        one row per (asset, d): ADF statistic, critical value, correlation with the input series
                 (retained memory), number of training observations;
    tradeoff     per d over the assets with history: share rejecting the unit root, median ADF, median
                 correlation, window width in bars and days, median critical value.
    """

    per_asset: dict
    fallback: float
    pooled: float
    has_history: pd.Series
    table: pd.DataFrame
    tradeoff: pd.DataFrame = field(repr=False)


def select_ffd_order(series: Mapping[str, pd.Series], *, train_end, grid: Sequence[float] = tuple(round(0.1 * i, 1) for i in range(11)),
                     thresh: float = 1e-4, min_obs: int = 200, pooled_quantile: float = 0.9,
                     bars_per_day: int = 6) -> FFDOrderSelection:
    """Choose the FFD order on the part of each series strictly before ``train_end`` (AFML §5.6).

    ``series`` maps asset -> series (usually ``_log_close``).  Nothing at or after ``train_end`` is used, so the
    choice can be applied to later data without look-ahead.
    """
    rows = []
    for a, s in series.items():
        train = s.loc[s.index < train_end].to_numpy()
        for d in grid:
            y, _ = frac_diff_ffd(train, d, thresh)
            stat, crit = adf_statistic(y, min_obs=min_obs)
            fin = np.isfinite(y)
            corr = np.corrcoef(train[fin], y[fin])[0, 1] if fin.sum() > min_obs else np.nan
            rows.append({"asset": a, "d": d, "adf": stat, "crit5": crit, "corr_with_log_price": corr,
                         "n_train_bars": int(fin.sum())})
    table = pd.DataFrame(rows)
    has_hist = table.groupby("asset", sort=False)["n_train_bars"].max() >= min_obs
    hist_assets = has_hist[has_hist].index
    per_asset = {}
    for a, g in table[table["asset"].isin(hist_assets)].groupby("asset"):
        ok = g.loc[g["adf"] < g["crit5"], "d"]
        per_asset[a] = float(ok.min()) if len(ok) else 1.0
    fallback = float(np.median(list(per_asset.values())))
    for a in series:
        per_asset.setdefault(a, fallback)
    target = np.quantile([per_asset[a] for a in hist_assets], pooled_quantile)
    pooled = float(min(grid, key=lambda d: abs(d - target)))

    sub = table[table["asset"].isin(hist_assets)]
    tradeoff = sub.groupby("d").agg(median_adf=("adf", "median"), median_corr=("corr_with_log_price", "median"))
    tradeoff.insert(0, "share_rejecting", sub.assign(rej=sub["adf"] < sub["crit5"]).groupby("d")["rej"].mean())
    tradeoff["window_bars"] = pd.Series({d: len(ffd_weights(d, thresh)) for d in grid})
    tradeoff["window_days"] = tradeoff["window_bars"] / bars_per_day
    tradeoff["crit5"] = table.groupby("d")["crit5"].median()
    return FFDOrderSelection(per_asset=per_asset, fallback=fallback, pooled=pooled, has_history=has_hist,
                             table=table, tradeoff=tradeoff)


def add_ffd_features(feats: Mapping[str, pd.DataFrame], d: float, *, thresh: float = 1e-4,
                     sources=DEFAULT_FFD_SOURCES, bars_per_day: int = 6, z_window_days: int = 365,
                     z_min_days: int = 60) -> dict[str, pd.DataFrame]:
    """Append ``<name>`` (FFD of order ``d`` of the forward-filled source) and ``<name>_z`` (z-score against the
    trailing ``z_window_days`` mean and sd, at least ``z_min_days`` of data) for every ``(source, name)``.
    """
    w, mp = z_window_days * bars_per_day, z_min_days * bars_per_day
    out = {}
    for a, f in feats.items():
        new = {}
        for src, name in sources:
            y, _ = frac_diff_ffd(f[src].ffill().to_numpy(), d, thresh)
            s = pd.Series(y, index=f.index)
            new[name] = s
            new[name + "_z"] = (s - s.rolling(w, min_periods=mp).mean()) / s.rolling(w, min_periods=mp).std()
        out[a] = pd.concat([f, pd.DataFrame(new, index=f.index)], axis=1)
    return out
