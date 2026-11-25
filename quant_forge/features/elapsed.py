"""Bar features measured in elapsed (calendar) time, for bars of any type and frequency, with panel features for
asynchronous bars.

The families of :mod:`quant_forge.features.asset` count horizons in bars (``days * bars_per_day``), which is exact
only for time bars.  Volume and dollar bars last as long as the market needs to trade the threshold, so a fixed
number of bars covers a varying span of time, and comparisons across sampling frequencies need the same horizon
for every bar series.  Here every horizon is elapsed time:

* a "d-day" window holds the bars whose **end** lies in ``(t - d days, t]`` (``t`` = the bar's end, the decision
  time of the strategies); sums over such windows (squared returns, volume, trades) are independent of how often
  the series is sampled, up to microstructure;
* a return over ``d`` days uses the close of the last bar that ended at or before ``t - d``;
* moving averages are duration-weighted (each close weighs the time since the previous bar end, capped at a day),
  i.e. time averages of the price path, which equal the plain SMA for time bars;
* calendar features use the timestamp of the bar end (hour with minutes, day of week);
* a feature needs ``min_history`` x its horizon of history since the asset's first bar, and windows without any
  bar are NaN.

Panel features (:func:`asynchronous_panel_features`).  Bars of different assets end at different times.  At the
decision time ``t`` of a bar of asset ``a``, each other asset enters with its **latest bar that ended at or before
t** (as-of alignment, never a later bar) and only if that bar is at most ``max_staleness`` old (inactive, halted or
not yet listed assets drop out).  With synchronous time bars this is the usual cross-section at ``t``.

Families and prefixes are those of the bar-count features (``v_`` V, ``t_`` T, ``f_`` F, ``cal_`` CAL, ``m_``/``b_``
M, ``x_`` R, ``xr_`` X), so :func:`quant_forge.features.feature_families`, ``orient_by_side`` and the meta-labeling
pipeline work unchanged.

=====================================  ======================================================================
``v_vol_{1,7,90}d``                    realised volatility per sqrt(day): sqrt(sum r^2 over the window / d)
``v_rvol_30d``                         sqrt(sum r^2 over 30 days) (the inverse-volatility sizing input)
``v_vol_ratio_1d_30d``, ``_7d_90d``    log ratios of daily-scale volatilities
``v_range_1d``                         log(max high / min low) over one day
``t_ret_{1,7,30,90}d`` (``_z``)        log return over d days (scaled by the 30-day daily volatility x sqrt(d))
``t_dd_{30,90}d``, ``t_up_30d``        log distance to the rolling high / low; ``t_pos_range_30d`` position in range
``t_sma{50,200}d_dist``                log(close / duration-weighted mean close over 50 / 200 days)
``t_eff_7d``                           7-day return / sum of absolute bar returns (signed efficiency)
``f_taker_imb_{1,7}d``                 2 * taker-buy volume / volume - 1 over the window
``f_qvol_rel_1d_30d``, ``_7d_30d``,    log of the quote volume per day of a short window over that of a long one
``f_qvol_rel_30d_365d``
``f_ntrades_rel_1d_30d``               the same for the number of trades
``f_amihud_7d``                        log(sum |r| / quote volume) over 7 days
``cal_*``                              hour of day (with minutes) and day of week (sin/cos), weekend flag
``m_*``, ``b_*``                       market state across assets as of t; BTC's own state as of t
``x_rs_{7,30}d``, ``x_rank_ret_30d``   relative strength vs the cross-sectional median, percentile rank
``xr_*``                               percentile ranks of own volatility, activity and 7-day momentum
=====================================  ======================================================================
"""

from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["elapsed_asset_features", "asynchronous_panel_features", "elapsed_bar_features", "ELAPSED_AUX_COLUMNS"]

DAY_NS = 86_400 * 10 ** 9
#: Auxiliary columns (inputs of the panel features, not model inputs) and the raw close for relative levels.
ELAPSED_AUX_COLUMNS = ("_above_sma50d", "_above_sma200d")


def _times_ns(index: pd.DatetimeIndex) -> np.ndarray:
    idx = index if index.tz is not None else index.tz_localize("UTC")
    return idx.tz_convert("UTC").as_unit("ns").asi8


def elapsed_asset_features(bars: pd.DataFrame, *, min_history: float = 0.8, duration_cap_days: float = 1.0) -> pd.DataFrame:
    """Families V, T, F and CAL of one asset in elapsed time (see the module docstring), plus ``raw_close`` and
    the auxiliaries of :data:`ELAPSED_AUX_COLUMNS`.  ``bars`` is a bar frame indexed by strictly increasing bar
    ends (time or event bars)."""
    idx = bars.index
    t = _times_ns(idx)
    n = len(t)
    c = bars["close"].to_numpy(float)
    hi = bars["high"].to_numpy(float)
    lo = bars["low"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.r_[np.nan, np.diff(np.log(c))]
    r2 = np.nan_to_num(lr ** 2)
    ar = np.nan_to_num(np.abs(lr))
    start_ns = (int(bars["bar_start_us"].iloc[0]) * 1000) if "bar_start_us" in bars.columns and n else (t[0] if n else 0)
    age = t - start_ns
    lo_cache: dict = {}

    def lo_idx(days: float) -> np.ndarray:
        if days not in lo_cache:
            lo_cache[days] = np.searchsorted(t, t - int(days * DAY_NS), side="right")
        return lo_cache[days]

    def wsum(x, days: float) -> np.ndarray:
        cs = np.r_[0.0, np.cumsum(np.nan_to_num(np.asarray(x, dtype=float)))]
        return cs[1:] - cs[lo_idx(days)]

    def ok(days: float) -> np.ndarray:
        cnt = np.arange(1, n + 1) - lo_idx(days)
        return (age >= min_history * days * DAY_NS) & (cnt > 0)

    def mask(x, days):
        return np.where(ok(days), x, np.nan)

    f: dict = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        vol = {d: mask(np.sqrt(wsum(r2, d) / d), d) for d in (1, 7, 30, 90)}
        vol = {d: np.where(v > 0, v, np.nan) for d, v in vol.items()}
        f["v_vol_1d"], f["v_vol_7d"], f["v_vol_90d"] = vol[1], vol[7], vol[90]
        f["v_rvol_30d"] = vol[30] * np.sqrt(30)
        f["v_vol_ratio_1d_30d"] = np.log(vol[1] / vol[30])
        f["v_vol_ratio_7d_90d"] = np.log(vol[7] / vol[90])
        hs, ls = pd.Series(hi, index=idx), pd.Series(lo, index=idx)
        f["v_range_1d"] = mask(np.log(hs.rolling("1D").max().to_numpy() / ls.rolling("1D").min().to_numpy()), 1)

        for d in (1, 7, 30, 90):
            p = np.searchsorted(t, t - int(d * DAY_NS), side="right") - 1
            valid = (p >= 0) & (age >= min_history * d * DAY_NS)
            ret = np.where(valid, np.log(c / c[np.maximum(p, 0)]), np.nan)
            f[f"t_ret_{d}d"] = ret
            f[f"t_ret_{d}d_z"] = ret / (vol[30] * np.sqrt(d))
        for d in (30, 90):
            H = hs.rolling(f"{d}D").max().to_numpy()
            L = ls.rolling(f"{d}D").min().to_numpy()
            f[f"t_dd_{d}d"] = mask(np.log(c / H), d)
            if d == 30:
                f["t_up_30d"] = mask(np.log(c / L), d)
                f["t_pos_range_30d"] = mask(np.where(H > L, (c - L) / (H - L), np.nan), d)
        w = np.r_[0.0, np.diff(t) / DAY_NS].clip(0, duration_cap_days)
        for d in (50, 200):
            twap = wsum(c * w, d) / wsum(w, d)
            f[f"t_sma{d}d_dist"] = mask(np.log(c / twap), d)
        f["t_eff_7d"] = f["t_ret_7d"] / np.where(wsum(ar, 7) > 0, wsum(ar, 7), np.nan)

        vb = bars["volume"].to_numpy(float)
        tb = bars["taker_buy_base"].to_numpy(float)
        qv = bars["quote_volume"].to_numpy(float)
        nt = bars["n_trades"].to_numpy(float)
        for d in (1, 7):
            v = wsum(vb, d)
            f[f"f_taker_imb_{d}d"] = mask(np.where(v > 0, 2 * wsum(tb, d) / v - 1, np.nan), d)
        q1, q7, q30, q365 = (wsum(qv, d) for d in (1, 7, 30, 365))
        pos = lambda x: np.where(x > 0, x, np.nan)   # noqa: E731
        f["f_qvol_rel_1d_30d"] = mask(np.log(pos(q1) / pos(q30 / 30)), 30)
        f["f_qvol_rel_7d_30d"] = mask(np.log(pos(q7 / 7) / pos(q30 / 30)), 30)
        f["f_qvol_rel_30d_365d"] = mask(np.log(pos(q30 / 30) / pos(q365 / 365)), 365)
        f["f_ntrades_rel_1d_30d"] = mask(np.log(pos(wsum(nt, 1)) / pos(wsum(nt, 30) / 30)), 30)
        f["f_amihud_7d"] = mask(np.log(pos(wsum(ar, 7)) / pos(q7)), 7)

    utc = idx.tz_convert("UTC") if idx.tz is not None else idx
    hr = np.asarray(utc.hour + utc.minute / 60.0)
    dow = np.asarray(utc.dayofweek)
    f["cal_hour_sin"], f["cal_hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    f["cal_dow_sin"], f["cal_dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    f["cal_weekend"] = (dow >= 5).astype(float)
    for d in (50, 200):
        s = f[f"t_sma{d}d_dist"]
        f[f"_above_sma{d}d"] = np.where(np.isnan(s), np.nan, (s > 0).astype(float))
    f["raw_close"] = c
    return pd.DataFrame(f, index=idx)


_PANEL_SOURCES = ("t_ret_1d", "t_ret_7d", "t_ret_30d", "t_ret_7d_z", "t_ret_30d_z", "v_vol_7d", "v_rvol_30d",
                  "v_vol_ratio_1d_30d", "t_dd_90d", "f_qvol_rel_1d_30d", "_above_sma50d", "_above_sma200d")


def _pct_rank(M: np.ndarray, own: np.ndarray) -> np.ndarray:
    """Percentile rank (average ties, like ``rank(pct=True)``) of ``own`` among the non-NaN values of each row."""
    valid = ~np.isnan(M)
    N = valid.sum(1)
    less = (M < own[:, None]).sum(1)
    eq = (M == own[:, None]).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (less + (eq + 1) / 2) / N
    return np.where(np.isnan(own) | (N == 0), np.nan, out)


def asynchronous_panel_features(feats: Mapping[str, pd.DataFrame], *, btc: str = "BTC",
                                max_staleness: str | pd.Timedelta = "3D") -> dict[str, pd.DataFrame]:
    """Market (``m_``, ``b_``), relative-strength (``x_``) and rank (``xr_``) features of every asset at its own bar
    ends, from the other assets' latest bars at or before each decision time (as-of alignment).

    ``feats`` maps asset -> output of :func:`elapsed_asset_features` (needs the columns of ``_PANEL_SOURCES``).
    Returns ``{asset: frame}`` with the panel columns only, indexed like ``feats[asset]``.
    """
    assets = list(feats)
    stale = int(pd.Timedelta(max_staleness).value)
    times = {a: _times_ns(feats[a].index) for a in assets}
    src = {a: {c: feats[a][c].to_numpy(float) for c in _PANEL_SOURCES} for a in assets}
    j_btc = assets.index(btc) if btc in assets else None
    out = {}
    for a in assets:
        T = times[a]
        n, A = len(T), len(assets)
        pos = np.zeros((n, A), dtype=np.int64)
        ok = np.zeros((n, A), dtype=bool)
        for j, b in enumerate(assets):
            tb_ = times[b]
            if not len(tb_):
                continue
            p = np.searchsorted(tb_, T, side="right") - 1
            v = p >= 0
            pp = np.maximum(p, 0)
            v &= (T - tb_[pp]) <= stale
            pos[:, j], ok[:, j] = pp, v

        def gather(col):
            M = np.full((n, A), np.nan)
            for j, b in enumerate(assets):
                if len(times[b]):
                    M[:, j] = np.where(ok[:, j], src[b][col][pos[:, j]], np.nan)
            return M

        f: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            R1, R7, R30 = gather("t_ret_1d"), gather("t_ret_7d"), gather("t_ret_30d")
            Z7 = gather("t_ret_7d_z")
            V30, V7 = gather("v_rvol_30d"), gather("v_vol_7d")
            Q1 = gather("f_qvol_rel_1d_30d")
            f["m_n_assets"] = ok.sum(1).astype(float)
            f["m_ret_1d"], f["m_ret_7d"], f["m_ret_30d"] = (np.nanmean(R, axis=1) for R in (R1, R7, R30))
            f["m_vol_7d"] = np.nanmedian(V7, axis=1)
            f["m_rvol_30d"] = np.nanmedian(V30, axis=1)
            f["m_breadth_sma50d"] = np.nanmean(gather("_above_sma50d"), axis=1)
            f["m_breadth_sma200d"] = np.nanmean(gather("_above_sma200d"), axis=1)
            f["m_dispersion_7d"] = np.nanstd(R7, axis=1)
            if j_btc is not None:
                for name, col in (("b_ret_7d_z", "t_ret_7d_z"), ("b_ret_30d_z", "t_ret_30d_z"),
                                  ("b_vol_ratio_1d_30d", "v_vol_ratio_1d_30d"), ("b_dd_90d", "t_dd_90d")):
                    bj = np.where(ok[:, j_btc], src[btc][col][pos[:, j_btc]], np.nan)
                    f[name] = bj
                f["m_alt_vs_btc_30d"] = f["m_ret_30d"] - R30[:, j_btc]
            own = src[a]
            f["x_rs_7d"] = own["t_ret_7d"] - np.nanmedian(R7, axis=1)
            f["x_rs_30d"] = own["t_ret_30d"] - np.nanmedian(R30, axis=1)
            f["x_rank_ret_30d"] = _pct_rank(R30, own["t_ret_30d"])
            f["xr_vol_30d"] = _pct_rank(V30, own["v_rvol_30d"])
            f["xr_qvol_rel_1d_30d"] = _pct_rank(Q1, own["f_qvol_rel_1d_30d"])
            f["xr_ret_7d_z"] = _pct_rank(Z7, own["t_ret_7d_z"])
        out[a] = pd.DataFrame(f, index=feats[a].index)
    return out


def elapsed_bar_features(bars: Mapping[str, pd.DataFrame], *, btc: str = "BTC", max_staleness="3D",
                         min_history: float = 0.8, n_jobs: int = 1, assets: Sequence[str] | None = None) -> dict[str, pd.DataFrame]:
    """``{asset: own families + panel families}`` for bars of any type (the elapsed-time counterpart of
    :func:`quant_forge.features.bar_features`).  ``n_jobs > 1`` computes the per-asset part in processes."""
    names = list(assets) if assets is not None else list(bars)
    if n_jobs == 1:
        own = {a: elapsed_asset_features(bars[a], min_history=min_history) for a in names}
    else:
        from joblib import Parallel, delayed

        frames = Parallel(n_jobs=n_jobs)(delayed(elapsed_asset_features)(bars[a], min_history=min_history) for a in names)
        own = dict(zip(names, frames))
    panel = asynchronous_panel_features(own, btc=btc, max_staleness=max_staleness)
    return {a: pd.concat([own[a], panel[a]], axis=1) for a in names}
