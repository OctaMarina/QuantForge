"""Single-asset bar features (families ``V`` volatility, ``T`` trend, ``F`` flow/activity, ``CAL`` calendar).

Input: one asset's bar frame indexed by bar-end timestamps (tz-aware), with ``open``, ``high``, ``low``,
``close``, ``volume``, ``quote_volume``, ``n_trades`` and ``taker_buy_base`` (the output of
:func:`quant_forge.data.klines_to_bars`).  Every value at bar ``t`` uses bars up to and including ``t``.

Horizons are given in days and converted with ``bars_per_day`` (6 for 4-hour bars).  Column names carry the
family prefix (``v_``, ``t_``, ``f_``, ``cal_``); columns starting with ``_`` are auxiliary series needed by the
panel-level features (:mod:`quant_forge.features.market`, :mod:`quant_forge.features.fracdiff`) and are not
model inputs.

=====================================  ======================================================================
``v_rvol_{1,3,7,30,90}d``              realised volatility per bar over the horizon (sqrt of mean r^2)
``v_rvol_ratio_*``                     log ratios of short to long realised volatility
``v_rvol_pct_1y``                      1-year rolling percentile of the 7-day volatility
``v_bb_width``, ``v_atr_rel``          Bollinger width (4 sd / mean, 20 bars), ATR(14)/close
``v_range_rel``, ``v_range_rel_6``     (high - low)/close, its 6-bar mean
``v_parkinson_7d``, ``v_gk_7d``        Parkinson and Garman-Klass volatility (7 days)
``v_vol_of_vol_30d``                   sd of the 1-day volatility over 30 days / 30-day volatility
``v_rvol_30d``                         30-day realised volatility (not per bar): the inverse-volatility input
``t_ret_{d}d`` / ``t_ret_{d}d_z``      log return over d days, raw and scaled by the 30-day volatility
``t_ema{n}_dist_atr``                  (close - EMA_n) / ATR
``t_sma{200,50}d_dist``                log distance to the 200/50-day SMA
``t_dd_{d}d``, ``t_up_{d}d``           log distance to the rolling high / low (drawdown, run-up)
``t_eff_{n}``                          efficiency ratio (net move / path length)
``t_rsi{7,14,28}``                     RSI / 100
``t_pos_range_{30,90}d``               position of the close in the rolling range
``t_bars_since_high_90d``              bars since the 90-day high / window length
``t_skew_30d``, ``t_kurt_30d``,        moments and lag-1 autocorrelation of bar returns
``t_acf1_30d``, ``t_green_share_7d``   share of up bars
``f_taker_imb_{1,6,18,42}``            taker buy imbalance 2*buy/volume - 1 and its means
``f_qvol_rel_{7,30,90}d``              log quote volume minus its rolling mean; ``f_qvol_trend_7d_30d``
``f_ntrades_rel_30d``, ``f_vol_z_20``  relative trade count, volume z-score
``f_amihud_7d``, ``f_corr_ret_vol_20`` Amihud illiquidity (log), return / volume-change correlation
``cal_*``                              hour of day and day of week (sin/cos), weekend flag
=====================================  ======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .kernels import bars_since_rolling_max, rsi, wilder_mean

__all__ = ["asset_features", "volatility_features", "trend_features", "flow_features", "calendar_features",
           "AUX_COLUMNS"]

#: Auxiliary columns added by :func:`asset_features` (inputs of the panel features, not model inputs).
AUX_COLUMNS = ("_r", "_above_sma50d", "_above_sma200d", "_log_close", "_log_open", "_log_volume")


class _Base:
    """Shared intermediates of one asset (log returns, ATR, realised volatilities)."""

    def __init__(self, g: pd.DataFrame, bpd: int):
        self.g, self.bpd = g, bpd
        self.o, self.h, self.l, self.c = g["open"], g["high"], g["low"], g["close"]
        self.r = np.log(self.c).diff()
        tr = pd.concat([self.h - self.l, (self.h - self.c.shift()).abs(), (self.l - self.c.shift()).abs()],
                       axis=1).max(axis=1)
        self.atr = pd.Series(wilder_mean(tr.to_numpy(), 14), index=g.index)
        self.rvd = {d: self._rv(d * bpd) for d in (1, 3, 7, 30, 90)}
        self.sig = self.rvd[30] / np.sqrt(30 * bpd)

    def _rv(self, n: int) -> pd.Series:
        return np.sqrt((self.r ** 2).rolling(n, min_periods=int(0.8 * n)).sum())


def _volatility(b: _Base, f: dict) -> None:
    bpd, h, l, c, o = b.bpd, b.h, b.l, b.c, b.o
    for d in (1, 3, 7, 30, 90):
        f[f"v_rvol_{d}d"] = b.rvd[d] / np.sqrt(d * bpd)
    f["v_rvol_ratio_1d_30d"] = np.log(f["v_rvol_1d"] / b.sig)
    f["v_rvol_ratio_3d_30d"] = np.log(f["v_rvol_3d"] / b.sig)
    f["v_rvol_ratio_7d_90d"] = np.log(f["v_rvol_7d"] / f["v_rvol_90d"])
    f["v_rvol_pct_1y"] = f["v_rvol_7d"].rolling(365 * bpd, min_periods=90 * bpd).rank(pct=True)
    f["v_bb_width"] = 4 * c.rolling(20).std(ddof=0) / c.rolling(20).mean()
    f["v_atr_rel"] = b.atr / c
    f["v_range_rel"] = (h - l) / c
    f["v_range_rel_6"] = f["v_range_rel"].rolling(6).mean()
    f["v_parkinson_7d"] = np.sqrt((np.log(h / l) ** 2).rolling(7 * bpd).mean() / (4 * np.log(2)))
    f["v_gk_7d"] = np.sqrt((0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2)
                           .rolling(7 * bpd).mean().clip(lower=0))
    f["v_vol_of_vol_30d"] = f["v_rvol_1d"].rolling(30 * bpd).std() / b.sig
    f["v_rvol_30d"] = b.rvd[30]


def _trend(b: _Base, f: dict) -> None:
    bpd, h, l, c, o, r, atr, sig = b.bpd, b.h, b.l, b.c, b.o, b.r, b.atr, b.sig
    for d in (1, 3, 7, 14, 30, 60, 90):
        k = d * bpd
        f[f"t_ret_{d}d"] = np.log(c / c.shift(k))
        f[f"t_ret_{d}d_z"] = f[f"t_ret_{d}d"] / (sig * np.sqrt(k))
    for n in (20, 50, 100, 200):
        ema = c.ewm(span=n, adjust=False, min_periods=n).mean()
        f[f"t_ema{n}_dist_atr"] = (c - ema) / atr
    f["t_sma200d_dist"] = np.log(c / c.rolling(200 * bpd, min_periods=150 * bpd).mean())
    f["t_sma50d_dist"] = np.log(c / c.rolling(50 * bpd, min_periods=40 * bpd).mean())
    for d in (30, 90, 180):
        f[f"t_dd_{d}d"] = np.log(c / h.rolling(d * bpd, min_periods=d * bpd // 2).max())
        f[f"t_up_{d}d"] = np.log(c / l.rolling(d * bpd, min_periods=d * bpd // 2).min())
    for n in (10, 20, 40):
        f[f"t_eff_{n}"] = c.diff(n) / c.diff().abs().rolling(n).sum()
    for n in (7, 14, 28):
        f[f"t_rsi{n}"] = rsi(c, n) / 100
    for d in (30, 90):
        hi, lo = h.rolling(d * bpd).max(), l.rolling(d * bpd).min()
        f[f"t_pos_range_{d}d"] = (c - lo) / (hi - lo)
    f["t_bars_since_high_90d"] = bars_since_rolling_max(h, 90 * bpd) / (90 * bpd)
    f["t_skew_30d"] = r.rolling(30 * bpd).skew()
    f["t_kurt_30d"] = r.rolling(30 * bpd).kurt()
    f["t_acf1_30d"] = r.rolling(30 * bpd).corr(r.shift())
    f["t_green_share_7d"] = (c > o).astype(float).rolling(7 * bpd).mean()


def _flow(b: _Base, f: dict) -> None:
    g, bpd, r = b.g, b.bpd, b.r
    imb = 2 * g["taker_buy_base"] / g["volume"].where(g["volume"] > 0) - 1
    for n in (1, 6, 18, 42):
        f[f"f_taker_imb_{n}"] = imb.rolling(n).mean() if n > 1 else imb
    lq = np.log(g["quote_volume"].where(g["quote_volume"] > 0))
    for d in (7, 30, 90):
        f[f"f_qvol_rel_{d}d"] = lq - lq.rolling(d * bpd, min_periods=int(0.7 * d * bpd)).mean()
    ln = np.log(g["n_trades"].where(g["n_trades"] > 0))
    f["f_ntrades_rel_30d"] = ln - ln.rolling(30 * bpd, min_periods=20 * bpd).mean()
    f["f_vol_z_20"] = (g["volume"] - g["volume"].rolling(20).mean()) / g["volume"].rolling(20).std()
    f["f_amihud_7d"] = np.log((r.abs() / g["quote_volume"].where(g["quote_volume"] > 0)).rolling(7 * bpd).mean())
    f["f_corr_ret_vol_20"] = r.rolling(20).corr(np.log(g["volume"].where(g["volume"] > 0)).diff())
    f["f_qvol_trend_7d_30d"] = f["f_qvol_rel_7d"] - f["f_qvol_rel_30d"]


def _calendar(index: pd.DatetimeIndex, f: dict) -> None:
    hr, dow = index.hour, index.dayofweek
    f["cal_hour_sin"], f["cal_hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    f["cal_dow_sin"], f["cal_dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    f["cal_weekend"] = (dow >= 5).astype(float)


def _aux(b: _Base, f: dict) -> None:
    c, bpd = b.c, b.bpd
    f["_r"] = b.r
    for n, name, mp in [(50 * bpd, "50d", 40 * bpd), (200 * bpd, "200d", 150 * bpd)]:
        sma = c.rolling(n, min_periods=mp).mean()
        f[f"_above_sma{name}"] = (c > sma).astype(float).where(sma.notna())
    f["_log_close"], f["_log_open"] = np.log(c), np.log(b.o)
    f["_log_volume"] = np.log(b.g["volume"].where(b.g["volume"] > 0))


def _frame(f: dict, index) -> pd.DataFrame:
    return pd.DataFrame({k: (v if isinstance(v, pd.Series) else pd.Series(np.asarray(v), index=index))
                         for k, v in f.items()}, index=index)


def asset_features(bars: pd.DataFrame, bars_per_day: int = 6, *, families=("V", "T", "F", "CAL"),
                   aux: bool = True) -> pd.DataFrame:
    """All single-asset features of one bar frame, in a fixed column order (V, T, F, CAL, then auxiliaries)."""
    b = _Base(bars, bars_per_day)
    f: dict = {}
    for fam, fn in (("V", _volatility), ("T", _trend), ("F", _flow)):
        if fam in families:
            fn(b, f)
    if "CAL" in families:
        _calendar(bars.index, f)
    if aux:
        _aux(b, f)
    return _frame(f, bars.index)


def volatility_features(bars: pd.DataFrame, bars_per_day: int = 6) -> pd.DataFrame:
    """Family ``V`` only."""
    return asset_features(bars, bars_per_day, families=("V",), aux=False)


def trend_features(bars: pd.DataFrame, bars_per_day: int = 6) -> pd.DataFrame:
    """Family ``T`` only."""
    return asset_features(bars, bars_per_day, families=("T",), aux=False)


def flow_features(bars: pd.DataFrame, bars_per_day: int = 6) -> pd.DataFrame:
    """Family ``F`` only."""
    return asset_features(bars, bars_per_day, families=("F",), aux=False)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Family ``CAL`` from bar-end timestamps."""
    f: dict = {}
    _calendar(index, f)
    return _frame(f, index)
