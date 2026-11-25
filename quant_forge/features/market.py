"""Panel features: market state (``M``), relative strength (``R``) and cross-sectional ranks (``X``).

All three are computed from the per-asset frames of :func:`quant_forge.features.asset_features` and are
point-in-time by construction: at time ``t`` only the assets that have a bar at ``t`` (and their history)
enter.

* :func:`market_state_features` -> one frame on the union of the bar times (``m_*`` and ``b_*`` for BTC):
  equal-weighted index returns and volatility, its 1-year volatility percentile, drawdown, breadth above the
  50/200-day SMA, cross-sectional dispersion, alt-vs-BTC return, BTC's own trend and volatility state and the
  average pairwise correlation of the assets over 30 days.
* :func:`cross_sectional_features` -> per asset: returns relative to the cross-sectional median
  (``x_rs_*``), correlation and beta with the index (``x_corr_mkt_30d``, ``x_beta_30d``), percentile rank of
  the 30-day return (``x_rank_ret_30d``) and percentile ranks of selected own features (``xr_*``).
* :func:`panel_features` -> the per-asset frames with R, X and M appended (the usual entry point).

The order of the assets in the input mapping fixes the column order of the internal panels; it changes
results only at floating-point rounding level.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["market_state_features", "cross_sectional_features", "panel_features", "DEFAULT_RANK_COLUMNS"]

#: Own features whose cross-sectional percentile rank is added as ``xr_<name without family prefix>``.
DEFAULT_RANK_COLUMNS = (
    "v_rvol_30d", "v_rvol_7d", "v_atr_rel", "v_bb_width", "v_rvol_ratio_1d_30d", "v_rvol_ratio_7d_90d", "v_range_rel",
    "t_ret_1d_z", "t_ret_7d_z", "t_ret_30d_z", "t_ret_90d_z", "t_rsi14", "t_ema50_dist_atr", "t_up_90d", "t_dd_90d",
    "t_dd_30d", "t_eff_20", "f_qvol_rel_30d", "f_qvol_rel_7d", "f_taker_imb_6", "f_taker_imb_18")


def _panel(feats: Mapping[str, pd.DataFrame], column: str, index=None) -> pd.DataFrame:
    p = pd.DataFrame({a: f[column] for a, f in feats.items()})
    return p.sort_index() if index is None else p.reindex(index)


def _average_pairwise_corr(R: pd.DataFrame, window: int, step: int, min_coverage: float) -> pd.Series:
    """Mean off-diagonal correlation of the trailing ``window`` rows, every ``step`` rows (forward-filled)."""
    values = R.to_numpy()
    out, pts = [], R.index[window::step]
    for pos in range(window, len(R), step):
        w = values[max(0, pos + 1 - window):pos + 1]
        w = w[:, (~np.isnan(w)).mean(axis=0) > min_coverage]
        cc = pd.DataFrame(w).corr().to_numpy()
        n = cc.shape[0]
        out.append((cc.sum() - np.trace(cc)) / (n * (n - 1)) if n > 1 else np.nan)
    return pd.Series(out, index=pts).reindex(R.index).ffill()


def market_state_features(feats: Mapping[str, pd.DataFrame], bars_per_day: int = 6, *, btc: str = "BTC",
                          corr_window_days: int = 30, corr_step: int | None = None,
                          corr_min_coverage: float = 0.9) -> pd.DataFrame:
    """Market-state features (family ``M``) on the union of the bar times of all assets.

    Needs the auxiliary columns ``_r``, ``_above_sma50d``, ``_above_sma200d`` and ``t_ret_{7,30,90}d`` of every
    asset, and ``t_ret_30d``, ``t_ret_{7,30}d_z``, ``v_rvol_ratio_1d_30d``, ``t_dd_90d`` of ``btc``.
    The average correlation is computed every ``corr_step`` bars (default: one day) and forward-filled.
    """
    bpd = bars_per_day
    R = _panel(feats, "_r")
    idx = R.index
    mr = R.mean(axis=1).fillna(0.0)
    cm = mr.cumsum()

    def mrv(n):
        return np.sqrt((mr ** 2).rolling(n, min_periods=int(0.8 * n)).sum())

    msig = mrv(30 * bpd) / np.sqrt(30 * bpd)
    m: dict = {"m_n_assets": R.notna().sum(axis=1).astype(float)}
    for d in (1, 7, 30, 90, 365):
        k = d * bpd
        m[f"m_ret_{d}d"] = cm - cm.shift(k)
        m[f"m_ret_{d}d_z"] = m[f"m_ret_{d}d"] / (msig * np.sqrt(k))
    m["m_rvol_7d"] = mrv(7 * bpd) / np.sqrt(7 * bpd)
    m["m_rvol_30d"] = msig
    m["m_rvol_ratio_7d_90d"] = np.log((mrv(7 * bpd) / np.sqrt(7 * bpd)) / (mrv(90 * bpd) / np.sqrt(90 * bpd)))
    m["m_rvol_pct_1y"] = m["m_rvol_7d"].rolling(365 * bpd, min_periods=90 * bpd).rank(pct=True)
    m["m_dd_180d"] = cm - cm.rolling(180 * bpd, min_periods=90 * bpd).max()
    for name in ["50d", "200d"]:
        m[f"m_breadth_sma{name}"] = _panel(feats, f"_above_sma{name}", idx).mean(axis=1)
    m["m_dispersion_7d"] = _panel(feats, "t_ret_7d", idx).std(axis=1)
    m["m_dispersion_30d"] = _panel(feats, "t_ret_30d", idx).std(axis=1)
    b = feats[btc]
    m["m_alt_vs_btc_30d"] = m["m_ret_30d"] - b["t_ret_30d"].reindex(idx)
    m["b_ret_7d_z"] = b["t_ret_7d_z"].reindex(idx)
    m["b_ret_30d_z"] = b["t_ret_30d_z"].reindex(idx)
    m["b_rvol_ratio_1d_30d"] = b["v_rvol_ratio_1d_30d"].reindex(idx)
    m["b_dd_90d"] = b["t_dd_90d"].reindex(idx)
    m["m_avg_corr_30d"] = _average_pairwise_corr(R, corr_window_days * bpd, corr_step or bpd, corr_min_coverage)
    return pd.DataFrame(m, index=idx)


def cross_sectional_features(feats: Mapping[str, pd.DataFrame], bars_per_day: int = 6, *,
                             rank_columns: Sequence[str] = DEFAULT_RANK_COLUMNS) -> dict[str, pd.DataFrame]:
    """Relative-strength (``R``: ``x_*``) and cross-sectional rank (``X``: ``xr_*``) features of every asset.

    Ranks are percentile ranks among the assets with a value at the same time (ties averaged).
    """
    bpd = bars_per_day
    R = _panel(feats, "_r")
    idx = R.index
    mr = R.mean(axis=1).fillna(0.0)
    med = {d: _panel(feats, f"t_ret_{d}d", idx).median(axis=1) for d in (7, 30, 90)}
    rank_ret30 = _panel(feats, "t_ret_30d", idx).rank(axis=1, pct=True)
    ranks = {c: _panel(feats, c, idx).rank(axis=1, pct=True) for c in rank_columns}
    w, mp = 30 * bpd, 20 * bpd
    out = {}
    for a, f in feats.items():
        x: dict = {}
        for d in (7, 30, 90):
            x[f"x_rs_{d}d"] = f[f"t_ret_{d}d"] - med[d].reindex(f.index)
        rr, mm = R[a].reindex(f.index), mr.reindex(f.index)
        x["x_corr_mkt_30d"] = rr.rolling(w, min_periods=mp).corr(mm)
        x["x_beta_30d"] = rr.rolling(w, min_periods=mp).cov(mm) / mm.rolling(w, min_periods=mp).var()
        x["x_rank_ret_30d"] = rank_ret30[a].reindex(f.index)
        for c in rank_columns:
            x[f"xr_{c[2:]}"] = ranks[c][a].reindex(f.index)
        out[a] = pd.DataFrame(x, index=f.index)
    return out


def panel_features(feats: Mapping[str, pd.DataFrame], bars_per_day: int = 6, *, btc: str = "BTC",
                   rank_columns: Sequence[str] = DEFAULT_RANK_COLUMNS, return_market: bool = False):
    """Per-asset frames extended with R, X and M columns (in this order).

    Returns ``{asset: frame}``, or ``(frames, market)`` with ``return_market=True``.
    """
    market = market_state_features(feats, bars_per_day, btc=btc)
    cross = cross_sectional_features(feats, bars_per_day, rank_columns=rank_columns)
    out = {a: pd.concat([f, cross[a], market.reindex(f.index)], axis=1) for a, f in feats.items()}
    return (out, market) if return_market else out
