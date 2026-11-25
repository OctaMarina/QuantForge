"""Trade-level features and transformations for meta-labeling datasets.

A *trade table* has one row per closed trade of a primary strategy with at least ``strategy``, ``asset``,
``side``, ``decision_time``, ``entry_time``, ``exit_time``, ``trade_return`` and the label ``y``
(see :func:`quant_forge.metalabeling.build_trade_panel`).

* :func:`track_record_features` - family ``S``: the strategy's recent record, from trades **closed strictly
  before** the decision (win rates and mean returns over the last N closed trades, per strategy, over all
  strategies and per strategy and asset; open positions, signals in the last 24 h and trades closed in the last
  30 days, normalised by the number of assets with data);
* :func:`orient_by_side` - multiply direction-carrying features by the trade side so that "in favour of the
  trade" has the same sign for long and short trades (centred features are centred first);
* :func:`relative_levels` - turn the strategy's price-level features into ratios to the signal close;
* :func:`feature_family` / :func:`feature_families` - family of a column from its prefix.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["track_record_features", "orient_by_side", "relative_levels", "feature_family", "feature_families",
           "FAMILY_PREFIXES", "DIRECTIONAL_PREFIXES", "CENTERED_PREFIXES", "LEVEL_FEATURES", "FAMILY_ORDER"]

#: Column prefix (text before the first ``_``) -> feature family.
FAMILY_PREFIXES = {"v": "V", "t": "T", "f": "F", "cal": "CAL", "m": "M", "b": "M", "x": "R", "xr": "X", "s": "S",
                   "ffd": "FFD", "raw": "RAW", "lib": "LIB"}
FAMILY_ORDER = ("LIB", "FFD", "V", "T", "F", "CAL", "M", "R", "X", "S")

#: Features whose sign means "up" (multiplied by ``side``).
DIRECTIONAL_PREFIXES = ("t_ret", "t_ema", "t_sma", "t_eff", "t_skew", "t_acf1", "x_rs", "m_ret", "b_ret", "f_taker",
                        "m_alt", "x_beta", "lib_ema_distance", "lib_di_diff", "lib_plus_di", "lib_minus_di",
                        "lib_long_breakout_atr", "lib_short_breakout_atr", "lib_htf_momentum", "lib_st_direction",
                        "ffd_log_close", "ffd_log_open")
#: Features in [0, 1] (or ratios around 0 for ``lib_*``) that are centred before multiplying by ``side``.
CENTERED_PREFIXES = ("t_rsi", "t_pos_range", "t_green_share", "x_rank_ret_30d", "xr_ret_", "xr_rsi14", "xr_ema50",
                     "xr_eff_20", "xr_taker", "m_breadth", "lib_dc_high_rel", "lib_dc_low_rel", "lib_ema_trend_rel",
                     "lib_sma_trend_rel", "lib_st_band_rel", "lib_st_ema_rel")

#: Price-level features of the strategy library: ``"ratio"`` -> ``lib_<name>_rel``, ``"drop"`` -> removed.
LEVEL_FEATURES = {"dc_high": "ratio", "dc_low": "ratio", "atr_value": "ratio", "ema_trend": "ratio",
                  "sma_trend": "ratio", "st_band": "ratio", "st_ema": "ratio", "sl_distance": "ratio",
                  "tp_distance": "ratio", "st_vol_ma": "drop"}
_DISTANCES = ("atr_value", "sl_distance", "tp_distance")   # already distances: divided by close, not minus 1


def _ns(x: pd.Series) -> np.ndarray:
    return x.dt.tz_convert("UTC").dt.tz_localize(None).values.astype("datetime64[ns]").astype("int64")


def track_record_features(trades: pd.DataFrame, *, clip: float = 0.2, n_assets_column: str = "m_n_assets",
                          windows_strategy: Sequence[int] = (30, 100), window_all: int = 100,
                          windows_asset: Sequence[int] = (10, 30)) -> pd.DataFrame:
    """Family ``S``, point-in-time: only trades whose ``exit_time`` is strictly before the decision enter.

    A rate over the last ``n`` closed trades is NaN until ``n // 2`` trades have closed.  Returns are clipped at
    ``+-clip`` for the mean-return features.  Index = ``trades.index``.
    """
    tr = trades
    out = pd.DataFrame(index=tr.index)
    clipped = tr["trade_return"].clip(-clip, clip)
    for scope, groups in [("s", tr.groupby("strategy")), ("a", [("ALL", tr)])]:
        for _, g in groups:
            ex = g.sort_values("exit_time")
            et = _ns(ex["exit_time"])
            yc = np.r_[0, ex["y"].cumsum().to_numpy()]
            rc = np.r_[0, clipped.loc[ex.index].cumsum().to_numpy()]
            k = np.searchsorted(et, _ns(g["decision_time"]), side="left")
            for n in (windows_strategy if scope == "s" else [window_all]):
                lo = np.maximum(k - n, 0)
                cnt = k - lo
                ok = cnt >= n // 2
                out.loc[g.index, f"s_{scope}wr_{n}"] = np.where(ok, (yc[k] - yc[lo]) / np.maximum(cnt, 1), np.nan)
                if scope == "s":
                    out.loc[g.index, f"s_mr_{n}"] = np.where(ok, (rc[k] - rc[lo]) / np.maximum(cnt, 1), np.nan)
    for _, g in tr.groupby(["strategy", "asset"]):
        ex = g.sort_values("exit_time")
        et = _ns(ex["exit_time"])
        yc = np.r_[0, ex["y"].cumsum().to_numpy()]
        k = np.searchsorted(et, _ns(g["decision_time"]), side="left")
        for n in windows_asset:
            lo = np.maximum(k - n, 0)
            out.loc[g.index, f"s_wr_asset_{n}"] = np.where(k - lo >= n // 2, (yc[k] - yc[lo]) / np.maximum(k - lo, 1), np.nan)
    day = 24 * 3600 * 10 ** 9
    for _, g in tr.groupby("strategy"):
        dt = _ns(g["decision_time"])
        en, ex = np.sort(_ns(g["entry_time"])), np.sort(_ns(g["exit_time"]))
        n_assets = g[n_assets_column]
        out.loc[g.index, "s_open_positions"] = (np.searchsorted(en, dt, side="right") - np.searchsorted(ex, dt, side="right")) / n_assets
        ds = np.sort(dt)
        out.loc[g.index, "s_signals_24h"] = (np.searchsorted(ds, dt, side="right") - np.searchsorted(ds, dt - day, side="right") - 1) / n_assets
        out.loc[g.index, "s_closed_30d"] = (np.searchsorted(ex, dt, side="left") - np.searchsorted(ex, dt - 30 * day, side="left")) / n_assets
    return out


def orient_by_side(trades: pd.DataFrame, *, directional: Sequence[str] = DIRECTIONAL_PREFIXES,
                   centered: Sequence[str] = CENTERED_PREFIXES, side_column: str = "side") -> pd.DataFrame:
    """Copy of ``trades`` with direction-carrying columns multiplied by ``side`` (no-op for long-only data).

    Centred columns become ``(x - c) * side`` with ``c = 0`` for ``lib_*`` ratios and ``0.5`` otherwise.
    Columns ending in ``_level`` are left alone.
    """
    out = trades.copy()
    directional, centered = tuple(directional), tuple(centered)
    side = out[side_column]
    new = {}
    for c in out.columns:
        if c.startswith(directional):
            new[c] = out[c] * side
        elif c.startswith(centered) and not c.endswith("_level"):
            centre = 0.0 if c.startswith("lib_") else 0.5
            new[c] = (out[c] - centre) * side
    for c, v in new.items():
        out[c] = v
    return out


def relative_levels(trades: pd.DataFrame, feature_names: Iterable[str], *, close_column: str = "raw_close",
                    levels: Mapping[str, str] = LEVEL_FEATURES, keep_levels: bool = True) -> pd.DataFrame:
    """Strategy features as model inputs: ``lib_<name>`` for scale-free features, ``lib_<name>_rel`` for price
    levels (``level / close - 1``; distances ``/ close``) and optionally ``lib_<name>_level`` (raw, for
    stationarity checks only).  Features marked ``"drop"`` are omitted.
    """
    out = {}
    for c in feature_names:
        if c in levels:
            if levels[c] == "ratio":
                out[f"lib_{c}_rel"] = (trades[c] / trades[close_column] - 1 if c not in _DISTANCES
                                       else trades[c] / trades[close_column])
                if keep_levels:
                    out[f"lib_{c}_level"] = trades[c]
        else:
            out[f"lib_{c}"] = trades[c]
    return pd.DataFrame(out, index=trades.index)


def feature_family(column: str, prefixes: Mapping[str, str] = FAMILY_PREFIXES) -> str | None:
    """Family code of a column from the text before its first underscore (``None`` if not a feature)."""
    return prefixes.get(column.split("_")[0])


def feature_families(columns: Iterable[str], prefixes: Mapping[str, str] = FAMILY_PREFIXES) -> dict[str, str]:
    """``{column: family}`` for the columns that belong to a family, in column order."""
    out = {}
    for c in columns:
        fam = feature_family(c, prefixes)
        if fam:
            out[c] = fam
    return out
