"""Point-in-time feature engineering for bar data and for trade tables.

    from quant_forge import features as qf

    feats = qf.bar_features(bars, bars_per_day=6)                  # {asset: V, T, F, CAL, R, X, M (+ aux)}
    sel = qf.select_ffd_order({a: f["_log_close"] for a, f in feats.items()}, train_end="2019-03-01")
    feats = qf.add_ffd_features(feats, sel.pooled)                 # FFD of log open / close / volume
    feats = qf.add_raw_levels(feats, bars)                         # raw_open, raw_close, ...

    feats = qf.elapsed_bar_features(bars)                          # any bar type: horizons in elapsed time,
                                                                   # panel features aligned as-of across assets

Modules: :mod:`.kernels` (smoothers), :mod:`.asset` (single-asset families), :mod:`.market` (panel families),
:mod:`.elapsed` (elapsed-time families and asynchronous panels for time / volume / dollar bars),
:mod:`.fracdiff` (AFML ch. 5), :mod:`.trades` (trade-level features and transformations),
:mod:`.pipeline` (one-call builders).  See ``docs/features.md``.
"""

from .asset import (AUX_COLUMNS, asset_features, calendar_features, flow_features, trend_features,
                    volatility_features)
from .elapsed import (ELAPSED_AUX_COLUMNS, asynchronous_panel_features, elapsed_asset_features,
                      elapsed_bar_features)
from .fracdiff import (DEFAULT_FFD_SOURCES, FFDOrderSelection, add_ffd_features, adf_statistic, ffd_weights,
                       frac_diff_ffd, select_ffd_order)
from .kernels import HAVE_NUMBA, bars_since_rolling_max, rsi, wilder_mean
from .market import DEFAULT_RANK_COLUMNS, cross_sectional_features, market_state_features, panel_features
from .pipeline import RAW_LEVEL_COLUMNS, add_raw_levels, bar_features, model_columns
from .trades import (CENTERED_PREFIXES, DIRECTIONAL_PREFIXES, FAMILY_ORDER, FAMILY_PREFIXES, LEVEL_FEATURES,
                     feature_families, feature_family, orient_by_side, relative_levels, track_record_features)

__all__ = [
    "bar_features", "add_raw_levels", "model_columns", "RAW_LEVEL_COLUMNS",
    "elapsed_bar_features", "elapsed_asset_features", "asynchronous_panel_features", "ELAPSED_AUX_COLUMNS",
    "asset_features", "volatility_features", "trend_features", "flow_features", "calendar_features", "AUX_COLUMNS",
    "market_state_features", "cross_sectional_features", "panel_features", "DEFAULT_RANK_COLUMNS",
    "ffd_weights", "frac_diff_ffd", "adf_statistic", "select_ffd_order", "FFDOrderSelection", "add_ffd_features",
    "DEFAULT_FFD_SOURCES",
    "wilder_mean", "rsi", "bars_since_rolling_max", "HAVE_NUMBA",
    "track_record_features", "orient_by_side", "relative_levels", "feature_family", "feature_families",
    "FAMILY_PREFIXES", "FAMILY_ORDER", "DIRECTIONAL_PREFIXES", "CENTERED_PREFIXES", "LEVEL_FEATURES",
]
