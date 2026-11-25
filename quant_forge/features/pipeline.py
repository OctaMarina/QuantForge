"""One-call construction of the bar-level feature panel."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from .asset import asset_features
from .market import DEFAULT_RANK_COLUMNS, panel_features

__all__ = ["bar_features", "add_raw_levels", "model_columns", "RAW_LEVEL_COLUMNS"]

RAW_LEVEL_COLUMNS = ("open", "close", "volume", "quote_volume", "n_trades")


def bar_features(bars: Mapping[str, pd.DataFrame], bars_per_day: int = 6, *, btc: str = "BTC",
                 rank_columns: Sequence[str] = DEFAULT_RANK_COLUMNS, n_jobs: int = 1,
                 return_market: bool = False):
    """``{asset: features}`` with families V, T, F, CAL (per asset) and R, X, M (panel), plus ``_``-auxiliaries.

    ``n_jobs > 1`` computes the per-asset part in parallel processes (joblib); the result is identical.
    """
    if n_jobs == 1:
        own = {a: asset_features(b, bars_per_day) for a, b in bars.items()}
    else:
        from joblib import Parallel, delayed

        frames = Parallel(n_jobs=n_jobs)(delayed(asset_features)(b, bars_per_day) for b in bars.values())
        own = dict(zip(bars, frames))
    return panel_features(own, bars_per_day, btc=btc, rank_columns=rank_columns, return_market=return_market)


def add_raw_levels(feats: Mapping[str, pd.DataFrame], bars: Mapping[str, pd.DataFrame],
                   columns: Sequence[str] = RAW_LEVEL_COLUMNS) -> dict[str, pd.DataFrame]:
    """Append raw bar levels as ``raw_<column>`` (family ``RAW``: kept to document non-stationarity, never
    meant as pooled model inputs)."""
    return {a: pd.concat([f, bars[a][list(columns)].add_prefix("raw_").reindex(f.index)], axis=1)
            for a, f in feats.items()}


def model_columns(frame: pd.DataFrame) -> list[str]:
    """Columns that are not auxiliaries (auxiliary columns start with ``_``)."""
    return [c for c in frame.columns if not c.startswith("_")]
