"""Meta-labeling datasets: primary-strategy trades of many assets with their decision-time features.

:func:`build_trade_panel` runs each primary strategy of :mod:`quant_forge.strategies` on each asset with the
bar-level features attached as ``context_columns`` (copied from the signal bar, so nothing after the decision
enters), keeps closed trades, converts the strategy's own level features to ratios, and labels each trade with
``y = 1[trade_return > 0]`` (net of fees, AFML §3.6).  :func:`prepare_meta_dataset` adds the point-in-time track
record (family ``S``), orients direction-carrying features by side and lists the candidate model inputs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..features.trades import (CENTERED_PREFIXES, DIRECTIONAL_PREFIXES, FAMILY_PREFIXES, LEVEL_FEATURES,
                               feature_families, orient_by_side, relative_levels, track_record_features)

__all__ = ["build_trade_panel", "assign_groups", "assign_periods", "prepare_meta_dataset", "MetaDataset",
           "candidates_for", "TRADE_COLUMNS"]

#: Trade columns kept from the strategy library's closed-trade table.
TRADE_COLUMNS = ("side", "decision_time", "entry_time", "exit_time", "entry_price", "trade_return",
                 "trade_return_gross", "holding_hours", "exit_reason")


def _run_asset(strategies, asset, frame, feature_columns, run_kwargs, tick_size, level_map, keep, close_column):
    import quant_forge.strategies as qs

    out = []
    for strat, name in strategies:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = qs.run_strategy(frame, strat, tick_size=tick_size, context_columns=feature_columns, **run_kwargs)
        tdf = res.closed_trades_frame()
        lib = relative_levels(tdf, list(strat.spec.feature_names), close_column=close_column, levels=level_map)
        ids = pd.DataFrame({"strategy": name, "asset": asset}, index=tdf.index)
        out.append((name, pd.concat([tdf[list(keep) + list(feature_columns)], lib, ids], axis=1)))
    return out


def build_trade_panel(bars: Mapping[str, pd.DataFrame], feats: Mapping[str, pd.DataFrame],
                      strategies: Mapping, *, feature_columns: Sequence[str], tick_sizes: Mapping[str, float],
                      fee_bps: float = 5.0, slippage_ticks: int = 0, level_map: Mapping[str, str] = LEVEL_FEATURES,
                      keep: Sequence[str] = TRADE_COLUMNS, close_column: str = "raw_close", n_jobs: int = 1,
                      **strategy_overrides) -> pd.DataFrame:
    """Closed trades of every ``{strategy member: short name}`` on every asset, sorted by decision time.

    feature_columns  bar-feature columns attached to each trade (must include ``close_column`` when the strategy
                     has level features).  Extra keyword arguments are strategy config overrides.
    n_jobs           assets in parallel (joblib processes); the result does not depend on it.
    Columns: ``keep`` + features + ``lib_*`` + ``strategy``, ``asset``, ``y``.  Index: 0..n-1 in
    ``(decision_time, strategy, asset)`` order.
    """
    items = list(strategies.items())
    run_kwargs = dict(fee_bps=fee_bps, slippage_ticks=slippage_ticks, **strategy_overrides)
    cols = list(feature_columns)

    def task(a):
        return (items, a, bars[a].join(feats[a][cols]), cols, run_kwargs, tick_sizes[a], level_map, keep,
                close_column)

    if n_jobs == 1:
        results = [_run_asset(*task(a)) for a in bars]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(delayed(_run_asset)(*task(a)) for a in bars)
    by_strategy = {name: [] for _, name in items}
    for res in results:
        for name, part in res:
            by_strategy[name].append(part)
    trades = pd.concat([p for name in by_strategy for p in by_strategy[name]], ignore_index=True)
    for c in ["decision_time", "entry_time", "exit_time"]:
        trades[c] = pd.to_datetime(trades[c], utc=True)
    trades = trades.sort_values(["decision_time", "strategy", "asset"]).reset_index(drop=True)
    return pd.concat([trades, trades["trade_return"].gt(0).astype(int).rename("y")], axis=1)


def assign_groups(trades: pd.DataFrame, groups: Mapping[str, Sequence[str]], column: str = "group",
                  asset_column: str = "asset") -> pd.Series:
    """Group label per trade from ``{label: assets}`` (e.g. ``{"SEEN": [...], "NEW": [...]}``)."""
    lab = pd.Series(np.nan, index=trades.index, dtype=object)
    for g, assets in groups.items():
        lab[trades[asset_column].isin(assets)] = g
    return lab.rename(column)


def assign_periods(times: pd.Series, boundaries: Sequence[tuple[str, pd.Timestamp]], last: str) -> np.ndarray:
    """Period label by time: ``[("A", t1), ("B", t2)]`` -> A before t1, B before t2, ``last`` afterwards."""
    return np.select([times < t for _, t in boundaries], [lab for lab, _ in boundaries], last)


@dataclass
class MetaDataset:
    """Trades with features (``trades``), feature family of every feature column (``family``) and the candidate
    model inputs (``candidates``: all family columns except raw levels and ``*_level`` diagnostics)."""

    trades: pd.DataFrame
    family: dict
    candidates: list

    def candidates_for(self, strategy: str, min_strategy_coverage: float = 0.5) -> list[str]:
        return candidates_for(self.trades, self.candidates, strategy, min_strategy_coverage)


def prepare_meta_dataset(trades: pd.DataFrame, *, track_record: bool = True, orient: bool = True,
                         directional: Sequence[str] = DIRECTIONAL_PREFIXES, centered: Sequence[str] = CENTERED_PREFIXES,
                         family_prefixes: Mapping[str, str] = FAMILY_PREFIXES,
                         exclude_families: Sequence[str] = ("RAW",), **track_record_kwargs) -> MetaDataset:
    """Add the ``S`` family, orient by side and collect the candidate features of a trade panel."""
    if track_record:
        trades = trades.join(track_record_features(trades, **track_record_kwargs))
    if orient:
        trades = orient_by_side(trades, directional=directional, centered=centered)
    family = feature_families(trades.columns, family_prefixes)
    candidates = [c for c in family if not c.endswith("_level") and family[c] not in exclude_families]
    return MetaDataset(trades, family, candidates)


def candidates_for(trades: pd.DataFrame, candidates: Sequence[str], strategy: str,
                   min_strategy_coverage: float = 0.5, strategy_column: str = "strategy") -> list[str]:
    """Candidates of one strategy: drop the ``lib_*`` features of other strategies (mostly missing for it)."""
    g = trades.loc[trades[strategy_column].eq(strategy)]
    return [c for c in candidates if not c.startswith("lib_") or g[c].notna().mean() > min_strategy_coverage]
