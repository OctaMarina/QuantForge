"""Purged K-fold with embargo on common calendar folds for a *panel* of events (several assets).

:class:`~quant_forge.validation.PurgedKFold` handles one series (unique, sorted timestamps).  A pooled
meta-model trains on events of many assets, so decision times repeat and interleave.  This splitter applies
the same overlap rule to such a panel:

* fold boundaries are calendar dates: the quantiles ``linspace(0, 1, n_splits + 1)`` of the decision times
  ``t0`` (identical for every asset); fold ``k`` holds ``cuts[k] <= t0 < cuts[k+1]``; the first fold is open to
  the left and the last to the right, so every event is tested exactly once.  ``cover_all=False`` reproduces
  the research-notebook version, where the outer edges are the float-rounded quantiles of int64 nanoseconds:
  events at the very first / last decision time can fall outside every test fold (they then only ever train);
* a training event is **purged** when its closed interval ``[t0, t1]`` intersects the validation information
  window ``[min t0, max t1]`` of the fold (AFML §7.4.1; same closed-interval rule as ``PurgedKFold``);
* a training event is **embargoed** when ``window_end < t0 <= window_end + h`` with
  ``h = embargo_frac * (max t0 - min t0)`` (a share of the calendar span, AFML §7.4.2);
* training uses events on both sides of the fold.  Empty folds are skipped.

Inputs are arrays of datetimes (any numpy/pandas datetime type or int64 nanoseconds) in any order; returned
indices are positional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

__all__ = ["CalendarPurgedKFold", "CalendarFold", "to_int64_ns"]


def to_int64_ns(t) -> np.ndarray:
    """Datetime-like array (tz-aware or naive, Series, Index, numpy) -> int64 nanoseconds since the epoch."""
    if isinstance(t, (pd.Series, pd.Index)) and isinstance(t.dtype, pd.DatetimeTZDtype):
        t = t.dt.tz_convert("UTC").dt.tz_localize(None) if isinstance(t, pd.Series) else t.tz_convert("UTC").tz_localize(None)
    a = np.asarray(t)
    if np.issubdtype(a.dtype, np.datetime64):
        return a.astype("datetime64[ns]").astype("int64")
    return a.astype("int64")


@dataclass(frozen=True)
class CalendarFold:
    """Positional indices of one fold: ``train``, ``test``, ``purged``, ``embargoed`` (disjoint) and its window."""

    train: np.ndarray
    test: np.ndarray
    purged: np.ndarray
    embargoed: np.ndarray
    window_start: int
    window_end: int


class CalendarPurgedKFold:
    """``CalendarPurgedKFold(n_splits=4, embargo_frac=0.01).split(t0, t1)`` yields ``(train, test)`` arrays."""

    def __init__(self, n_splits: int = 4, embargo_frac: float = 0.01, cover_all: bool = True):
        if int(n_splits) < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0 <= embargo_frac < 1:
            raise ValueError("embargo_frac must be in [0, 1)")
        self.n_splits, self.embargo_frac, self.cover_all = int(n_splits), float(embargo_frac), bool(cover_all)

    def get_n_splits(self, *args, **kwargs) -> int:
        return self.n_splits

    def split_with_diagnostics(self, t0, t1) -> Iterator[CalendarFold]:
        t0, t1 = to_int64_ns(t0), to_int64_ns(t1)
        if t0.shape != t1.shape:
            raise ValueError("t0 and t1 must have the same length")
        if np.any(t1 < t0):
            raise ValueError("every end time must be >= its start time")
        cuts = np.quantile(t0, np.linspace(0, 1, self.n_splits + 1)).astype("int64")
        h = int(self.embargo_frac * (t0.max() - t0.min()))
        last = self.n_splits - 1
        for k in range(self.n_splits):
            if self.cover_all:
                test = ((t0 >= cuts[k]) if k > 0 else np.ones(len(t0), bool)) & ((t0 < cuts[k + 1]) if k < last else True)
            else:
                test = (t0 >= cuts[k]) & ((t0 < cuts[k + 1]) if k < last else (t0 <= cuts[k + 1]))
            if test.sum() == 0:
                continue
            ws, we = t0[test].min(), t1[test].max()
            overlap = (t0 <= we) & (t1 >= ws) & ~test
            embargo = (t0 > we) & (t0 <= we + h) & ~test & ~overlap
            train = ~(test | overlap | embargo)
            yield CalendarFold(np.flatnonzero(train), np.flatnonzero(test), np.flatnonzero(overlap),
                               np.flatnonzero(embargo), int(ws), int(we))

    def split(self, t0, t1) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for fold in self.split_with_diagnostics(t0, t1):
            yield fold.train, fold.test
