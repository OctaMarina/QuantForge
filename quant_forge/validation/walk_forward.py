"""Walk-forward (expanding-window) evaluation windows for event data.

A window ``[w0, w1)`` is evaluated on the events *decided* inside it, by a model trained on the events that
*ended* (exited) before ``w0``: no label of the training set reaches into the window.

    windows = calendar_windows("2019-03-01", end="2026-11-19", step="12MS")   # 12-month windows, last one ends at `end`
    train, test = expanding_split(trades, *windows[0], decision_col="decision_time", end_col="exit_time")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["calendar_windows", "expanding_split"]


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts


def calendar_windows(first_start, *, end, years: int | None = 1, months: int | None = None,
                     n: int | None = None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Consecutive windows of ``years`` (or ``months``) starting at ``first_start``; the last window ends at ``end``.

    With ``n`` the number of window starts is fixed (the last start may be followed by a shorter window);
    otherwise starts are generated while they are before ``end``.
    """
    first_start, end = _utc(first_start), _utc(end)
    offset = pd.DateOffset(months=months) if months else pd.DateOffset(years=years or 1)
    starts = []
    i = 0
    while (n is None and first_start + i * offset < end) or (n is not None and i < n):
        starts.append(first_start + i * offset)
        i += 1
    return list(zip(starts[:-1], starts[1:])) + [(starts[-1], end)]


def expanding_split(events: pd.DataFrame, w0, w1, *, decision_col: str = "decision_time",
                    end_col: str = "exit_time", train_mask=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(train, test)``: training events ended before ``w0`` (and in ``train_mask`` if given), test events decided
    in ``[w0, w1)``."""
    tr = events[end_col] < w0
    if train_mask is not None:
        tr &= np.asarray(train_mask, dtype=bool)
    te = (events[decision_col] >= w0) & (events[decision_col] < w1)
    return events.loc[tr], events.loc[te]
