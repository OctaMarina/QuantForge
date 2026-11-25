"""Input validation and segmentation of a bar DataFrame.

:func:`prepare_bars` turns the caller's DataFrame into a :class:`PreparedBars`
object holding only the numpy arrays the strategies need.  The input frame is
never mutated and only the columns that are actually used are copied.

Conventions (identical to the reference notebook):

* ``bar_start_us`` / ``bar_end_us`` are integer microseconds since the UTC epoch.
* ``bar_start_us`` must be finite and strictly increasing after a stable sort.
* a bar is *valid* when OHLC is finite and consistent, timestamps are
  consistent and (if present) ``complete`` is truthy;
* two consecutive bars are *adjacent* when both are valid, the second does not
  start before the first ended, ``bar_seq`` increases by exactly one (if present)
  and ``segment_id`` is unchanged (if present);
* a *segment* is a maximal run of adjacent bars.  Indicators restart in every
  segment; positions and pending orders do not survive a segment break.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BASE_COLUMNS = ("open", "high", "low", "close", "bar_start_us", "bar_end_us")
OPTIONAL_COLUMNS = (
    "first_trade_ts_us", "bar_available_us", "decision_time_us",
    "complete", "bar_seq", "segment_id", "bar_id",
)
_SINGLE_VALUE_COLUMNS = ("series_id", "bar_type")


@dataclass(frozen=True)
class PreparedBars:
    n: int
    source_row: np.ndarray       # position of each bar in the caller's DataFrame
    bar_id: np.ndarray | None
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray | None
    start_us: np.ndarray         # float64 (exact for microsecond epochs < 2**53)
    end_us: np.ndarray
    first_us: np.ndarray         # first trade timestamp, falls back to start
    ready_us: np.ndarray         # max(end, bar_available, decision_time)
    valid: np.ndarray
    adjacent: np.ndarray
    segment: np.ndarray          # segment number of each bar
    segment_start: np.ndarray    # index of the first bar of each bar's segment
    context: pd.DataFrame        # caller-requested extra columns, same order
    columns_used: tuple[str, ...] = field(default=())

    @property
    def segment_offset(self) -> np.ndarray:
        """0-based position of every bar inside its segment."""
        return np.arange(self.n) - self.segment_start

    def lag(self, values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """``values`` shifted by one bar, never reaching across a segment break."""
        out = np.full(self.n, fill, dtype=float)
        if self.n > 1:
            out[1:] = values[:-1]
        out[~self.adjacent] = fill
        return out

    def segments(self):
        """Yield index arrays of each segment, in order."""
        if self.n == 0:
            return
        starts = np.flatnonzero(~self.adjacent)
        ends = np.r_[starts[1:], self.n]
        for a, b in zip(starts, ends):
            yield np.arange(a, b)


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def prepare_bars(
    df: pd.DataFrame,
    *,
    volume_column: str | None = None,
    context_columns: tuple[str, ...] | list[str] = (),
) -> PreparedBars:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    required = list(BASE_COLUMNS) + ([volume_column] if volume_column else [])
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for col in _SINGLE_VALUE_COLUMNS:
        if col in df.columns and df[col].nunique(dropna=False) > 1:
            raise ValueError(f"Expected only one {col} in input DataFrame")
    context_columns = tuple(dict.fromkeys(context_columns))
    absent = [c for c in context_columns if c not in df.columns]
    if absent:
        raise ValueError(f"Missing context columns: {absent}")

    optional = [c for c in OPTIONAL_COLUMNS if c in df.columns]
    used = list(dict.fromkeys([*required, *optional, *context_columns]))
    sub = df.loc[:, used]  # selection copy: the caller's frame is never touched

    start_raw = _numeric(sub, "bar_start_us")
    order = np.argsort(start_raw, kind="stable")  # NaN sorts last, like sort_values
    sub = sub.iloc[order].reset_index(drop=True)
    n = len(sub)
    start = start_raw[order]
    if n and (not np.all(np.isfinite(start)) or np.any(np.diff(start) <= 0)):
        raise ValueError("bar_start_us must be finite and strictly increasing")

    def micros(column: str, fallback: np.ndarray | None = None) -> np.ndarray | None:
        if column not in sub:
            return None if fallback is None else np.array(fallback, copy=True)
        values = _numeric(sub, column)
        return values if fallback is None else np.where(np.isfinite(values), values, fallback)

    end = micros("bar_end_us")
    first = micros("first_trade_ts_us", start)
    available = micros("bar_available_us", end)
    decision = micros("decision_time_us", end)
    ready = np.maximum.reduce([end, available, decision]) if n else end

    o, h, l, c = (_numeric(sub, k) for k in ("open", "high", "low", "close"))
    v = _numeric(sub, volume_column) if volume_column else None
    valid = (
        np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
        & np.isfinite(end) & np.isfinite(first) & np.isfinite(ready)
        & (o > 0) & (l > 0) & (l <= np.minimum(o, c)) & (h >= np.maximum(o, c))
        & (end >= start) & (first >= start) & (first <= end)
    )
    if v is not None:
        valid &= np.isfinite(v) & (v >= 0)
    if "complete" in sub:
        valid &= sub["complete"].fillna(False).astype(bool).to_numpy()

    adjacent = np.zeros(n, dtype=bool)
    if n > 1:
        adjacent[1:] = valid[:-1] & valid[1:] & (start[1:] >= end[:-1])
        if "bar_seq" in sub:
            seq = _numeric(sub, "bar_seq")
            adjacent[1:] &= np.isfinite(seq[1:]) & np.isfinite(seq[:-1]) & (seq[1:] == seq[:-1] + 1)
        if "segment_id" in sub:
            s = sub["segment_id"]
            adjacent[1:] &= (s.iloc[1:].reset_index(drop=True)
                             .eq(s.iloc[:-1].reset_index(drop=True))
                             .fillna(False).to_numpy(bool))
    segment = np.cumsum(~adjacent)
    segment_start = np.maximum.accumulate(np.where(~adjacent, np.arange(n), 0)) if n else np.zeros(0, int)

    return PreparedBars(
        n=n, source_row=order.astype(np.int64),
        bar_id=sub["bar_id"].to_numpy(dtype=object) if "bar_id" in sub else None,
        o=o, h=h, l=l, c=c, v=v, start_us=start, end_us=end, first_us=first, ready_us=ready,
        valid=valid, adjacent=adjacent, segment=segment, segment_start=segment_start,
        context=sub.loc[:, list(context_columns)].copy(), columns_used=tuple(used),
    )
