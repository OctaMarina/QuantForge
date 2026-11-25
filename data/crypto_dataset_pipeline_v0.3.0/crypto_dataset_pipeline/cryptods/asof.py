"""Backward as-of joins on *availability* time, with staleness limits and explicit tie rules.

Never nearest, never forward: a row at decision time d may only see observations whose
``available_at`` is < d (default, ``exclude_equal``) or <= d (``include_equal``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .nulls import Col, CODE, VALID


def asof_index(avail: np.ndarray, decision: np.ndarray, tie: str = "exclude_equal") -> np.ndarray:
    """Index of the last observation available before/at each decision time (-1 if none).
    ``avail`` must be sorted ascending."""
    avail = np.asarray(avail, dtype=np.int64)
    side = "left" if tie == "exclude_equal" else "right"
    return np.searchsorted(avail, np.asarray(decision, dtype=np.int64), side=side) - 1


def asof_values(obs: pd.DataFrame, value_col: str, decision: np.ndarray, max_age_us: int | None,
                tie: str = "exclude_equal", age_from: str = "available_at_us",
                valid_col: str | None = None) -> tuple[Col, np.ndarray, np.ndarray]:
    """Return (Col of values, age_us, source row index).

    ``age_from``: column whose time defines staleness (e.g. the observation's period_end or its
    availability); ``obs`` is sorted by ``available_at_us`` here.
    """
    n = len(decision)
    col = Col(n, "no_observation")
    age = np.full(n, -1, dtype=np.int64)
    if obs is None or len(obs) == 0:
        col.reason[:] = CODE["missing_source_data"]
        return col, age, np.full(n, -1)
    o = obs.sort_values("available_at_us", kind="stable").reset_index(drop=True)
    idx = asof_index(o["available_at_us"].to_numpy(), decision, tie)
    has = idx >= 0
    j = np.where(has, idx, 0)
    vals = o[value_col].to_numpy(np.float64)[j]
    ref_t = o[age_from].to_numpy(np.int64)[j]
    a = np.asarray(decision, dtype=np.int64) - ref_t
    age[has] = a[has]
    ok = has & np.isfinite(vals)
    if valid_col is not None:
        ok &= o[valid_col].to_numpy(bool)[j]
    stale = np.zeros(n, dtype=bool)
    if max_age_us is not None:
        stale = ok & (a > max_age_us)
    col.set_valid(ok & ~stale, vals)
    col.reason[stale] = CODE["stale_input"]
    col.reason[has & ~np.isfinite(vals)] = CODE["missing_source_data"]
    return col, age, np.where(has, idx, -1)


def point_in_time_latest(obs: pd.DataFrame, key: str = "economic_date", value: str = "value") -> pd.DataFrame:
    """From a vintage table (one row per (economic key, version) with available_at_us), build the
    sequence of states "latest economic key known, with its latest known version" ordered by
    availability.  A revision of an *older* key never masquerades as new information about the
    latest key."""
    if obs is None or len(obs) == 0:
        return pd.DataFrame(columns=["available_at_us", key, value])
    o = obs.sort_values(["available_at_us"], kind="stable")
    known: dict = {}
    rows = []
    for r in o.itertuples(index=False):
        known[getattr(r, key)] = getattr(r, value)
        latest = max(known)
        rows.append({"available_at_us": int(r.available_at_us), key: latest, value: known[latest]})
    df = pd.DataFrame(rows)
    # keep the last state per availability instant
    return df.groupby("available_at_us", as_index=False).last()
