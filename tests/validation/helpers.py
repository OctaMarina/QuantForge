"""Shared builders and an independent brute-force reference for the PurgedKFold tests."""

from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

BASE = pd.Timestamp("2024-01-01")


def day(i, tz=None):
    return (BASE + pd.Timedelta(days=int(i))).tz_localize(tz) if tz else BASE + pd.Timedelta(days=int(i))


def make_info(starts, ends, tz=None) -> pd.Series:
    """Series of event end times indexed by event start times; starts/ends are day offsets."""
    index = pd.DatetimeIndex([day(s, tz) for s in starts])
    return pd.Series([day(e, tz) for e in ends], index=index, name="t1")


def make_X(info: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"f": np.arange(len(info), dtype=float)}, index=info.index)


def folds(cv, info):
    return [(tr.tolist(), te.tolist()) for tr, te in cv.split(make_X(info))]


def kfold_bounds(n, k):
    """Independent re-derivation of KFold(shuffle=False) test ranges."""
    out, a = [], 0
    for f in range(k):
        size = n // k + (1 if f < n % k else 0)
        out.append((a, a + size))
        a += size
    return out


def brute_force_split(info: pd.Series, n_splits: int, pct_embargo: float = 0.0):
    """Pure-python reference: pairwise closed-interval overlap + bisect_left embargo (Snippet 7.3)."""
    starts, ends = list(info.index), list(info.values)
    n = len(starts)
    m = int(np.floor(n * pct_embargo + 1e-9))
    result = []
    for a, b in kfold_bounds(n, n_splits):
        test = list(range(a, b))
        t1 = max(ends[j] for j in test)
        p = bisect.bisect_left(starts, t1)
        embargo = set(range(p, min(p + m, n))) if m else set()
        train = []
        for i in range(n):
            if i in test or i in embargo:
                continue
            if any(starts[i] <= ends[j] and starts[j] <= ends[i] for j in test):   # pairwise, closed
                continue
            train.append(i)
        result.append((train, test))
    return result


def count_overlaps(info: pd.Series, train, test) -> int:
    """Number of (train, test) pairs whose closed intervals intersect."""
    s, e = info.index.to_numpy(), info.to_numpy()
    return sum(int(s[i] <= e[j] and s[j] <= e[i]) for i in train for j in test)
