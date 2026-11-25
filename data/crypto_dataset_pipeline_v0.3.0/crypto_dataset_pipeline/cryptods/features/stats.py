"""Numerical helpers shared by feature families (all window-local, hence start-point invariant)."""
from __future__ import annotations

import numpy as np

from ..nulls import Col, CODE, VALID

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f


@njit(cache=True)
def window_sums(x, valid, w):
    """For each i: sum of x[i-w+1..i] computed directly inside the window (no running-sum drift)
    and a flag telling whether all w entries are valid.  Results do not depend on where the series
    starts, which keeps incremental and full recomputation bit-identical."""
    n = x.shape[0]
    out = np.full(n, np.nan)
    ok = np.zeros(n, dtype=np.bool_)
    for i in range(w - 1, n):
        s = 0.0
        good = True
        for j in range(i - w + 1, i + 1):
            if not valid[j]:
                good = False
                break
            s += x[j]
        if good:
            out[i] = s
            ok[i] = True
    return out, ok


def causal_zscore(x: np.ndarray, valid: np.ndarray, n_window: int, min_obs: int, ddof: int = 1) -> Col:
    """z_t = (x_t - mean(prev obs)) / std(prev obs) over the previous ``n_window`` *valid observations*
    (current value excluded).  Missing observations are skipped, never filled.  std == 0 -> null."""
    x = np.asarray(x, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(x)
    n = x.size
    col = Col(n, "missing_source_data")
    idx = np.flatnonzero(valid)
    vals = x[idx]
    for pos, i in enumerate(idx):
        lo = max(0, pos - n_window)
        hist = vals[lo:pos]
        if hist.size < min_obs:
            col.reason[i] = CODE["insufficient_history"]
            continue
        sd = hist.std(ddof=ddof)
        if not np.isfinite(sd) or sd == 0:
            col.reason[i] = CODE["zero_denominator"]
            continue
        col.values[i] = (x[i] - hist.mean()) / sd
        col.reason[i] = VALID
    return col


def ols_no_intercept_from_sums(n, sxx, sxy, syy, min_n: int):
    """OLS y = b x + e without intercept from sufficient statistics.

    b = Sxy/Sxx, RSS = Syy - b*Sxy, s^2 = RSS/(n-1), se = sqrt(s^2/Sxx), t = b/se.
    Returns (b, t, reason_b, reason_t) arrays.  Classical (non-robust) standard errors.
    """
    n = np.asarray(n, dtype=np.float64)
    sxx = np.asarray(sxx, dtype=np.float64)
    sxy = np.asarray(sxy, dtype=np.float64)
    syy = np.asarray(syy, dtype=np.float64)
    b = np.full(n.shape, np.nan)
    t = np.full(n.shape, np.nan)
    rb = np.full(n.shape, CODE["insufficient_history"], dtype=np.int8)
    rt = rb.copy()
    enough = n >= max(min_n, 2)
    zero_var = enough & ~(sxx > 0)
    rb[zero_var] = CODE["invalid_regression"]
    rt[zero_var] = CODE["invalid_regression"]
    ok = enough & (sxx > 0)
    b[ok] = sxy[ok] / sxx[ok]
    rb[ok] = VALID
    rss = syy - np.where(ok, b, 0.0) * sxy
    s2 = rss / np.maximum(n - 1, 1)
    okt = ok & (s2 > 0) & np.isfinite(s2)
    se = np.sqrt(np.where(okt, s2, np.nan) / np.where(okt, sxx, np.nan))
    t[okt] = b[okt] / se[okt]          # coefficient / sqrt(variance)  (NOT / variance)
    rt[okt] = VALID
    rt[ok & ~okt] = CODE["invalid_regression"]
    return b, t, rb, rt


def safe_log_ratio(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.full(a.shape, np.nan)
    ok = (a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b)
    out[ok] = np.log(a[ok] / b[ok])
    reason = np.where(ok, VALID, CODE["missing_source_data"]).astype(np.int8)
    nonpos = np.isfinite(a) & np.isfinite(b) & ~((a > 0) & (b > 0))
    reason[nonpos] = CODE["nonpositive_log_input"]
    return out, reason


def safe_div(num, den, reason_zero="zero_denominator"):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.full(num.shape, np.nan)
    fin = np.isfinite(num) & np.isfinite(den)
    ok = fin & (den != 0)
    out[ok] = num[ok] / den[ok]
    reason = np.full(num.shape, CODE["missing_source_data"], dtype=np.int8)
    reason[ok] = VALID
    reason[fin & (den == 0)] = CODE[reason_zero]
    return out, reason


def imbalance(a, b):
    """(a - b) / (a + b); null when a + b == 0."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return safe_div(a - b, a + b)
