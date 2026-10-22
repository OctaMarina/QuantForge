"""Structural statistics vs brute-force references; endpoint causality."""
import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.stats.diagnostic import recursive_olsresiduals

from cryptods.features.structural import sadf_family, sdfc_stat, csw_stats, bde_stat


def series(n=160, seed=0, explosive_from=110):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[0] = 10.0
    for i in range(1, n):
        rho = 1.0 if i < explosive_from else 1.02
        y[i] = 10 + rho * (y[i - 1] - 10) + 0.01 * rng.standard_normal() + (0.002 if i >= explosive_from else 0)
    return y


def adf_ref(y, t0, t, p, const=True):
    rows = range(t0 + p + 1, t + 1)
    dy = np.array([y[i] - y[i - 1] for i in rows])
    X = []
    for i in rows:
        x = ([1.0] if const else []) + [y[i - 1]] + [y[i - j] - y[i - j - 1] for j in range(1, p + 1)]
        X.append(x)
    res = sm.OLS(dy, np.array(X)).fit()
    return res.tvalues[1 if const else 0]


@pytest.mark.parametrize("p", [0, 1, 2])
def test_sadf_qadf_cadf_vs_bruteforce(p):
    y = series()
    n = len(y)
    W, tau, q = 60, 20, 0.9
    chain = np.zeros(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)
    s, qq, c, nst = sadf_family(y, chain, valid, W, tau, p, True, False, q, 1)
    for t in (W + 5, 120, n - 1):
        lo = t - W + 1
        vals = [adf_ref(y, t0, t, p) for t0 in range(lo, t - p - tau + 1)]
        vals = np.array(vals)
        assert nst[t] == len(vals)  # exact grid of start points, no thinning
        assert s[t] == pytest.approx(vals.max(), rel=1e-8)
        qref = np.quantile(vals, q)  # numpy default = linear (Hyndman-Fan type 7)
        assert qq[t] == pytest.approx(qref, rel=1e-8)
        assert c[t] == pytest.approx(vals[vals >= qref].mean(), rel=1e-8)


def test_sadf_detects_explosive_segment_right_tail():
    y = series()
    chain = np.zeros(len(y), dtype=np.int64)
    s, _, _, _ = sadf_family(y, chain, np.ones(len(y), bool), 60, 20, 1, True, False, 0.95, 1)
    assert np.nanmax(s[140:]) > np.nanmax(s[60:105])


def test_sdfc_vs_bruteforce():
    y = series(seed=3)
    n, W, trim = len(y), 80, 0.15
    chain = np.zeros(n, dtype=np.int64)
    out, arg = sdfc_stat(y, chain, np.ones(n, bool), W, trim, 1)
    t = n - 1
    lo = t - W + 1
    w = y[lo:t + 1]
    dy = np.diff(w)
    ylag = w[:-1]
    best, barg = -np.inf, None
    for tau in range(max(2, int(np.ceil(trim * W))), int(np.floor((1 - trim) * W)) + 1):
        r = np.arange(2, W + 1)  # relative index of dy (1-based)
        x = ylag * (r >= tau)
        tv = sm.OLS(dy, x).fit().tvalues[0]
        if tv > best:
            best, barg = tv, tau
    assert out[t] == pytest.approx(best, rel=1e-8)
    assert arg[t] == W - barg


def test_csw_vs_bruteforce():
    y = series(seed=5)
    n, W = len(y), 50
    up, dn, two = csw_stats(y, np.zeros(n, np.int64), np.ones(n, bool), W, 1)
    t = n - 1
    w = y[t - W + 1:t + 1]
    sig = np.sqrt(np.sum(np.diff(w) ** 2) / (W - 1))
    S = np.array([(w[-1] - w[k]) / (sig * np.sqrt(W - 1 - k)) for k in range(0, W - 1)])
    assert up[t] == pytest.approx(S.max()) and dn[t] == pytest.approx((-S).max()) and two[t] == pytest.approx(np.abs(S).max())


def test_bde_vs_statsmodels_recursive_residuals():
    rng = np.random.default_rng(9)
    r = rng.standard_normal(300) * 0.01
    n, W, p = len(r), 120, 1
    out, nw = bde_stat(r, np.zeros(n, np.int64), np.ones(n, bool), W, p, 1)
    t = n - 1
    idx = np.arange(t - W + 1, t + 1)
    y = r[idx]
    X = np.column_stack([np.ones(W), r[idx - 1]])
    res = sm.OLS(y, X).fit()
    rr = recursive_olsresiduals(res, skip=2)
    assert nw[t] == W - 2
    assert out[t] == pytest.approx(rr[5][-1], rel=1e-9)  # final rcusum (sd ddof=1 of scaled residuals)


def test_endpoint_causality_future_perturbation():
    y = series(seed=1)
    n = len(y)
    chain = np.zeros(n, np.int64)
    ok = np.ones(n, bool)
    cut = 120
    y2 = y.copy()
    y2[cut + 1:] = y2[cut + 1:] * 1.5 + 3  # drastic change strictly after the cutoff
    a = [f(y, chain, ok) for f in _fns()]
    b = [f(y2, chain, ok) for f in _fns()]
    for x, z in zip(a, b):
        for xa, za in zip(x, z):
            np.testing.assert_array_equal(np.asarray(xa)[:cut + 1], np.asarray(za)[:cut + 1])


def test_sdfc_not_assigned_to_candidate_date():
    """The statistic evaluated with sample ending at t is stored at t only: truncating the sample at
    an earlier candidate tau must reproduce the earlier row exactly (no retrospective assignment)."""
    y = series(seed=2)
    n = len(y)
    out, _ = sdfc_stat(y, np.zeros(n, np.int64), np.ones(n, bool), 60, 0.15, 1)
    for tau in (80, 100, 130):
        tr, _ = sdfc_stat(y[:tau + 1], np.zeros(tau + 1, np.int64), np.ones(tau + 1, bool), 60, 0.15, 1)
        assert tr[tau] == out[tau]


def _fns():
    return [lambda y, c, v: sadf_family(y, c, v, 50, 15, 1, True, False, 0.95, 1),
            lambda y, c, v: sdfc_stat(y, c, v, 50, 0.15, 1),
            lambda y, c, v: csw_stats(y, c, v, 50, 1),
            lambda y, c, v: bde_stat(np.concatenate([[0], np.diff(y)]), c, np.concatenate([[False], v[1:]]), 50, 1, 1)]


def test_chain_breaks_restart_windows():
    y = series(seed=4)
    n = len(y)
    chain = np.zeros(n, np.int64)
    chain[100:] = 1
    s, _, _, nst = sadf_family(y, chain, np.ones(n, bool), 60, 20, 1, True, False, 0.95, 1)
    # right after the break there is not enough history inside the new chain
    assert np.isnan(s[100:100 + 21]).all() and np.isfinite(s[130])
