"""Structural-break / explosiveness statistics (task section 5.1-5.5; AFML 2018 ch. 17).

All statistics are evaluated at endpoint t using only observations <= t inside a finite window of
the last ``W`` bars of the current unbroken chain (segment).  Hence the names carry ``_w{W}``.

* BDE: recursive residuals of r_i = c + sum_j phi_j r_(i-j) + e_i (bar log returns), initialised with
  the first k observations of the window; w_i = (y_i - x_i'b_(i-1)) / sqrt(1 + x_i'(X'X)_(i-1)^-1 x_i);
  S_t = sum w_j / sd(w) with sd over the residuals available at t (ddof=1).
* CSW: S_(n,t) = (y_t - y_n) / (sigma_t sqrt(t-n)), sigma_t^2 = (t-1)^-1 sum_(i=2..t) (dy_i)^2,
  indices relative to the window start; sup over n in [1, t-1], separate up / down / two-sided.
* SADF (windowed, backward start points at fixed endpoint t; PSY's "BSADF_t", AFML's SADF_t):
  dy_i = a + b y_(i-1) + sum_j phi_j dy_(i-j) [+ trend] + e; ADF_(t0,t) = b / SE(b);
  SADF_t = max over all start points t0 with >= tau effective observations (exact grid, no thinning).
* QADF_t: empirical q-quantile (linear interpolation, Hyndman-Fan type 7) of the same ADF set.
* CADF_t: mean of ADF_(t0,t) >= QADF_t (inclusive tail; never empty because the max is in it).
* SDFC (Chow-type DF): dy_i = delta y_(i-1) 1{i >= tau} + e_i, no intercept, no lags, candidates
  tau in [ceil(trim*n), floor((1-trim)*n)] within the window ending at t; SDFC_t = max DFC_tau.  The
  statistic is attached to t only (never to the candidate date tau).
"""
from __future__ import annotations

import numpy as np

from ..nulls import Col, CODE, VALID
from ..registry import FeatureSpec
from .basic import previous_tick_close
from .context import Ctx

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

AFML17 = ("AFML (2018) ch. 17 'Structural Breaks'.  Primary text NOT accessible here (O'Reilly preview ends "
          "before 17.3, checked 2026-10-20).  Form = task specification; cross-checked against MizarLabs "
          "sadf/sdfc source (implementation, not authority) and statsmodels recursive_olsresiduals.")


@njit(cache=True)
def _solve_inv(A, k):
    """Inverse of a small SPD matrix by Gauss-Jordan with partial pivoting; returns (inv, ok)."""
    M = np.zeros((k, 2 * k))
    for i in range(k):
        for j in range(k):
            M[i, j] = A[i, j]
        M[i, k + i] = 1.0
    for c in range(k):
        p = c
        best = abs(M[c, c])
        for r in range(c + 1, k):
            if abs(M[r, c]) > best:
                best = abs(M[r, c])
                p = r
        if best < 1e-300:
            return M[:, k:], False
        if p != c:
            for j in range(2 * k):
                tmp = M[c, j]
                M[c, j] = M[p, j]
                M[p, j] = tmp
        piv = M[c, c]
        for j in range(2 * k):
            M[c, j] /= piv
        for r in range(k):
            if r != c:
                f = M[r, c]
                if f != 0.0:
                    for j in range(2 * k):
                        M[r, j] -= f * M[c, j]
    return M[:, k:].copy(), True


@njit(cache=True)
def _adf_set_at(y, t, lo, p, const, trend, tau, out):
    """ADF statistics for all start points t0 in [lo, ...] with endpoint t.

    Regression rows i in [t0+p+1, t]: dy_i on [1, y_(i-1)-y_t, dy_(i-1..i-p), (i-t)].
    Returns the number of statistics written to ``out``.  Rows are added one at a time while t0
    moves backwards, so all start points are evaluated exactly.
    """
    k = (1 if const else 0) + 1 + p + (1 if trend else 0)
    XtX = np.zeros((k, k))
    Xty = np.zeros(k)
    yty = 0.0
    x = np.zeros(k)
    cnt = 0
    m = 0
    yc = y[t]
    # first row index added is i = t, then t-1, ...; start t0 = i - p - 1
    i = t
    while True:
        t0 = i - p - 1
        if t0 < lo:
            break
        dy = y[i] - y[i - 1]
        c = 0
        if const:
            x[c] = 1.0
            c += 1
        x[c] = y[i - 1] - (yc if const else 0.0)
        bpos = c
        c += 1
        for j in range(1, p + 1):
            x[c] = y[i - j] - y[i - j - 1]
            c += 1
        if trend:
            x[c] = float(i - t)
            c += 1
        for a in range(k):
            Xty[a] += x[a] * dy
            for bb in range(k):
                XtX[a, bb] += x[a] * x[bb]
        yty += dy * dy
        cnt += 1
        if cnt >= tau and cnt > k:
            inv, ok = _solve_inv(XtX, k)
            if ok:
                beta = np.zeros(k)
                for a in range(k):
                    s = 0.0
                    for bb in range(k):
                        s += inv[a, bb] * Xty[bb]
                    beta[a] = s
                rss = yty
                for a in range(k):
                    rss -= beta[a] * Xty[a]
                s2 = rss / (cnt - k)
                v = s2 * inv[bpos, bpos]
                if s2 > 0 and v > 0:
                    out[m] = beta[bpos] / np.sqrt(v)
                    m += 1
        i -= 1
    return m


@njit(cache=True)
def sadf_family(y, chain, valid, W, tau, p, const, trend, q, every):
    n = y.shape[0]
    sadf = np.full(n, np.nan)
    qadf = np.full(n, np.nan)
    cadf = np.full(n, np.nan)
    nst = np.zeros(n, dtype=np.int64)
    buf = np.empty(W + 2)
    # start of current valid chain
    cs = 0
    for t in range(n):
        if t == 0 or chain[t] != chain[t - 1]:
            cs = t
        if not valid[t]:
            cs = t + 1
            continue
        if every > 1 and (t % every) != 0:
            continue
        lo = max(cs, t - W + 1)
        if t - lo + 1 < tau + p + 1:
            continue
        m = _adf_set_at(y, t, lo, p, const, trend, tau, buf)
        if m == 0:
            continue
        vals = np.sort(buf[:m])
        sadf[t] = vals[m - 1]
        # type-7 quantile
        h = (m - 1) * q
        lo_i = int(np.floor(h))
        hi_i = min(lo_i + 1, m - 1)
        qq = vals[lo_i] + (h - lo_i) * (vals[hi_i] - vals[lo_i])
        qadf[t] = qq
        s = 0.0
        c = 0
        for j in range(m):
            if vals[j] >= qq:
                s += vals[j]
                c += 1
        cadf[t] = s / c
        nst[t] = m
    return sadf, qadf, cadf, nst


@njit(cache=True)
def sdfc_stat(y, chain, valid, W, trim, every):
    n = y.shape[0]
    out = np.full(n, np.nan)
    arg = np.full(n, -1, dtype=np.int64)
    cs = 0
    for t in range(n):
        if t == 0 or chain[t] != chain[t - 1]:
            cs = t
        if not valid[t]:
            cs = t + 1
            continue
        if every > 1 and (t % every) != 0:
            continue
        lo = t - W + 1
        if lo < cs:
            continue
        nn = W  # points y[lo..t]; regression obs i = 2..W (relative)
        N = nn - 1
        sdy2 = 0.0
        for i in range(lo + 1, t + 1):
            d = y[i] - y[i - 1]
            sdy2 += d * d
        # suffix sums over relative index r = 2..nn
        a_lo = int(np.ceil(trim * nn))
        a_hi = int(np.floor((1.0 - trim) * nn))
        if a_lo < 2:
            a_lo = 2
        best = -np.inf
        bi = -1
        sxy = 0.0
        sxx = 0.0
        for r in range(nn, 1, -1):  # relative index r <-> absolute lo + r - 1
            ia = lo + r - 1
            d = y[ia] - y[ia - 1]
            sxy += d * y[ia - 1]
            sxx += y[ia - 1] * y[ia - 1]
            if r <= a_hi and r >= a_lo:
                if sxx > 0:
                    dl = sxy / sxx
                    rss = sdy2 - dl * sxy
                    s2 = rss / (N - 1)
                    if s2 > 0:
                        st = dl / np.sqrt(s2 / sxx)
                        if st > best:
                            best = st
                            bi = r
        if bi >= 0:
            out[t] = best
            arg[t] = nn - bi  # bars between candidate break and endpoint
    return out, arg


@njit(cache=True)
def csw_stats(y, chain, valid, W, every):
    n = y.shape[0]
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    two = np.full(n, np.nan)
    cs = 0
    for t in range(n):
        if t == 0 or chain[t] != chain[t - 1]:
            cs = t
        if not valid[t]:
            cs = t + 1
            continue
        if every > 1 and (t % every) != 0:
            continue
        lo = t - W + 1
        if lo < cs:
            continue
        s2 = 0.0
        for i in range(lo + 1, t + 1):
            d = y[i] - y[i - 1]
            s2 += d * d
        s2 /= (W - 1)  # (t-1)^-1 sum_{i=2..t}, t = W relative
        if not (s2 > 0):
            continue
        sig = np.sqrt(s2)
        bu = -np.inf
        bd = -np.inf
        bt = -np.inf
        for nrel in range(1, W):  # n in [1, t-1] relative; y_n = y[lo + nrel - 1]
            v = (y[t] - y[lo + nrel - 1]) / (sig * np.sqrt(W - nrel))
            if v > bu:
                bu = v
            if -v > bd:
                bd = -v
            if abs(v) > bt:
                bt = abs(v)
        up[t] = bu
        dn[t] = bd
        two[t] = bt
    return up, dn, two


@njit(cache=True)
def bde_stat(r, chain, valid, W, p, every):
    """BDE CUSUM on the window of the last W bar returns ending at t (regression with intercept)."""
    n = r.shape[0]
    out = np.full(n, np.nan)
    nw = np.zeros(n, dtype=np.int64)
    k = 1 + p
    cs = 0
    w = np.empty(W)
    x = np.zeros(k)
    for t in range(n):
        if t == 0 or chain[t] != chain[t - 1]:
            cs = t
        if not valid[t]:
            cs = t + 1
            continue
        if every > 1 and (t % every) != 0:
            continue
        lo = t - W + 1
        if lo - p < cs:
            continue
        # observations i = lo..t, regressors [1, r_(i-1..i-p)]
        XtX = np.zeros((k, k))
        Xty = np.zeros(k)
        m = 0
        cnt = 0
        okw = True
        for i in range(lo, t + 1):
            x[0] = 1.0
            for j in range(1, p + 1):
                x[j] = r[i - j]
            yi = r[i]
            if cnt >= k:
                inv, ok = _solve_inv(XtX, k)
                if not ok:
                    okw = False
                    break
                pred = 0.0
                h = 0.0
                for a in range(k):
                    ba = 0.0
                    for bb in range(k):
                        ba += inv[a, bb] * Xty[bb]
                    pred += x[a] * ba
                    for bb in range(k):
                        h += x[a] * inv[a, bb] * x[bb]
                w[m] = (yi - pred) / np.sqrt(1.0 + h)
                m += 1
            for a in range(k):
                Xty[a] += x[a] * yi
                for bb in range(k):
                    XtX[a, bb] += x[a] * x[bb]
            cnt += 1
        if not okw or m < 3:
            continue
        mu = 0.0
        for j in range(m):
            mu += w[j]
        mu /= m
        v = 0.0
        for j in range(m):
            v += (w[j] - mu) ** 2
        sd = np.sqrt(v / (m - 1))
        if sd > 0:
            out[t] = (mu * m) / sd
            nw[t] = m
    return out, nw


def compute(ctx: Ctx):
    sc = ctx.cfg["features"]["structural"]
    n = ctx.n
    if not sc.get("enabled", True):
        return
    gate = sc["formula_gate"]
    pt, ptv, _, chain = previous_tick_close(ctx)
    ok = ptv & (pt > 0)
    y = np.where(ok, np.log(np.where(ok, pt, 1.0)), np.nan)
    W, tau, p = int(sc["window_bars"]), int(sc["min_length"]), int(sc["adf_lags"])
    every = int(sc["eval_every"])
    const, trend = bool(sc["adf_constant"]), bool(sc["adf_trend"])
    q = float(sc["quantile_q"])
    clock = f"{ctx.bar_type} bars"
    tag = f"w{W}_tau{tau}_p{p}{'_c' if const else '_nc'}{'t' if trend else ''}"
    common = dict(provider=ctx.spec.exchange, inputs="trades -> bar previous-tick log close", clock=clock,
                  availability="bar close (evaluated at endpoint t only)", nature="test_statistic",
                  reference=AFML17)

    def add(name, vals, valid_mask, spec_kw, reason_default="insufficient_history"):
        c = Col(n, reason_default)
        if gate == "primary_verified_only":
            c.reason[:] = CODE["unverified_formula"]
        else:
            c.set_valid(valid_mask & np.isfinite(vals), vals)
            c.reason[~ok] = CODE["feed_gap"]
            if every > 1:
                skipped = (np.arange(n) % every) != 0
                c.reason[skipped & ok] = CODE["not_evaluated"]
            c.values[c.reason != VALID] = np.nan
        ctx.add(FeatureSpec(name=name, **common, **spec_kw), c)

    yy = np.where(ok, y, 0.0)
    sadf, qadf, cadf, nst = sadf_family(yy, chain, ok, W, tau, p, const, trend, q, every)
    nulls = ["insufficient_history", "feed_gap", "invalid_regression", "unverified_formula"]
    spec_adf = f"dy_i = {'a + ' if const else ''}b y_(i-1) + sum_(j<={p}) phi_j dy_(i-j){' + c*i' if trend else ''} + e"
    add(f"sadf_{tag}", sadf, nst > 0, dict(family="structural_sadf",
        formula=f"max over all start points t0 (>= {tau} effective obs, window {W}) of ADF_(t0,t) = b/SE(b); {spec_adf}",
        unit="t-statistic (right tail = explosive)", windows=f"W={W} bars, tau={tau}", null_reasons=nulls,
        variant="windowed backward SADF at endpoint t (PSY: BSADF_t; AFML: SADF_t with start points limited to W)"))
    add(f"qadf_q{int(q * 100)}_{tag}", qadf, nst > 0, dict(family="structural_sadf",
        formula=f"type-7 empirical {q}-quantile of {{ADF_(t0,t)}} (same set as SADF)", unit="t-statistic",
        windows=f"W={W}", null_reasons=nulls))
    add(f"cadf_q{int(q * 100)}_{tag}", cadf, nst > 0, dict(family="structural_sadf",
        formula=f"mean of ADF_(t0,t) >= QADF_t (inclusive tail; ties at the quantile included)", unit="t-statistic",
        windows=f"W={W}", null_reasons=nulls, notes="AFML tail convention (>= vs >) not verified against the primary text."))
    c = Col(n, "insufficient_history")
    c.set_valid(nst > 0, nst.astype(float))
    ctx.add(FeatureSpec(name=f"sadf_n_starts_{tag}", family="structural_sadf", formula="number of ADF start points evaluated",
                        unit="count", **{k: v for k, v in common.items() if k != "nature"}, nature="observed",
                        null_reasons=["insufficient_history"]), c)

    trim = float(sc["sdfc_trim"])
    sd, arg = sdfc_stat(yy, chain, ok, W, trim, every)
    add(f"sdfc_w{W}_trim{int(trim * 100)}", sd, np.isfinite(sd), dict(family="structural_sdfc",
        formula="max over candidates tau of delta/SE(delta) in dy_i = delta y_(i-1) 1{i>=tau} + e_i (no intercept, no lags)",
        unit="t-statistic", windows=f"W={W}, trim={trim}", null_reasons=nulls,
        notes="Dummy uses i >= tau (MizarLabs and secondary sources); the task's 1{i>tau} is the same family "
              "shifted by one index.  Value is assigned to the endpoint t, never to the candidate date."))
    ca = Col(n, "insufficient_history")
    ca.set_valid(arg >= 0, arg.astype(float))
    ctx.add(FeatureSpec(name=f"sdfc_argmax_bars_ago_w{W}", family="structural_sdfc",
                        formula="t - argmax_tau DFC_tau (bars), known at t", unit="bars",
                        **{k: v for k, v in common.items() if k != "nature"}, nature="test_statistic",
                        null_reasons=["insufficient_history"]), ca)

    Wc = int(sc["csw_window"])
    up, dn, two = csw_stats(yy, chain, ok, Wc, every)
    for nm, v, side in (("up", up, "sup_n S_(n,t)"), ("down", dn, "sup_n -S_(n,t)"), ("two", two, "sup_n |S_(n,t)|")):
        add(f"csw_{nm}_w{Wc}", v, np.isfinite(v), dict(family="structural_csw",
            formula=f"{side}, S_(n,t)=(y_t-y_n)/(sigma_t sqrt(t-n)), sigma_t^2=(t-1)^-1 sum_(i=2..t) dy_i^2, window-relative indices",
            unit="statistic", windows=f"W={Wc} bars", null_reasons=nulls,
            notes="No critical boundary is emitted (the b_alpha constant was not verified for this sampling)."))

    Wb, pb = int(sc["bde_window"]), int(sc["bde_lags"])
    lr = np.full(n, np.nan)
    okr = np.zeros(n, dtype=bool)
    okr[1:] = ok[1:] & ok[:-1] & (chain[1:] == chain[:-1])
    lr[1:] = np.where(okr[1:], yy[1:] - yy[:-1], np.nan)
    # BDE needs valid returns; break chain at invalid return
    chain_r = chain.copy()
    bde, nw = bde_stat(np.where(okr, lr, 0.0), chain_r, okr, Wb, pb, every)
    add(f"bde_cusum_w{Wb}_p{pb}", bde, nw > 0, dict(family="structural_bde",
        formula=f"S_t = sum_j w_j / sd(w), recursive residuals of r_i = c + sum_(j<={pb}) phi_j r_(i-j) + e over the last {Wb} bars",
        unit="statistic (~N(0, n_w) under H0)", windows=f"W={Wb} bars", null_reasons=nulls,
        variant=f"windowed BDE; regression configured as AR({pb}) with intercept on bar log returns (test configuration)",
        notes="sd(w) uses ddof=1 over residuals available at t (prefix), never the future sample.  "
              "AFML's exact standardisation (denominator, sigma_eps in f_t) not verified against the book."))
    c = Col(n, "insufficient_history")
    c.set_valid(nw > 0, nw.astype(float))
    ctx.add(FeatureSpec(name=f"bde_n_resid_w{Wb}_p{pb}", family="structural_bde", formula="number of recursive residuals",
                        unit="count", **{k: v for k, v in common.items() if k != "nature"}, nature="observed",
                        null_reasons=["insufficient_history"]), c)
    if every > 1:
        ctx.quality["structural"] = f"evaluated every {every} bars; other rows null with reason not_evaluated (no carry-forward)"
