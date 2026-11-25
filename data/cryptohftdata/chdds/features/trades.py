"""Trade-derived features (families F1 price/trajectory, F2 volatility on regular grids, F3 activity/flow).

Clock: availability (received_time). A window W=(t-w, t] contains the canonical trades with
recv in (t-w, t] and canonical position <= cursor (the bar's closing trade / last trade before a
time-bar boundary). Validity per window (shared mask `tw_state_<w>`):
  0 valid, 1 warmup (window starts before the first canonical trade of the history),
  2 source_gap (a trade-continuity gap intersects the window), 3 no_trade_in_window (where needed)
Values that need a positive denominator are NaN when it is zero ("undefined_denominator").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pyarrow.parquet as pq
from numba import njit

from .. import paths as P

TRADES_DIR = P.DERIVED / "canonical" / "trades"
NS = 1_000_000_000


def wname(w):
    w = float(w)
    if w >= 3600 and w % 3600 == 0:
        return f"{int(w // 3600)}h"
    if w >= 60 and w % 60 == 0:
        return f"{int(w // 60)}m"
    if w < 1:
        return f"{int(round(w * 1000))}ms"
    return f"{w:g}s"


@dataclass
class TradeFeatureConfig:
    flow_windows_s: tuple = (5, 30, 120, 900, 3600)
    return_windows_s: tuple = (5, 30, 120, 900, 3600, 14400, 86400)
    quantile_windows_s: tuple = (30, 120)
    rv_grids: dict = field(default_factory=lambda: {0.1: (5, 30), 1.0: (30, 120, 900, 3600), 60.0: (14400, 86400)})
    path_eff_windows_s: tuple = (120, 900, 3600)
    max_price_age_s: float = 10.0            # freshness of the last trade price at window ends
    min_bv_returns: int = 30
    sign_acf_n: int = 1000
    sign_acf_lags: tuple = (1, 10)
    sign_acf_min_pairs: int = 100
    burst_windows_s: tuple = (120, 900)
    burst_min_cells: int = 30
    large_trade_quantile: float = 0.95       # of previous UTC day's trade notionals


def load_trade_arrays(d: date, warm_days: int = 1):
    tabs = []
    for k in range(warm_days, -1, -1):
        p = TRADES_DIR / f"{(d - timedelta(days=k)).isoformat()}.parquet"
        if p.exists():
            tabs.append(pq.read_table(p, columns=["seq", "recv_ns", "price_e2", "qty_e3", "sign", "seg"]))
    if not tabs:
        return None
    import pyarrow as pa
    t = pa.concat_tables(tabs)
    seq = t["seq"].to_numpy()
    recv = t["recv_ns"].to_numpy()
    px = t["price_e2"].to_numpy() / 100.0
    q = t["qty_e3"].to_numpy() / 1000.0
    sg = t["sign"].to_numpy().astype(np.float64)
    seg = t["seg"].to_numpy()
    pe2 = t["price_e2"].to_numpy()
    qe3 = t["qty_e3"].to_numpy()
    return dict(seq=seq, recv=recv, px=px, q=q, d=px * q, sg=sg, seg=seg, lx=np.log(px), qe3=qe3, de5=pe2 * qe3)


def load_gaps():
    p = P.DERIVED / "canonical" / "trades_gaps.parquet"
    if not p.exists():
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    g = pq.read_table(p).to_pydict()
    t0 = np.array([x if x is not None else 0 for x in g.get("t0", [])], np.int64)
    t1 = np.array([x if x is not None else (1 << 62) for x in g.get("t1", [])], np.int64)
    o = np.argsort(t0)
    return t0[o], t1[o]


def window_gap_mask(t, w_ns, g0, g1):
    """True where (t-w, t] intersects any gap interval [g0, g1] (gaps sorted by g0)."""
    if g0.size == 0:
        return np.zeros(t.size, bool)
    s = t - w_ns
    # candidate gaps start before t; check the latest-starting ones and the running max of ends
    j = np.searchsorted(g0, t, side="right") - 1
    endmax = np.maximum.accumulate(g1)
    ok = j >= 0
    res = np.zeros(t.size, bool)
    res[ok] = endmax[j[ok]] > s[ok]
    return res


# --------------------------------------------------------------------------- numba helpers
@njit(cache=True)
def _seg_build(x, op_max):
    n = x.shape[0]
    size = 1
    while size < n:
        size *= 2
    tree = np.full(2 * size, -np.inf if op_max else np.inf)
    for i in range(n):
        tree[size + i] = x[i]
    for i in range(size - 1, 0, -1):
        a = tree[2 * i]
        b = tree[2 * i + 1]
        if op_max:
            tree[i] = a if a > b else b
        else:
            tree[i] = a if a < b else b
    return tree, size


@njit(cache=True)
def _seg_query(tree, size, lo, hi, op_max):
    """Inclusive-exclusive [lo, hi)."""
    res = -np.inf if op_max else np.inf
    lo += size
    hi += size
    while lo < hi:
        if lo & 1:
            v = tree[lo]
            if op_max:
                res = v if v > res else res
            else:
                res = v if v < res else res
            lo += 1
        if hi & 1:
            hi -= 1
            v = tree[hi]
            if op_max:
                res = v if v > res else res
            else:
                res = v if v < res else res
        lo //= 2
        hi //= 2
    return res


@njit(cache=True)
def _range_minmax(tmax, tmin, size, lo, hi):
    n = lo.shape[0]
    H = np.full(n, np.nan)
    L = np.full(n, np.nan)
    for i in range(n):
        if hi[i] > lo[i]:
            H[i] = _seg_query(tmax, size, lo[i], hi[i], True)
            L[i] = _seg_query(tmin, size, lo[i], hi[i], False)
    return H, L


@njit(cache=True)
def _window_quantiles(x, lo, hi, qs, max_n):
    n = lo.shape[0]
    out = np.full((n, qs.shape[0]), np.nan)
    for i in range(n):
        m = hi[i] - lo[i]
        if m <= 0 or m > max_n:
            continue
        v = np.sort(x[lo[i]:hi[i]])
        for j in range(qs.shape[0]):
            # linear interpolation (numpy default "linear")
            pos = qs[j] * (m - 1)
            a = int(np.floor(pos))
            b = a + 1 if a + 1 < m else a
            fr = pos - a
            out[i, j] = v[a] * (1 - fr) + v[b] * fr
    return out


@njit(cache=True)
def _run_starts(sg, seg):
    n = sg.shape[0]
    rs = np.empty(n, np.int64)
    for i in range(n):
        if i == 0 or sg[i] != sg[i - 1] or seg[i] != seg[i - 1]:
            rs[i] = i
        else:
            rs[i] = rs[i - 1]
    return rs


def _iprefix(x):
    out = np.zeros(x.size + 1, np.int64)
    np.cumsum(x, out=out[1:])
    return out


def _prefix(x):
    out = np.zeros(x.size + 1, np.float64)
    np.cumsum(x, out=out[1:])
    return out


def _grid_series(recv, lx, g_times):
    """As-of log price at grid times (last trade with recv <= g)."""
    idx = np.searchsorted(recv, g_times, side="right") - 1
    v = np.full(g_times.size, np.nan)
    ok = idx >= 0
    v[ok] = lx[idx[ok]]
    return v, idx


# --------------------------------------------------------------------------- main
def compute_trade_features(d: date, eval_t: np.ndarray, cursor_seq: np.ndarray, cfg: TradeFeatureConfig,
                           history_start_ns: int, gaps=None, large_thr_usdt: float | None = None):
    """Features for queries whose eval_t lies in day d. Returns dict name -> array."""
    A = load_trade_arrays(d, warm_days=1)
    n = eval_t.size
    out = {}
    if A is None or n == 0:
        return out
    g0, g1 = gaps if gaps is not None else load_gaps()
    recv, px, q, dn, sg, lx, seq = A["recv"], A["px"], A["q"], A["d"], A["sg"], A["lx"], A["seq"]
    # hi = number of chunk trades with seq <= cursor (window end, inclusive of the closing trade)
    hi = np.searchsorted(seq, cursor_seq, side="right")
    hi = np.where(cursor_seq < 0, 0, hi)
    has_last = hi > 0
    P_last = np.where(has_last, px[np.maximum(hi - 1, 0)], np.nan)
    t_last = np.where(has_last, recv[np.maximum(hi - 1, 0)], 0)
    fresh_last = has_last & (eval_t - t_last <= cfg.max_price_age_s * NS)
    out["last_trade_price"] = P_last
    out["last_trade_age_ms"] = np.where(has_last, (eval_t - t_last) / 1e6, np.nan)
    buy = sg > 0
    # exact integer prefix sums (qty in 1e-3 BTC, notional in 1e-5 USDT) -> window sums are exact
    qe3 = A["qe3"]
    de5 = A["de5"]
    cN = _iprefix(np.ones(recv.size, np.int64))
    cV = _iprefix(qe3)
    cD = _iprefix(de5)
    cVb = _iprefix(qe3 * buy)
    cDb = _iprefix(de5 * buy)
    cNb = _iprefix(buy.astype(np.int64))
    cQ2 = _prefix(q * q)
    if large_thr_usdt is not None:
        big = dn > large_thr_usdt
        cDL = _prefix(dn * big)
        cSL = _prefix(dn * big * sg)
    tmax, size = _seg_build(px, True)
    tmin, _ = _seg_build(px, False)

    def wstate(w_ns, tend=None):
        """Window (tend-w, tend] validity; if tend < eval_t also requires no gap in (tend, eval_t]."""
        te = eval_t if tend is None else tend
        st = np.zeros(n, np.int8)
        st[te - w_ns < history_start_ns] = 1
        gm = window_gap_mask(te, w_ns, g0, g1)
        if tend is not None:
            gm |= window_gap_mask(eval_t, eval_t - te, g0, g1)
        st[(st == 0) & gm] = 2
        return st

    states = {}
    all_w = sorted(set(cfg.flow_windows_s) | set(cfg.return_windows_s) | set(cfg.quantile_windows_s) |
                   set(cfg.burst_windows_s) | set(cfg.path_eff_windows_s) | {86400})
    for w in all_w:
        w_ns = int(w * NS)
        st = wstate(w_ns)
        states[w] = st
        out[f"tw_state_{wname(w)}"] = st
    # ---- flows / activity
    for w in cfg.flow_windows_s:
        w_ns = int(w * NS)
        lo = np.searchsorted(recv, eval_t - w_ns, side="right")
        lo = np.minimum(lo, hi)
        ok = states[w] == 0
        N = (cN[hi] - cN[lo]).astype(np.float64)
        V = (cV[hi] - cV[lo]) / 1e3
        D = (cD[hi] - cD[lo]) / 1e5
        Vb = (cVb[hi] - cVb[lo]) / 1e3
        Db = (cDb[hi] - cDb[lo]) / 1e5
        Nb = (cNb[hi] - cNb[lo]).astype(np.float64)
        s = wname(w)
        nan = np.nan
        out[f"trade_count_{s}"] = np.where(ok, N, nan)
        out[f"volume_{s}_btc"] = np.where(ok, V, nan)
        out[f"notional_{s}_usdt"] = np.where(ok, D, nan)
        out[f"trade_rate_{s}_per_s"] = np.where(ok, N / w, nan)
        out[f"volume_rate_{s}_btc_s"] = np.where(ok, V / w, nan)
        out[f"notional_rate_{s}_usdt_s"] = np.where(ok, D / w, nan)
        out[f"tfi_{s}_btc"] = np.where(ok, 2 * Vb - V, nan)
        out[f"tfi_{s}_usdt"] = np.where(ok, 2 * Db - D, nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"tfi_ratio_{s}"] = np.where(ok & (V > 0), (2 * Vb - V) / V, nan)
            out[f"count_imbalance_{s}"] = np.where(ok & (N > 0), (2 * Nb - N) / N, nan)
            out[f"mean_trade_size_{s}_btc"] = np.where(ok & (N > 0), V / N, nan)
            out[f"size_concentration_{s}"] = np.where(ok & (V > 0), (cQ2[hi] - cQ2[lo]) / (V * V), nan)
            vw = np.where(V > 0, D / V, nan)
            out[f"vwap_{s}"] = np.where(ok & (V > 0), vw, nan)
            out[f"vwap_distance_{s}_bps"] = np.where(ok & (V > 0) & fresh_last, 1e4 * (P_last / vw - 1), nan)
            if large_thr_usdt is not None:
                DL = cDL[hi] - cDL[lo]
                out[f"large_trade_share_{s}"] = np.where(ok & (D > 0), DL / D, nan)
                out[f"signed_large_flow_{s}_usdt"] = np.where(ok, cSL[hi] - cSL[lo], nan)
            else:
                out[f"large_trade_share_{s}"] = np.full(n, nan)
                out[f"signed_large_flow_{s}_usdt"] = np.full(n, nan)
        # range-based trajectory features on trades in the window
        H, L = _range_minmax(tmax, tmin, size, lo.astype(np.int64), hi.astype(np.int64))
        with np.errstate(invalid="ignore", divide="ignore"):
            okr = ok & (N > 0) & fresh_last
            out[f"range_log_{s}"] = np.where(ok & (N > 0), np.log(H / L), nan)
            out[f"close_location_{s}"] = np.where(okr & (H > L), (P_last - L) / (H - L), nan)
            out[f"drawdown_from_high_{s}"] = np.where(okr, np.log(P_last / H), nan)
            out[f"rebound_from_low_{s}"] = np.where(okr, np.log(P_last / L), nan)
    # ---- size quantiles (exact, bounded window size)
    qs = np.array([0.5, 0.9, 0.99])
    for w in cfg.quantile_windows_s:
        w_ns = int(w * NS)
        lo = np.minimum(np.searchsorted(recv, eval_t - w_ns, side="right"), hi)
        ok = states[w] == 0
        s = wname(w)
        Qb = _window_quantiles(q, lo.astype(np.int64), hi.astype(np.int64), qs, 5_000_000)
        Qd = _window_quantiles(dn, lo.astype(np.int64), hi.astype(np.int64), qs, 5_000_000)
        for j, nm in enumerate(("p50", "p90", "p99")):
            out[f"trade_size_{nm}_{s}_btc"] = np.where(ok, Qb[:, j], np.nan)
            out[f"trade_size_{nm}_{s}_usdt"] = np.where(ok, Qd[:, j], np.nan)
    # ---- returns on the trade price (endpoint as-of, both ends fresh)
    for w in cfg.return_windows_s:
        w_ns = int(w * NS)
        lo = np.searchsorted(recv, eval_t - w_ns, side="right")
        has0 = lo > 0
        p0 = np.where(has0, px[np.maximum(lo - 1, 0)], np.nan)
        t0 = np.where(has0, recv[np.maximum(lo - 1, 0)], 0)
        fresh0 = has0 & ((eval_t - w_ns) - t0 <= cfg.max_price_age_s * NS)
        # defined by two prices only: a gap inside the window is irrelevant, a gap between a price and its endpoint is not,
        # because the true last trade at that endpoint may be one of the missing trades
        end_ok = ~window_gap_mask(eval_t, eval_t - t_last, g0, g1)
        start_ok = ~window_gap_mask(eval_t - w_ns, (eval_t - w_ns) - t0, g0, g1)
        ok = (states[w] != 1) & fresh0 & fresh_last & end_ok & start_ok
        out[f"ret_trade_{wname(w)}"] = np.where(ok, np.log(P_last / p0), np.nan)
    # ---- realized variance on regular grids of the last trade price
    t_lo_all = eval_t.min() - 86400 * NS - 2 * NS
    for dg, wins in cfg.rv_grids.items():
        dg_ns = int(round(dg * NS))
        g_start = (max(t_lo_all, int(recv[0])) // dg_ns) * dg_ns
        g_end = (int(eval_t.max()) // dg_ns) * dg_ns
        gt = np.arange(g_start, g_end + 1, dg_ns, dtype=np.int64)
        gv, _ = _grid_series(recv, lx, gt)
        r = np.diff(gv)
        r = np.where(np.isfinite(r), r, 0.0)
        okr = np.isfinite(np.diff(gv))
        c_r2 = _prefix(r * r)
        c_r2n = _prefix(r * r * (r < 0))
        c_r2p = _prefix(r * r * (r > 0))
        c_abs = _prefix(np.abs(r))
        bvp = np.zeros(r.size)
        bvp[1:] = np.abs(r[1:]) * np.abs(r[:-1])
        c_bv = _prefix(bvp)
        c_ok = _prefix(okr.astype(np.float64))
        ge = (eval_t // dg_ns) * dg_ns              # last grid point <= t
        je = ((ge - g_start) // dg_ns).astype(np.int64)   # index of that grid point
        tmax_r, sz_r = _seg_build(np.abs(r), True)
        for w in wins:
            nret = int(round(w / dg))
            js = je - nret
            # validity on the grid window (ge-w, ge] *and* (ge, t] (no gap between grid end and decision)
            ok = (wstate(int(w * NS), ge) == 0) & (js >= 0)
            ok &= (c_ok[np.clip(je, 0, r.size)] - c_ok[np.clip(js, 0, r.size)]) == nret
            jsc = np.clip(js, 0, r.size)
            jec = np.clip(je, 0, r.size)
            rv = c_r2[jec] - c_r2[jsc]
            s = f"{wname(w)}_grid{wname(dg)}"
            out[f"rv_trade_{s}"] = np.where(ok, rv, np.nan)
            out[f"rv_neg_trade_{s}"] = np.where(ok, c_r2n[jec] - c_r2n[jsc], np.nan)
            out[f"rv_pos_trade_{s}"] = np.where(ok, c_r2p[jec] - c_r2p[jsc], np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"rv_asym_trade_{s}"] = np.where(ok & (rv > 0), (out[f"rv_pos_trade_{s}"] - out[f"rv_neg_trade_{s}"]) / rv, np.nan)
            out[f"rv_n_returns_{s}"] = np.where(ok, nret, np.nan)
            # bipower: pairs fully inside the window -> indices js+1 .. je-1 of bvp (pair uses r[j], r[j-1])
            bv_sum = c_bv[jec] - c_bv[np.clip(jsc + 1, 0, r.size)]
            okbv = ok & (nret >= cfg.min_bv_returns)
            bv = (np.pi / 2) * (nret / max(nret - 1, 1)) * bv_sum
            out[f"bv_trade_{s}"] = np.where(okbv, bv, np.nan)
            out[f"jump_trade_{s}"] = np.where(okbv, np.maximum(rv - bv, 0.0), np.nan)
            mx, _ = _range_minmax(tmax_r, tmax_r, sz_r, jsc.astype(np.int64), jec.astype(np.int64))
            out[f"max_abs_ret_trade_{s}"] = np.where(ok, mx, np.nan)
        # path efficiency on the 1 s grid only
        if dg == 1.0:
            for w in cfg.path_eff_windows_s:
                nret = int(w)
                js = je - nret
                ok = (wstate(int(w * NS), ge) == 0) & (js >= 0)
                jsc = np.clip(js, 0, r.size)
                jec = np.clip(je, 0, r.size)
                ok &= (c_ok[jec] - c_ok[jsc]) == nret
                num = gv[np.clip(je, 0, gv.size - 1)] - gv[np.clip(js, 0, gv.size - 1)]
                den = c_abs[jec] - c_abs[jsc]
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[f"path_efficiency_trade_{wname(w)}_grid1s"] = np.where(ok & (den > 0), num / den, np.nan)
    # ---- sign persistence over the last N trades (same continuity segment)
    seg = A["seg"]
    cE = _prefix(sg)
    N_ = cfg.sign_acf_n
    for lag in cfg.sign_acf_lags:
        prod = np.zeros(sg.size)
        prod[lag:] = sg[lag:] * sg[:-lag]
        cP = _prefix(prod)
        b_lo = hi - (N_ - lag)          # b indices [b_lo, hi), a = b - lag
        okk = (b_lo - lag >= 0) & has_last
        b_loc = np.clip(b_lo, 0, sg.size)
        a_lo = np.clip(b_lo - lag, 0, sg.size)
        a_hi = np.clip(hi - lag, 0, sg.size)
        npair = (hi - b_loc).astype(np.float64)
        okk &= npair >= cfg.sign_acf_min_pairs
        same_seg = np.zeros(n, bool)
        idx_first = np.clip(b_lo - lag, 0, sg.size - 1)
        same_seg[okk] = seg[idx_first[okk]] == seg[np.maximum(hi - 1, 0)[okk]]
        okk &= same_seg
        Sa = cE[a_hi] - cE[a_lo]
        Sb = cE[hi] - cE[b_loc]
        Sab = cP[hi] - cP[b_loc]
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = Sab - Sa * Sb / npair
            va = npair - Sa * Sa / npair
            vb = npair - Sb * Sb / npair
            rho = cov / np.sqrt(va * vb)
        out[f"sign_acf_lag{lag}_n{N_}"] = np.where(okk & (va > 0) & (vb > 0), rho, np.nan)
    rs = _run_starts(sg, seg)
    last = np.maximum(hi - 1, 0)
    rstart = rs[last]
    out["run_length_trades"] = np.where(has_last, (hi - rstart).astype(np.float64), np.nan)
    out["run_sign"] = np.where(has_last, sg[last], np.nan)
    out["run_volume_btc"] = np.where(has_last, (cV[hi] - cV[rstart]) / 1e3, np.nan)
    out["run_duration_s"] = np.where(has_last, (recv[last] - recv[rstart]) / 1e9, np.nan)
    out["run_censored"] = np.where(has_last, (rstart == 0).astype(np.float64), np.nan)  # run started before chunk
    # ---- burstiness: Fano factor of 1 s trade counts over complete cells in the window
    c0 = (int(recv[0]) // NS) * NS
    cells = ((recv - c0) // NS).astype(np.int64)
    ncell = int((eval_t.max() - c0) // NS) + 2
    cnt = np.bincount(cells, minlength=ncell)[:ncell].astype(np.float64)
    cC = _prefix(cnt)
    cC2 = _prefix(cnt * cnt)
    for w in cfg.burst_windows_s:
        ce = ((eval_t - c0) // NS).astype(np.int64)        # cells fully before t: [0, ce)
        cs = ((eval_t - int(w * NS) - c0 + NS - 1) // NS).astype(np.int64)
        okb = (states[w] == 0) & (cs >= 0) & (ce - cs >= cfg.burst_min_cells)
        csc = np.clip(cs, 0, ncell)
        cec = np.clip(ce, 0, ncell)
        m = (cec - csc).astype(np.float64)
        S1 = cC[cec] - cC[csc]
        S2 = cC2[cec] - cC2[csc]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = S1 / m
            var = S2 / m - mean * mean
            out[f"burstiness_fano_{wname(w)}_cell1s"] = np.where(okb & (mean > 0), var / mean, np.nan)
    # ---- activity relative to the trailing 24 h
    w24 = 86400 * NS
    lo24 = np.minimum(np.searchsorted(recv, eval_t - w24, side="right"), hi)
    D24 = (cD[hi] - cD[lo24]) / 1e5
    ok24 = states[86400] == 0
    for w in cfg.flow_windows_s:
        s = wname(w)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"notional_rate_ratio_{s}_vs_24h"] = np.where(ok24 & (D24 > 0), out[f"notional_rate_{s}_usdt_s"] / (D24 / 86400.0), np.nan)
    return out


def previous_day_large_threshold(d: date, quantile: float) -> float | None:
    p = TRADES_DIR / f"{(d - timedelta(days=1)).isoformat()}.parquet"
    if not p.exists():
        return None
    t = pq.read_table(p, columns=["price_e2", "qty_e3"])
    dn = t["price_e2"].to_numpy() / 100.0 * t["qty_e3"].to_numpy() / 1000.0
    return float(np.quantile(dn, quantile)) if dn.size else None
