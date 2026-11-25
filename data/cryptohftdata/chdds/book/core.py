"""Numba core of the L2 replay (state, atomic message application, integrals, query emission).

Column order emitted by `_emit_query` MUST match ReplayConfig.columns(); a test asserts it.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .chain import Q_ANCHOR, Q_EMPTY, Q_HEAL, Q_INVALIDATE, Q_UPDATE, Q_VERIFY
from .replay import (BS_AWAIT, BS_CROSSED, BS_EMPTY, BS_HEALED, BS_REBUILT, BS_REBUILT_UNC, BS_SETTLE, BS_STALE,
                     BS_UNANCHORED, BS_VALID, F_INDEX,
                     I_HEALED, I_HEAL_T, I_NHEAL, I_EPOCH, I_HEXT, I_LEXT, I_UNANCH, I_EMPTY_T, I_NEMPTY, I_FORCED, I_QUAR,
                     F_MARK, I_ANCH, I_ANCHOR_SYNTH, I_BA, I_BB, I_CERT_HI, I_CERT_LO, I_CROSS_SINCE, I_ESCOUNT,
                     I_ESHEAD, I_KEVT, I_KHAS, I_KT, I_LAST_APPLY_T, I_LAST_EVT, I_LAST_T, I_LAST_TXN, I_MOCOUNT,
                     I_MOHEAD, I_MODE_OK, I_NANCHOR, I_NAPPLY, I_NCROSS, I_NCROSS_INVAL, I_NES, I_NINVAL, I_NLEV,
                     I_NMO, I_NMSG, I_NOFFTICK_POS, I_NVERIFY, I_NVERIFY_FAIL, I_QHEAD, I_RCOUNT, I_RHEAD,
                     I_SETTLE_UNTIL, I_TMAX_A, I_TMAX_B, I_TMIN_A, I_TMIN_B, I_VERIFY_MISMATCH_LEVELS, NF, NI,
                     NK, NML, NT, ReplayConfig)

TICKF = 0.1


class ReplayState:
    """All mutable replay state as numpy arrays (checkpointable)."""

    def __init__(self, cfg: ReplayConfig):
        nb = len(cfg.bands_bps)
        self.n_tw = 3 + NML + 2 * len(cfg.tw_band_bps)
        self.n_step = NML + 6
        self.nc = 2 * self.n_tw + self.n_step + 6
        self.nx = 2 * self.n_tw + 9
        R = cfg.ring_size
        nh = len(cfg.markout_h_s)
        self.qb = np.zeros(NT, np.int64)
        self.qa = np.zeros(NT, np.int64)
        self.rep_b = np.zeros(NT, np.int32)     # anchor epoch in which the bid tick was last reported
        self.rep_a = np.zeros(NT, np.int32)
        self.si = np.zeros(NI, np.int64)
        self.si[I_BB] = -1
        self.si[I_BA] = -1
        self.si[I_CROSS_SINCE] = -1
        self.si[I_CERT_LO] = NT
        self.si[I_CERT_HI] = -1
        self.si[I_TMIN_B] = NT
        self.si[I_TMIN_A] = NT
        self.si[I_TMAX_B] = -1
        self.si[I_TMAX_A] = -1
        self.sf = np.full(NF, np.nan)
        self.bands = np.zeros((nb, 8), np.int64)
        self.topb = np.full((NK, 2), -1, np.int64)
        self.topa = np.full((NK, 2), -1, np.int64)
        self.ptop = np.full((NML, 4), -1, np.int64)   # previous ranks: bid px, bid q, ask px, ask q
        self.ring_t = np.zeros(R, np.int64)
        self.ring_c = np.zeros((R, self.nc), np.float64)
        self.ring_x = np.zeros((R, self.nx), np.float64)
        self.ring_v = np.full((R, 4), np.nan)
        self.cum = np.zeros(self.nc, np.float64)
        self.cur = np.full(self.n_tw + 4, np.nan)     # tw primitives + 4 basis values
        self.curv = np.zeros(self.n_tw + 1, np.int64)  # validity per tw primitive + basis structural validity
        self.es_t = np.zeros(1 << 18, np.int64)
        self.es_c = np.zeros((1 << 18, 5), np.float64)
        self.es_cum = np.zeros(5, np.float64)
        self.mo_t = np.zeros(1 << 19, np.int64)
        self.mo_c = np.zeros((1 << 19, 3 * nh), np.float64)
        self.mo_cum = np.zeros(3 * nh, np.float64)
        self.mo_fifo = np.zeros((nh, 1 << 18, 4), np.float64)
        self.mo_ht = np.zeros((nh, 2), np.int64)
        self.quote = np.zeros((1024, 6), np.int64)    # eff_t, txn, m2, valid, evt, bb/ba packed unused

    def kernel_args(self):
        return (self.qb, self.qa, self.rep_b, self.rep_a, self.si, self.sf, self.bands, self.topb, self.topa, self.ptop, self.ring_t,
                self.ring_c, self.ring_x, self.ring_v, self.cum, self.cur, self.curv, self.es_t, self.es_c,
                self.es_cum, self.mo_t, self.mo_c, self.mo_cum, self.mo_fifo, self.mo_ht, self.quote)

    def save(self, path):
        np.savez_compressed(path, **{k: v for k, v in self.__dict__.items() if isinstance(v, np.ndarray)})

    def load_into(self, path):
        z = np.load(path)
        for k in z.files:
            dst = getattr(self, k)
            if k == "si" and z[k].size < dst.size:
                dst[:z[k].size] = z[k]      # checkpoint written before the rebuilt-mode fields existed
            else:
                dst[...] = z[k]


# --------------------------------------------------------------------------- small helpers
@njit(cache=True)
def _band_edges(m2, bc):
    lo = (m2 * (1000000 - bc) + 2000000 - 1) // 2000000
    hi = (m2 * (1000000 + bc)) // 2000000
    return lo, hi


@njit(cache=True)
def _recompute_bands(qb, qa, bands, bands_c, m2):
    for b in range(bands.shape[0]):
        lo, hi = _band_edges(m2, bands_c[b])
        sq = 0
        sn = 0
        sq2 = 0
        top = m2 // 2
        for p in range(max(lo, 0), min(top, qb.shape[0] - 1) + 1):
            q = qb[p]
            sq += q
            sn += q * p
            sq2 += q * q
        bands[b, 0] = lo
        bands[b, 2] = sq
        bands[b, 4] = sn
        bands[b, 6] = sq2
        sq = 0
        sn = 0
        sq2 = 0
        for p in range(max((m2 + 1) // 2, 0), min(hi, qa.shape[0] - 1) + 1):
            q = qa[p]
            sq += q
            sn += q * p
            sq2 += q * q
        bands[b, 1] = hi
        bands[b, 3] = sq
        bands[b, 5] = sn
        bands[b, 7] = sq2


@njit(cache=True)
def _move_bands(qb, qa, bands, bands_c, m2):
    for b in range(bands.shape[0]):
        lo, hi = _band_edges(m2, bands_c[b])
        olo = bands[b, 0]
        if lo < olo:
            for p in range(lo, olo):
                q = qb[p]
                bands[b, 2] += q
                bands[b, 4] += q * p
                bands[b, 6] += q * q
        elif lo > olo:
            for p in range(olo, lo):
                q = qb[p]
                bands[b, 2] -= q
                bands[b, 4] -= q * p
                bands[b, 6] -= q * q
        bands[b, 0] = lo
        ohi = bands[b, 1]
        if hi > ohi:
            for p in range(ohi + 1, hi + 1):
                q = qa[p]
                bands[b, 3] += q
                bands[b, 5] += q * p
                bands[b, 7] += q * q
        elif hi < ohi:
            for p in range(hi + 1, ohi + 1):
                q = qa[p]
                bands[b, 3] -= q
                bands[b, 5] -= q * p
                bands[b, 7] -= q * q
        bands[b, 1] = hi


@njit(cache=True)
def _scan_top(q, start, step, top, lo_lim, hi_lim):
    n = 0
    p = start
    while n < top.shape[0] and p >= lo_lim and p <= hi_lim and p >= 0 and p < q.shape[0]:
        if q[p] > 0:
            top[n, 0] = p
            top[n, 1] = q[p]
            n += 1
        p += step
    for j in range(n, top.shape[0]):
        top[j, 0] = -1
        top[j, 1] = 0
    return n


@njit(cache=True)
def _ofi(pb, pbq, pa, paq, b, bq, a, aq):
    e = 0.0
    if b >= pb:
        e += bq
    if b <= pb:
        e -= pbq
    if a <= pa:
        e -= aq
    if a >= pa:
        e += paq
    return e


@njit(cache=True)
def _ring_find(ring_t, head, count, t):
    """Ring index of the last entry with time <= t; -1 if none (or ring does not reach back to t)."""
    R = ring_t.shape[0]
    if count == 0:
        return -1
    start = (head - count) % R
    if ring_t[start] > t:
        return -1
    lo = 0
    hi = count - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ring_t[(start + mid) % R] <= t:
            lo = mid
        else:
            hi = mid - 1
    return (start + lo) % R


@njit(cache=True)
def _advance(acc, t0, t1, x, base, n_tw):
    """Add integrals over (t0, t1] given the rate record x (layout ring_x; time bounds are int offsets
    relative to `base`, exact in float64 for |offset| < 2**52 ns)."""
    if t1 <= t0:
        return
    if x[2 * n_tw + 8] < 0.5:      # structurally invalid
        return
    vf = base + np.int64(x[2 * n_tw + 5])
    vu = base + np.int64(x[2 * n_tw + 6])
    a = t0 if t0 > vf else vf
    b = t1 if t1 < vu else vu
    if b <= a:
        return
    dt = (b - a) / 1e9
    nst = acc.shape[0] - 2 * n_tw - 6
    for j in range(n_tw):
        if x[n_tw + 4 + j] > 0.5:
            acc[j] += x[j] * dt
            acc[n_tw + j] += dt
    bd = 2 * n_tw + nst
    acc[bd] += dt
    if x[2 * n_tw + 4] > 0.5:
        bu = base + np.int64(x[2 * n_tw + 7])
        b2 = t1 if t1 < bu else bu
        if b2 > a:
            dt2 = (b2 - a) / 1e9
            for j in range(4):
                acc[bd + 1 + j] += x[n_tw + j] * dt2
            acc[bd + 5] += dt2


OFF_MIN = -(1 << 52)
OFF_MAX = (1 << 52)


@njit(cache=True)
def _off(v, base):
    d = v - base
    if d < OFF_MIN:
        return float(OFF_MIN)
    if d > OFF_MAX:
        return float(OFF_MAX)
    return float(d)


@njit(cache=True)
def _rate_record(x, cur, curv, si, sf, n_tw, max_age_ns, mark_age_ns, base):
    for j in range(n_tw + 4):
        x[j] = cur[j]
    for j in range(n_tw):
        x[n_tw + 4 + j] = curv[j]
    struct = si[I_ANCH] == 1 and si[I_MODE_OK] == 1 and si[I_CROSS_SINCE] < 0 and si[I_BB] >= 0 and si[I_BA] >= 0
    if struct and si[I_UNANCH] == 1 and si[I_FORCED] == 0:
        struct = si[I_BB] >= si[I_CERT_LO] and si[I_BA] <= si[I_CERT_HI]
    x[2 * n_tw + 4] = 1.0 if (struct and curv[n_tw] == 1) else 0.0
    x[2 * n_tw + 5] = _off(si[I_SETTLE_UNTIL], base)
    vu = si[I_LAST_EVT] * 1000000 + max_age_ns
    x[2 * n_tw + 6] = _off(vu, base)
    bu = si[I_KEVT] * 1000000 + mark_age_ns
    if bu > vu:
        bu = vu
    x[2 * n_tw + 7] = _off(bu, base)
    x[2 * n_tw + 8] = 1.0 if struct else 0.0


@njit(cache=True)
def _state_code(si, t, max_age_ns):
    if si[I_ANCH] == 0:
        return BS_UNANCHORED
    if si[I_MODE_OK] == 0:
        return BS_AWAIT
    if si[I_CROSS_SINCE] >= 0:
        return BS_CROSSED
    if si[I_BB] < 0 or si[I_BA] < 0:
        return BS_EMPTY
    if t - si[I_LAST_EVT] * 1000000 > max_age_ns:
        return BS_STALE
    if t < si[I_SETTLE_UNTIL]:
        return BS_SETTLE
    if si[I_HEALED] == 1:
        return BS_HEALED
    if si[I_UNANCH] == 1:
        if si[I_BB] >= si[I_CERT_LO] and si[I_BA] <= si[I_CERT_HI]:
            return BS_REBUILT
        return BS_REBUILT_UNC
    return BS_VALID


@njit(cache=True)
def _lim_lo(si):
    return 0 if si[I_FORCED] == 1 else si[I_CERT_LO]


@njit(cache=True)
def _lim_hi(si):
    return NT - 1 if si[I_FORCED] == 1 else si[I_CERT_HI]


@njit(cache=True)
def _emit_ok(si, st):
    if st == BS_VALID or st == BS_REBUILT:
        return True
    return si[I_FORCED] == 1 and (st == BS_HEALED or st == BS_REBUILT_UNC)


@njit(cache=True)
def _cost(qside, best, step, need_e3, lim_lo, lim_hi, mid_ticks, buy):
    if best < 0 or need_e3 <= 0:
        return np.nan
    rem = need_e3
    acc = 0.0
    p = best
    while rem > 0:
        if p < lim_lo or p > lim_hi or p < 0 or p >= qside.shape[0]:
            return np.nan
        q = qside[p]
        if q > 0:
            take = q if q < rem else rem
            acc += take * float(p)
            rem -= take
        p += step
    avg = acc / need_e3
    if buy:
        return 1e4 * (avg / mid_ticks - 1.0)
    return 1e4 * (1.0 - avg / mid_ticks)


# --------------------------------------------------------------------------- primitives refresh
@njit(cache=True)
def _refresh(qb, qa, si, sf, bands, bands_c, topb, topa, cur, curv, tw_bands, mm_band, n_tw, ring_vrow):
    """Recompute current primitive values after a state change. ring_vrow receives min/max values."""
    bb = si[I_BB]
    ba = si[I_BA]
    for j in range(cur.shape[0]):
        cur[j] = np.nan
    for j in range(curv.shape[0]):
        curv[j] = 0
    for j in range(4):
        ring_vrow[j] = np.nan
    if bb < 0 or ba < 0 or si[I_ANCH] == 0:
        return
    lo_lim = _lim_lo(si)
    hi_lim = _lim_hi(si)
    nbk = _scan_top(qb, bb, -1, topb, lo_lim, NT)
    nak = _scan_top(qa, ba, 1, topa, 0, hi_lim)
    top_ok = bb >= lo_lim and ba <= hi_lim
    m2 = bb + ba
    mid = m2 * 0.5
    qb1 = qb[bb] / 1000.0
    qa1 = qa[ba] / 1000.0
    if top_ok:
        cur[0] = 1e4 * (ba - bb) / mid
        cur[1] = (qb1 - qa1) / (qb1 + qa1)
        cur[2] = 0.5 * (qb1 + qa1)
        curv[0] = 1
        curv[1] = 1
        curv[2] = 1
        ring_vrow[0] = cur[0]
        ring_vrow[1] = cur[2]
    for k in range(NML):
        if k < nbk and k < nak:
            cur[3 + k] = 0.5 * (topb[k, 1] + topa[k, 1]) / 1000.0
            curv[3 + k] = 1
    for j in range(tw_bands.shape[0]):
        b = tw_bands[j]
        if bands[b, 0] >= lo_lim:
            cur[3 + NML + 2 * j] = bands[b, 2] / 1000.0
            curv[3 + NML + 2 * j] = 1
        if bands[b, 1] <= hi_lim:
            cur[3 + NML + 2 * j + 1] = bands[b, 3] / 1000.0
            curv[3 + NML + 2 * j + 1] = 1
    if bands[mm_band, 0] >= lo_lim:
        ring_vrow[2] = bands[mm_band, 2] / 1000.0
    if bands[mm_band, 1] <= hi_lim:
        ring_vrow[3] = bands[mm_band, 3] / 1000.0
    # basis values (need mark/index)
    if top_ok and np.isfinite(sf[F_MARK]) and np.isfinite(sf[F_INDEX]) and sf[F_INDEX] > 0 and sf[F_MARK] > 0:
        midp = mid * TICKF
        bmi = 1e4 * (midp / sf[F_INDEX] - 1.0)
        bmm = 1e4 * (midp / sf[F_MARK] - 1.0)
        cur[n_tw] = bmi
        cur[n_tw + 1] = bmi * bmi
        cur[n_tw + 2] = bmm
        cur[n_tw + 3] = bmm * bmm
        curv[n_tw] = 1


@njit(cache=True)
def _find_best_bid_from(qb, start):
    p = start
    while p >= 0 and qb[p] == 0:
        p -= 1
    return p


@njit(cache=True)
def _find_best_ask_from(qa, start):
    p = start
    while p < qa.shape[0] and qa[p] == 0:
        p += 1
    return p if p < qa.shape[0] else -1


@njit(cache=True)
def _cert_rebuilt(si, rep_b, rep_a):
    """Exact certificate WITHOUT an anchor. A book started empty is a subset of the exchange book (every level held was
    reported, so it is exact), hence true_best_bid >= b and true_best_ask <= a, and the exchange book is never crossed:
    b <= b* < a* <= a. A hidden bid can therefore only sit strictly between b and a; if every tick there has been reported
    on the bid side, b is the true best bid (symmetrically for the ask). With a 1-tick spread there is nothing between."""
    bb = si[I_BB]
    ba = si[I_BA]
    if bb < 0 or ba < 0 or bb >= ba or ba - bb > 4000:
        return
    if bb >= si[I_CERT_LO] and ba <= si[I_CERT_HI]:
        return
    ep = si[I_EPOCH]
    bid_ok = True
    ask_ok = True
    for p in range(bb + 1, ba):
        if rep_b[p] != ep:
            bid_ok = False
        if rep_a[p] != ep:
            ask_ok = False
    if bid_ok:
        if bb < si[I_CERT_LO]:
            si[I_CERT_LO] = bb
        if bb > si[I_HEXT]:
            si[I_HEXT] = bb
    if ask_ok:
        if ba > si[I_CERT_HI]:
            si[I_CERT_HI] = ba
        if si[I_LEXT] < 0 or ba < si[I_LEXT]:
            si[I_LEXT] = ba


@njit(cache=True)
def _reset_empty(si, qb, qa, bands, t):
    _clear_side_range(qb, si[I_TMIN_B], si[I_TMAX_B])
    _clear_side_range(qa, si[I_TMIN_A], si[I_TMAX_A])
    si[I_TMIN_B] = NT
    si[I_TMAX_B] = -1
    si[I_TMIN_A] = NT
    si[I_TMAX_A] = -1
    si[I_BB] = -1
    si[I_BA] = -1
    si[I_CERT_LO] = NT
    si[I_CERT_HI] = -1
    si[I_HEXT] = -1
    si[I_LEXT] = -1
    si[I_EPOCH] += 1
    si[I_ANCH] = 1
    si[I_MODE_OK] = 1
    si[I_UNANCH] = 1
    si[I_HEALED] = 0
    si[I_CROSS_SINCE] = -1
    si[I_EMPTY_T] = t
    si[I_NEMPTY] += 1
    si[I_ANCHOR_SYNTH] = 0
    for b in range(bands.shape[0]):
        for j in range(bands.shape[1]):
            bands[b, j] = 0


@njit(cache=True)
def _nth_level(q, best, step, n):
    """Tick of the n-th occupied level from `best` moving by `step` (deepest found if fewer exist)."""
    p = best
    found = 0
    last = best
    while p >= 0 and p < q.shape[0] and found < n:
        if q[p] > 0:
            found += 1
            last = p
        p += step
    return last


@njit(cache=True)
def _emit_top(xo, iq, qb, qa, bb, ba, lo, hi):
    """Best levels (tick, qty) per side inside [lo, hi] limits; -1 beyond the limit (not verifiable)."""
    K = xo.shape[1] // 4
    n = 0
    p = bb
    while n < K and p >= lo and p >= 0:
        if qb[p] > 0:
            xo[iq, 2 * n] = p
            xo[iq, 2 * n + 1] = qb[p]
            n += 1
        p -= 1
    for j in range(n, K):
        xo[iq, 2 * j] = -1
        xo[iq, 2 * j + 1] = -1
    n = 0
    p = ba
    o = 2 * K
    while n < K and p <= hi and p >= 0 and p < qa.shape[0]:
        if qa[p] > 0:
            xo[iq, o + 2 * n] = p
            xo[iq, o + 2 * n + 1] = qa[p]
            n += 1
        p += 1
    for j in range(n, K):
        xo[iq, o + 2 * j] = -1
        xo[iq, o + 2 * j + 1] = -1


@njit(cache=True)
def _advance_frontiers(si, rep_b, rep_a):
    """Exact extension of the certified domain (see docs/ARCHITECTURE.md §5):
    bids are certified on [F_bid, inf), asks on (-inf, F_ask]. Asks at prices <= a certified best bid must be empty
    (no crossing) -> known; bids >= a certified best ask -> known; a tick reported since the anchor is known.
    The frontier only moves across contiguous known ticks."""
    ep = si[I_EPOCH]
    for _ in range(3):
        bb = si[I_BB]
        ba = si[I_BA]
        moved = False
        if bb >= 0 and bb >= si[I_CERT_LO] and bb > si[I_HEXT]:
            si[I_HEXT] = bb
        if ba >= 0 and ba <= si[I_CERT_HI] and (si[I_LEXT] < 0 or ba < si[I_LEXT]):
            si[I_LEXT] = ba
        f = si[I_CERT_HI]
        if si[I_HEXT] > f:
            f = si[I_HEXT]
        while f + 1 < rep_a.shape[0] and rep_a[f + 1] == ep:
            f += 1
        if f != si[I_CERT_HI]:
            si[I_CERT_HI] = f
            moved = True
        g = si[I_CERT_LO]
        if si[I_LEXT] >= 0 and si[I_LEXT] < g:
            g = si[I_LEXT]
        while g - 1 >= 0 and rep_b[g - 1] == ep:
            g -= 1
        if g != si[I_CERT_LO]:
            si[I_CERT_LO] = g
            moved = True
        if not moved:
            break


@njit(cache=True)
def _clear_side_range(q, lo, hi):
    if hi < lo:
        return
    for p in range(max(lo, 0), min(hi, q.shape[0] - 1) + 1):
        q[p] = 0


# --------------------------------------------------------------------------- the kernel
@njit(cache=True)
def replay_kernel(seq_msg, seq_time, seq_kind, seq_aux,
                  m_evt, m_txn, m_synth, row_start, r_side, r_idx, r_qty, r_off,
                  k_t, k_evt, k_mark, k_index,
                  tr_t, tr_tt, tr_px, tr_q, tr_sign,
                  q_t, out, xo,
                  g_t, g_out,
                  qb, qa, rep_b, rep_a, si, sf, bands, topb, topa, ptop, ring_t, ring_c, ring_x, ring_v, cum, cur, curv,
                  es_t, es_c, es_cum, mo_t, mo_c, mo_cum, mo_fifo, mo_ht, quote,
                  ip, bands_c, ks, win_ns, win_ml, win_mm, tw_bands, cost_q, cost_d, bwin_ns, eswin_ns, mo_h_ns,
                  mowin_ns, min_cov, verify_log):
    nb = ip[0]
    si[I_FORCED] = ip[22]
    mm_band = ip[4]
    ar_band = ip[5]
    conc_band = ip[6]
    es_max_age_ms = ip[11]
    max_age_ns = ip[14] * 1000000
    mark_age_ns = ip[15] * 1000000
    cross_tol_ns = ip[16] * 1000000
    settle_ns = ip[17] * 1000000
    n_tw = cur.shape[0] - 4
    nst = cum.shape[0] - 2 * n_tw - 6
    C_STEP = 2 * n_tw
    R = ring_t.shape[0]
    RE = es_t.shape[0]
    RM = mo_t.shape[0]
    nh = mo_h_ns.shape[0]
    FC = mo_fifo.shape[1]
    xrow = np.zeros(ring_x.shape[1], np.float64)
    vrow = np.full(4, np.nan)
    tmp = np.zeros(cum.shape[0], np.float64)
    tmp2 = np.zeros(cum.shape[0], np.float64)
    cumq = np.zeros(cum.shape[0], np.float64)
    nseq = seq_msg.shape[0]
    nk = k_t.shape[0]
    ntr = tr_t.shape[0]
    nq = q_t.shape[0]
    ngr = g_t.shape[0]
    i_s = 0
    i_k = 0
    i_t = 0
    i_q = 0
    i_g = 0
    nver = 0
    BIG = np.int64(9223372036854775807)
    while True:
        ts = seq_time[i_s] if i_s < nseq else BIG
        tk = k_t[i_k] if i_k < nk else BIG
        tt_ = tr_t[i_t] if i_t < ntr else BIG
        tq = q_t[i_q] if i_q < nq else BIG
        tg = g_t[i_g] if i_g < ngr else BIG
        tm = BIG
        hm = -1
        for h in range(nh):
            if mo_ht[h, 0] < mo_ht[h, 1]:
                v = np.int64(mo_fifo[h, mo_ht[h, 0] % FC, 0])
                if v < tm:
                    tm = v
                    hm = h
        # priority at equal time: trade < book < mark < maturity < grid < query
        t = tt_
        cls = 0
        if ts < t:
            t = ts
            cls = 1
        if tk < t:
            t = tk
            cls = 2
        if tm < t:
            t = tm
            cls = 3
        if tg < t:
            t = tg
            cls = 4
        if tq < t:
            t = tq
            cls = 5
        if t == BIG:
            break
        if cls == 3 and ts == BIG and tk == BIG and tt_ == BIG and tq == BIG and tg == BIG:
            # maturities past the end of this batch wait for the next batch: draining them here would evaluate them on the
            # end-of-batch state and leave mo_t unsorted around the batch boundary
            break
        # integrate running cumulative values up to t with the state in force
        # Integrals advance ONLY at state-changing events (book application, mark update), so the floating-point
        # summation order — and therefore every output bit — is independent of which queries/grid points/trades are
        # evaluated (exact comparability between bar configurations). Queries integrate on a temporary copy.
        if cls == 1 or cls == 2:
            _rate_record(xrow, cur, curv, si, sf, n_tw, max_age_ns, mark_age_ns, si[I_LAST_T])
            _advance(cum, si[I_LAST_T], t, xrow, si[I_LAST_T], n_tw)
            if t > si[I_LAST_T]:
                si[I_LAST_T] = t
        if cls == 0:
            # ----------------------------------------------------------- trade: effective spread + markouts
            tq_ = tr_t[i_t]
            ttm = tr_tt[i_t]
            px = tr_px[i_t]
            sg = tr_sign[i_t]
            qq = tr_q[i_t]
            # latest applied quote strictly before (availability) and strictly before in exchange time
            mpre = -1.0
            qh = si[I_QHEAD]
            QN = quote.shape[0]
            for back in range(1, QN):
                j = (qh - back) % QN
                if quote[j, 0] == 0:
                    break
                if quote[j, 0] < tq_ and quote[j, 1] < ttm:
                    if quote[j, 3] == 1 and ttm - quote[j, 1] <= es_max_age_ms:
                        mpre = quote[j, 2] * 0.5 * TICKF
                    break
            if mpre > 0:
                es = 2e4 * sg * (px - mpre) / mpre
                es_cum[0] += es * qq
                es_cum[1] += es * qq * px
                es_cum[2] += qq
                es_cum[3] += qq * px
                es_cum[4] += 1
                h = si[I_ESHEAD]
                if si[I_ESCOUNT] == 0:
                    # first entry: the zero base survives rebases as a ring record, a literal 0 would not
                    es_t[h] = tq_ - 1
                    for j in range(5):
                        es_c[h, j] = 0.0
                    h = (h + 1) % RE
                    si[I_ESHEAD] = h
                    si[I_ESCOUNT] = 1
                es_t[h] = tq_
                for j in range(5):
                    es_c[h, j] = es_cum[j]
                si[I_ESHEAD] = (h + 1) % RE
                if si[I_ESCOUNT] < RE:
                    si[I_ESCOUNT] += 1
                si[I_NES] += 1
                for hh in range(nh):
                    tl = mo_ht[hh, 1]
                    if tl - mo_ht[hh, 0] < FC:
                        mo_fifo[hh, tl % FC, 0] = tq_ + mo_h_ns[hh]
                        mo_fifo[hh, tl % FC, 1] = sg
                        mo_fifo[hh, tl % FC, 2] = mpre
                        mo_fifo[hh, tl % FC, 3] = qq
                        mo_ht[hh, 1] = tl + 1
            i_t += 1
            continue
        if cls == 1:
            # ----------------------------------------------------------- book application
            kind = seq_kind[i_s]
            m = seq_msg[i_s]
            i_s += 1
            prev_struct = si[I_ANCH] == 1 and si[I_MODE_OK] == 1 and si[I_CROSS_SINCE] < 0 and si[I_BB] >= 0 \
                and si[I_BA] >= 0
            ofi_ok = prev_struct
            if kind == Q_INVALIDATE:
                si[I_ANCH] = 0
                si[I_MODE_OK] = 0
                si[I_NINVAL] += 1
                ofi_ok = False
            elif kind == Q_EMPTY:
                # restart from an empty book: only levels reported from now on are held (see _cert_rebuilt)
                _reset_empty(si, qb, qa, bands, t)
                ofi_ok = False
            elif kind == Q_HEAL:
                # experimental: keep the (possibly stale) state and continue; flagged until verified/rebuilt
                if si[I_ANCH] == 1:
                    si[I_HEALED] = 1
                    si[I_HEAL_T] = t
                    si[I_NHEAL] += 1
                    si[I_MODE_OK] = 1
                ofi_ok = False
            elif kind == Q_ANCHOR or kind == Q_VERIFY:
                rs = row_start[m]
                re_ = row_start[m + 1]
                if kind == Q_VERIFY and prev_struct:
                    # Compare the snapshot (state at L) with the replayed state. If aux>=0 the replayed state is
                    # after message aux (U <= L <= u): state(u) = snapshot overlaid with aux, so levels touched by
                    # aux are excluded from the comparison (exact Binance rule).
                    aux = seq_aux[i_s - 1]
                    na_ = 0
                    if aux >= 0:
                        na_ = row_start[aux + 1] - row_start[aux]
                    ex_k = np.empty(na_, np.int64)
                    for j in range(na_):
                        r = row_start[aux] + j
                        ex_k[j] = r_idx[r] * 4 + r_side[r]
                    ex_k = np.sort(ex_k)
                    # domain of the snapshot
                    lo_b = NT
                    hi_a = -1
                    for r in range(rs, re_):
                        if r_off[r] or r_idx[r] < 0 or r_idx[r] >= NT:
                            continue
                        if r_side[r] == 0 and r_idx[r] < lo_b:
                            lo_b = r_idx[r]
                        if r_side[r] == 1 and r_idx[r] > hi_a:
                            hi_a = r_idx[r]
                    if ip[23] > 0 and re_ - rs > 2 * ip[23]:
                        bl = np.empty(re_ - rs, np.int64)
                        al = np.empty(re_ - rs, np.int64)
                        nb_ = 0
                        na2 = 0
                        for r in range(rs, re_):
                            if r_off[r] or r_idx[r] < 0 or r_idx[r] >= NT or r_qty[r] <= 0:
                                continue
                            if r_side[r] == 0:
                                bl[nb_] = r_idx[r]
                                nb_ += 1
                            elif r_side[r] == 1:
                                al[na2] = r_idx[r]
                                na2 += 1
                        if nb_ > ip[23]:
                            lo_b = np.sort(bl[:nb_])[nb_ - ip[23]]
                        if na2 > ip[23]:
                            hi_a = np.sort(al[:na2])[ip[23] - 1]
                    # comparison only where BOTH the snapshot and our state are certified:
                    # bids in [max(lo_b, cert_lo), +inf), asks in (-inf, min(hi_a, cert_hi)]
                    cb = lo_b if lo_b > si[I_CERT_LO] else si[I_CERT_LO]
                    ca = hi_a if hi_a < si[I_CERT_HI] else si[I_CERT_HI]
                    mis = 0
                    nsb = 0
                    nsa = 0
                    for r in range(rs, re_):
                        if r_off[r] or r_idx[r] < 0 or r_idx[r] >= NT:
                            continue
                        p = r_idx[r]
                        kx = p * 4 + r_side[r]
                        if na_ > 0:
                            jj = np.searchsorted(ex_k, kx)
                            if jj < na_ and ex_k[jj] == kx:
                                continue
                        if r_side[r] == 0 and p >= cb:
                            if r_qty[r] > 0:
                                nsb += 1
                            if qb[p] != r_qty[r]:
                                mis += 1
                        elif r_side[r] == 1 and p <= ca:
                            if r_qty[r] > 0:
                                nsa += 1
                            if qa[p] != r_qty[r]:
                                mis += 1
                    nob = 0
                    if si[I_BB] >= cb:
                        for p in range(cb, si[I_BB] + 1):
                            if qb[p] > 0:
                                kx = p * 4
                                if na_ > 0:
                                    jj = np.searchsorted(ex_k, kx)
                                    if jj < na_ and ex_k[jj] == kx:
                                        continue
                                nob += 1
                    noa = 0
                    if si[I_BA] >= 0 and ca >= si[I_BA]:
                        for p in range(si[I_BA], ca + 1):
                            if qa[p] > 0:
                                kx = p * 4 + 1
                                if na_ > 0:
                                    jj = np.searchsorted(ex_k, kx)
                                    if jj < na_ and ex_k[jj] == kx:
                                        continue
                                noa += 1
                    mis += abs(nob - nsb) + abs(noa - nsa)
                    # mismatches near the top (within 10 bps of our mid) and top-of-book agreement
                    near = 0
                    m2v = si[I_BB] + si[I_BA]
                    lo10 = (m2v * (1000000 - 1000)) // 2000000
                    hi10 = (m2v * (1000000 + 1000)) // 2000000
                    sbb = -1
                    sba = NT
                    for r in range(rs, re_):
                        if r_off[r] or r_qty[r] <= 0 or r_idx[r] < 0 or r_idx[r] >= NT:
                            continue
                        p = r_idx[r]
                        if r_side[r] == 0 and p > sbb:
                            sbb = p
                        if r_side[r] == 1 and p < sba:
                            sba = p
                        kx = p * 4 + r_side[r]
                        if na_ > 0:
                            jj = np.searchsorted(ex_k, kx)
                            if jj < na_ and ex_k[jj] == kx:
                                continue
                        if p >= lo10 and p <= hi10:
                            if (r_side[r] == 0 and qb[p] != r_qty[r]) or (r_side[r] == 1 and qa[p] != r_qty[r]):
                                near += 1
                    for p in range(max(lo10, 0), min(hi10, NT - 1) + 1):
                        kx0 = p * 4
                        if qb[p] > 0:
                            found = False
                            for r in range(rs, re_):
                                if r_side[r] == 0 and r_idx[r] == p and r_qty[r] > 0:
                                    found = True
                                    break
                            if not found:
                                ex_ok = False
                                if na_ > 0:
                                    jj = np.searchsorted(ex_k, kx0)
                                    ex_ok = jj < na_ and ex_k[jj] == kx0
                                if not ex_ok:
                                    near += 1
                        if qa[p] > 0:
                            found = False
                            for r in range(rs, re_):
                                if r_side[r] == 1 and r_idx[r] == p and r_qty[r] > 0:
                                    found = True
                                    break
                            if not found:
                                ex_ok = False
                                if na_ > 0:
                                    jj = np.searchsorted(ex_k, kx0 + 1)
                                    ex_ok = jj < na_ and ex_k[jj] == kx0 + 1
                                if not ex_ok:
                                    near += 1
                    si[I_NVERIFY] += 1
                    if nver < verify_log.shape[0]:
                        verify_log[nver, 0] = seq_time[i_s - 1]
                        verify_log[nver, 1] = mis
                        verify_log[nver, 2] = nsb + nsa
                        verify_log[nver, 3] = si[I_BB]
                        verify_log[nver, 4] = 1 if aux >= 0 else 0
                        if verify_log.shape[1] > 5:
                            verify_log[nver, 5] = si[I_HEALED] + 2 * si[I_UNANCH]
                            verify_log[nver, 6] = (seq_time[i_s - 1] - si[I_HEAL_T]) // 1000000 if si[I_HEALED] == 1 else -1
                            verify_log[nver, 7] = near
                            verify_log[nver, 8] = 1 if (aux < 0 and sbb == si[I_BB] and sba == si[I_BA]) else (0 if aux < 0 else -1)
                        nver += 1
                    if mis == 0 and si[I_UNANCH] == 0:
                        si[I_HEALED] = 0
                        if lo_b < si[I_CERT_LO] or hi_a > si[I_CERT_HI]:
                            # adopt the snapshot outside our certified domain (levels there were unknown);
                            # levels touched by the overlay message keep their (newer) values
                            if lo_b < si[I_CERT_LO]:
                                for p in range(lo_b, si[I_CERT_LO]):
                                    kx = p * 4
                                    keep = False
                                    if na_ > 0:
                                        jj = np.searchsorted(ex_k, kx)
                                        keep = jj < na_ and ex_k[jj] == kx
                                    if not keep:
                                        qb[p] = 0
                            if hi_a > si[I_CERT_HI]:
                                for p in range(si[I_CERT_HI] + 1, hi_a + 1):
                                    kx = p * 4 + 1
                                    keep = False
                                    if na_ > 0:
                                        jj = np.searchsorted(ex_k, kx)
                                        keep = jj < na_ and ex_k[jj] == kx
                                    if not keep:
                                        qa[p] = 0
                            for r in range(rs, re_):
                                if r_off[r] or r_idx[r] < 0 or r_idx[r] >= NT:
                                    continue
                                p = r_idx[r]
                                kx = p * 4 + r_side[r]
                                if na_ > 0:
                                    jj = np.searchsorted(ex_k, kx)
                                    if jj < na_ and ex_k[jj] == kx:
                                        continue
                                if r_side[r] == 0 and p < si[I_CERT_LO]:
                                    qb[p] = r_qty[r]
                                    if p < si[I_TMIN_B]:
                                        si[I_TMIN_B] = p
                                elif r_side[r] == 1 and p > si[I_CERT_HI]:
                                    qa[p] = r_qty[r]
                                    if p > si[I_TMAX_A]:
                                        si[I_TMAX_A] = p
                            if lo_b < si[I_CERT_LO]:
                                si[I_CERT_LO] = lo_b
                            if hi_a > si[I_CERT_HI]:
                                si[I_CERT_HI] = hi_a
                            if hi_a > si[I_HEXT]:
                                si[I_HEXT] = hi_a
                            if si[I_LEXT] < 0 or lo_b < si[I_LEXT]:
                                si[I_LEXT] = lo_b
                            if si[I_BB] >= 0 and si[I_BA] >= 0:
                                _recompute_bands(qb, qa, bands, bands_c, si[I_BB] + si[I_BA])
                        _refresh(qb, qa, si, sf, bands, bands_c, topb, topa, cur, curv, tw_bands, mm_band, n_tw, vrow)
                        continue
                    if mis > 0:
                        si[I_NVERIFY_FAIL] += 1
                        si[I_VERIFY_MISMATCH_LEVELS] += mis
                        if ip[20] == 1:
                            # experiment: measure only, keep the (wrong) replayed state
                            _refresh(qb, qa, si, sf, bands, bands_c, topb, topa, cur, curv, tw_bands, mm_band, n_tw, vrow)
                            continue
                    # mismatch, or a rebuilt book reaching a snapshot: the snapshot (overlaid with aux) is authoritative
                    # -> the state is rebuilt from it and the book returns to the anchored tier
                # ---- anchor: reset state from snapshot rows
                _clear_side_range(qb, si[I_TMIN_B], si[I_TMAX_B])
                _clear_side_range(qa, si[I_TMIN_A], si[I_TMAX_A])
                lo_b = NT
                hi_b = -1
                lo_a = NT
                hi_a = -1
                for r in range(rs, re_):
                    if r_off[r]:
                        continue
                    p = r_idx[r]
                    if p < 0 or p >= NT:
                        continue
                    if r_side[r] == 0:
                        qb[p] = r_qty[r]
                        if p < lo_b:
                            lo_b = p
                        if p > hi_b and r_qty[r] > 0:
                            hi_b = p
                    elif r_side[r] == 1:
                        qa[p] = r_qty[r]
                        if p > hi_a:
                            hi_a = p
                        if p < lo_a and r_qty[r] > 0:
                            lo_a = p
                si[I_TMIN_B] = lo_b
                si[I_TMAX_B] = hi_b
                si[I_TMIN_A] = lo_a
                si[I_TMAX_A] = hi_a
                si[I_CERT_LO] = lo_b
                si[I_CERT_HI] = hi_a
                si[I_EPOCH] += 1
                si[I_HEXT] = hi_a
                si[I_LEXT] = lo_b
                si[I_BB] = hi_b
                si[I_BA] = lo_a
                if ip[23] > 0 and hi_b >= 0 and lo_a < NT:
                    nb_lim = _nth_level(qb, hi_b, -1, ip[23])
                    na_lim = _nth_level(qa, lo_a, 1, ip[23])
                    if nb_lim > lo_b:
                        si[I_CERT_LO] = nb_lim
                        lo_b = nb_lim
                    if na_lim < hi_a:
                        si[I_CERT_HI] = na_lim
                        hi_a = na_lim
                si[I_HEXT] = hi_a
                si[I_LEXT] = lo_b
                si[I_ANCH] = 1
                si[I_UNANCH] = 0
                si[I_QUAR] = 0
                si[I_HEALED] = 0
                si[I_MODE_OK] = 1 if kind == Q_VERIFY else 0
                si[I_CROSS_SINCE] = -1
                si[I_NANCHOR] += 1
                si[I_ANCHOR_SYNTH] = 1 if m_synth[m] else 0
                si[I_LAST_EVT] = m_evt[m]
                si[I_LAST_TXN] = m_txn[m]
                if kind == Q_VERIFY and seq_aux[i_s - 1] >= 0:
                    aux = seq_aux[i_s - 1]
                    for r in range(row_start[aux], row_start[aux + 1]):
                        if r_off[r] or r_idx[r] < 0 or r_idx[r] >= NT:
                            continue
                        p = r_idx[r]
                        if r_side[r] == 0:
                            qb[p] = r_qty[r]
                        elif r_side[r] == 1:
                            qa[p] = r_qty[r]
                    hi_b = _find_best_bid_from(qb, max(hi_b, si[I_TMAX_B]))
                    lo_a = _find_best_ask_from(qa, lo_a if lo_a < NT else 0)
                    si[I_BB] = hi_b
                    si[I_BA] = lo_a
                if hi_b >= 0 and lo_a >= 0 and lo_a < NT:
                    _recompute_bands(qb, qa, bands, bands_c, hi_b + lo_a)
                ofi_ok = False
            else:
                # ----- Q_UPDATE: atomic application of all levels of the message
                if si[I_QUAR] == 1 and t >= si[I_SETTLE_UNTIL]:
                    # end of the crossing quarantine: nothing held so far is trusted, the certificate must be earned again
                    _reset_empty(si, qb, qa, bands, t)
                    si[I_QUAR] = 0
                    ofi_ok = False
                if si[I_ANCH] == 1 and si[I_MODE_OK] == 0:
                    si[I_MODE_OK] = 1      # first connecting update after an anchor
                    if si[I_ANCHOR_SYNTH] == 1:
                        si[I_SETTLE_UNTIL] = seq_time[i_s - 1] + settle_ns
                    ofi_ok = False
                if si[I_ANCH] == 1:
                    bb = si[I_BB]
                    ba = si[I_BA]
                    have_mid = bb >= 0 and ba >= 0
                    m2p = bb + ba
                    if have_mid:
                        ar_lo, ar_hi = _band_edges(m2p, bands_c[ar_band])
                    else:
                        ar_lo = NT
                        ar_hi = -1
                    rs = row_start[m]
                    re_ = row_start[m + 1]
                    nbest = bb
                    nbest_a = ba
                    add_b = 0
                    rem_b = 0
                    add_a = 0
                    rem_a = 0
                    for r in range(rs, re_):
                        if r_off[r]:
                            if r_qty[r] > 0:
                                si[I_NOFFTICK_POS] += 1
                            continue
                        p = r_idx[r]
                        if p < 0 or p >= NT:
                            continue
                        nq_ = r_qty[r]
                        if r_side[r] == 0:
                            old = qb[p]
                            d = nq_ - old
                            qb[p] = nq_
                            kn = si[I_UNANCH] == 0 or rep_b[p] == si[I_EPOCH] or p >= si[I_CERT_LO]
                            rep_b[p] = si[I_EPOCH]
                            if have_mid:
                                for b in range(nb):
                                    if p >= bands[b, 0]:
                                        bands[b, 2] += d
                                        bands[b, 4] += d * p
                                        bands[b, 6] += nq_ * nq_ - old * old
                            if p >= ar_lo and kn:
                                if d > 0:
                                    add_b += d
                                else:
                                    rem_b -= d
                            if nq_ > 0 and p > nbest:
                                nbest = p
                            if p < si[I_TMIN_B]:
                                si[I_TMIN_B] = p
                            if p > si[I_TMAX_B]:
                                si[I_TMAX_B] = p
                        elif r_side[r] == 1:
                            old = qa[p]
                            d = nq_ - old
                            qa[p] = nq_
                            kn = si[I_UNANCH] == 0 or rep_a[p] == si[I_EPOCH] or p <= si[I_CERT_HI]
                            rep_a[p] = si[I_EPOCH]
                            if have_mid:
                                for b in range(nb):
                                    if p <= bands[b, 1]:
                                        bands[b, 3] += d
                                        bands[b, 5] += d * p
                                        bands[b, 7] += nq_ * nq_ - old * old
                            if p <= ar_hi and kn:
                                if d > 0:
                                    add_a += d
                                else:
                                    rem_a -= d
                            if nq_ > 0 and (nbest_a < 0 or p < nbest_a):
                                nbest_a = p
                            if p < si[I_TMIN_A]:
                                si[I_TMIN_A] = p
                            if p > si[I_TMAX_A]:
                                si[I_TMAX_A] = p
                        si[I_NLEV] += 1
                    # new best levels
                    if nbest >= 0 and qb[nbest] == 0:
                        p = nbest
                        while p >= 0 and qb[p] == 0 and nbest - p < 2000000:
                            p -= 1
                        nbest = p if (p >= 0 and qb[p] > 0) else -1
                    if nbest_a >= 0 and qa[nbest_a] == 0:
                        p = nbest_a
                        while p < NT and qa[p] == 0 and p - nbest_a < 2000000:
                            p += 1
                        nbest_a = p if (p < NT and qa[p] > 0) else -1
                    si[I_BB] = nbest
                    si[I_BA] = nbest_a
                    if nbest >= 0 and nbest_a >= 0:
                        if nbest >= nbest_a:
                            if ip[24] == 1:
                                # the premise of the certificate failed: drop it for both tiers until the restart
                                si[I_SETTLE_UNTIL] = seq_time[i_s - 1] + ip[25] * 1000000
                                si[I_QUAR] = 1
                                si[I_UNANCH] = 1
                                si[I_CERT_LO] = NT
                                si[I_CERT_HI] = -1
                                si[I_HEXT] = -1
                                si[I_LEXT] = -1
                            if si[I_CROSS_SINCE] < 0:
                                si[I_CROSS_SINCE] = seq_time[i_s - 1]
                                si[I_NCROSS] += 1
                            elif seq_time[i_s - 1] - si[I_CROSS_SINCE] > cross_tol_ns:
                                si[I_NCROSS_INVAL] += 1
                                if si[I_UNANCH] == 1:
                                    _reset_empty(si, qb, qa, bands, t)   # rebuilt tier: restart (a crossed subset is impossible)
                                else:
                                    si[I_ANCH] = 0     # persistent crossing invalidates the anchor
                                    si[I_MODE_OK] = 0
                        else:
                            si[I_CROSS_SINCE] = -1
                            if have_mid:
                                _move_bands(qb, qa, bands, bands_c, nbest + nbest_a)
                            else:
                                _recompute_bands(qb, qa, bands, bands_c, nbest + nbest_a)
                            if si[I_QUAR] == 0:
                                if si[I_UNANCH] == 1:
                                    _cert_rebuilt(si, rep_b, rep_a)
                                if ip[21] == 1 and si[I_MODE_OK] == 1:
                                    _advance_frontiers(si, rep_b, rep_a)
                    cum[C_STEP + NML] += add_b / 1000.0 if ofi_ok else 0.0
                    cum[C_STEP + NML + 1] += rem_b / 1000.0 if ofi_ok else 0.0
                    cum[C_STEP + NML + 2] += add_a / 1000.0 if ofi_ok else 0.0
                    cum[C_STEP + NML + 3] += rem_a / 1000.0 if ofi_ok else 0.0
                    cum[C_STEP + NML + 4] += 1.0
                    cum[C_STEP + NML + 5] += (re_ - rs)
                si[I_LAST_EVT] = m_evt[m]
                si[I_LAST_TXN] = m_txn[m]
                si[I_NAPPLY] += 1
            si[I_LAST_APPLY_T] = t
            _refresh(qb, qa, si, sf, bands, bands_c, topb, topa, cur, curv, tw_bands, mm_band, n_tw, vrow)
            now_struct = si[I_ANCH] == 1 and si[I_MODE_OK] == 1 and si[I_CROSS_SINCE] < 0 and si[I_BB] >= 0 \
                and si[I_BA] >= 0
            # OFI per rank between two consecutive valid states (never across anchors/gaps)
            if ofi_ok and now_struct and kind == Q_UPDATE:
                for k in range(NML):
                    if ptop[k, 0] >= 0 and ptop[k, 2] >= 0 and topb[k, 0] >= 0 and topa[k, 0] >= 0:
                        cum[C_STEP + k] += _ofi(ptop[k, 0], ptop[k, 1], ptop[k, 2], ptop[k, 3],
                                                topb[k, 0], topb[k, 1], topa[k, 0], topa[k, 1]) / 1000.0
            for k in range(NML):
                if now_struct:
                    ptop[k, 0] = topb[k, 0]
                    ptop[k, 1] = topb[k, 1]
                    ptop[k, 2] = topa[k, 0]
                    ptop[k, 3] = topa[k, 1]
                else:
                    ptop[k, 0] = -1
                    ptop[k, 2] = -1
            # quote history for effective spread
            qh = si[I_QHEAD]
            QN = quote.shape[0]
            quote[qh, 0] = t
            quote[qh, 1] = si[I_LAST_TXN]
            quote[qh, 2] = si[I_BB] + si[I_BA]
            top_ok = now_struct and si[I_BB] >= _lim_lo(si) and si[I_BA] <= _lim_hi(si) and \
                t >= si[I_SETTLE_UNTIL]
            quote[qh, 3] = 1 if top_ok else 0
            quote[qh, 4] = si[I_LAST_EVT]
            si[I_QHEAD] = (qh + 1) % QN
        elif cls == 2:
            # ----------------------------------------------------------- mark/index update
            sf[F_MARK] = k_mark[i_k]
            sf[F_INDEX] = k_index[i_k]
            si[I_KEVT] = k_evt[i_k]
            si[I_KT] = k_t[i_k]
            si[I_KHAS] = 1
            i_k += 1
            _refresh(qb, qa, si, sf, bands, bands_c, topb, topa, cur, curv, tw_bands, mm_band, n_tw, vrow)
        elif cls == 3:
            # ----------------------------------------------------------- markout maturity
            h = hm
            slot = mo_ht[h, 0] % FC
            sg = mo_fifo[h, slot, 1]
            mpre = mo_fifo[h, slot, 2]
            qq = mo_fifo[h, slot, 3]
            mo_ht[h, 0] += 1
            st = _state_code(si, t, max_age_ns)
            if _emit_ok(si, st) and si[I_BB] >= _lim_lo(si) and si[I_BA] <= _lim_hi(si):
                mpost = (si[I_BB] + si[I_BA]) * 0.5 * TICKF
                mo = 1e4 * sg * np.log(mpost / mpre)
                mo_cum[3 * h] += mo * qq
                mo_cum[3 * h + 1] += qq
                mo_cum[3 * h + 2] += 1
                si[I_NMO] += 1
            hd = si[I_MOHEAD]
            if si[I_MOCOUNT] == 0:
                mo_t[hd] = t - 1
                for j in range(3 * nh):
                    mo_c[hd, j] = 0.0
                hd = (hd + 1) % RM
                si[I_MOHEAD] = hd
                si[I_MOCOUNT] = 1
            mo_t[hd] = t
            for j in range(3 * nh):
                mo_c[hd, j] = mo_cum[j]
            si[I_MOHEAD] = (hd + 1) % RM
            if si[I_MOCOUNT] < RM:
                si[I_MOCOUNT] += 1
            continue
        elif cls == 4:
            # ----------------------------------------------------------- 100ms grid sample
            st = _state_code(si, t, max_age_ns)
            g_out[i_g, 0] = t
            g_out[i_g, 1] = st
            g_out[i_g, 2] = si[I_BB] + si[I_BA] if (si[I_BB] >= 0 and si[I_BA] >= 0) else -1
            g_out[i_g, 3] = si[I_BB]
            g_out[i_g, 4] = si[I_BA]
            g_out[i_g, 5] = qb[si[I_BB]] if si[I_BB] >= 0 else -1
            g_out[i_g, 6] = qa[si[I_BA]] if si[I_BA] >= 0 else -1
            g_out[i_g, 7] = (t - si[I_LAST_EVT] * 1000000) // 1000000 if si[I_LAST_EVT] > 0 else -1
            # top-of-book certified flag folded into state: uncertified top -> state 7
            if st == BS_VALID and not (si[I_BB] >= si[I_CERT_LO] and si[I_BA] <= si[I_CERT_HI]):
                g_out[i_g, 1] = 7
            i_g += 1
            continue
        else:
            # ----------------------------------------------------------- query emission
            for j in range(cum.shape[0]):
                cumq[j] = cum[j]
            _rate_record(xrow, cur, curv, si, sf, n_tw, max_age_ns, mark_age_ns, si[I_LAST_T])
            _advance(cumq, si[I_LAST_T], t, xrow, si[I_LAST_T], n_tw)
            if xo.shape[1] > 0:
                if si[I_ANCH] == 1 and si[I_BB] >= 0 and si[I_BA] >= 0:
                    _emit_top(xo, i_q, qb, qa, si[I_BB], si[I_BA], _lim_lo(si), _lim_hi(si))
                else:
                    for j in range(xo.shape[1]):
                        xo[i_q, j] = -1
            _emit_query(out, i_q, t, qb, qa, si, sf, bands, topb, topa, cur, curv, cumq, ring_t, ring_c, ring_x,
                        ring_v, es_t, es_c, es_cum, mo_t, mo_c, mo_cum, ip, ks, win_ns, win_ml, win_mm, tw_bands,
                        cost_q, cost_d, bwin_ns, eswin_ns, mowin_ns, nh, min_cov, max_age_ns, mark_age_ns,
                        conc_band, tmp, tmp2, mo_h_ns)
            i_q += 1
            continue
        # ---- append ring entry after a state-changing event (book / mark)
        _rate_record(xrow, cur, curv, si, sf, n_tw, max_age_ns, mark_age_ns, t)
        h = si[I_RHEAD]
        ring_t[h] = t
        for j in range(cum.shape[0]):
            ring_c[h, j] = cum[j]
        for j in range(xrow.shape[0]):
            ring_x[h, j] = xrow[j]
        for j in range(4):
            ring_v[h, j] = vrow[j]
        si[I_RHEAD] = (h + 1) % R
        if si[I_RCOUNT] < R:
            si[I_RCOUNT] += 1
    return nver


@njit(cache=True)
def _window_cum(ring_t, ring_c, ring_x, si, s, n_tw, outv):
    """Cumulative values at time s (exact): last ring entry <= s plus extrapolation. Returns False if the ring
    does not reach back to s."""
    k = _ring_find(ring_t, si[I_RHEAD], si[I_RCOUNT], s)
    if k < 0:
        return False
    for j in range(outv.shape[0]):
        outv[j] = ring_c[k, j]
    _advance(outv, ring_t[k], s, ring_x[k], ring_t[k], n_tw)
    return True


@njit(cache=True)
def _emit_query(out, iq, t, qb, qa, si, sf, bands, topb, topa, cur, curv, cum, ring_t, ring_c, ring_x, ring_v,
                es_t, es_c, es_cum, mo_t, mo_c, mo_cum, ip, ks, win_ns, win_ml, win_mm, tw_bands, cost_q, cost_d,
                bwin_ns, eswin_ns, mowin_ns, nh, min_cov, max_age_ns, mark_age_ns, conc_band, tmp, tmp2, mo_h_ns):
    nb = ip[0]
    n_tw = cur.shape[0] - 4
    nst = cum.shape[0] - 2 * n_tw - 6
    C_STEP = 2 * n_tw
    BD = 2 * n_tw + nst
    st = _state_code(si, t, max_age_ns)
    bb = si[I_BB]
    ba = si[I_BA]
    lo_lim = _lim_lo(si)
    hi_lim = _lim_hi(si)
    valid = _emit_ok(si, st)
    top_ok = valid and bb >= lo_lim and ba <= hi_lim
    c = 0
    out[iq, c] = st if (st != BS_VALID or top_ok) else 7
    c += 1
    out[iq, c] = (t - si[I_LAST_APPLY_T]) / 1e6 if si[I_LAST_APPLY_T] > 0 else np.nan
    c += 1
    out[iq, c] = (t - si[I_LAST_EVT] * 1000000) / 1e6 if si[I_LAST_EVT] > 0 else np.nan
    c += 1
    if top_ok:
        m2 = bb + ba
        mid_t = m2 * 0.5
        qb1 = qb[bb] / 1000.0
        qa1 = qa[ba] / 1000.0
        out[iq, c] = bb * TICKF
        out[iq, c + 1] = ba * TICKF
        out[iq, c + 2] = mid_t * TICKF
        out[iq, c + 3] = 1e4 * (ba - bb) / mid_t
        out[iq, c + 4] = qb1
        out[iq, c + 5] = qa1
        out[iq, c + 6] = (qb1 - qa1) / (qb1 + qa1)
        out[iq, c + 7] = 1e4 * ((ba * qb1 + bb * qa1) / (qb1 + qa1) - mid_t) / mid_t
        out[iq, c + 8] = 1e4 * (topa[1, 0] - topa[0, 0]) / mid_t if topa[1, 0] >= 0 else np.nan
        out[iq, c + 9] = 1e4 * (topb[0, 0] - topb[1, 0]) / mid_t if topb[1, 0] >= 0 else np.nan
    else:
        for j in range(10):
            out[iq, c + j] = np.nan
    c += 10
    for kk in range(ks.shape[0]):
        K = ks[kk]
        if top_ok and topb[K - 1, 0] >= 0 and topa[K - 1, 0] >= 0:
            db = 0.0
            da = 0.0
            for j in range(K):
                db += topb[j, 1]
                da += topa[j, 1]
            out[iq, c] = db / 1000.0
            out[iq, c + 1] = da / 1000.0
            out[iq, c + 2] = (db - da) / (db + da)
        else:
            out[iq, c] = np.nan
            out[iq, c + 1] = np.nan
            out[iq, c + 2] = np.nan
        c += 3
    for b in range(nb):
        okb = top_ok and bands[b, 0] >= lo_lim
        oka = top_ok and bands[b, 1] <= hi_lim
        db = bands[b, 2] / 1000.0
        da = bands[b, 3] / 1000.0
        out[iq, c] = db if okb else np.nan
        out[iq, c + 1] = da if oka else np.nan
        out[iq, c + 2] = bands[b, 4] * 1e-4 if okb else np.nan
        out[iq, c + 3] = bands[b, 5] * 1e-4 if oka else np.nan
        out[iq, c + 4] = (db - da) / (db + da) if (okb and oka and db + da > 0) else np.nan
        c += 5
    # concentration in band
    b = conc_band
    if top_ok and bands[b, 0] >= lo_lim and bands[b, 2] > 0:
        out[iq, c] = bands[b, 6] / (bands[b, 2] * 1.0 * bands[b, 2])
    else:
        out[iq, c] = np.nan
    if top_ok and bands[b, 1] <= hi_lim and bands[b, 3] > 0:
        out[iq, c + 1] = bands[b, 7] / (bands[b, 3] * 1.0 * bands[b, 3])
    else:
        out[iq, c + 1] = np.nan
    c += 2
    mid_t = (bb + ba) * 0.5 if top_ok else np.nan
    for j in range(cost_q.shape[0]):
        if top_ok:
            out[iq, c] = _cost(qa, ba, 1, cost_q[j], 0, hi_lim, mid_t, True)
            out[iq, c + 1] = _cost(qb, bb, -1, cost_q[j], lo_lim, 1 << 40, mid_t, False)
        else:
            out[iq, c] = np.nan
            out[iq, c + 1] = np.nan
        c += 2
    for j in range(cost_d.shape[0]):
        if top_ok:
            need = np.int64(np.round(cost_d[j] / (mid_t * TICKF) * 1000.0))
            out[iq, c] = _cost(qa, ba, 1, need, 0, hi_lim, mid_t, True)
            out[iq, c + 1] = _cost(qb, bb, -1, need, lo_lim, 1 << 40, mid_t, False)
        else:
            out[iq, c] = np.nan
            out[iq, c + 1] = np.nan
        c += 2
    if top_ok:
        tl = si[I_CERT_LO]
        th = si[I_CERT_HI]
        out[iq, c] = 1e4 * (1.0 - tl / mid_t) if tl <= bb else np.nan
        out[iq, c + 1] = 1e4 * (th / mid_t - 1.0) if th >= ba else np.nan
    else:
        out[iq, c] = np.nan
        out[iq, c + 1] = np.nan
    c += 2
    # ------------------------------------------------------------- windows
    for wi in range(win_ns.shape[0]):
        w = win_ns[wi]
        s = t - w
        ok = _window_cum(ring_t, ring_c, ring_x, si, s, n_tw, tmp)
        wsec = w / 1e9
        cov = (cum[BD] - tmp[BD]) / wsec if ok else np.nan
        full = ok and cov >= min_cov - 1e-7
        if full and ip[24] == 1 and si[I_FORCED] == 0:
            # rebuilt mode: the first step after a valid state begins has no previous state, so it must lie before the window.
            # The coverage tolerance alone (1e-7 of w) would admit it.
            if not _window_cum(ring_t, ring_c, ring_x, si, s - 1000000, n_tw, tmp2) or \
                    cum[BD] - tmp2[BD] < wsec + 0.001 - 1e-7 * wsec:
                full = False
        out[iq, c] = cov
        c += 1
        if full:
            ofi = cum[C_STEP] - tmp[C_STEP]
            out[iq, c] = ofi
            out[iq, c + 1] = ofi / wsec
            dd = cum[n_tw + 2] - tmp[n_tw + 2]
            d1 = (cum[2] - tmp[2]) / dd if dd > 0 else np.nan
            out[iq, c + 2] = ofi / d1 if (d1 > 0) else np.nan
        else:
            out[iq, c] = np.nan
            out[iq, c + 1] = np.nan
            out[iq, c + 2] = np.nan
        c += 3
        if win_ml[wi] == 1:
            for k in range(NML):
                if full:
                    v = cum[C_STEP + k] - tmp[C_STEP + k]
                    dd = cum[n_tw + 3 + k] - tmp[n_tw + 3 + k]
                    dk = (cum[3 + k] - tmp[3 + k]) / dd if dd > 0 else np.nan
                    if ip[24] == 1 and si[I_FORCED] == 0 and dd < wsec * min_cov - 1e-7:
                        v = np.nan
                    out[iq, c] = v
                    out[iq, c + 1] = v / dk if dk > 0 else np.nan
                else:
                    out[iq, c] = np.nan
                    out[iq, c + 1] = np.nan
                c += 2
        # TW means: spread, qi1, depth1, tw bands (each needs its own full coverage)
        for j in (0, 1, 2):
            dd = cum[n_tw + j] - tmp[n_tw + j]
            out[iq, c] = (cum[j] - tmp[j]) / dd if (ok and dd >= wsec * min_cov - 1e-7 and dd > 0) else np.nan
            c += 1
        for jb in range(tw_bands.shape[0]):
            for sd in range(2):
                j = 3 + NML + 2 * jb + sd
                dd = cum[n_tw + j] - tmp[n_tw + j]
                out[iq, c] = (cum[j] - tmp[j]) / dd if (ok and dd >= wsec * min_cov - 1e-7 and dd > 0) else np.nan
                c += 1
        if win_mm[wi] == 1:
            if full:
                # values in force during (s, t]: entry at/before s and all entries after
                k0 = _ring_find(ring_t, si[I_RHEAD], si[I_RCOUNT], s)
                R = ring_t.shape[0]
                mx_s = -np.inf
                mn_d = np.inf
                mn_b = np.inf
                mn_a = np.inf
                k = k0
                while True:
                    v0 = ring_v[k, 0]
                    if v0 == v0 and v0 > mx_s:
                        mx_s = v0
                    v1 = ring_v[k, 1]
                    if v1 == v1 and v1 < mn_d:
                        mn_d = v1
                    v2 = ring_v[k, 2]
                    if v2 == v2 and v2 < mn_b:
                        mn_b = v2
                    v3 = ring_v[k, 3]
                    if v3 == v3 and v3 < mn_a:
                        mn_a = v3
                    if k == (si[I_RHEAD] - 1) % R:
                        break
                    k = (k + 1) % R
                out[iq, c] = mx_s if mx_s > -np.inf else np.nan
                out[iq, c + 1] = mn_d if mn_d < np.inf else np.nan
                out[iq, c + 2] = mn_b if mn_b < np.inf else np.nan
                out[iq, c + 3] = mn_a if mn_a < np.inf else np.nan
                if ip[24] == 1 and si[I_FORCED] == 0:
                    # rebuilt mode: a band minimum over part of the window is not the window minimum
                    okb_ = False
                    oka_ = False
                    for jb in range(tw_bands.shape[0]):
                        if tw_bands[jb] == ip[4]:
                            jm = 3 + NML + 2 * jb
                            okb_ = cum[n_tw + jm] - tmp[n_tw + jm] >= wsec * min_cov - 1e-7
                            oka_ = cum[n_tw + jm + 1] - tmp[n_tw + jm + 1] >= wsec * min_cov - 1e-7
                    if not okb_:
                        out[iq, c + 2] = np.nan
                    if not oka_:
                        out[iq, c + 3] = np.nan
            else:
                for j in range(4):
                    out[iq, c + j] = np.nan
            c += 4
        ar_full = full
        if ip[24] == 1 and si[I_FORCED] == 0 and full:
            # rebuilt mode: a first touch outside the certified domain has an unknown previous size, so adds/removes are
            # complete only if the band stayed inside the domain (widest tw band, both sides) for the whole window
            jw = 3 + NML + 2 * (tw_bands.shape[0] - 1)
            if (cum[n_tw + jw] - tmp[n_tw + jw] < wsec * min_cov - 1e-7) or \
                    (cum[n_tw + jw + 1] - tmp[n_tw + jw + 1] < wsec * min_cov - 1e-7):
                ar_full = False
        for j in range(6):
            okj = ar_full if j < 4 else full
            out[iq, c + j] = (cum[C_STEP + NML + j] - tmp[C_STEP + NML + j]) if okj else np.nan
        c += 6
    # ------------------------------------------------------------- basis (book mid vs mark/index)
    mark_ok = si[I_KHAS] == 1 and (t - si[I_KEVT] * 1000000) <= mark_age_ns
    if top_ok and mark_ok and np.isfinite(cur[n_tw]):
        out[iq, c] = cur[n_tw]
        out[iq, c + 1] = cur[n_tw + 2]
    else:
        out[iq, c] = np.nan
        out[iq, c + 1] = np.nan
    out[iq, c + 2] = (t - si[I_KT]) / 1e6 if si[I_KHAS] == 1 else np.nan
    out[iq, c + 3] = (t - si[I_KEVT] * 1000000) / 1e6 if si[I_KHAS] == 1 else np.nan
    c += 4
    for wi in range(bwin_ns.shape[0]):
        w = bwin_ns[wi]
        s = t - w
        ok = _window_cum(ring_t, ring_c, ring_x, si, s, n_tw, tmp)
        wsec = w / 1e9
        dd = cum[BD + 5] - tmp[BD + 5] if ok else 0.0
        cov = dd / wsec if ok else np.nan
        out[iq, c] = cov
        if ok and dd >= wsec * min_cov - 1e-7 and dd > 0:
            m1 = (cum[BD + 1] - tmp[BD + 1]) / dd
            m2_ = (cum[BD + 2] - tmp[BD + 2]) / dd
            m3 = (cum[BD + 3] - tmp[BD + 3]) / dd
            m4 = (cum[BD + 4] - tmp[BD + 4]) / dd
            out[iq, c + 1] = m1
            out[iq, c + 2] = np.sqrt(max(m2_ - m1 * m1, 0.0))
            out[iq, c + 3] = m3
            out[iq, c + 4] = np.sqrt(max(m4 - m3 * m3, 0.0))
        else:
            for j in range(1, 5):
                out[iq, c + j] = np.nan
        c += 5
    # ------------------------------------------------------------- effective spread (trade-weighted)
    for wi in range(eswin_ns.shape[0]):
        s = t - eswin_ns[wi]
        k = _ring_find(es_t, si[I_ESHEAD], si[I_ESCOUNT], s)
        okb = _window_cum(ring_t, ring_c, ring_x, si, s, n_tw, tmp)
        if k < 0:
            if si[I_ESCOUNT] == es_t.shape[0]:
                okb = False
            elif si[I_ESCOUNT] > 0:
                k = (si[I_ESHEAD] - si[I_ESCOUNT]) % es_t.shape[0]   # nothing before the first entry: it holds the initial base
        cov = (cum[BD] - tmp[BD]) / (eswin_ns[wi] / 1e9) if okb else 0.0
        if okb:
            # a trade needs a valid quote older than itself in exchange time, so the book must also cover the largest
            # admissible quote age before the window
            wh = eswin_ns[wi] + ip[11] * 1000000
            okb = _window_cum(ring_t, ring_c, ring_x, si, t - wh, n_tw, tmp2)
            if okb:
                cov = min(cov, (cum[BD] - tmp2[BD]) / (wh / 1e9))
        if okb and cov >= min_cov - 1e-7:
            base = np.zeros(5)
            if k >= 0:
                for j in range(5):
                    base[j] = es_c[k, j]
            dq = es_cum[2] - base[2]
            dd_ = es_cum[3] - base[3]
            out[iq, c] = (es_cum[0] - base[0]) / dq if dq > 0 else np.nan
            out[iq, c + 1] = (es_cum[1] - base[1]) / dd_ if dd_ > 0 else np.nan
            out[iq, c + 2] = es_cum[4] - base[4]
        else:
            out[iq, c] = np.nan
            out[iq, c + 1] = np.nan
            out[iq, c + 2] = np.nan
        c += 3
    # ------------------------------------------------------------- matured markouts (by maturity time)
    for h in range(nh):
        for wi in range(mowin_ns.shape[0]):
            s = t - mowin_ns[wi]
            k = _ring_find(mo_t, si[I_MOHEAD], si[I_MOCOUNT], s)
            okb = _window_cum(ring_t, ring_c, ring_x, si, s, n_tw, tmp)
            if k < 0:
                if si[I_MOCOUNT] == mo_t.shape[0]:
                    okb = False
                elif si[I_MOCOUNT] > 0:
                    k = (si[I_MOHEAD] - si[I_MOCOUNT]) % mo_t.shape[0]
            cov = (cum[BD] - tmp[BD]) / (mowin_ns[wi] / 1e9) if okb else 0.0
            if okb:
                # a markout maturing in the window started h earlier, so the book must cover w + h, plus the largest
                # admissible quote age (see effective spread)
                wh = mowin_ns[wi] + mo_h_ns[h] + ip[11] * 1000000
                okb = _window_cum(ring_t, ring_c, ring_x, si, t - wh, n_tw, tmp2)
                if okb:
                    cov = min(cov, (cum[BD] - tmp2[BD]) / (wh / 1e9))
            if okb and cov >= min_cov - 1e-7:
                b0 = mo_c[k, 3 * h] if k >= 0 else 0.0
                b1 = mo_c[k, 3 * h + 1] if k >= 0 else 0.0
                b2 = mo_c[k, 3 * h + 2] if k >= 0 else 0.0
                dq = mo_cum[3 * h + 1] - b1
                out[iq, c] = (mo_cum[3 * h] - b0) / dq if dq > 0 else np.nan
                out[iq, c + 1] = mo_cum[3 * h + 2] - b2
            else:
                out[iq, c] = np.nan
                out[iq, c + 1] = np.nan
            c += 2
    return c


@njit(cache=True)
def rebase(cum, ring_c, si, es_cum, es_c, mo_cum, mo_c):
    """Subtract current cumulative values from every stored cumulative record (all window results are
    differences, so they are unchanged); keeps float64 magnitudes small over a multi-month replay."""
    R = ring_c.shape[0]
    for i in range(si[I_RCOUNT]):
        k = (si[I_RHEAD] - 1 - i) % R
        for j in range(cum.shape[0]):
            ring_c[k, j] -= cum[j]
    for j in range(cum.shape[0]):
        cum[j] = 0.0
    RE = es_c.shape[0]
    for i in range(si[I_ESCOUNT]):
        k = (si[I_ESHEAD] - 1 - i) % RE
        for j in range(es_cum.shape[0]):
            es_c[k, j] -= es_cum[j]
    for j in range(es_cum.shape[0]):
        es_cum[j] = 0.0
    RM = mo_c.shape[0]
    for i in range(si[I_MOCOUNT]):
        k = (si[I_MOHEAD] - 1 - i) % RM
        for j in range(mo_cum.shape[0]):
            mo_c[k, j] -= mo_cum[j]
    for j in range(mo_cum.shape[0]):
        mo_cum[j] = 0.0
