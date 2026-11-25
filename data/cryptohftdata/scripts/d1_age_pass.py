"""D1 large-sample complement to the explicit parallel books.

If the chain is continuous, a book started empty at T holds exactly the levels reported since T, so
    E_T(t) = R(t) restricted to levels whose last report is >= T.
Top-K of E_T(t) therefore equals top-K of R(t) iff the oldest of those K levels was last reported at or after T, that is
iff max_age_K(t) <= t - T. One pass over the reference gives the answer for EVERY possible start T at once, with a book
implementation independent of the production kernel. The identity itself is checked against the explicit runs
(scripts/d1_analyze.py, section identity).

For each continuous reference segment (from derived/d1/chain_pass, corrected strict resolver) it writes to
derived/d1/age/<segment>/: ages.parquet (1 s: age of the K-th oldest prefix, per side) and short.parquet (10 s: share of
each band's depth held by levels younger than each delta, only where the band is inside the certified domain).
Usage: .venv/bin/python scripts/d1_age_pass.py [workers] [max_hours_per_segment]
"""
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402
from chdds.book import chain as CH  # noqa: E402
from chdds.book.driver import _concat_batch, _take_pending, load_book_hour  # noqa: E402

NT = 4_000_000
KMAX = 50
KS = (1, 5, 10, 50)
CERT_LEVELS = 500
DELTAS_S = np.array([5, 10, 30, 60, 120, 300, 600, 900, 1800, 3600], np.int64)
BANDS_X100 = np.array([100, 500, 1000, 2500], np.int64)      # 1, 5, 10, 25 bps
OUT = P.DERIVED / "d1" / "age"
# state indices
S_BB, S_BA, S_LO, S_HI, S_TA, S_ANCH, S_HEXT, S_LEXT, S_LAST, S_NI, S_NANCH, S_NV = range(12)


@njit(cache=True)
def _frontier(s, lb, la):
    ta = s[S_TA]
    if s[S_BB] >= s[S_LO] and s[S_BB] > s[S_HEXT]:
        s[S_HEXT] = s[S_BB]
    if s[S_BA] >= 0 and s[S_BA] <= s[S_HI] and (s[S_LEXT] < 0 or s[S_BA] < s[S_LEXT]):
        s[S_LEXT] = s[S_BA]
    f = s[S_HI]
    if s[S_HEXT] > f:
        f = s[S_HEXT]
    while f + 1 < NT and la[f + 1] >= ta:
        f += 1
    s[S_HI] = f
    g = s[S_LO]
    if s[S_LEXT] >= 0 and s[S_LEXT] < g:
        g = s[S_LEXT]
    while g - 1 >= 0 and lb[g - 1] >= ta:
        g -= 1
    s[S_LO] = g


@njit(cache=True)
def _limit(ticks, n, keep_high, lim):
    """Tick of the lim-th best level (deepest certified) among n ticks."""
    a = np.sort(ticks[:n])
    if n <= lim:
        return a[0] if keep_high else a[n - 1]
    return a[n - lim] if keep_high else a[lim - 1]


@njit(cache=True)
def _best_bid(q, start):
    p = start
    while p >= 0 and q[p] == 0:
        p -= 1
    return p


@njit(cache=True)
def _best_ask(q, start):
    p = start
    while p < NT and q[p] == 0:
        p += 1
    return p if p < NT else -1


@njit(cache=True)
def _sample(i, t, s, qb, qa, lb, la, ages, spread, ok, tas):
    tas[i] = s[S_TA]
    bb = s[S_BB]
    ba = s[S_BA]
    good = s[S_ANCH] == 1 and bb >= 0 and ba >= 0 and bb < ba and bb >= s[S_LO] and ba <= s[S_HI] and t - s[S_LAST] <= 1000000000
    ok[i] = 1 if good else 0
    if not good:
        return
    spread[i] = ba - bb
    ta = s[S_TA]
    n = 0
    p = bb
    while n < KMAX and p >= s[S_LO]:
        if qb[p] > 0:
            ages[i, n] = (t - lb[p]) // 1000000 if lb[p] >= ta else (t - ta) // 1000000 + 1
            n += 1
        p -= 1
    n = 0
    p = ba
    while n < KMAX and p <= s[S_HI]:
        if qa[p] > 0:
            ages[i, KMAX + n] = (t - la[p]) // 1000000 if la[p] >= ta else (t - ta) // 1000000 + 1
            n += 1
        p += 1


@njit(cache=True)
def _short(i, t, s, qb, qa, lb, la, deltas, bands, tot, frac):
    bb = s[S_BB]
    ba = s[S_BA]
    ta = s[S_TA]
    m2 = bb + ba
    nd = deltas.shape[0]
    acc = np.zeros(nd, np.int64)
    for j in range(bands.shape[0]):
        lo = (m2 * (1000000 - bands[j])) // 2000000 + 1
        hi = (m2 * (1000000 + bands[j]) - 1) // 2000000
        for sd in range(2):
            o = j * 2 + sd
            if (sd == 0 and lo < s[S_LO]) or (sd == 1 and hi > s[S_HI]):
                continue
            total = 0
            acc[:] = 0
            a = lo if sd == 0 else ba
            b = bb if sd == 0 else hi
            for p in range(a, b + 1):
                q = qb[p] if sd == 0 else qa[p]
                if q > 0:
                    total += q
                    lr = lb[p] if sd == 0 else la[p]
                    age = t - lr if lr >= ta else t - ta + 1
                    for m in range(nd):
                        if age <= deltas[m]:
                            acc[m] += q
            tot[i, o] = total / 1000.0
            if total > 0:
                for m in range(nd):
                    frac[i, o, m] = acc[m] / total


@njit(cache=True)
def age_kernel(seq_msg, seq_time, seq_kind, seq_aux, row_start, side, idx, qty, off, qb, qa, lb, la, s, samp, ages, spread,
               ok, tas, short_every, deltas, bands, tot, frac, vlog):
    i = 0
    n = seq_msg.shape[0]
    ns = samp.shape[0]
    bt = np.empty(200000, np.int64)
    at = np.empty(200000, np.int64)
    for k in range(n):
        t = seq_time[k]
        while s[S_NI] < ns and samp[s[S_NI]] < t:
            j = s[S_NI]
            _sample(j, samp[j], s, qb, qa, lb, la, ages, spread, ok, tas)
            if j % short_every == 0 and ok[j] == 1:
                _short(j // short_every, samp[j], s, qb, qa, lb, la, deltas, bands, tot, frac)
            s[S_NI] += 1
        kind = seq_kind[k]
        m = seq_msg[k]
        if kind == 3:      # invalidate
            s[S_ANCH] = 0
            continue
        if kind == 5:      # empty restart (no-reference mode): only what is visible since T can be measured
            qb[:] = 0
            qa[:] = 0
            s[S_TA] = t
            s[S_ANCH] = 1
            s[S_LO] = 0
            s[S_HI] = NT - 1
            s[S_BB] = -1
            s[S_BA] = -1
            s[S_HEXT] = -1
            s[S_LEXT] = -1
            s[S_LAST] = t
            s[S_NANCH] += 1
            continue
        if kind == 1 or kind == 2:
            rs = row_start[m]
            re_ = row_start[m + 1]
            nb = 0
            na = 0
            for r in range(rs, re_):
                if off[r] or idx[r] < 0 or idx[r] >= NT or qty[r] <= 0:
                    continue
                if side[r] == 0:
                    bt[nb] = idx[r]
                    nb += 1
                elif side[r] == 1:
                    at[na] = idx[r]
                    na += 1
            if nb == 0 or na == 0:
                continue
            lo = _limit(bt, nb, True, 500)
            hi = _limit(at, na, False, 500)
            if kind == 1 or s[S_ANCH] == 0:
                qb[:] = 0
                qa[:] = 0
                for r in range(rs, re_):
                    if off[r] or idx[r] < 0 or idx[r] >= NT:
                        continue
                    if side[r] == 0:
                        qb[idx[r]] = qty[r]
                    elif side[r] == 1:
                        qa[idx[r]] = qty[r]
                s[S_TA] = t
                s[S_NANCH] += 1
                s[S_LO] = lo
                s[S_HI] = hi
                s[S_HEXT] = hi
                s[S_LEXT] = lo
                s[S_ANCH] = 1
                s[S_BB] = _best_bid(qb, np.max(bt[:nb]))
                s[S_BA] = _best_ask(qa, np.min(at[:na]))
                s[S_LAST] = t
            else:
                # verification, independent of the production kernel: inside both domains the state must equal the snapshot
                # (levels touched by the overlay message are newer than the snapshot and are skipped)
                a = seq_aux[k]
                cb = max(lo, s[S_LO])
                ca = min(hi, s[S_HI])
                mis = 0
                nsb = 0
                nsa = 0
                for r in range(rs, re_):
                    if off[r] or idx[r] < 0 or idx[r] >= NT:
                        continue
                    if a >= 0:
                        skip = False
                        for x in range(row_start[a], row_start[a + 1]):
                            if idx[x] == idx[r] and side[x] == side[r]:
                                skip = True
                        if skip:
                            continue
                    if side[r] == 0 and idx[r] >= cb:
                        if qty[r] > 0:
                            nsb += 1
                        if qb[idx[r]] != qty[r]:
                            mis += 1
                    elif side[r] == 1 and idx[r] <= ca:
                        if qty[r] > 0:
                            nsa += 1
                        if qa[idx[r]] != qty[r]:
                            mis += 1
                nob = 0
                for p in range(cb, s[S_BB] + 1):
                    if qb[p] > 0:
                        nob += 1
                noa = 0
                for p in range(s[S_BA], ca + 1):
                    if qa[p] > 0:
                        noa += 1
                if a >= 0:
                    for x in range(row_start[a], row_start[a + 1]):
                        if idx[x] >= 0 and idx[x] < NT and not off[x]:
                            if side[x] == 0 and idx[x] >= cb and qb[idx[x]] > 0:
                                nob -= 1
                            elif side[x] == 1 and idx[x] <= ca and qa[idx[x]] > 0:
                                noa -= 1
                mis += abs(nob - nsb) + abs(noa - nsa)
                j = s[S_NV]
                if j < vlog.shape[0]:
                    vlog[j, 0] = t
                    vlog[j, 1] = mis
                    vlog[j, 2] = nsb + nsa
                    vlog[j, 3] = 1 if a >= 0 else 0
                    s[S_NV] += 1
                # levels outside the domain are adopted (age unknown -> older than T_a)
                if lo < s[S_LO]:
                    for p in range(lo, s[S_LO]):
                        qb[p] = 0
                if hi > s[S_HI]:
                    for p in range(s[S_HI] + 1, hi + 1):
                        qa[p] = 0
                for r in range(rs, re_):
                    if off[r] or idx[r] < 0 or idx[r] >= NT:
                        continue
                    if side[r] == 0 and idx[r] < s[S_LO] and idx[r] >= lo:
                        qb[idx[r]] = qty[r]
                    elif side[r] == 1 and idx[r] > s[S_HI] and idx[r] <= hi:
                        qa[idx[r]] = qty[r]
                if lo < s[S_LO]:
                    s[S_LO] = lo
                if hi > s[S_HI]:
                    s[S_HI] = hi
                a = seq_aux[k]
                if a >= 0:
                    for r in range(row_start[a], row_start[a + 1]):
                        if off[r] or idx[r] < 0 or idx[r] >= NT:
                            continue
                        if side[r] == 0:
                            qb[idx[r]] = qty[r]
                        elif side[r] == 1:
                            qa[idx[r]] = qty[r]
            continue
        if kind != 0 or s[S_ANCH] == 0:
            continue
        bb = s[S_BB]
        ba = s[S_BA]
        for r in range(row_start[m], row_start[m + 1]):
            if off[r] or idx[r] < 0 or idx[r] >= NT:
                continue
            p = idx[r]
            if side[r] == 0:
                qb[p] = qty[r]
                lb[p] = t
                if qty[r] > 0 and p > bb:
                    bb = p
            elif side[r] == 1:
                qa[p] = qty[r]
                la[p] = t
                if qty[r] > 0 and (ba < 0 or p < ba):
                    ba = p
        if bb >= 0 and qb[bb] == 0:
            bb = _best_bid(qb, bb)
        if ba >= 0 and qa[ba] == 0:
            ba = _best_ask(qa, ba)
        s[S_BB] = bb
        s[S_BA] = ba
        s[S_LAST] = t
        if bb >= 0 and ba >= 0 and bb < ba:
            _frontier(s, lb, la)
    while s[S_NI] < ns and n > 0 and samp[s[S_NI]] <= seq_time[n - 1]:
        j = s[S_NI]
        _sample(j, samp[j], s, qb, qa, lb, la, ages, spread, ok, tas)
        s[S_NI] += 1


def hours_between(t0, t1):
    out = []
    t = t0 - t0 % P.HOUR_NS
    while t <= t1:
        dt = datetime.fromtimestamp(t / 1e9, timezone.utc)
        out.append((dt.date(), dt.hour))
        t += P.HOUR_NS
    return out


def run_segment(seg_id, t0, t1, samples=None, write=True, short_every=10, empty=False):
    """Replays one reference segment. samples: optional explicit sample times (identity check).
    empty=True: no reference at all. The book starts empty and restarts after every true gap, snapshots are ignored, and
    ages describe only the levels it can see. For delta well below the elapsed time this estimates the same quantity
    without a reference (the method is itself checked in the months that have one)."""
    d = (OUT.parent / "age_empty" if empty else OUT) / seg_id
    if write and (d / "DONE").exists():
        return seg_id, 0.0, 0
    tstart = time.time()
    qb = np.zeros(NT, np.int64)
    qa = np.zeros(NT, np.int64)
    lb = np.zeros(NT, np.int64)
    la = np.zeros(NT, np.int64)
    cst, ru, rh = CH.new_state()
    carry = None
    res_a, res_t, res_s, res_ok, res_tot, res_fr, res_ts, res_ta = [], [], [], [], [], [], [], []
    vlog = np.zeros((200000, 4), np.int64)
    s = np.zeros(12, np.int64)
    s[S_BB] = -1
    s[S_BA] = -1
    s[S_LEXT] = -1
    for (dd, h) in hours_between(t0, t1):
        cur = load_book_hour(dd.isoformat(), h)
        h0 = P.hour_start_ns(dd, h)
        if samples is None:
            samp = np.arange(h0 + 10**9, h0 + P.HOUR_NS + 1, 10**9, dtype=np.int64)
        else:
            samp = samples[(samples >= h0) & (samples < h0 + P.HOUR_NS)]
        samp = samp[samp <= t1]
        if cur is None:
            cst[CH.ST_MODE] = CH.M_NONE
            cst[CH.ST_GAP_SINCE] = -1
            carry = None
            s[S_ANCH] = 0
            continue
        batch, ncar = _concat_batch(carry, cur)
        r = CH.resolve(batch["recv"], batch["et"], batch["U"], batch["u"], batch["pu"], batch["L"], batch["hash"], cst, ru, rh,
                       ncar, 60 * 10**9, 2 if empty else 0, 3 if empty else 0, 0, 0)
        seq_msg, seq_time, seq_kind, seq_aux = r[:4]
        carry = _take_pending(batch, r[-1])
        n = samp.size
        ages = np.full((n, 2 * KMAX), -1, np.int32)
        spread = np.zeros(n, np.int32)
        ok = np.zeros(n, np.int8)
        tas = np.zeros(n, np.int64)
        nsh = (n + short_every - 1) // short_every
        tot = np.full((nsh, 2 * BANDS_X100.size), np.nan, np.float32)
        fr = np.full((nsh, 2 * BANDS_X100.size, DELTAS_S.size), np.nan, np.float32)
        s[S_NI] = 0
        age_kernel(seq_msg, seq_time, seq_kind, seq_aux, batch["row_start"], batch["side"], batch["idx"], batch["qty"],
                   batch["off"], qb, qa, lb, la, s, samp, ages, spread, ok, tas, short_every, DELTAS_S * 10**9, BANDS_X100, tot, fr, vlog)
        res_a.append(ages)
        res_t.append(samp)
        res_s.append(spread)
        res_ok.append(ok)
        res_ta.append(tas)
        res_tot.append(tot)
        res_fr.append(fr)
        res_ts.append(samp[::short_every])
    t = np.concatenate(res_t)
    a = np.concatenate(res_a)
    sp = np.concatenate(res_s)
    okk = np.concatenate(res_ok).astype(bool)
    if not write:
        return t, a, okk
    d.mkdir(parents=True, exist_ok=True)
    nv = int(s[S_NV])
    pl.DataFrame({"t": vlog[:nv, 0], "mis": vlog[:nv, 1], "levels": vlog[:nv, 2], "overlay": vlog[:nv, 3]}).write_parquet(d / "verify.parquet")
    tab = {"t": t[okk], "ta": np.concatenate(res_ta)[okk], "spread_ticks": sp[okk]}
    v = a[okk]
    for si, sd in enumerate(("b", "a")):
        x = v[:, si * KMAX:(si + 1) * KMAX]
        present = x >= 0
        mx = np.maximum.accumulate(np.where(present, x, -1), axis=1)
        nlev = present.sum(axis=1)
        tab[f"nlev_{sd}"] = nlev.astype(np.int16)
        for K in KS:
            tab[f"age{K}_{sd}"] = np.where(nlev >= K, mx[:, K - 1], -1).astype(np.int32)
    pl.DataFrame(tab).write_parquet(d / "ages.parquet", compression="zstd")
    tt = np.concatenate(res_ts)
    tot = np.concatenate(res_tot)
    fr = np.concatenate(res_fr)
    good = np.isfinite(tot).any(axis=1)
    st = {"t": tt[good]}
    for j, b in enumerate((1, 5, 10, 25)):
        for si, sd in enumerate(("b", "a")):
            o = j * 2 + si
            st[f"tot{b}_{sd}"] = tot[good, o]
            for m, dl in enumerate(DELTAS_S):
                st[f"f{b}_{sd}_{dl}"] = fr[good, o, m]
    pl.DataFrame(st).write_parquet(d / "short.parquet", compression="zstd")
    (d / "DONE").write_text(json.dumps(dict(t0=t0, t1=t1, n=int(okk.sum()), n_all=int(okk.size), secs=time.time() - tstart,
                                            n_anchors=int(s[S_NANCH]), n_verify=nv, mismatches=int(vlog[:nv, 1].sum()))))
    return seg_id, time.time() - tstart, int(okk.sum())


def chunks(max_h=72):
    """Reference segments cut at a snapshot every ~max_h hours, so they can run in parallel. Each chunk re-anchors by itself."""
    tr = pl.read_parquet(P.DERIVED / "d1" / "chain_pass" / "transitions_strict.parquet").sort("t")
    snaps = tr.filter(pl.col("kind").is_in(["anchor", "verify"]))["t"].to_numpy()
    out = []
    for a, b in segments():
        c = a
        while True:
            j = np.searchsorted(snaps, c + max_h * P.HOUR_NS)
            if j >= snaps.size or snaps[j] >= b - P.HOUR_NS:
                out.append((c, b))
                break
            out.append((c, int(snaps[j])))
            c = int(snaps[j])
    return out


def segments():
    """Continuous reference segments of the corrected strict resolver: [anchor, invalidation)."""
    tr = pl.read_parquet(P.DERIVED / "d1" / "chain_pass" / "transitions_strict.parquet").sort("t")
    seg, cur = [], None
    for t, k in tr.select("t", "kind").rows():
        if k == "anchor":
            if cur is None:
                cur = t
        elif k in ("invalidate", "invalidate_missing_file") and cur is not None:
            seg.append((cur, t))
            cur = None
    if cur is not None:
        seg.append((cur, P.hour_start_ns(P.HISTORY_END, 23) + P.HOUR_NS))
    return seg


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    min_h = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    empty = len(sys.argv) > 3 and sys.argv[3] == "empty"
    if empty:
        # one UTC day out of every 5 over the whole history, plus the two days of the 2025-10-10 crash; the book is not
        # warm during the first hour of a chunk, which the analysis discards (elapsed filter)
        step = 5 * 24 * P.HOUR_NS
        a0 = P.hour_start_ns(P.HISTORY_START, 0)
        a1 = P.hour_start_ns(P.HISTORY_END, 23) + P.HOUR_NS - 1
        days = set(range(a0, a1, step)) | {P.hour_start_ns(date(2025, 10, 10), 0), P.hour_start_ns(date(2025, 10, 11), 0)}
        seg = [(a, min(a + 24 * P.HOUR_NS - 1, a1)) for a in sorted(days)]
    else:
        seg = [(a, b) for a, b in chunks() if (b - a) / P.HOUR_NS >= min_h]
        seg.sort(key=lambda x: x[0] - x[1])      # longest first
    print(len(seg), "segments", sum((b - a) / P.HOUR_NS for a, b in seg), "hours", flush=True)
    t0 = time.time()
    # spawn, not fork: the parent has already used polars, whose thread pool does not survive a fork
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        futs = {ex.submit(run_segment, f"s{a}", a, b, empty=empty): a for a, b in seg}
        for i, f in enumerate(as_completed(futs)):
            try:
                r = f.result()
                if i % 20 == 0:
                    print(i, r, f"{time.time() - t0:.0f}s", flush=True)
            except Exception as e:
                print("FAILED", futs[f], repr(e), flush=True)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
