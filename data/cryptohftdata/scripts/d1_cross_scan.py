"""D1: exhaustive scan of crossed states in the stream itself, over the whole history.

A book started empty holds only reported levels, so if the exchange stream's book is never crossed this book cannot cross.
Any crossing is therefore a property of the stream (or an undetected hole), and the protocol certificate is not guaranteed
while it lasts. Independent minimal book, one fresh start per UTC day, restart after every true gap, snapshots ignored.
Output: derived/d1/cross/<day>.parquet (one row per episode), derived/d1/cross/_days.parquet
Usage: .venv/bin/python scripts/d1_cross_scan.py [workers]
"""
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402
from chdds.book import chain as CH  # noqa: E402
from chdds.book.driver import _concat_batch, _take_pending, load_book_hour  # noqa: E402

NT = 4_000_000
OUT = P.DERIVED / "d1" / "cross"


@njit(cache=True)
def scan(seq_msg, seq_time, seq_kind, row_start, side, idx, qty, off, qb, qa, s, ep):
    """s: 0 bb, 1 ba, 2 crossing since (ns, -1 none), 3 worst depth in ticks, 4 n episodes, 5 messages, 6 crossed messages"""
    for k in range(seq_msg.shape[0]):
        kind = seq_kind[k]
        if kind == 5:
            qb[:] = 0
            qa[:] = 0
            s[0] = -1
            s[1] = -1
            if s[2] >= 0 and s[4] < ep.shape[0]:
                ep[s[4], 0] = s[2]
                ep[s[4], 1] = seq_time[k]
                ep[s[4], 2] = s[3]
                ep[s[4], 3] = 1
                s[4] += 1
            s[2] = -1
            s[3] = 0
            continue
        if kind != 0:
            continue
        m = seq_msg[k]
        bb = s[0]
        ba = s[1]
        for r in range(row_start[m], row_start[m + 1]):
            if off[r] or idx[r] < 0 or idx[r] >= NT:
                continue
            p = idx[r]
            if side[r] == 0:
                qb[p] = qty[r]
                if qty[r] > 0 and p > bb:
                    bb = p
            elif side[r] == 1:
                qa[p] = qty[r]
                if qty[r] > 0 and (ba < 0 or p < ba):
                    ba = p
        if bb >= 0 and qb[bb] == 0:
            while bb >= 0 and qb[bb] == 0:
                bb -= 1
        if ba >= 0 and qa[ba] == 0:
            while ba < NT and qa[ba] == 0:
                ba += 1
            if ba >= NT:
                ba = -1
        s[0] = bb
        s[1] = ba
        s[5] += 1
        t = seq_time[k]
        if bb >= 0 and ba >= 0 and bb >= ba:
            s[6] += 1
            if s[2] < 0:
                s[2] = t
                s[3] = 0
            if bb - ba > s[3]:
                s[3] = bb - ba
        elif s[2] >= 0:
            if s[4] < ep.shape[0]:
                ep[s[4], 0] = s[2]
                ep[s[4], 1] = t
                ep[s[4], 2] = s[3]
                ep[s[4], 3] = 0
                s[4] += 1
            s[2] = -1
            s[3] = 0


def run_day(d):
    p = OUT / f"{d.isoformat()}.parquet"
    if p.exists():
        return d, -1
    qb = np.zeros(NT, np.int64)
    qa = np.zeros(NT, np.int64)
    s = np.array([-1, -1, -1, 0, 0, 0, 0], np.int64)
    ep = np.zeros((200000, 4), np.int64)
    cst, ru, rh = CH.new_state()
    carry = None
    gaps = 0
    for h in range(24):
        cur = load_book_hour(d.isoformat(), h)
        if cur is None:
            cst[CH.ST_MODE] = CH.M_NONE
            cst[CH.ST_GAP_SINCE] = -1
            carry = None
            continue
        b, ncar = _concat_batch(carry, cur)
        r = CH.resolve(b["recv"], b["et"], b["U"], b["u"], b["pu"], b["L"], b["hash"], cst, ru, rh, ncar, 60 * 10**9, 2, 3, 0, 0)
        carry = _take_pending(b, r[-1])
        gaps += int((r[5] == CH.E_GAP).sum())
        scan(r[0], r[1], r[2], b["row_start"], b["side"], b["idx"], b["qty"], b["off"], qb, qa, s, ep)
    n = int(s[4])
    pl.DataFrame({"t0": ep[:n, 0], "t1": ep[:n, 1], "worst_ticks": ep[:n, 2], "ended_by_restart": ep[:n, 3]}).write_parquet(p)
    (OUT / f"{d.isoformat()}.meta").write_text(f"{int(s[5])} {int(s[6])} {gaps}")
    return d, n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    days = [P.HISTORY_START + timedelta(days=i) for i in range((P.HISTORY_END - P.HISTORY_START).days + 1)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        futs = {ex.submit(run_day, d): d for d in days}
        tot = 0
        for i, f in enumerate(as_completed(futs)):
            try:
                _, n = f.result()
                tot += max(n, 0)
                if i % 25 == 0:
                    print(i, futs[f], n, tot, flush=True)
            except Exception as e:
                print("FAILED", futs[f], repr(e), flush=True)
    print("done", tot)


if __name__ == "__main__":
    main()
