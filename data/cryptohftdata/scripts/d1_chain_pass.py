"""D1 step 1: effect of the resolver corrections and the list of true gaps, over the WHOLE history, without touching any
dataset. Only the message metadata of each hourly book file is used (no book state is built).

Three resolvers see the same messages:
  legacy         strict, behaviour before 2026-11-04 (no client buffer while unanchored; a snapshot ahead of the chain
                 cannot bridge an unpublished update)
  strict         strict, corrected
  chain_nosnap   rebuild, snapshots ignored: continuous-chain segments, i.e. what an empty-book restart would have to
                 live with, including the months without anchors
Outputs in derived/d1/chain_pass/: hours.parquet (per hour and resolver), transitions_<variant>.parquet, events_<variant>.parquet
Usage: .venv/bin/python scripts/d1_chain_pass.py [workers]
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402
from chdds.book import chain as CH  # noqa: E402
from chdds.book.driver import load_book_hour  # noqa: E402

META = ("recv", "evt", "et", "U", "u", "pu", "L", "hash")
VARIANTS = {"legacy": dict(heal=0, probe=0, legacy=1), "strict": dict(heal=0, probe=0, legacy=0),
            "chain_nosnap": dict(heal=2, probe=3, legacy=0)}
TIMEOUT_NS = 60 * 10**9


def load_meta(day_iso, hour):
    b = load_book_hour(day_iso, hour)
    if b is None:
        return None
    m = {k: b[k] for k in META}
    upd = b["et"] == CH.ET_UPDATE
    rows = np.diff(b["row_start"])
    e = np.sort(b["evt"][upd])
    d = np.diff(e) if e.size > 1 else np.zeros(1, np.int64)
    d = d[d > 0] if (d > 0).any() else np.zeros(1, np.int64)
    m["stats"] = dict(n_upd=int(upd.sum()), n_snap=int((b["et"] == CH.ET_SNAPSHOT).sum()),
                      rows_upd=int(rows[upd].sum()), cad_p50=float(np.median(d)), cad_p99=float(np.percentile(d, 99)),
                      cad_max=float(d.max()))
    return m


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    out = P.DERIVED / "d1" / "chain_pass"
    out.mkdir(parents=True, exist_ok=True)
    hours = list(P.iter_hours(P.HISTORY_START, P.HISTORY_END))
    st = {v: CH.new_state() for v in VARIANTS}
    carry = {v: None for v in VARIANTS}
    rec_h, trans, events = [], {v: [] for v in VARIANTS}, {v: [] for v in VARIANTS}
    t0 = time.time()
    ahead = 2 * workers
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {}

        def sub(i):
            if i < len(hours):
                futs[i] = ex.submit(load_meta, hours[i][0].isoformat(), hours[i][1])

        for i in range(ahead):
            sub(i)
        for i, (d, h) in enumerate(hours):
            sub(i + ahead)
            cur = futs.pop(i).result()
            h0 = P.hour_start_ns(d, h)
            base = dict(day=d.isoformat(), hour=h, file=cur is not None)
            if cur is not None:
                base.update(cur["stats"])
            for v, kw in VARIANTS.items():
                cst, ru, rh = st[v]
                if cur is None:
                    if cst[CH.ST_MODE] != CH.M_NONE:
                        trans[v].append((h0, "invalidate_missing_file"))
                    cst[CH.ST_MODE] = CH.M_NONE
                    cst[CH.ST_GAP_SINCE] = -1
                    carry[v] = None
                    rec_h.append(dict(base, variant=v, mode_end=int(cst[CH.ST_MODE])))
                    continue
                c = carry[v]
                ncar = 0 if c is None else c["recv"].size
                b = cur if c is None else {k: np.concatenate([c[k], cur[k]]) for k in META}
                r = CH.resolve(b["recv"], b["et"], b["U"], b["u"], b["pu"], b["L"], b["hash"], cst, ru, rh, ncar,
                               TIMEOUT_NS, kw["heal"], kw["probe"], kw["legacy"], 0)
                seq_msg, seq_time, seq_kind, _, action, ev_kind, ev_t0, ev_t1, ev_u0, ev_u1, pend = r
                pend = np.sort(pend)
                carry[v] = {k: b[k][pend] for k in META} if pend.size else None
                sel = seq_kind != CH.Q_UPDATE
                for tt, kk in zip(seq_time[sel].tolist(), seq_kind[sel].tolist()):
                    trans[v].append((tt, {CH.Q_ANCHOR: "anchor", CH.Q_VERIFY: "verify", CH.Q_INVALIDATE: "invalidate",
                                          CH.Q_EMPTY: "empty"}.get(kk, str(kk))))
                for j in range(ev_kind.size):
                    events[v].append((CH.EVENT_NAMES[int(ev_kind[j])], int(ev_t0[j]), int(ev_t1[j]), int(ev_u0[j]),
                                      int(ev_u1[j])))
                ac = np.bincount(action, minlength=len(CH.ACTION_NAMES))
                rec_h.append(dict(base, variant=v, mode_end=int(cst[CH.ST_MODE]),
                                  **{f"act_{n}": int(ac[k]) for k, n in enumerate(CH.ACTION_NAMES)}))
            if i % 500 == 0:
                print(d, h, f"{time.time() - t0:.0f}s", {v: len(trans[v]) for v in VARIANTS}, flush=True)
            if i % 1000 == 999 or i == len(hours) - 1:
                pl.from_dicts(rec_h, infer_schema_length=None).write_parquet(out / "hours.parquet")
                for v in VARIANTS:
                    pl.DataFrame(trans[v], schema=["t", "kind"], orient="row").write_parquet(out / f"transitions_{v}.parquet")
                    pl.DataFrame(events[v], schema=["kind", "t0", "t1", "u0", "u1"], orient="row").write_parquet(
                        out / f"events_{v}.parquet")
    (out / "DONE.json").write_text(json.dumps(dict(secs=time.time() - t0, hours=len(hours))))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
