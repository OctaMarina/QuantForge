"""Canonical economic trade stream (single source for all four bar engines).

Rules (report §3.3, §7.1):
  * identity = trade_id; identical copies -> keep the first authentic reception (earliest
    received_time that is not synthetic); same id with different payload -> quarantine
    (excluded from the economic stream, logged, continuity segment broken).
  * order_type NA with price=qty=0 are not executions: excluded from prices/volume/count,
    kept for ID continuity. Non-positive price/qty with another type: excluded + logged.
  * canonical order = availability order (received_time, trade_id); market time kept.
  * continuity: a missing trade_id (after counting NA ids) breaks the segment; the
    availability interval between the neighbouring trades is recorded as a trade gap.
Output: derived/canonical/trades/YYYY-MM-DD.parquet + trades_gaps.parquet + trades_build.json
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import paths as P
from ..audit.run import staged_path

OUT = P.DERIVED / "canonical" / "trades"
SCHEMA = pa.schema([("seq", pa.int64()), ("recv_ns", pa.int64()), ("trade_time_ms", pa.int64()),
                    ("trade_id", pa.int64()), ("price_e2", pa.int64()), ("qty_e3", pa.int64()),
                    ("sign", pa.int8()), ("seg", pa.int32()), ("n_copies", pa.int8()),
                    ("src_hour", pa.int32())])


def _load_hour(day, hour):
    sp = staged_path("trades", day, hour)
    if not sp.exists():
        return None
    return pq.read_table(sp).to_pydict() if False else pq.read_table(sp)


def build(start: date = P.HISTORY_START, end: date = P.HISTORY_END, log=print):
    OUT.mkdir(parents=True, exist_ok=True)
    seq = 0
    seg = 0
    last_id = None          # last trade_id (any type, incl. NA) in id order seen so far
    last_recv = None
    last_tt = None
    prev_ids = np.zeros(0, np.int64)
    gaps = []
    quarantine = []
    counters = dict(rows=0, copies_dropped=0, cross_file_dups=0, na=0, nonpos=0, conflicts=0, economic=0,
                    synthetic_recv=0, id_gaps=0, missing_ids=0, missing_files=0, inversions=0)
    t0 = time.time()
    d = start
    while d <= end:
        day_tables = []
        for h in range(24):
            sp = staged_path("trades", d, h)
            hidx = int(P.hour_index(P.hour_start_ns(d, h)))
            if not sp.exists():
                if P.raw_path(d, h, "trades").exists():
                    raise RuntimeError(f"staged trades missing for {d} {h} — run the audit first")
                counters["missing_files"] += 1
                if last_id is not None:
                    # a missing hour file is a continuity break even if ids later look contiguous
                    gaps.append(dict(kind="missing_file", t0=int(last_recv), t1=None, id0=int(last_id), id1=None,
                                     missing=None, hour=f"{d} {h:02d}"))
                    last_id = None
                continue
            t = pq.read_table(sp)
            n = t.num_rows
            counters["rows"] += n
            recv = t["recv_ns"].to_numpy()
            evt = t["event_ms"].to_numpy()
            tt = t["trade_time_ms"].to_numpy()
            tid = t["trade_id"].to_numpy()
            px = t["price_e2"].to_numpy()
            qx = t["qty_e3"].to_numpy()
            ibm = t["is_buyer_maker"].to_numpy()
            ot = t["order_type"].to_numpy()
            pflags = t["parse_flags"].to_numpy()
            synth = recv == evt * P.MS_NS
            # --- copies within file: sort by (id, synthetic, recv); first row per id is canonical
            o = np.lexsort((recv, synth, tid))
            tid_o = tid[o]
            first = np.ones(n, bool)
            first[1:] = tid_o[1:] != tid_o[:-1]
            canon = o[first]
            ncop = np.diff(np.append(np.nonzero(first)[0], n)).astype(np.int8)
            counters["copies_dropped"] += int(n - canon.size)
            # conflicts: any copy with different payload than the canonical
            if canon.size < n:
                grp = np.cumsum(first) - 1
                ci = canon[grp]
                diff = ((px[o] != px[ci]) | (qx[o] != qx[ci]) | (tt[o] != tt[ci]) | (ibm[o] != ibm[ci]) |
                        (ot[o] != ot[ci]))
                bad_ids = np.unique(tid_o[diff])
            else:
                bad_ids = np.zeros(0, np.int64)
            # --- cross-file duplicates (ids already emitted from the previous hour)
            ids_c = tid[canon]
            dupx = np.isin(ids_c, prev_ids) if prev_ids.size else np.zeros(canon.size, bool)
            counters["cross_file_dups"] += int(dupx.sum())
            keep = canon[~dupx]
            ncop = ncop[~dupx]
            # --- id continuity (id order, all types incl. NA)
            ids_k = tid[keep]
            oi = np.argsort(ids_k, kind="stable")
            ids_sorted = ids_k[oi]
            rec_sorted = recv[keep][oi]
            tt_sorted = tt[keep][oi]
            seg_start_id = []
            if last_id is not None and ids_sorted.size and ids_sorted[0] != last_id + 1:
                miss = int(ids_sorted[0] - last_id - 1)
                gaps.append(dict(kind="id_gap", t0=int(last_recv), t1=int(rec_sorted[0]), id0=int(last_id),
                                 id1=int(ids_sorted[0]), missing=miss, m0=int(last_tt), m1=int(tt_sorted[0]),
                                 hour=f"{d} {h:02d}"))
                seg_start_id.append(ids_sorted[0])
            elif last_id is None and ids_sorted.size:
                seg_start_id.append(ids_sorted[0])
            dd = np.diff(ids_sorted)
            gi = np.nonzero(dd > 1)[0]
            for i in gi:
                gaps.append(dict(kind="id_gap", t0=int(rec_sorted[i]), t1=int(rec_sorted[i + 1]), id0=int(ids_sorted[i]),
                                 id1=int(ids_sorted[i + 1]), missing=int(dd[i] - 1), m0=int(tt_sorted[i]),
                                 m1=int(tt_sorted[i + 1]), hour=f"{d} {h:02d}"))
                seg_start_id.append(ids_sorted[i + 1])
            # conflicting ids also break continuity (their content is unknown)
            for b in bad_ids:
                quarantine.append(dict(trade_id=int(b), hour=f"{d} {h:02d}"))
                seg_start_id.append(int(b) + 1)
            counters["conflicts"] += int(bad_ids.size)
            if ids_sorted.size:
                last_id = int(ids_sorted[-1])
                last_recv = int(rec_sorted[-1])
                last_tt = int(tt_sorted[-1])
            prev_ids = ids_k
            # --- economic filter
            is_na = ot[keep] == 1
            pos = (px[keep] > 0) & (qx[keep] > 0)
            bad = np.isin(tid[keep], bad_ids) | (pflags[keep] != 0)
            counters["na"] += int(is_na.sum())
            counters["nonpos"] += int(np.count_nonzero(~is_na & ~pos))
            ek = (~is_na) & pos & ~bad
            sel = keep[ek]
            ncop_e = ncop[ek]
            # --- availability order
            ao = np.lexsort((tid[sel], recv[sel]))
            sel = sel[ao]
            ncop_e = ncop_e[ao]
            counters["inversions"] += int(np.count_nonzero(np.diff(tid[sel]) < 0))
            counters["synthetic_recv"] += int(synth[sel].sum())
            # --- segment ids: a segment starts at the first trade (in availability order) whose id >= start id
            ids_e = tid[sel]
            seg_arr = np.zeros(sel.size, np.int32)
            if seg_start_id:
                starts = np.array(sorted(seg_start_id), np.int64)
                # number of segment starts <= id  (in id terms) -> segment increments
                k = np.searchsorted(starts, ids_e, side="right")
                seg_arr = (seg + k).astype(np.int32)
                seg += len(starts)
            else:
                seg_arr[:] = seg
            # ensure segment ids non-decreasing in availability order (inversions across a gap are flagged)
            seg_arr = np.maximum.accumulate(seg_arr)
            m = sel.size
            counters["economic"] += m
            day_tables.append(pa.table({
                "seq": np.arange(seq, seq + m, dtype=np.int64), "recv_ns": recv[sel], "trade_time_ms": tt[sel],
                "trade_id": ids_e, "price_e2": px[sel], "qty_e3": qx[sel],
                "sign": np.where(ibm[sel], -1, 1).astype(np.int8), "seg": seg_arr, "n_copies": ncop_e,
                "src_hour": np.full(m, hidx, np.int32)}, schema=SCHEMA))
            seq += m
        if day_tables:
            tab = pa.concat_tables(day_tables)
            # availability order must hold across hour files too
            r = tab["recv_ns"].to_numpy()
            if np.any(np.diff(r) < 0):
                counters.setdefault("cross_file_recv_regress", 0)
                counters["cross_file_recv_regress"] += int(np.count_nonzero(np.diff(r) < 0))
            pq.write_table(tab, OUT / f"{d.isoformat()}.parquet", compression="zstd", compression_level=3)
        if d.day == 1 or d == end:
            log(f"  trades canonical {d} seq={seq:,} seg={seg} gaps={len(gaps)} {time.time()-t0:.0f}s")
        d += timedelta(days=1)
    counters["id_gaps"] = sum(1 for g in gaps if g["kind"] == "id_gap")
    counters["missing_ids"] = int(sum(g["missing"] or 0 for g in gaps if g["kind"] == "id_gap"))
    counters["segments"] = seg
    pq.write_table(pa.Table.from_pylist(gaps) if gaps else pa.table({"kind": pa.array([], pa.string())}),
                   P.DERIVED / "canonical" / "trades_gaps.parquet")
    (P.DERIVED / "canonical" / "trades_quarantine.json").write_text(json.dumps(quarantine))
    (P.DERIVED / "canonical" / "trades_build.json").write_text(json.dumps(counters, indent=1))
    log(json.dumps(counters))
    return counters
