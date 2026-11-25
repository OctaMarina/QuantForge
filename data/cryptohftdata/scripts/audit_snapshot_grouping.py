"""Exhaustive audit of how snapshots are written in the raw files, independent of the production grouping.

For every book file containing snapshot rows, each snapshot is identified as the vendor recommends
(event_time, last_update_id), NOT by received_time. For each snapshot it records:
  - distinct received_time values and the rows/sides of each part (detects "two halves" vs identical copies)
  - union of distinct levels over all parts vs the largest part (a true split has union > largest part)
  - file position of the parts vs the surrounding update messages (published replay order)
  - the update message containing L (U <= L <= u) or following it (pu == L): position in FILE order and whether it was
    RECEIVED before the snapshot's (non-synthetic) receive time -> what a live client must have buffered
Usage: .venv/bin/python scripts/audit_snapshot_grouping.py [workers]
Output: derived/audit/snapshot_grouping.parquet
"""
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402
from chdds.decimal import parse_decimal  # noqa: E402
from chdds.rawio import col_i64, read_raw  # noqa: E402

COLS = ["received_time", "event_time", "event_type", "first_update_id", "final_update_id", "prev_final_update_id",
        "last_update_id", "side", "price", "quantity"]


def scan(day_iso: str, hour: int):
    d = date.fromisoformat(day_iso)
    t, _ = read_raw(P.raw_path(d, hour, "orderbook"), columns=COLS, with_hash=False)
    recv, _ = col_i64(t, "received_time")
    evt, _ = col_i64(t, "event_time")
    U, _ = col_i64(t, "first_update_id", -1)
    u, _ = col_i64(t, "final_update_id", -1)
    pu, _ = col_i64(t, "prev_final_update_id", -1)
    L, _ = col_i64(t, "last_update_id", -1)
    is_s = pc.equal(t.column("event_type"), "snapshot").to_numpy(zero_copy_only=False)
    side = pc.equal(t.column("side"), "bid").to_numpy(zero_copy_only=False)
    price, _, _ = parse_decimal(t.column("price"), 2)
    qty, _, _ = parse_decimal(t.column("quantity"), 3)
    rows = np.arange(t.num_rows)
    # update messages in file order: first file row of each distinct u
    ur = rows[~is_s]
    uu, first = np.unique(u[ur], return_index=True)
    m_row = ur[first]
    m_u, m_U, m_pu, m_recv = u[m_row], U[m_row], pu[m_row], recv[m_row]
    o = np.argsort(m_row)
    m_row, m_u, m_U, m_pu, m_recv = m_row[o], m_u[o], m_U[o], m_pu[o], m_recv[o]
    h0 = P.hour_start_ns(d, hour)
    out = []
    srows = rows[is_s]
    key = np.stack([evt[srows], L[srows]], axis=1)
    uniq = np.unique(key, axis=0)
    for e_, l_ in uniq:
        sel = srows[(evt[srows] == e_) & (L[srows] == l_)]
        parts = np.unique(recv[sel])
        sizes, nb, na = [], [], []
        for r_ in parts:
            x = sel[recv[sel] == r_]
            sizes.append(x.size)
            nb.append(int(side[x].sum()))
            na.append(int((~side[x]).sum()))
        lev = np.unique(price[sel] * 2 + side[sel])
        real = parts[parts != e_ * 1_000_000]
        r_real = int(real.min()) if real.size else -1
        # connecting message
        cont = (m_U <= l_) & (l_ <= m_u)
        nxt = m_pu == l_
        conn = np.nonzero(cont | nxt)[0]
        srow0, srow1 = int(sel.min()), int(sel.max())
        rec = dict(day=day_iso, hour=hour, evt=int(e_), L=int(l_), n_parts=int(parts.size), rows=int(sel.size),
                   largest_part=int(max(sizes)), union_levels=int(lev.size), part_nbid=str(nb), part_nask=str(na),
                   synthetic_part=bool(np.any(parts == e_ * 1_000_000)), real_recv=r_real,
                   contiguous=bool(srow1 - srow0 + 1 == sel.size), srow0=srow0, srow1=srow1,
                   n_file_rows=int(t.num_rows), is_hour_start=bool(np.any(parts == h0)))
        i_after = np.searchsorted(m_row, srow1)
        i_before = i_after - 1
        rec["prev_u_file"] = int(m_u[i_before]) if i_before >= 0 else -1
        rec["prev_recv_file"] = int(m_recv[i_before]) if i_before >= 0 else -1
        rec["next_U_file"] = int(m_U[i_after]) if i_after < m_u.size else -1
        rec["next_u_file"] = int(m_u[i_after]) if i_after < m_u.size else -1
        rec["next_pu_file"] = int(m_pu[i_after]) if i_after < m_u.size else -1
        rec["next_recv_file"] = int(m_recv[i_after]) if i_after < m_u.size else -1
        rec["has_conn_msg"] = bool(conn.size > 0)
        if conn.size:
            c = conn[0]
            rec["conn_kind"] = "contains_L" if cont[c] else "pu_eq_L"
            rec["conn_row_after_snapshot"] = bool(m_row[c] > srow1)
            rec["conn_recv_minus_real_ms"] = float((m_recv[c] - r_real) / 1e6) if r_real > 0 else None
        else:
            rec["conn_kind"] = ""
            rec["conn_row_after_snapshot"] = None
            rec["conn_recv_minus_real_ms"] = None
        rec["first_u_in_file"] = int(m_u[0]) if m_u.size else -1
        rec["last_u_in_file"] = int(m_u[-1]) if m_u.size else -1
        out.append(rec)
    return out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    f = pl.read_parquet(P.DERIVED / "audit" / "files_orderbook.parquet", columns=["day", "hour", "s_n_msgs_snapshot"])
    todo = f.filter(pl.col("s_n_msgs_snapshot") > 0).select(["day", "hour"]).rows()
    print(len(todo), "files with snapshots", flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(scan, d, h) for d, h in todo]
        for i, fu in enumerate(as_completed(futs)):
            res += fu.result()
            if i % 200 == 0:
                print(i, len(res), flush=True)
    df = pl.from_dicts(res, infer_schema_length=None).sort(["day", "hour", "evt"])
    df.write_parquet(P.DERIVED / "audit" / "snapshot_grouping.parquet")
    print(df.height, "snapshots")


if __name__ == "__main__":
    main()
