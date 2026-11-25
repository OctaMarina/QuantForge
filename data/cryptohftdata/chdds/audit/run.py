"""Parallel, resumable exhaustive audit of every raw file.

For each raw file: read all rows, compute stats/events, write staged exact-integer copy
(for all streams except orderbook) and finally a per-file JSON result (the commit marker).
Re-running skips files whose JSON exists with identical (size, mtime_ns).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

from .. import paths as P
from ..rawio import read_raw

AUDIT_DIR = P.DERIVED / "audit" / "per_file"
STAGED_DIR = P.DERIVED / "staged"


def result_path(stream, day: date, hour: int) -> Path:
    return AUDIT_DIR / stream / day.isoformat() / f"{hour:02d}.json"


def staged_path(stream, day: date, hour: int) -> Path:
    return STAGED_DIR / stream / day.isoformat() / f"{hour:02d}.parquet"


def _atomic_write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def audit_one(stream: str, day_iso: str, hour: int, write_staged: bool = True) -> dict:
    from .filestats import AUDITORS
    day = date.fromisoformat(day_iso)
    rp = result_path(stream, day, hour)
    src = P.raw_path(day, hour, stream)
    st = src.stat()
    t0 = time.perf_counter()
    table, info = read_raw(src)
    t_read = time.perf_counter() - t0
    h0 = P.hour_start_ns(day, hour)
    stats, events, staged = AUDITORS[stream](table, info, h0)
    nrows = table.num_rows
    del table
    if staged is not None and write_staged:
        sp = staged_path(stream, day, hour)
        sp.parent.mkdir(parents=True, exist_ok=True)
        tmp = sp.with_name(sp.name + f".tmp{os.getpid()}")
        pq.write_table(staged, tmp, compression="zstd", compression_level=3)
        os.replace(tmp, sp)
    res = dict(stream=stream, day=day_iso, hour=hour, key=P.raw_key(day, hour, stream), path=str(src),
               size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=info.sha256, created_by=info.created_by,
               kv_metadata=json.dumps(info.kv_metadata, sort_keys=True), schema_fp=info.schema_fp,
               columns=json.dumps(info.columns), num_row_groups=info.num_row_groups,
               parquet_num_rows=info.num_rows, read_s=t_read, total_s=time.perf_counter() - t0,
               stats=stats, events=events)
    _atomic_write_bytes(rp, json.dumps(res, default=_json_default).encode())
    return dict(stream=stream, day=day_iso, hour=hour, rows=nrows, s=res["total_s"])


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def _safe(stream, day_iso, hour, write_staged):
    try:
        return audit_one(stream, day_iso, hour, write_staged)
    except Exception:
        return dict(stream=stream, day=day_iso, hour=hour, error=traceback.format_exc())


def pending(streams, start=P.HISTORY_START, end=P.HISTORY_END):
    todo = []
    for day, hour in P.iter_hours(start, end):
        for s in streams:
            src = P.raw_path(day, hour, s)
            if not src.exists():
                continue
            rp = result_path(s, day, hour)
            if rp.exists():
                try:
                    with rp.open() as f:
                        head = json.load(f)
                    st = src.stat()
                    if head.get("size") == st.st_size and head.get("mtime_ns") == st.st_mtime_ns:
                        continue
                except Exception:
                    pass
            todo.append((s, day.isoformat(), hour))
    return todo


def run(streams=P.STREAMS, workers: int = 12, start=P.HISTORY_START, end=P.HISTORY_END, write_staged=True,
        log=print):
    todo = pending(streams, start, end)
    # biggest files first for better load balance
    todo.sort(key=lambda t: (t[0] != "orderbook", t[1], t[2]))
    log(f"audit: {len(todo)} files pending, workers={workers}")
    t0 = time.time()
    done = errors = rows = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_safe, s, d, h, write_staged) for s, d, h in todo]
        for f in as_completed(futs):
            r = f.result()
            done += 1
            if "error" in r:
                errors += 1
                log(f"ERROR {r['stream']} {r['day']} {r['hour']}: {r['error'].splitlines()[-1]}")
                (P.DERIVED / "audit").mkdir(parents=True, exist_ok=True)
                with (P.DERIVED / "audit" / "errors.log").open("a") as fh:
                    fh.write(json.dumps(r) + "\n")
            else:
                rows += r["rows"]
            if done % 500 == 0 or done == len(todo):
                el = time.time() - t0
                log(f"  {done}/{len(todo)} files, {rows/1e9:.2f}e9 rows, {el:.0f}s elapsed, errors={errors}")
    return done, errors


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    streams = sys.argv[2].split(",") if len(sys.argv) > 2 else P.STREAMS
    run(streams, w)
