"""Consolidate per-file audit JSON into columnar tables (files_<stream>.parquet, events_<stream>.parquet)."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .. import paths as P
from .run import AUDIT_DIR

OUT = P.DERIVED / "audit"


def consolidate(streams=P.STREAMS, log=print):
    for s in streams:
        files = sorted((AUDIT_DIR / s).rglob("*.json"))
        rows, evs = [], []
        for f in files:
            r = json.loads(f.read_text())
            st = r.pop("stats")
            ev = r.pop("events")
            row = {**r, **{f"s_{k}": v for k, v in st.items()}}
            rows.append(row)
            for e in ev:
                e = dict(e)
                e["day"] = r["day"]
                e["hour"] = r["hour"]
                evs.append(e)
        if not rows:
            continue
        df = pl.from_dicts(rows, infer_schema_length=None)
        df.write_parquet(OUT / f"files_{s}.parquet")
        if evs:
            ed = pl.from_dicts(evs, infer_schema_length=None)
            ed.write_parquet(OUT / f"events_{s}.parquet")
        log(f"  consolidated {s}: {df.height} files, {len(evs)} events")


def load(stream):
    return pl.read_parquet(OUT / f"files_{stream}.parquet")


def load_events(stream):
    p = OUT / f"events_{stream}.parquet"
    return pl.read_parquet(p) if p.exists() else None
