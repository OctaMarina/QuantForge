"""Inventory: hour x stream grid confronted with manifests and filesystem.

Sources:
  rest_manifest.csv               — REST downloader ledger (status ok/unavailable, path, size, rows)
  rest_unavailable_or_errors.csv  — subset with status != ok
  download_manifest.csv           — older S3 downloader log
  filesystem                      — actual files under RAW_ROOT
"""
from __future__ import annotations

import csv
from collections import Counter

import polars as pl

from .. import paths as P


def load_rest_manifest():
    return pl.read_csv(P.RAW_ROOT / "rest_manifest.csv", schema_overrides={"size": pl.Int64, "rows": pl.Int64,
                                                                          "mtime": pl.Int64})


def build_inventory():
    man = load_rest_manifest()
    unav = pl.read_csv(P.RAW_ROOT / "rest_unavailable_or_errors.csv", schema_overrides={"size": pl.Int64,
                                                                                         "rows": pl.Int64,
                                                                                         "mtime": pl.Int64})
    dl = pl.read_csv(P.RAW_ROOT / "download_manifest.csv", infer_schema_length=0)
    rows = []
    for day, hour in P.iter_hours():
        for s in P.STREAMS:
            key = P.raw_key(day, hour, s)
            p = P.raw_path(day, hour, s)
            exists = p.exists()
            rows.append(dict(key=key, day=day.isoformat(), hour=hour, stream=s,
                             fs_exists=exists, fs_path=str(p.relative_to(P.RAW_ROOT)) if exists else None,
                             fs_size=p.stat().st_size if exists else None,
                             fs_format=("zst" if p.suffix == ".zst" else "parquet") if exists else None))
    inv = pl.DataFrame(rows)
    inv = inv.join(man.select(pl.col("key"), pl.col("status").alias("rest_status"), pl.col("path").alias("rest_path"),
                              pl.col("size").alias("rest_size"), pl.col("rows").alias("rest_rows"),
                              pl.col("note").alias("rest_note")), on="key", how="left")
    # S3 log: normalise .zst suffix; keep last status per key
    dl2 = (dl.with_columns(pl.col("source_key").str.replace(r"\.zst$", "").alias("key"))
             .group_by("key").agg(pl.col("status").last().alias("s3_status"),
                                  pl.col("status").n_unique().alias("s3_n_status")))
    inv = inv.join(dl2, on="key", how="left")
    checks = {
        "manifest_keys": man.height,
        "manifest_unique_keys": man["key"].n_unique(),
        "manifest_status": dict(Counter(man["status"].to_list())),
        "unavailable_file_rows": unav.height,
        "unavailable_equals_manifest_subset": set(unav["key"]) == set(man.filter(pl.col("status") != "ok")["key"]),
        "grid_hours": inv.height // len(P.STREAMS),
        "grid_keys_missing_from_manifest": int(inv["rest_status"].null_count()),
        "ok_but_missing_on_disk": int(inv.filter((pl.col("rest_status") == "ok") & ~pl.col("fs_exists")).height),
        "on_disk_but_not_ok": int(inv.filter((pl.col("rest_status") != "ok") & pl.col("fs_exists")).height),
        "size_mismatch_vs_manifest": int(inv.filter(pl.col("fs_exists") & (pl.col("fs_size") != pl.col("rest_size"))).height),
        "ok_with_zero_rows": int(inv.filter((pl.col("rest_status") == "ok") & (pl.col("rest_rows") == 0)).height),
        "s3_log_rows": dl.height,
        "s3_log_status": dict(Counter(dl["status"].to_list())),
        "s3_ok_keys_not_ok_in_rest": int(inv.filter((pl.col("s3_status") == "ok") & (pl.col("rest_status") != "ok")).height),
        "fs_formats": dict(Counter(x for x in inv["fs_format"].to_list() if x)),
    }
    return inv, checks


def unavailable_intervals(inv: pl.DataFrame):
    """Contiguous runs of hours where a stream file is not available (end exclusive)."""
    out = []
    for s in P.STREAMS:
        sub = inv.filter(pl.col("stream") == s).sort(["day", "hour"])
        ok = sub["fs_exists"].to_list()
        days = sub["day"].to_list()
        hours = sub["hour"].to_list()
        i = 0
        n = len(ok)
        while i < n:
            if not ok[i]:
                j = i
                while j < n and not ok[j]:
                    j += 1
                end = f"{days[j]} {hours[j]:02d}:00" if j < n else "end"
                out.append(dict(stream=s, start=f"{days[i]} {hours[i]:02d}:00", end_excl=end, hours=j - i))
                i = j
            else:
                i += 1
    return pl.DataFrame(out)
