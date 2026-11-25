"""D1 coverage. From the chain pass (no book is built):
  live segments of a book restarted empty after every true gap, usable duration after a warm-up, and the share of each month
  that a feature family would cover, alone and together with the corrected strict book. A family needs warm-up + window.
Warm-up counted from the availability of the restart (conservative: the messages buffered during the timeout are ignored).
Outputs: derived/d1/results/correction_effect.csv, segments.parquet, segment_stats.csv, coverage_by_month.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402

CP = P.DERIVED / "d1" / "chain_pass"
RES = P.DERIVED / "d1" / "results"
T0 = P.hour_start_ns(P.HISTORY_START, 0)
T1 = P.hour_start_ns(P.HISTORY_END, 23) + P.HOUR_NS


def months():
    out = []
    for y, m in [(2025, x) for x in range(6, 13)] + [(2026, x) for x in range(1, 11)]:
        a = np.datetime64(f"{y}-{m:02d}-01", "ns").astype(np.int64)
        n = (y + 1, 1) if m == 12 else (y, m + 1)
        b = np.datetime64(f"{n[0]}-{n[1]:02d}-01", "ns").astype(np.int64)
        out.append((f"{y}-{m:02d}", max(int(a), T0), min(int(b), T1)))
    return out


def clip_len(iv, a, b):
    lo = np.maximum(iv[:, 0], a)
    hi = np.minimum(iv[:, 1], b)
    return np.maximum(hi - lo, 0).sum()


def union(*ivs):
    x = np.concatenate([i for i in ivs if i.size]) if any(i.size for i in ivs) else np.zeros((0, 2), np.int64)
    if x.size == 0:
        return x
    x = x[np.argsort(x[:, 0])]
    out = [x[0].tolist()]
    for a, b in x[1:].tolist():
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.array(out, np.int64)


def anchored(variant):
    """[connected anchor, invalidation) of a strict resolver."""
    tr = pl.read_parquet(CP / f"transitions_{variant}.parquet").sort("t")
    ev = pl.read_parquet(CP / f"events_{variant}.parquet")
    conn = ev.filter(pl.col("kind") == "anchor_connected").select(pl.col("t1").alias("t"), pl.lit("conn").alias("kind"))
    x = pl.concat([tr.filter(pl.col("kind").str.starts_with("invalidate")).select("t", "kind"), conn]).sort("t")
    out, cur = [], None
    for t, k in x.rows():
        if k == "conn":
            if cur is None:
                cur = t
        elif cur is not None:
            if t > cur:
                out.append((cur, t))
            cur = None
    if cur is not None:
        out.append((cur, T1))
    return np.array(out, np.int64).reshape(-1, 2)


def live_segments():
    """Book started empty: [restart, next hole first observed)."""
    tr = pl.read_parquet(CP / "transitions_chain_nosnap.parquet").sort("t")
    ev = pl.read_parquet(CP / "events_chain_nosnap.parquet").filter(pl.col("kind") == "true_gap").sort("t0")
    starts = tr.filter(pl.col("kind") == "empty")["t"].to_numpy()
    miss = tr.filter(pl.col("kind") == "invalidate_missing_file")["t"].to_numpy()
    ends = np.sort(np.concatenate([ev["t0"].to_numpy(), miss]))
    seg = []
    for i, a in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < starts.size else T1
        j = np.searchsorted(ends, a, side="right")
        b = min(int(ends[j]) if j < ends.size else T1, int(nxt))
        seg.append((int(a), b))
    return np.array(seg, np.int64)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    leg, fix = anchored("legacy"), anchored("strict")
    trl = pl.read_parquet(CP / "transitions_legacy.parquet")
    trs = pl.read_parquet(CP / "transitions_strict.parquet")
    evl = pl.read_parquet(CP / "events_legacy.parquet")
    evs = pl.read_parquet(CP / "events_strict.parquet")
    rows = []
    for m, a, b in months():
        def cnt(df, col, name):
            return df.filter((pl.col(col) >= a) & (pl.col(col) < b) & (pl.col("kind") == name)).height
        rows.append(dict(month=m, legacy_anch=clip_len(leg, a, b) / (b - a), fixed_anch=clip_len(fix, a, b) / (b - a),
                         legacy_anchors=cnt(trl, "t", "anchor"), fixed_anchors=cnt(trs, "t", "anchor"),
                         legacy_verify=cnt(trl, "t", "verify"), fixed_verify=cnt(trs, "t", "verify"),
                         legacy_gaps=cnt(evl, "t0", "true_gap"), fixed_gaps=cnt(evs, "t0", "true_gap"),
                         legacy_noconnect=cnt(evl, "t0", "anchor_noconnect"), fixed_noconnect=cnt(evs, "t0", "anchor_noconnect")))
    tot = T1 - T0
    rows.append(dict(month="ALL", legacy_anch=clip_len(leg, T0, T1) / tot, fixed_anch=clip_len(fix, T0, T1) / tot,
                     legacy_anchors=trl.filter(pl.col("kind") == "anchor").height, fixed_anchors=trs.filter(pl.col("kind") == "anchor").height,
                     legacy_verify=trl.filter(pl.col("kind") == "verify").height, fixed_verify=trs.filter(pl.col("kind") == "verify").height,
                     legacy_gaps=evl.filter(pl.col("kind") == "true_gap").height, fixed_gaps=evs.filter(pl.col("kind") == "true_gap").height,
                     legacy_noconnect=evl.filter(pl.col("kind") == "anchor_noconnect").height,
                     fixed_noconnect=evs.filter(pl.col("kind") == "anchor_noconnect").height))
    pl.from_dicts(rows).write_csv(RES / "correction_effect.csv")
    seg = live_segments()
    dur = (seg[:, 1] - seg[:, 0]) / 1e9
    pl.DataFrame({"t0": seg[:, 0], "t1": seg[:, 1], "dur_s": dur}).write_parquet(RES / "segments.parquet")
    rows = []
    dl = (0, 10, 60, 300, 900, 1800)
    for m, a, b in months():
        i = (seg[:, 0] >= a) & (seg[:, 0] < b)
        d = dur[i]
        r = dict(month=m, n=int(i.sum()), per_day=float(i.sum() / ((b - a) / 86400e9)))
        if d.size:
            r.update(p10_min=float(np.quantile(d, .1) / 60), p50_min=float(np.quantile(d, .5) / 60), p90_min=float(np.quantile(d, .9) / 60),
                     max_h=float(d.max() / 3600), under_1m=float((d < 60).mean()), under_5m=float((d < 300).mean()),
                     under_15m=float((d < 900).mean()))
        rows.append(r)
    pl.from_dicts(rows, infer_schema_length=None).write_csv(RES / "segment_stats.csv")
    rows = []
    for m, a, b in months():
        r = dict(month=m, strict_legacy=clip_len(leg, a, b) / (b - a), strict_fixed=clip_len(fix, a, b) / (b - a),
                 chain_live=clip_len(seg, a, b) / (b - a))
        for w in dl:
            iv = np.stack([seg[:, 0] + int(w * 1e9), seg[:, 1]], axis=1)
            iv = iv[iv[:, 1] > iv[:, 0]]
            r[f"rebuilt_{w}s"] = clip_len(iv, a, b) / (b - a)
            r[f"union_{w}s"] = clip_len(union(iv, fix), a, b) / (b - a)
        rows.append(r)
    pl.from_dicts(rows).write_csv(RES / "coverage_by_month.csv")
    print(pl.read_csv(RES / "correction_effect.csv"))
    print(pl.read_csv(RES / "segment_stats.csv"))
    print(pl.read_csv(RES / "coverage_by_month.csv").select("month", "strict_fixed", "chain_live", "rebuilt_60s", "rebuilt_300s", "union_300s"))


if __name__ == "__main__":
    main()
