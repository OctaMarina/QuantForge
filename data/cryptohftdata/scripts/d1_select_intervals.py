"""D1: pre-declared plan of the experiment, written BEFORE any comparison is run (configs/d1_intervals.json).

Reference candidates: continuous segments of the corrected strict resolver, cut at a snapshot, at least 1 hour long.
Strata: collection regime x volatility tercile x liquidity tercile, terciles taken over all reference hours.
Split: whole UTC days go to calibration or validation (seeded, alternating inside each regime), so no day is in both.
Starts per interval: the anchor itself, three seeded random interior moments, and the start of the sharpest minute of the
interval (rule-based, chosen to look for failures, not for a favourable case).
Extra cell "stress": the most volatile and the least liquid reference hours of each split, regardless of the cell quota.
Usage: .venv/bin/python scripts/d1_select_intervals.py [seed]
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402
from d1_age_pass import segments  # noqa: E402

LEN_H = 2.0
MIN_H = 1.0
PER_CELL_SPLIT = 2
N_STRESS = 3
N_POSTGAP = 12
HORIZON_S = 3600
INTERIOR = 3
SEC = 10**9


def regime(day):
    if day < "2025-08-01":
        return "R102_2025-06"
    if day < "2026-05-01":
        return "R52_2026-04"
    if day < "2026-06-01":
        return "R26_2026-05"
    if day < "2026-09-01":
        return "R26_2026-06_08"
    return "R26_2026-09_10"


def day_of(t):
    return datetime.fromtimestamp(t / 1e9, timezone.utc).date().isoformat()


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20261104
    rng = np.random.default_rng(seed)
    tr = pl.read_parquet(P.DERIVED / "d1" / "chain_pass" / "transitions_strict.parquet").sort("t")
    snaps = tr.filter(pl.col("kind").is_in(["anchor", "verify"]))["t"].to_numpy()
    segs = segments()
    ht = pl.read_parquet(P.DERIVED / "d1" / "hour_table.parquet").filter(pl.col("n_min") >= 50)
    # candidates: a snapshot inside a segment with >= MIN_H of chain after it
    cand = []
    for a, b in segs:
        s = snaps[(snaps >= a) & (snaps <= b - int(MIN_H * P.HOUR_NS))]
        last = -1
        for t in s:
            if t - last < P.HOUR_NS:      # at most one candidate per hour of a segment
                continue
            last = int(t)
            cand.append((int(t), min(int(t) + int(LEN_H * P.HOUR_NS), int(b) - SEC)))
    c = pl.DataFrame(cand, schema=["t_anchor", "t_end"], orient="row").with_columns(
        hour_ns=pl.col("t_anchor") // P.HOUR_NS * P.HOUR_NS).join(ht, on="hour_ns", how="inner").filter(
        pl.col("depth5").is_not_null())
    c = c.with_columns(day=pl.col("t_anchor").map_elements(day_of, return_dtype=pl.Utf8))
    c = c.with_columns(regime=pl.col("day").map_elements(regime, return_dtype=pl.Utf8))
    assert c["rv_1h"].is_finite().all() and c["depth5"].is_finite().all(), "strata need finite measures"
    q = dict(vol=np.quantile(c["rv_1h"].to_numpy(), [1 / 3, 2 / 3]), liq=np.quantile(c["depth5"].to_numpy(), [1 / 3, 2 / 3]))
    c = c.with_columns(vol=pl.Series(np.digitize(c["rv_1h"].to_numpy(), q["vol"])),
                       liq=pl.Series(np.digitize(c["depth5"].to_numpy(), q["liq"])))
    # day split, seeded and alternating inside each regime
    split = {}
    for r in sorted(c["regime"].unique().to_list()):
        days = sorted(c.filter(pl.col("regime") == r)["day"].unique().to_list())
        rng.shuffle(days)
        for i, d in enumerate(days):
            split[d] = "calibration" if i % 2 == 0 else "validation"
    c = c.with_columns(split=pl.col("day").replace_strict(split))
    picked = {}
    for key, g in c.group_by(["regime", "vol", "liq", "split"], maintain_order=True):
        idx = rng.permutation(g.height)[:PER_CELL_SPLIT]
        for row in g[idx.tolist()].iter_rows(named=True):
            picked[row["t_anchor"]] = dict(row, cell="grid")
    for sp in ("calibration", "validation"):
        g = c.filter(pl.col("split") == sp)
        for row in g.sort("rv_1h", descending=True).head(N_STRESS).iter_rows(named=True):
            picked.setdefault(row["t_anchor"], dict(row, cell="stress_vol"))
        for row in g.sort("depth5").head(N_STRESS).iter_rows(named=True):
            picked.setdefault(row["t_anchor"], dict(row, cell="stress_liq"))
    # real gaps inside reference periods: the reconnect snapshot gives R an anchor, the empty book restarts after the timeout
    gaps = pl.read_parquet(P.DERIVED / "d1" / "chain_pass" / "events_chain_nosnap.parquet").filter(pl.col("kind") == "true_gap")
    anch = tr.filter(pl.col("kind") == "anchor")["t"].to_numpy()
    pg = []
    for t0 in gaps["t0"].to_list():
        j = np.searchsorted(anch, t0 - 2 * SEC)
        if j < anch.size and anch[j] <= t0 + 2 * SEC:
            pg.append(int(anch[j]))
    g = c.select("day", "regime", "split").unique("day")
    pgdf = pl.DataFrame({"t_anchor": pg}).with_columns(hour_ns=pl.col("t_anchor") // P.HOUR_NS * P.HOUR_NS).join(ht, on="hour_ns", how="inner")
    pgdf = pgdf.filter(pl.col("depth5").is_not_null()).with_columns(
        day=pl.col("t_anchor").map_elements(day_of, return_dtype=pl.Utf8)).join(g, on="day", how="inner")
    pgdf = pgdf.with_columns(t_end=pl.col("t_anchor") + int(LEN_H * P.HOUR_NS),
                             vol=pl.Series(np.digitize(pgdf["rv_1h"].to_numpy(), q["vol"])),
                             liq=pl.Series(np.digitize(pgdf["depth5"].to_numpy(), q["liq"])))
    post = {}
    for sp in ("calibration", "validation"):
        x = pgdf.filter(pl.col("split") == sp)
        for row in x[rng.permutation(x.height)[:N_POSTGAP].tolist()].iter_rows(named=True):
            post[row["t_anchor"]] = dict(row, cell="post_gap")
    # sharpest minute of each interval from the 1-minute bars
    bars = pl.concat([pl.read_parquet(f, columns=["t_end_ns", "bar_close"]) for f in sorted(
        (P.DERIVED / "datasets" / "btcusdt_initial").glob("time_1m-*/*.parquet"))]).sort("t_end_ns").with_columns(
        r=pl.col("bar_close").log().diff().abs())
    bt, br = bars["t_end_ns"].to_numpy(), bars["r"].fill_null(0).fill_nan(0).to_numpy()
    out = []
    for t, row in sorted(post.items()):
        Ta = row["t_anchor"]
        out.append(dict(id=f"pg_{row['day']}_{Ta // SEC % 86400:05d}", t_anchor=Ta, t_end=row["t_end"], horizon_s=HORIZON_S,
                        regime=row["regime"], vol=int(row["vol"]), liq=int(row["liq"]), split=row["split"], cell="post_gap",
                        rv_1h=row["rv_1h"], depth5=row["depth5"], notional=row["notional"],
                        starts=[dict(name="postgap", t=Ta + 60 * SEC)]))
    for t, row in sorted(picked.items()):
        Ta, Tb = row["t_anchor"], row["t_end"]
        lo = Ta + 60 * SEC
        hi = max(lo + SEC, Tb - 600 * SEC)
        starts = [dict(name="anchor", t=Ta)]
        for j in range(INTERIOR):
            starts.append(dict(name=f"in{j}", t=int(rng.integers(lo, hi))))
        m = (bt > lo) & (bt < hi)
        if m.any():
            ts = int(bt[m][np.argmax(br[m])]) - 60 * SEC
            starts.append(dict(name="stress", t=max(ts, lo)))
        out.append(dict(id=f"{row['day']}_{Ta // SEC % 86400:05d}", t_anchor=Ta, t_end=Tb,
                        horizon_s=HORIZON_S, regime=row["regime"], vol=int(row["vol"]), liq=int(row["liq"]), split=row["split"],
                        cell=row["cell"], rv_1h=row["rv_1h"], depth5=row["depth5"], notional=row["notional"], starts=starts))
    # Extension declared after the first sample turned out concentrated in the reconnect storm of 2026-05-24..27 (seen from
    # reference metadata only): real gaps followed by at least one hour without another hole. Separate RNG, so the intervals
    # above stay exactly as declared. The rule looks only at the chain, not at any comparison.
    rng2 = np.random.default_rng(seed + 1)
    g0 = np.sort(gaps["t0"].to_numpy())
    calm = []
    for t in pgdf["t_anchor"].to_list():
        j = np.searchsorted(g0, t + 120 * SEC)
        if j >= g0.size or g0[j] > t + 3720 * SEC:
            calm.append(t)
    x0 = pgdf.filter(pl.col("t_anchor").is_in(calm) & ~pl.col("day").is_in(["2026-05-24", "2026-05-25", "2026-05-26", "2026-05-27"]))
    for sp in ("calibration", "validation"):
        x = x0.filter(pl.col("split") == sp)
        for row in x[rng2.permutation(x.height)[:N_POSTGAP].tolist()].iter_rows(named=True):
            Ta = row["t_anchor"]
            out.append(dict(id=f"pgc_{row['day']}_{Ta // SEC % 86400:05d}", t_anchor=Ta, t_end=row["t_end"], horizon_s=HORIZON_S,
                            regime=row["regime"], vol=int(row["vol"]), liq=int(row["liq"]), split=row["split"], cell="post_gap_calm",
                            rv_1h=row["rv_1h"], depth5=row["depth5"], notional=row["notional"],
                            starts=[dict(name="postgap", t=Ta + 60 * SEC)]))
    plan = dict(seed=seed, created=datetime.now(timezone.utc).isoformat(), len_h=LEN_H, horizon_s=HORIZON_S,
                terciles={k: v.tolist() for k, v in q.items()}, split_days=split, intervals=out,
                n_candidates=c.height, n_segments=len(segs))
    Path("configs/d1_intervals.json").write_text(json.dumps(plan, indent=1))
    t = pl.DataFrame([{k: i[k] for k in ("regime", "vol", "liq", "split", "cell")} for i in out])
    print(len(out), "intervals of", c.height, "candidates")
    print(t.group_by(["regime", "split"]).len().sort(["regime", "split"]))
    print(t.group_by(["cell", "split"]).len().sort(["cell", "split"]))


if __name__ == "__main__":
    main()
