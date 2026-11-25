"""D1: how far the results on reference periods can be carried to the months without a reference.

  strict_valid_by_month.csv   real validity of the corrected strict book (fresh, inside the domain), 1 s samples
  age_by_month.csv            P(a top-K level is older than delta). "ref": exact, from the reference. "visible": from a book
                              started empty, only levels it can see, samples at least 1 h after a restart. The visible estimate
                              needs no reference; it is checked against "ref" on months that have both.
  conditions.csv              market and feed conditions, reference hours vs hours without reference
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402

RES = P.DERIVED / "d1" / "results"
POL = json.loads((RES / "policy.json").read_text())
KD = [(int(k), int(v)) for k, v in POL["top"].items()]
EXTRA = [(5, 10), (10, 10), (50, 10), (5, 60), (10, 60), (50, 60)]


def month(c="t"):
    return pl.from_epoch(pl.col(c), time_unit="ns").dt.strftime("%Y-%m").alias("month")


def load(root, cols):
    out = []
    for f in sorted(glob.glob(str(root / "*" / "ages.parquet"))):
        out.append(pl.read_parquet(f, columns=cols))
    return pl.concat(out)


def main():
    cols = ["t", "ta"] + [f"age{K}_{s}" for K in (1, 5, 10, 50) for s in "ba"]
    ref = load(P.DERIVED / "d1" / "age", cols).unique("t", keep="first").with_columns(month())
    emp = load(P.DERIVED / "d1" / "age_empty", cols).with_columns(month())
    secs = {}
    t0 = P.hour_start_ns(P.HISTORY_START, 0)
    hrs = pl.read_parquet(P.DERIVED / "d1" / "chain_pass" / "hours.parquet").filter(pl.col("variant") == "strict")
    hrs = hrs.with_columns(month=pl.col("day").str.slice(0, 7))
    secs = dict(hrs.group_by("month").len().rows())
    a = ref.group_by("month").len().sort("month")
    leg = pl.read_csv(P.DERIVED / "audit" / "book_replay_monthly.csv").select("month", pl.col("s0%").alias("legacy_valid_pct"))
    v = a.with_columns(fixed_valid_pct=pl.col("len") * 100 / (pl.col("month").replace_strict(secs, return_dtype=pl.Int64) * 3600)).drop("len")
    v = leg.join(v, on="month", how="left").with_columns(pl.col("fixed_valid_pct").fill_null(0.0))
    tot = sum(secs.values()) * 3600
    print("overall strict valid corrected", ref.height / tot)
    v.write_csv(RES / "strict_valid_by_month.csv")
    rows = []
    for K, dl in KD + EXTRA:
        for lab, d in (("ref", ref), ("visible", emp.filter(pl.col("t") - pl.col("ta") >= 3600 * 10**9))):
            x = d.filter((pl.col(f"age{K}_b") >= 0) & (pl.col(f"age{K}_a") >= 0))
            if lab == "ref":
                x = x.filter(pl.col("t") - pl.col("ta") >= dl * 10**9)
            x = x.with_columns(fail=pl.max_horizontal(f"age{K}_b", f"age{K}_a") > dl * 1000)
            g = x.group_by("month").agg(pl.len().alias("n"), pl.col("fail").mean())
            rows += [dict(K=K, delta=dl, source=lab, month=m, n=n, fail=f) for m, n, f in g.rows()]
    pl.from_dicts(rows).sort("K", "delta", "month", "source").write_csv(RES / "age_by_month.csv")
    ht = pl.read_parquet(P.DERIVED / "d1" / "hour_table.parquet").with_columns(month("hour_ns"))
    ref_h = set((ref["t"] // P.HOUR_NS * P.HOUR_NS).unique().to_list())
    ht = ht.with_columns(has_ref=pl.col("hour_ns").is_in(list(ref_h)))
    q = lambda c, p: pl.col(c).quantile(p).alias(f"{c}_p{int(p * 100)}")
    c = ht.group_by("has_ref").agg(pl.len().alias("hours"), q("rv_1h", .5), q("rv_1h", .99), pl.col("rv_1h").max().alias("rv_max"), q("notional", .5),
                                   q("notional", .99), q("n_trades", .5), q("n_trades", .99))
    c.write_csv(RES / "conditions.csv")
    print(c)
    m = hrs.group_by("month").agg(pl.col("cad_p50").median(), (pl.col("rows_upd").sum() / pl.col("n_upd").sum()).alias("rows_per_msg"),
                                  (pl.col("n_upd").sum() / pl.len() / 3600).alias("msgs_per_s")).sort("month")
    m.write_csv(RES / "feed_by_month.csv")


if __name__ == "__main__":
    main()
