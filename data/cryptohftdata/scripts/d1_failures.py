"""D1: worst hours of the time-only policy on validation days, from the large sample. Output: results/validation_worst_hours.csv"""
import glob
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402

RES = P.DERIVED / "d1" / "results"


def main():
    pol = json.loads((RES / "policy.json").read_text())
    plan = json.loads(Path("configs/d1_intervals.json").read_text())
    val = {d for d, s in plan["split_days"].items() if s == "validation"}
    ht = pl.read_parquet(P.DERIVED / "d1" / "hour_table.parquet").select("hour_ns", "rv_1h", "depth5", "notional")
    rows = []
    for f in sorted(glob.glob(str(P.DERIVED / "d1" / "age" / "*" / "ages.parquet"))):
        a = pl.read_parquet(f).with_columns(day=pl.from_epoch("t", time_unit="ns").dt.strftime("%Y-%m-%d"),
                                            hour_ns=pl.col("t") // P.HOUR_NS * P.HOUR_NS).filter(pl.col("day").is_in(list(val)))
        if a.height == 0:
            continue
        for K, dl in pol["top"].items():
            x = a.filter((pl.col(f"age{K}_b") >= 0) & (pl.col(f"age{K}_a") >= 0) & (pl.col("t") - pl.col("ta") >= int(dl) * 10**9))
            x = x.with_columns(m=pl.max_horizontal(f"age{K}_b", f"age{K}_a"))
            g = x.group_by("hour_ns", "day").agg(pl.len().alias("n"), (pl.col("m") > int(dl) * 1000).sum().alias("bad"),
                                                 (pl.col("m").max() / 1000).alias("oldest_level_s"))
            rows.append(g.filter(pl.col("bad") > 0).with_columns(K=pl.lit(int(K)), delta=pl.lit(int(dl))))
    r = pl.concat(rows).with_columns(fail=pl.col("bad") / pl.col("n")).join(ht, on="hour_ns", how="left")
    r = r.with_columns(hour=pl.from_epoch("hour_ns", time_unit="ns").dt.strftime("%Y-%m-%d %H")).sort("fail", descending=True)
    r.select("K", "delta", "hour", "n", "bad", "fail", "oldest_level_s", "rv_1h", "depth5").write_csv(RES / "validation_worst_hours.csv")
    print(r.group_by("K").agg(pl.len().alias("hours_with_failure"), pl.col("fail").max(), pl.col("oldest_level_s").max()).sort("K"))
    print(r.select("K", "delta", "hour", "bad", "fail", "oldest_level_s", "rv_1h", "depth5").head(12))


if __name__ == "__main__":
    main()
