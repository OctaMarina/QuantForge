"""D1: optional rebuilt mode vs the strict dataset on the demonstration subsets. Output: results/demo_comparison.csv"""
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402

COLS = ["spread_bps", "qi_1", "depth_bid_top10_btc", "depth_bid_5bps_btc", "depth_bid_25bps_btc", "ofi_30s_btc", "ofi_2m_btc", "ofi_15m_btc", "tw_spread_bps_2m"]
OUT = P.DERIVED / "d1" / "results"


def main():
    rows = []
    for name, a, b in (("rebuilt_demo", "2025-12-14", "2025-12-16"), ("rebuilt_demo_crash", "2025-10-09", "2025-10-11")):
        for sd in sorted(glob.glob(str(P.DERIVED / "datasets" / name / "*-*"))):
            spec = Path(sd).name
            f = pl.concat([pl.read_parquet(x) for x in sorted(glob.glob(sd + "/*.parquet"))])
            s = pl.concat([pl.read_parquet(x, columns=["bar_id", "eval_time_ns", "book_state"] + COLS) for x in sorted(glob.glob(str(P.DERIVED / "datasets" / "btcusdt_initial" / spec / "*.parquet")))
                           if Path(x).stem in {a[:7], b[:7]}]) if (P.DERIVED / "datasets" / "btcusdt_initial" / spec).exists() else None
            lo = np.datetime64(a, "ns").astype(np.int64)
            hi = np.datetime64(b, "ns").astype(np.int64) + 86400 * 10**9
            s = s.filter((pl.col("eval_time_ns") >= int(lo)) & (pl.col("eval_time_ns") < int(hi)))
            st = f["book_state"].to_numpy()
            r = dict(subset=name, bars=spec.split("-")[0], n=f.height, state9=float((st == 9).mean()), state10=float((st == 10).mean()), stale=float((st == 2).mean()),
                     strict_rows=s.height, strict_state0=float((s["book_state"].to_numpy() == 0).mean()))
            for c in COLS:
                r[c] = float(np.isfinite(f[c].to_numpy()).mean())
                r["strict_" + c] = float(np.isfinite(s[c].to_numpy()).mean())
            rows.append(r)
    d = pl.from_dicts(rows)
    d.write_csv(OUT / "demo_comparison.csv")
    pl.Config.set_tbl_cols(14)
    pl.Config.set_tbl_width_chars(170)
    print(d.select("subset", "bars", "n", "state9", "state10", "stale", "strict_state0", "spread_bps", "strict_spread_bps", "depth_bid_5bps_btc", "depth_bid_25bps_btc", "ofi_2m_btc", "ofi_15m_btc"))


if __name__ == "__main__":
    main()
