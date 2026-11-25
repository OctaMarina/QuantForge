"""Summary of the regen_dry_* datasets: independent verification, provenance, and availability vs the existing dataset on the same days.
Output: derived/regen_check/dry_run.json"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402

COLS = ["ret_trade_1h", "ret_trade_4h", "ret_trade_24h", "rv_trade_1h_grid1s", "tfi_1h_btc", "spread_bps", "ofi_2m_btc", "markout_30s_qw_bps_15m", "oi_btc_asof", "oi_log_change_known_1h",
        "tw_oi_known_1h_btc", "liq_count_1h", "liq_count_15m", "time_since_buy_short_liq_s", "funding_interval_s_observed", "funding_rz_7d", "basis_mark_index_bps"]


def main():
    rep = {}
    for d in sorted(glob.glob(str(P.DERIVED / "datasets" / "regen_dry_*"))):
        name = Path(d).name
        ver = json.load(open(d + "/verification.json"))
        man = json.load(open(d + "/manifest.json"))
        tot = dict(bars=0, conservation=0, brute=0, brute_err=0, leak=0, unregistered=0, ambiguous=0)
        for b in ver["bar_sets"].values():
            tot["unregistered"] += len(b.get("unregistered", []))
            tot["ambiguous"] += len(b.get("ambiguous", []))
            for c in b["checks"]:
                tot["bars"] += c["bars_checked"]
                tot["conservation"] += c["conservation_mismatches"]
                tot["brute"] += c["bruteforce_comparisons"]
                tot["brute_err"] += c["bruteforce_errors"]
                tot["leak"] += c.get("book_values_when_invalid", 0) + sum(v for k, v in c.items() if k.endswith("_when_oi_state_invalid") or k.endswith("_when_mark_state_invalid") or k.endswith("_when_liq_state_15m_invalid"))
        spec = [s for s in man["bar_sets"] if s.startswith("time_1m")][0]
        new = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{d}/{spec}/*.parquet"))])
        lo, hi = new["eval_time_ns"].min(), new["eval_time_ns"].max()
        months = sorted({Path(f).stem for f in glob.glob(f"{d}/{spec}/*.parquet")})
        old = pl.concat([pl.read_parquet(P.DERIVED / "datasets" / "btcusdt_initial" / spec / f"{m}.parquet") for m in months]).filter(pl.col("eval_time_ns").is_between(lo, hi))
        cmp_ = {c: [round(float(np.isfinite(old[c].to_numpy().astype(float)).mean()), 4), round(float(np.isfinite(new[c].to_numpy().astype(float)).mean()), 4)] for c in COLS if c in new.columns and c in old.columns}
        same = [c for c in new.columns if c in old.columns and new[c].dtype.is_numeric()]
        st = new["book_state"].to_numpy()
        rep[name] = dict(verification=tot, provenance=man.get("book_provenance", {}).get("book_mode"), rows_1m=new.height, book_state={str(int(a)): int(b) for a, b in zip(*np.unique(st[np.isfinite(st)], return_counts=True))},
                         null_book_state=int(np.isnan(st).sum()), availability_old_new_1m=cmp_, columns=len(new.columns), old_columns=len(old.columns))
    Path(P.DERIVED / "regen_check" / "dry_run.json").write_text(json.dumps(rep, indent=1))
    for k, v in rep.items():
        print(k, json.dumps(v)[:1250])


if __name__ == "__main__":
    main()
