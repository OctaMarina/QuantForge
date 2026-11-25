"""Measured coverage of an assembled dataset: share of bars with a value, per family of the registry and bar set, in the whole history, the months without
anchors, the crash and the rest. Quality metadata and bar identity are excluded. Also key features, book tiers per scope, and bars touched by quality incidents.
Outputs: derived/datasets/<name>/coverage.csv, coverage_key.csv, coverage_book_state.csv
Usage: .venv/bin/python scripts/coverage_report.py <name> [workers]"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402
from chdds.dataset import META_COLS, QUALITY_COLS, incident_rows, load_incidents  # noqa: E402
from chdds.registry import match  # noqa: E402

D = lambda s: int(np.datetime64(s, "ns").astype(np.int64))  # noqa: E731
SCOPES = {"all": (D("2025-06-28"), D("2026-11-01")), "no_anchors": (D("2025-07-01"), D("2026-04-11")), "crash_2025-10-10": (D("2025-10-10"), D("2025-10-11")),
          "crash_window_2025-10-09_11": (D("2025-10-09"), D("2025-10-12"))}
KEY = ["spread_bps", "qi_1", "depth_bid_top10_btc", "depth_bid_5bps_btc", "depth_bid_10bps_btc", "depth_bid_25bps_btc", "depth_bid_50bps_btc", "cost_buy_1btc_bps", "ofi_30s_btc", "ofi_2m_btc", "ofi_15m_btc",
       "mlofi_l5_30s_btc", "tw_spread_bps_2m", "add_bid_10bps_2m_btc", "eff_spread_qw_bps_2m", "markout_30s_qw_bps_15m", "ret_trade_1h", "ret_trade_24h", "tfi_1h_btc", "rv_trade_1h_grid1s", "oi_btc_asof",
       "liq_count_15m", "liq_count_1h", "funding_rate_asof", "basis_mark_index_bps"]


def one(args):
    root, spec = args
    files = sorted(Path(root, spec).glob("*.parquet"))
    acc = {}
    states = {}
    inc = load_incidents()
    n_inc = 0
    for f in files:
        pf = pq.ParquetFile(f)
        names = [c for c in pf.schema_arrow.names if c not in META_COLS]
        t = pq.read_table(f, columns=["eval_time_ns", "t_start_ns", "book_state"])
        et = t["eval_time_ns"].to_numpy()
        st = t["book_state"].to_numpy(zero_copy_only=False)
        n_inc += int(incident_rows(t["t_start_ns"].to_numpy(), et, inc).sum())
        masks = {k: (et >= a) & (et < b) for k, (a, b) in SCOPES.items()}
        masks = {k: m for k, m in masks.items() if m.any()}
        for k, m in masks.items():
            v, c = np.unique(np.nan_to_num(st[m].astype(float), nan=-1), return_counts=True)
            for a, b in zip(v, c):
                states[(k, int(a))] = states.get((k, int(a)), 0) + int(b)
        for i in range(0, len(names), 64):
            part = names[i:i + 64]
            tb = pq.read_table(f, columns=part)
            for c in part:
                x = tb[c]
                arr = x.to_numpy(zero_copy_only=False)
                ok = np.isfinite(arr.astype(float)) if arr.dtype.kind in "fiub" else np.array([v is not None for v in arr])
                for k, m in masks.items():
                    a = acc.setdefault((k, c), [0, 0])
                    a[0] += int(ok[m].sum())
                    a[1] += int(m.sum())
    return spec, acc, states, n_inc


def main():
    name = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    root = P.DERIVED / "datasets" / name
    man = json.loads((root / "manifest.json").read_text())
    specs = list(man["bar_sets"])
    fam = {}
    rows, keyrows, strows, incrows = [], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for spec, acc, states, n_inc in ex.map(one, [(str(root), s) for s in specs]):
            bars = man["bar_sets"][spec]["name"]
            incrows.append(dict(bars=bars, rows_touching_incident=n_inc))
            per = {}
            for (k, c), (a, b) in acc.items():
                if c in QUALITY_COLS or "_state" in c or b == 0:      # state columns are always filled and would inflate the mean
                    continue
                if c not in fam:
                    h = match(c)
                    fam[c] = h[0].family if h else "?"
                per.setdefault((k, fam[c]), []).append(a / b)
                if c in KEY:
                    keyrows.append(dict(bars=bars, scope=k, feature=c, coverage=a / b, rows=b))
            for (k, f), v in per.items():
                rows.append(dict(bars=bars, scope=k, family=f, columns=len(v), mean_coverage=float(np.mean(v)), median_coverage=float(np.median(v))))
            for (k, s), n in states.items():
                strows.append(dict(bars=bars, scope=k, book_state=s, rows=n))
    pl.from_dicts(rows).sort("bars", "scope", "family").write_csv(root / "coverage.csv")
    pl.from_dicts(keyrows).sort("bars", "scope", "feature").write_csv(root / "coverage_key.csv")
    pl.from_dicts(strows).sort("bars", "scope", "book_state").write_csv(root / "coverage_book_state.csv")
    pl.from_dicts(incrows).write_csv(root / "coverage_incidents.csv")
    print("written", root)


if __name__ == "__main__":
    main()
