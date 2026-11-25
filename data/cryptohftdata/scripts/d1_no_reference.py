"""D1: what the optional rebuilt mode delivers where no reference exists, with the same measure on control days that do have one.

The production kernel runs in book_mode=rebuilt and NEVER sees a snapshot (so control days are comparable), on seeded sample
days of every month plus the two crash days. Queries every 5 s. Per day: book states, availability of every feature column,
reach of the certificate, restarts, crossings, and effective vs quoted spread. Effective spread uses trades, an independent
stream: a wrong top of book would show up as a gap between the two.
Outputs: derived/d1/no_ref/<day>.parquet (per column) and <day>.json. Usage: .venv/bin/python scripts/d1_no_reference.py [workers]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402
from chdds.book.driver import run_replay  # noqa: E402
from chdds.book.replay import ReplayConfig  # noqa: E402

OUT = P.DERIVED / "d1" / "no_ref"
SEED = 20261105
CRASH = [date(2025, 10, 10), date(2025, 10, 11)]


def sample_days():
    rng = np.random.default_rng(SEED)
    out = {}
    for y, m in [(2025, x) for x in range(7, 13)] + [(2026, x) for x in range(1, 11)]:
        first = date(y, m, 1)
        nxt = date(y + (m == 12), m % 12 + 1, 1)
        days = [first + timedelta(days=i) for i in range((nxt - first).days)]
        if (y, m) == (2026, 4):
            days = [d for d in days if d < date(2026, 3, 31) or d > date(2026, 4, 11)]
        n = 3 if (y, m) < (2026, 4) else 2
        for i in rng.choice(len(days), n, replace=False):
            out[days[int(i)]] = "unanchored" if days[int(i)] < date(2026, 4, 11) else "control"
    for d in CRASH:
        out[d] = "crash"
    return out


def run_day(d, group):
    p = OUT / f"{d.isoformat()}.json"
    if p.exists():
        return d, "cached"
    h0 = P.hour_start_ns(d, 0)
    q = np.arange(h0 + 5 * 10**9, h0 + 24 * P.HOUR_NS, 5 * 10**9, dtype=np.int64)
    cfg = ReplayConfig(book_mode="rebuilt", checkpoint_cert_levels=500)
    out, cols, stats, cnt, st = run_replay(d, d, q, OUT / "_tmp", cfg, workers=0, write_grid=False, chain_probe=3, log=lambda *_: None)
    ci = {c: i for i, c in enumerate(cols)}
    s = out[:, ci["book_state"]]
    v = np.unique(s[np.isfinite(s)], return_counts=True)
    ev = [e for r in stats if r.get("kind") != "verify" for e in json.loads(r["events"])]
    gaps = [e for e in ev if e["kind"] == "true_gap"]
    fl = out[:, ci["cert_bid_floor_dist_bps"]]
    ce = out[:, ci["cert_ask_ceiling_dist_bps"]]
    tw = out[:, ci["tw_spread_bps_2m"]]
    es = out[:, ci["eff_spread_qw_bps_2m"]]
    m = np.isfinite(tw) & np.isfinite(es) & (tw > 0)
    rec = dict(day=d.isoformat(), group=group, n=int(q.size), states={str(int(a)): int(b) for a, b in zip(*v)}, gaps=len(gaps),
               restarts=int(st.si[__import__("chdds.book.replay", fromlist=["I_NEMPTY"]).I_NEMPTY]), crossings=cnt["ncross"],
               cross_restarts=cnt["ncross_inval"], floor_q=[float(np.nanquantile(fl, x)) if np.isfinite(fl).any() else None for x in (.1, .5, .9)],
               ceil_q=[float(np.nanquantile(ce, x)) if np.isfinite(ce).any() else None for x in (.1, .5, .9)],
               eff_over_quoted=float(np.median(es[m] / tw[m])) if m.any() else None,
               eff_minus_quoted_bps_p99=float(np.quantile(np.abs(es[m] - tw[m]), .99)) if m.any() else None, counters=cnt)
    pl.DataFrame({"feature": cols, "share": [float(np.isfinite(out[:, j]).mean()) for j in range(len(cols))]}).write_parquet(OUT / f"{d.isoformat()}.parquet")
    p.write_text(json.dumps(rec, indent=1))
    return d, "ok"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    days = sample_days()
    print(len(days), "days", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_day, d, g): d for d, g in sorted(days.items())}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:
                print("FAILED", futs[f], repr(e), flush=True)


if __name__ == "__main__":
    main()
