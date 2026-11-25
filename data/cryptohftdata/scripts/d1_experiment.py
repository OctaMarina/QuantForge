"""D1 experiment: a reference book vs books started EMPTY inside intervals with a valid reference.

For every interval of the pre-declared plan (scripts/d1_select_intervals.py -> configs/d1_intervals.json):
  R        strict replay from a snapshot + continuous updates, checkpoints certified only to their best 500 levels.
  E(T)     independent book started empty at T: never sees a snapshot, receives exactly the same messages from T on.
           forced run  : every feature is emitted regardless of the certificate (measures the time warm-up)
           cert run    : what the optional mode would emit (only inside the protocol certificate)
Comparison happens only where R itself is valid and certified. Outside R's certified domain nothing is claimed.

Per E run, derived/d1/runs/<interval>/<start>_<kind>.parquet holds, for every query:
  elapsed, E state, top-level agreement per side (first differing rank, verifiable levels, levels missing/wrong/extra) and the
  raw E and R values of the selected features. <start>_<kind>_summary.parquet holds, for EVERY feature column and elapsed bin,
  counts of both-present / equal / E-only / R-only and absolute error sums.
Usage: .venv/bin/python scripts/d1_experiment.py [--plan configs/d1_intervals.json] [--workers 4] [--only id,id]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402
from chdds.book import chain as CH  # noqa: E402
from chdds.book.driver import TradeSource, load_book_hour, run_replay  # noqa: E402
from chdds.book.replay import ReplayConfig  # noqa: E402

K_TOP = 50
CERT_LEVELS = 500
DENSE = np.array([.05, .1, .2, .3, .5, .75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 90, 120, 150,
                  180, 240, 300])
EL_BINS = np.array([0, 1, 5, 10, 30, 60, 120, 300, 600, 900, 1800, 3600, 1e9])
RUNS = P.DERIVED / "d1" / "runs"

SELECT = ["best_bid", "best_ask", "mid", "spread_bps", "qty_bid1_btc", "qty_ask1_btc", "qi_1", "weighted_mid_offset_bps",
          "depth_bid_top5_btc", "depth_ask_top5_btc", "qi_top5", "depth_bid_top10_btc", "depth_ask_top10_btc", "qi_top10",
          "depth_bid_top20_btc", "depth_ask_top20_btc", "qi_top20", "depth_bid_1bps_btc", "depth_ask_1bps_btc", "qi_1bps",
          "depth_bid_5bps_btc", "depth_ask_5bps_btc", "qi_5bps", "depth_bid_10bps_btc", "depth_ask_10bps_btc", "qi_10bps",
          "depth_bid_25bps_btc", "depth_ask_25bps_btc", "qi_25bps", "depth_bid_50bps_btc", "depth_ask_50bps_btc", "qi_50bps",
          "cost_buy_1btc_bps", "cost_sell_1btc_bps", "cost_buy_5btc_bps", "cost_sell_5btc_bps",
          "cert_bid_floor_dist_bps", "cert_ask_ceiling_dist_bps"]
for _w in ("1s", "5s", "30s", "2m", "15m"):
    SELECT += [f"book_coverage_{_w}", f"ofi_{_w}_btc", f"ofi_norm_{_w}", f"tw_spread_bps_{_w}", f"tw_qi_1_{_w}",
               f"tw_depth1_{_w}_btc", f"tw_depth_bid_5bps_{_w}_btc", f"tw_depth_ask_5bps_{_w}_btc",
               f"tw_depth_bid_25bps_{_w}_btc", f"tw_depth_ask_25bps_{_w}_btc",
               f"add_bid_10bps_{_w}_btc", f"rem_bid_10bps_{_w}_btc", f"add_ask_10bps_{_w}_btc", f"rem_ask_10bps_{_w}_btc",
               f"book_msg_count_{_w}"]
for _w in ("5s", "30s"):
    SELECT += [f"mlofi_l{_r}_{_w}_btc" for _r in (1, 2, 5)]
SELECT += ["eff_spread_qw_bps_30s", "eff_spread_qw_bps_2m", "basis_mid_index_bps"]

_cache = {}
_trades = TradeSource()


def hour_loader(day_iso, hour):
    k = (day_iso, hour)
    if k not in _cache:
        while len(_cache) >= 6:
            _cache.pop(next(iter(_cache)))
        _cache[k] = load_book_hour(day_iso, hour)
    return _cache[k]


def hours_between(t0, t1):
    a = P.hour_index(t0) if hasattr(P, "hour_index") else None
    out = []
    t = t0 - t0 % P.HOUR_NS
    while t <= t1:
        dt = datetime.fromtimestamp(t / 1e9, timezone.utc)
        out.append((dt.date(), dt.hour))
        t += P.HOUR_NS
    return out


def _cfg(**kw):
    return ReplayConfig(checkpoint_cert_levels=CERT_LEVELS, **kw)


def run_one(hours, q, cfg, **kw):
    ex = {}
    out, cols, stats, counters, st = run_replay(
        hours[0][0], hours[-1][0], q, RUNS, cfg, workers=0, hours=hours, write_grid=False, skip_unanchored=False,
        top_levels=K_TOP, loader=hour_loader, trades=_trades, extras=ex, log=lambda *_: None, **kw)
    ev = []
    for r in stats:
        if r.get("kind") != "verify":
            ev += json.loads(r["events"])
    ver = [r for r in stats if r.get("kind") == "verify"]
    return out, cols, ex["top"], ev, ver, counters


@njit(cache=True)
def compare_top(xr, xe, K):
    """Per query and side: first differing rank (= how many leading levels agree), levels verifiable in R, and inside R's
    verifiable range the levels missing in E, present with another size, and present in E but absent from R."""
    n = xr.shape[0]
    res = np.zeros((n, 10), np.int32)
    for i in range(n):
        for sd in range(2):
            o = 2 * K * sd
            nr = 0
            while nr < K and xr[i, o + 2 * nr] >= 0:
                nr += 1
            mr = 0
            while mr < nr and xe[i, o + 2 * mr] == xr[i, o + 2 * mr] and xe[i, o + 2 * mr + 1] == xr[i, o + 2 * mr + 1]:
                mr += 1
            miss = 0
            wrong = 0
            extra = 0
            a = 0
            b = 0
            ne = 0
            while ne < K and xe[i, o + 2 * ne] >= 0:
                ne += 1
            while a < nr and b < ne:
                pr = xr[i, o + 2 * a]
                pe = xe[i, o + 2 * b]
                if pr == pe:
                    if xr[i, o + 2 * a + 1] != xe[i, o + 2 * b + 1]:
                        wrong += 1
                    a += 1
                    b += 1
                elif (sd == 0 and pr > pe) or (sd == 1 and pr < pe):
                    miss += 1
                    a += 1
                else:
                    extra += 1
                    b += 1
            miss += nr - a
            res[i, 5 * sd] = mr
            res[i, 5 * sd + 1] = nr
            res[i, 5 * sd + 2] = miss
            res[i, 5 * sd + 3] = wrong
            res[i, 5 * sd + 4] = extra
    return res


def feature_summary(cols, outR, outE, el):
    """All feature columns: agreement by elapsed bin, restricted to queries where R is valid."""
    b = np.digitize(el, EL_BINS) - 1
    rows = []
    for j, c in enumerate(cols):
        if c in ("book_state", "book_recv_age_ms", "book_mkt_age_ms"):
            continue
        r, e = outR[:, j], outE[:, j]
        hr, he = np.isfinite(r), np.isfinite(e)
        both = hr & he
        d = np.abs(r - e)
        eq = both & (d <= 1e-9 * (1 + np.abs(r)))
        for k in np.unique(b):
            m = b == k
            mb = m & both
            rows.append((c, float(EL_BINS[k]), int(m.sum()), int(mb.sum()), int((m & eq).sum()), int((m & he & ~hr).sum()),
                         int((m & hr & ~he).sum()), float(d[mb].sum()), float(d[mb].max()) if mb.any() else 0.0,
                         float(np.abs(r[mb]).sum())))
    return pl.DataFrame(rows, orient="row", schema=["feature", "el_lo", "n", "n_both", "n_eq", "n_e_only", "n_r_only", "sum_abs",
                                                     "max_abs", "sum_abs_ref"])


def run_interval(spec):
    t_start = time.time()
    d = RUNS / spec["id"]
    d.mkdir(parents=True, exist_ok=True)
    if (d / "DONE.json").exists():
        return spec["id"], 0.0, "cached"
    Ta, Tb, H = spec["t_anchor"], spec["t_end"], spec["horizon_s"]
    # queries: a regular 1 s grid + dense early offsets per start; all aligned between R and E
    grid = {}
    allq = []
    for s in spec["starts"]:
        T = s["t"]
        end = min(T + int(H * 1e9), Tb)
        q = np.concatenate([T + (DENSE * 1e9).astype(np.int64),
                            np.arange(T + 10**9, end + 1, 10**9, dtype=np.int64)])
        q = np.unique(q[q <= end])
        grid[s["name"]] = q
        allq.append(q)
    allq = np.unique(np.concatenate(allq))
    hR = hours_between(Ta, Tb)
    if spec.get("t_ref_start"):
        hR = hours_between(spec["t_ref_start"], Tb)
    # ---- reference
    # same kernel settings as E (only the markout boundary handling differs from strict); the resolver is strict
    outR, cols, topR, evR, verR, cR = run_one(hR, allq, _cfg(book_mode="rebuilt"), chain_heal=0)
    ci = {c: i for i, c in enumerate(cols)}
    sel = [c for c in SELECT if c in ci]
    missing = [c for c in SELECT if c not in ci]
    assert not missing, missing
    stR = outR[:, ci["book_state"]]
    meta = dict(id=spec["id"], ref_counters=cR, ref_valid_share=float((stR == 0).mean()),
                ref_verifications=len(verR), ref_mismatches=int(sum(v["mismatches"] for v in verR)),
                ref_events=[e for e in evR if e["kind"] in ("true_gap", "anchor_noconnect", "dup_conflict")], runs={})
    for s in spec["starts"]:
        T = s["t"]
        q = grid[s["name"]]
        iq = np.searchsorted(allq, q)
        hE = hours_between(T, int(q[-1]))
        el = (q - T) / 1e9
        for kind, forced in (("forced", True), ("cert", False)):
            outE, _, topE, evE, verE, cE = run_one(
                hE, q, _cfg(book_mode="rebuilt", experiment_emit_healed=forced), chain_probe=3, start_after=T)
            gaps = [e for e in evE if e["kind"] == "true_gap" and e["t0"] > T]
            life_end = min((e["t0"] for e in gaps), default=int(q[-1]) + 1)
            keep = q < life_end
            Rk = outR[iq][keep]
            Ek = outE[keep]
            cmp_ = compare_top(np.ascontiguousarray(topR[iq][keep]), np.ascontiguousarray(topE[keep]), K_TOP)
            okR = Rk[:, ci["book_state"]] == 0
            tab = {"t": q[keep], "el": el[keep], "state_e": Ek[:, ci["book_state"]].astype(np.int8),
                   "state_r": Rk[:, ci["book_state"]].astype(np.int8)}
            for j, nme in enumerate(("mr_b", "nr_b", "miss_b", "wrong_b", "extra_b", "mr_a", "nr_a", "miss_a", "wrong_a",
                                     "extra_a")):
                tab[nme] = cmp_[:, j]
            for c in sel:
                tab["e_" + c] = Ek[:, ci[c]].astype(np.float64 if c in ("best_bid", "best_ask", "mid") else np.float32)
                tab["r_" + c] = Rk[:, ci[c]].astype(np.float64 if c in ("best_bid", "best_ask", "mid") else np.float32)
            pl.DataFrame(tab).write_parquet(d / f"{s['name']}_{kind}.parquet", compression="zstd")
            feature_summary(cols, Rk[okR], Ek[okR], el[keep][okR]).write_parquet(d / f"{s['name']}_{kind}_summary.parquet")
            meta["runs"][f"{s['name']}_{kind}"] = dict(
                t=T, life_end=life_end, n=int(keep.sum()), n_ref_valid=int(okR.sum()), n_gaps=len(gaps), counters=cE,
                ncross=cE["ncross"], restart_after_gap=len(gaps))
    (d / "DONE.json").write_text(json.dumps(meta, indent=1))
    _cache.clear()
    return spec["id"], time.time() - t_start, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="configs/d1_intervals.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    specs = plan["intervals"]
    if a.only:
        ids = set(a.only.split(","))
        specs = [s for s in specs if s["id"] in ids]
    print(len(specs), "intervals", flush=True)
    if a.workers <= 1:
        for s in specs:
            print(run_interval(s), flush=True)
        return
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_interval, s): s["id"] for s in specs}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:  # keep going: a failed interval is reported, never silently dropped
                print("FAILED", futs[f], repr(e), flush=True)


if __name__ == "__main__":
    main()
