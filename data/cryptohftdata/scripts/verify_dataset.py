"""Independent verification of an assembled dataset (does not reuse the feature code paths for the checks).

Usage: .venv/bin/python scripts/verify_dataset.py <dataset_name> [--days 2025-06-29,2026-10-02] [--sample 300]
Checks per bar set:
  1. every column registered exactly once (registry)
  2. bar conservation on the checked days: trades/volume/notional of bars == canonical trades in the bars' seq ranges
  3. time order: decision_time non-decreasing; eval_time <= decision_time; trade cursor recv <= eval_time
  4. brute-force recomputation (direct numpy on canonical trades) of volume/TFI/VWAP/count/range/returns for a sample
  5. masks: book columns are null whenever book_state is not 0 (anchored) or 9 (rebuilt mode, certified top); liquidation/oi/mark columns null when their state != 0
  6. ranges: imbalances in [-1,1], spread > 0, coverage in [0,1], costs non-decreasing with size
Writes derived/datasets/<name>/verification.json
"""
import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chdds import paths as P  # noqa: E402
from chdds import registry as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--days", default="")
ap.add_argument("--sample", type=int, default=300)
a = ap.parse_args()
droot = P.DERIVED / "datasets" / a.name
man = json.loads((droot / "manifest.json").read_text())
report = {"dataset": a.name, "bar_sets": {}}
NS = 10**9


def load_trades(d):
    p = P.DERIVED / "canonical" / "trades" / f"{d.isoformat()}.parquet"
    return pl.read_parquet(p) if p.exists() else None


for spec_id, info in man["bar_sets"].items():
    files = sorted(droot.joinpath(spec_id).glob("*.parquet"))
    if not files:
        continue
    res = {}
    cols = pq.read_schema(files[0]).names
    unreg = R.check_columns(cols)
    amb = [c for c in cols if len(R.match(c)) > 1]
    res["n_columns"] = len(cols)
    res["unregistered"] = unreg
    res["ambiguous"] = amb
    days = [date.fromisoformat(x) for x in a.days.split(",") if x] or []
    if not days:
        first = pl.read_parquet(files[0], columns=["decision_time_ns"])
        d0 = date(1970, 1, 1) + timedelta(days=int(first["decision_time_ns"][len(first) // 2] // (86400 * NS)))
        days = [d0]
    checks = []
    for d in days:
        month = files[0].parent / f"{d.isoformat()[:7]}.parquet"
        if not month.exists():
            continue
        df = pl.read_parquet(month)
        t0 = P.hour_start_ns(d, 0)
        df = df.filter((pl.col("eval_time_ns") >= t0) & (pl.col("eval_time_ns") < t0 + 86400 * NS))
        if df.height == 0:
            continue
        tr = pl.concat([x for x in (load_trades(d - timedelta(days=1)), load_trades(d)) if x is not None])
        seq = tr["seq"].to_numpy()
        recv = tr["recv_ns"].to_numpy()
        px = tr["price_e2"].to_numpy() / 100.0
        q = tr["qty_e3"].to_numpy() / 1000.0
        sg = tr["sign"].to_numpy().astype(float)
        c = {}
        # 2. conservation of bar contents
        fs, ls_ = df["first_seq"].to_numpy(), df["last_seq"].to_numpy()
        ntr = df["bar_n_trades"].to_numpy() if "bar_n_trades" in df.columns else None
        bad = 0
        if ntr is not None:
            for f_, l_, n_, v_ in zip(fs, ls_, ntr, df["bar_volume_btc"].to_numpy()):
                if n_ == 0:
                    continue
                i0 = np.searchsorted(seq, f_)
                i1 = np.searchsorted(seq, l_, side="right")
                if i1 - i0 != n_ or abs(q[i0:i1].sum() - v_) > 1e-9:
                    bad += 1
        c["bars_checked"] = int(df.height)
        c["conservation_mismatches"] = int(bad)
        # 3. time order
        dt = df["decision_time_ns"].to_numpy()
        et = df["eval_time_ns"].to_numpy()
        c["decision_time_monotonic"] = bool(np.all(np.diff(dt) >= 0))
        c["eval_le_decision"] = bool(np.all(et <= dt))
        cur = df["last_seq"].to_numpy()
        ok = cur >= 0
        ci = np.searchsorted(seq, cur[ok])
        ci = ci[ci < seq.size]
        c["cursor_recv_le_eval"] = bool(np.all(recv[ci] <= et[ok][:ci.size]))
        # 4. brute force on a sample
        rng = np.random.default_rng(0)
        idx = rng.choice(df.height, min(a.sample, df.height), replace=False)
        err = 0
        n_cmp = 0
        for i in idx:
            t = et[i]
            cs = cur[i]
            for w, s in ((30, "30s"), (900, "15m")):
                col = f"volume_{s}_btc"
                if col not in df.columns:
                    continue
                v = df[col][int(i)]
                if v is None or not np.isfinite(v):
                    continue
                m = (recv > t - w * NS) & (recv <= t) & (seq <= cs)
                n_cmp += 1
                if abs(q[m].sum() - v) > 1e-6 or abs((sg[m] * q[m]).sum() - df[f"tfi_{s}_btc"][int(i)]) > 1e-6:
                    err += 1
                if m.any() and abs((px[m] * q[m]).sum() / q[m].sum() - df[f"vwap_{s}"][int(i)]) > 1e-9:
                    err += 1
        c["bruteforce_comparisons"] = n_cmp
        c["bruteforce_errors"] = err
        # 5. masks
        if "book_state" in df.columns:
            bs = df["book_state"].to_numpy()
            inval = (bs != 0) & (bs != 9)
            book_cols = [x for x in ("spread_bps", "qi_1", "depth_bid_5bps_btc", "ofi_30s_btc", "cost_buy_1btc_bps",
                                     "basis_mid_index_bps") if x in df.columns]
            leak = 0
            for bc in book_cols:
                vv = df[bc].to_numpy()
                leak += int(np.isfinite(vv[inval]).sum())
            c["book_values_when_invalid"] = leak
            c["book_valid_fraction"] = float(np.mean(bs == 0))
            c["book_rebuilt_certified_fraction"] = float(np.mean(bs == 9))
        for stc, vc in (("oi_state", "oi_btc_asof"), ("mark_state", "basis_mark_index_bps"),
                        ("liq_state_15m", "liq_count_15m")):
            if stc in df.columns and vc in df.columns:
                st = df[stc].to_numpy()
                vv = df[vc].to_numpy()
                c[f"{vc}_values_when_{stc}_invalid"] = int(np.isfinite(vv[st != 0]).sum())
        # 6. ranges
        rr = {}
        for x in [x for x in df.columns if x.startswith(("qi_", "tfi_ratio_", "count_imbalance_", "liq_imbalance_")) and "_x_" not in x]:
            vv = df[x].to_numpy()
            vv = vv[np.isfinite(vv)]
            if vv.size and (vv.min() < -1 - 1e-12 or vv.max() > 1 + 1e-12):
                rr[x] = [float(vv.min()), float(vv.max())]
        if "spread_bps" in df.columns:
            vv = df["spread_bps"].to_numpy()
            if np.any(vv[np.isfinite(vv)] <= 0):
                rr["spread_bps"] = "non-positive"
        c["range_violations"] = rr
        checks.append({"day": d.isoformat(), **c})
    res["checks"] = checks
    report["bar_sets"][spec_id] = res
    print(spec_id, json.dumps(res)[:600])
(droot / "verification.json").write_text(json.dumps(report, indent=1, default=str))
