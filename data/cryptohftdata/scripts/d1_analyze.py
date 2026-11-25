"""D1 analysis. Every command reads only what it is told to, so the warm-up policy can be fixed on calibration days before
validation days are looked at:

  runs --split S       explicit parallel books: integrity, certificate soundness, convergence, feature errors
  age  --split S       large-sample ages: pointwise mismatch by delta, persistence for every start, band shortfall
  policy               chooses the warm-ups from CALIBRATION tables only (rule below), writes results/policy.json
  validate             applies results/policy.json unchanged to VALIDATION tables and lists every failure
Outputs: derived/d1/results/
Rule (fixed before any table was seen), top-K family: smallest delta on the grid with aggregate mismatch <= 0.1 % AND the
99th percentile of per-hour mismatch <= 1 %. Depth bands: smallest delta with 99th percentile shortfall <= 1 % of the band.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402

RES = P.DERIVED / "d1" / "results"
RUNS = P.DERIVED / "d1" / "runs"
AGE = P.DERIVED / "d1" / "age"
KS = (1, 5, 10, 50)
EL = np.array([0, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 900, 1800, 3601], float)
LAG = np.array([0, 1, 5, 10, 30, 60, 120, 300, 600, 900, 1800, 3601], float)
DELTAS = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600, 900, 1200, 1800, 2700, 3600]
WINDOWS = {"1s": 1, "5s": 5, "30s": 30, "2m": 120, "15m": 900}
INSTANT = ["spread_bps", "qty_bid1_btc", "qty_ask1_btc", "qi_1", "weighted_mid_offset_bps", "depth_bid_top5_btc", "qi_top5",
           "depth_bid_top10_btc", "qi_top10", "depth_bid_top20_btc", "qi_top20", "depth_bid_1bps_btc", "depth_ask_1bps_btc",
           "qi_1bps", "depth_bid_5bps_btc", "depth_ask_5bps_btc", "qi_5bps", "depth_bid_10bps_btc", "depth_ask_10bps_btc",
           "qi_10bps", "depth_bid_25bps_btc", "depth_ask_25bps_btc", "qi_25bps", "cost_buy_1btc_bps", "cost_sell_1btc_bps",
           "cost_buy_5btc_bps", "cost_sell_5btc_bps"]
WIN_FAM = ["ofi_{w}_btc", "ofi_norm_{w}", "tw_spread_bps_{w}", "tw_qi_1_{w}", "tw_depth1_{w}_btc", "tw_depth_bid_5bps_{w}_btc",
           "tw_depth_bid_25bps_{w}_btc", "add_bid_10bps_{w}_btc", "rem_bid_10bps_{w}_btc", "book_msg_count_{w}"]


def plan():
    return json.loads(Path("configs/d1_intervals.json").read_text())


def day_of_ns(t):
    return (t // (86400 * 10**9))


def split_of_days(pl_):
    from datetime import date
    return {(date.fromisoformat(d) - date(1970, 1, 1)).days: s for d, s in pl_["split_days"].items()}


def q(x, p):
    return float(np.quantile(x, p)) if x.size else float("nan")


def load_forced(path):
    """Rows of a run with the reference valid, elapsed time measured from the first message the book really received. If the
    feed is dead at T the book starts later (state 1 until then), and counting from T would overstate its age."""
    f = pl.read_parquet(path)
    if f.height == 0:
        return f
    idle = f.filter(pl.col("state_e") == 1)
    d0 = float(idle["el"].max()) if idle.height else 0.0
    return f.filter((pl.col("state_r") == 0) & (pl.col("el") > d0)).with_columns((pl.col("el") - d0).alias("el"))


# ------------------------------------------------------------------------------------------------ explicit runs
def runs(split):
    pln = plan()
    ivs = [i for i in pln["intervals"] if split == "all" or i["split"] == split]
    conv, integ, feat_acc, win_acc, cert_viol, cert_rows = [], [], {}, {}, [], []
    for iv in ivs:
        d = RUNS / iv["id"]
        if not (d / "DONE.json").exists():
            print("missing", iv["id"])
            continue
        meta = json.loads((d / "DONE.json").read_text())
        for s in iv["starts"]:
            nm = s["name"]
            base = dict(interval=iv["id"], start=nm, regime=iv["regime"], vol=iv["vol"], liq=iv["liq"], cell=iv["cell"],
                        split=iv["split"], ref_valid=meta["ref_valid_share"], ref_ver=meta["ref_verifications"],
                        ref_mis=meta["ref_mismatches"])
            f = load_forced(d / f"{nm}_forced.parquet")
            r_info = meta["runs"][f"{nm}_forced"]
            if f.height == 0:
                continue
            el = f["el"].to_numpy()
            mrb, mra = f["mr_b"].to_numpy(), f["mr_a"].to_numpy()
            nrb, nra = f["nr_b"].to_numpy(), f["nr_a"].to_numpy()
            integ.append(dict(base, n=f.height, wrong=int(f["wrong_b"].sum() + f["wrong_a"].sum()),
                              extra=int(f["extra_b"].sum() + f["extra_a"].sum()), crossings=r_info["ncross"],
                              gaps=r_info["n_gaps"], horizon=float(el.max())))
            rec = dict(base, horizon=float(el.max()))
            for K in KS:
                ver = (nrb >= K) & (nra >= K)
                ex = ver & (mrb >= K) & (mra >= K)
                bad = ver & ~ex
                rec[f"n_ver{K}"] = int(ver.sum())
                rec[f"first{K}"] = float(el[ex][0]) if ex.any() else float("nan")
                rec[f"last_fail{K}"] = float(el[bad][-1]) if bad.any() else 0.0
                rec[f"relapse{K}"] = int(bad.any() and ex.any() and el[bad][-1] > el[ex][0])
            conv.append(rec)
            for c in INSTANT:
                e, r = f["e_" + c].to_numpy(), f["r_" + c].to_numpy()
                m = np.isfinite(e) & np.isfinite(r)
                b = np.digitize(el[m], EL) - 1
                err = np.abs(e[m] - r[m])
                rel = err / np.maximum(np.abs(r[m]), 1e-9)
                for k in np.unique(b):
                    feat_acc.setdefault((c, int(k)), []).append((err[b == k], rel[b == k]))
            for w, ws in WINDOWS.items():
                for tmpl in WIN_FAM:
                    c = tmpl.format(w=w)
                    e, r = f["e_" + c].to_numpy(), f["r_" + c].to_numpy()
                    m = np.isfinite(e) & np.isfinite(r)
                    if not m.any():
                        continue
                    lag = el[m] - ws
                    b = np.digitize(lag, LAG) - 1
                    for k in np.unique(b):
                        sel = b == k
                        win_acc.setdefault((tmpl, w, int(k)), []).append((e[m][sel], r[m][sel]))
            # certificate tier: every emitted value must equal the reference
            g = pl.read_parquet(d / f"{nm}_cert.parquet").filter(pl.col("state_r") == 0)
            top = g.filter(pl.col("state_e") == 9)
            bad_top = int(((top["mr_b"] < 1) | (top["mr_a"] < 1)).sum()) if top.height else 0
            sm = pl.read_parquet(d / f"{nm}_cert_summary.parquet")
            sv = sm.group_by("feature").agg(pl.col("n_both").sum(), pl.col("n_eq").sum(), pl.col("n_e_only").sum(),
                                            pl.col("n_r_only").sum(), pl.col("max_abs").max())
            cert_rows.append(sv.with_columns(interval=pl.lit(iv["id"]), start=pl.lit(nm)))
            cert_viol.append(dict(base, n=g.height, n_top_cert=top.height, top_cert_share=top.height / max(g.height, 1),
                                  top_wrong=bad_top, wrong=int(g["wrong_b"].sum() + g["wrong_a"].sum()),
                                  extra=int(g["extra_b"].sum() + g["extra_a"].sum()),
                                  first_cert=float(top["el"][0]) if top.height else float("nan"),
                                  floor_p50=float(top["e_cert_bid_floor_dist_bps"].median() or 0) if top.height else 0.0,
                                  ceil_p50=float(top["e_cert_ask_ceiling_dist_bps"].median() or 0) if top.height else 0.0))
    RES.mkdir(parents=True, exist_ok=True)
    pl.from_dicts(conv).write_csv(RES / f"runs_convergence_{split}.csv")
    pl.from_dicts(integ).write_csv(RES / f"runs_integrity_{split}.csv")
    pl.from_dicts(cert_viol).write_csv(RES / f"runs_certificate_{split}.csv")
    cf = pl.concat(cert_rows).group_by("feature").agg(pl.col("n_both").sum(), pl.col("n_eq").sum(), pl.col("n_e_only").sum(),
                                                      pl.col("n_r_only").sum(), pl.col("max_abs").max())
    cf.with_columns(n_diff=pl.col("n_both") - pl.col("n_eq")).sort("n_diff", descending=True).write_csv(
        RES / f"runs_certificate_features_{split}.csv")
    rows = []
    for (c, k), v in sorted(feat_acc.items()):
        err = np.concatenate([x[0] for x in v])
        rel = np.concatenate([x[1] for x in v])
        rows.append(dict(feature=c, el_lo=EL[k], n=err.size, equal=float((err <= 1e-6 * 1).mean()), err_p50=q(err, .5), err_p99=q(err, .99),
                         err_max=float(err.max()), rel_p50=q(rel, .5), rel_p99=q(rel, .99), rel_max=float(rel.max())))
    pl.from_dicts(rows).write_csv(RES / f"runs_features_{split}.csv")
    rows = []
    for (tmpl, w, k), v in sorted(win_acc.items()):
        e = np.concatenate([x[0] for x in v]).astype(float)
        r = np.concatenate([x[1] for x in v]).astype(float)
        err = np.abs(e - r)
        sc = max(q(np.abs(r), .5), 1e-9)
        corr = float(np.corrcoef(e, r)[0, 1]) if e.size > 2 and e.std() > 0 and r.std() > 0 else float("nan")
        rows.append(dict(feature=tmpl, window=w, lag_lo=LAG[k], n=e.size, equal=float((err <= 1e-5 * (1 + np.abs(r))).mean()),
                         err_p50=q(err, .5), err_p99=q(err, .99), err_max=float(err.max()), scale_p50=sc,
                         err_p99_over_scale=q(err, .99) / sc, sign_agree=float((np.sign(e) == np.sign(r)).mean()), corr=corr))
    pl.from_dicts(rows).write_csv(RES / f"runs_windows_{split}.csv")
    print(split, len(conv), "runs")


# ------------------------------------------------------------------------------------------------ large sample
@njit(cache=True)
def persist(ts, age, H, stride):
    """For starts T = ts[0::stride], using exact_at(t) <=> age(t) <= t - T (ms): first exact time and end of the last failure
    within the horizon. Starts whose horizon runs past the data are marked -2."""
    n = ts.shape[0]
    ns = (n + stride - 1) // stride
    first = np.full(ns, -1, np.int64)
    last = np.zeros(ns, np.int64)
    for j in range(ns):
        T = ts[j * stride]
        if T + H > ts[n - 1]:
            first[j] = -2
            continue
        i = j * stride
        lf = 0
        fe = -1
        while i < n and ts[i] <= T + H:
            el = ts[i] - T
            if age[i] <= el:
                if fe < 0:
                    fe = el
            else:
                lf = el
            i += 1
        first[j] = fe
        last[j] = lf
    return first, last


def age_tables(split):
    pln = plan()
    sd = split_of_days(pln)
    ht = pl.read_parquet(P.DERIVED / "d1" / "hour_table.parquet").select("hour_ns", "rv_1h", "depth5", "notional")
    tv, tl = pln["terciles"]["vol"], pln["terciles"]["liq"]
    point, starts, short = [], [], []
    from d1_select_intervals import regime
    for dpath in sorted(glob.glob(str(AGE / "*"))):
        p = Path(dpath) / "ages.parquet"
        if not p.exists():
            continue
        a = pl.read_parquet(p)
        if a.height < 3600:
            continue
        a = a.with_columns(day=(pl.col("t") // (86400 * 10**9)), hour_ns=pl.col("t") // P.HOUR_NS * P.HOUR_NS)
        a = a.with_columns(split=pl.col("day").replace_strict(sd, default="unassigned"))
        if split != "all":
            a = a.filter(pl.col("split") == split)
        if a.height == 0:
            continue
        for K in KS:
            a = a.with_columns(pl.max_horizontal(f"age{K}_b", f"age{K}_a").alias(f"m{K}"),
                               (pl.min_horizontal(f"age{K}_b", f"age{K}_a") >= 0).alias(f"v{K}"))
        point.append(a.select(["t", "ta", "hour_ns", "spread_ticks"] + [f"m{K}" for K in KS] + [f"v{K}" for K in KS]))
        # persistence on 5 s starts, per contiguous anchor
        for key, g in a.group_by("ta", maintain_order=True):
            if g.height < 3700:
                continue
            ts = g["t"].to_numpy() // 1_000_000
            for K in KS:
                v = g[f"v{K}"].to_numpy()
                tk = ts[v]
                if tk.size < 3700:
                    continue
                f_, l_ = persist(tk, g[f"m{K}"].to_numpy()[v].astype(np.int64), 3_600_000, 5)
                m = f_ != -2
                t0 = tk[::5][m]
                starts.append(pl.DataFrame({"t_ms": t0, "K": np.full(t0.size, K, np.int16), "first_ms": f_[m], "last_fail_ms": l_[m]}))
        s = Path(dpath) / "short.parquet"
        if s.exists():
            sh = pl.read_parquet(s).join(a.select("t", "ta"), on="t", how="inner")
            short.append(sh)
    pt = pl.concat(point)
    pt = pt.with_columns(reg=(pl.col("t") // (86400 * 10**9)))
    dayiso = {x: None for x in pt["reg"].unique().to_list()}
    from datetime import date, timedelta
    reg_map = {k: regime((date(1970, 1, 1) + timedelta(days=k)).isoformat()) for k in dayiso}
    pt = pt.with_columns(regime=pl.col("reg").replace_strict(reg_map)).join(ht, on="hour_ns", how="left")
    pt = pt.with_columns(vol=pl.when(pl.col("rv_1h") < tv[0]).then(0).when(pl.col("rv_1h") < tv[1]).then(1).otherwise(2),
                         liq=pl.when(pl.col("depth5").is_null()).then(-1).when(pl.col("depth5") < tl[0]).then(0)
                         .when(pl.col("depth5") < tl[1]).then(1).otherwise(2))
    RES.mkdir(parents=True, exist_ok=True)
    rows, hours_rows = [], []
    for K in KS:
        g = pt.filter(pl.col(f"v{K}"))
        for dl in DELTAS:
            x = g.filter((pl.col("t") - pl.col("ta")) >= dl * 10**9).with_columns(fail=(pl.col(f"m{K}") > dl * 1000))
            for lab, sub in [("ALL", x)] + [(r[0], t) for r, t in x.group_by(["regime"])] + \
                    [(f"vol{r[0]}", t) for r, t in x.group_by(["vol"])] + [(f"liq{r[0]}", t) for r, t in x.group_by(["liq"])]:
                h = sub.group_by("hour_ns").agg(pl.col("fail").mean())["fail"].to_numpy()
                rows.append(dict(K=K, delta=dl, group=lab, n=sub.height, fail=float(sub["fail"].mean()), hours=h.size,
                                 hour_p50=q(h, .5), hour_p99=q(h, .99), hour_max=float(h.max()) if h.size else float("nan"),
                                 hours_zero=float((h == 0).mean()) if h.size else float("nan")))
    pl.from_dicts(rows).sort(["group", "K", "delta"]).write_csv(RES / f"age_pointwise_{split}.csv")
    # worst hours at each delta are kept for the failure tables
    st = pl.concat(starts).with_columns(hour_ns=(pl.col("t_ms") * 1_000_000) // P.HOUR_NS * P.HOUR_NS)
    st = st.with_columns(reg=(pl.col("t_ms") // 86_400_000))
    reg_map2 = {k: regime((date(1970, 1, 1) + timedelta(days=k)).isoformat()) for k in st["reg"].unique().to_list()}
    st = st.with_columns(regime=pl.col("reg").replace_strict(reg_map2))
    rows = []
    for K in KS:
        g = st.filter(pl.col("K") == K)
        for lab, sub in [("ALL", g)] + [(r[0], t) for r, t in g.group_by(["regime"])]:
            f_ = sub["first_ms"].to_numpy() / 1e3
            l_ = sub["last_fail_ms"].to_numpy() / 1e3
            ok = f_ >= 0
            rows.append(dict(K=K, group=lab, starts=sub.height, never_exact=float((~ok).mean()), first_p50=q(f_[ok], .5),
                             first_p99=q(f_[ok], .99), first_max=float(f_[ok].max()) if ok.any() else float("nan"),
                             stable_p50=q(l_, .5), stable_p90=q(l_, .9), stable_p99=q(l_, .99), stable_p999=q(l_, .999),
                             stable_max=float(l_.max()), relapse=float((l_[ok] > f_[ok]).mean()) if ok.any() else float("nan")))
    pl.from_dicts(rows).sort(["group", "K"]).write_csv(RES / f"age_persistence_{split}.csv")
    sh = pl.concat(short, how="diagonal")
    rows = []
    for b in (1, 5, 10, 25):
        for sd_ in ("b", "a"):
            for dl in (5, 10, 30, 60, 120, 300, 600, 900, 1800, 3600):
                c = f"f{b}_{sd_}_{dl}"
                x = sh.filter(((pl.col("t") - pl.col("ta")) >= dl * 10**9) & pl.col(c).is_not_nan() & pl.col(c).is_not_null())[c].to_numpy()
                if x.size == 0:
                    continue
                s_ = 1 - x
                rows.append(dict(band=b, side=sd_, delta=dl, n=x.size, exact=float((s_ <= 0).mean()), short_p50=q(s_, .5), short_p99=q(s_, .99),
                                 short_p999=q(s_, .999), short_max=float(s_.max())))
    pl.from_dicts(rows).write_csv(RES / f"age_bands_{split}.csv")
    print(split, "age samples", pt.height, "starts", st.height)


# ------------------------------------------------------------------------------------------------ policy
def policy():
    t = pl.read_csv(RES / "age_pointwise_calibration.csv").filter(pl.col("group") == "ALL")
    b = pl.read_csv(RES / "age_bands_calibration.csv")
    pol = {"rule": "top-K: aggregate mismatch <= 0.1% and p99 of per-hour mismatch <= 1%; bands: p99 shortfall <= 1%",
           "top": {}, "band": {}}
    for K in KS:
        g = t.filter((pl.col("K") == K) & (pl.col("fail") <= 1e-3) & (pl.col("hour_p99") <= 1e-2)).sort("delta")
        pol["top"][str(K)] = int(g["delta"][0]) if g.height else None
    for bd in (1, 5, 10, 25):
        g = b.filter(pl.col("band") == bd).group_by("delta").agg(pl.col("short_p99").max()).filter(pl.col("short_p99") <= 1e-2).sort("delta")
        pol["band"][str(bd)] = int(g["delta"][0]) if g.height else None
    (RES / "policy.json").write_text(json.dumps(pol, indent=1))
    print(json.dumps(pol, indent=1))


def validate():
    pol = json.loads((RES / "policy.json").read_text())
    t = pl.read_csv(RES / "age_pointwise_validation.csv")
    b = pl.read_csv(RES / "age_bands_validation.csv")
    rows = []
    for K, dl in pol["top"].items():
        if dl is None:
            rows.append(dict(family=f"top{K}", delta=None, group="ALL", n=0, fail=None, hour_p99=None, hour_max=None, passes=False))
            continue
        g = t.filter((pl.col("K") == int(K)) & (pl.col("delta") == dl))
        for r in g.iter_rows(named=True):
            rows.append(dict(family=f"top{K}", delta=dl, group=r["group"], n=r["n"], fail=r["fail"], hour_p99=r["hour_p99"],
                             hour_max=r["hour_max"], passes=bool(r["fail"] <= 1e-3 and r["hour_p99"] <= 1e-2)))
    pl.from_dicts(rows).write_csv(RES / "validation_top.csv")
    rows = []
    for bd, dl in pol["band"].items():
        if dl is None:
            continue
        for r in b.filter((pl.col("band") == int(bd)) & (pl.col("delta") == dl)).iter_rows(named=True):
            rows.append(dict(r, passes=bool(r["short_p99"] <= 1e-2)))
    if rows:
        pl.from_dicts(rows).write_csv(RES / "validation_bands.csv")
    print(pl.read_csv(RES / "validation_top.csv").filter(pl.col("group") == "ALL"))


def family_delta(pol):
    """feature template -> (warm-up family, window seconds). Windows need the warm-up of their state plus the window itself."""
    top, band = pol["top"], pol["band"]
    m = {"spread_bps": top["1"], "qty_bid1_btc": top["1"], "qty_ask1_btc": top["1"], "qi_1": top["1"],
         "weighted_mid_offset_bps": top["1"], "depth_bid_top5_btc": top["5"], "qi_top5": top["5"], "depth_bid_top10_btc": top["10"],
         "qi_top10": top["10"], "depth_bid_top20_btc": top["50"], "qi_top20": top["50"], "cost_buy_1btc_bps": top["10"],
         "cost_sell_1btc_bps": top["10"], "cost_buy_5btc_bps": top["50"], "cost_sell_5btc_bps": top["50"]}
    for b in (1, 5, 10, 25):
        for c in (f"depth_bid_{b}bps_btc", f"depth_ask_{b}bps_btc", f"qi_{b}bps"):
            m[c] = band[str(b)]
    w = {}
    for ws, sec in WINDOWS.items():
        for tmpl, d in (("ofi_{w}_btc", top["1"]), ("ofi_norm_{w}", top["1"]), ("tw_spread_bps_{w}", top["1"]), ("tw_qi_1_{w}", top["1"]),
                        ("tw_depth1_{w}_btc", top["1"]), ("tw_depth_bid_5bps_{w}_btc", band["5"]), ("tw_depth_bid_25bps_{w}_btc", band["25"]),
                        ("add_bid_10bps_{w}_btc", band["10"]), ("rem_bid_10bps_{w}_btc", band["10"]), ("book_msg_count_{w}", 0)):
            w[tmpl.format(w=ws)] = (d, sec)
    return m, w


def validate_runs(split="validation"):
    """Policy applied unchanged to the explicit parallel books: value errors after the warm-up, and every run that fails."""
    pol = json.loads((RES / "policy.json").read_text())
    inst, win = family_delta(pol)
    pln = plan()
    acc, fails = {}, []
    for iv in pln["intervals"]:
        if iv["split"] != split:
            continue
        d = RUNS / iv["id"]
        if not (d / "DONE.json").exists():
            continue
        for st in iv["starts"]:
            f = load_forced(d / f"{st['name']}_forced.parquet")
            if f.height == 0:
                continue
            el = f["el"].to_numpy()
            for K, dl in pol["top"].items():
                if dl is None:
                    continue
                K_ = int(K)
                m = (el >= dl) & (f["nr_b"].to_numpy() >= K_) & (f["nr_a"].to_numpy() >= K_)
                bad = m & ((f["mr_b"].to_numpy() < K_) | (f["mr_a"].to_numpy() < K_))
                if bad.any():
                    fails.append(dict(interval=iv["id"], start=st["name"], regime=iv["regime"], cell=iv["cell"], vol=iv["vol"], liq=iv["liq"],
                                      family=f"top{K}", delta=dl, n=int(m.sum()), n_bad=int(bad.sum()), first_bad_el=float(el[bad][0]),
                                      last_bad_el=float(el[bad][-1]),
                                      missing_mean=float((f["miss_b"].to_numpy()[bad] + f["miss_a"].to_numpy()[bad]).mean())))
            for c, dl in list(inst.items()) + [(k, v[0] + v[1] if v[0] is not None else None) for k, v in win.items()]:
                if dl is None:
                    continue
                e, r = f["e_" + c].to_numpy().astype(float), f["r_" + c].to_numpy().astype(float)
                m = np.isfinite(e) & np.isfinite(r) & (el >= dl)
                if m.any():
                    acc.setdefault(c, []).append((e[m], r[m], dl))
    rows = []
    for c, v in sorted(acc.items()):
        e = np.concatenate([x[0] for x in v])
        r = np.concatenate([x[1] for x in v])
        err = np.abs(e - r)
        sc = max(q(np.abs(r), .5), 1e-9)
        rows.append(dict(feature=c, min_elapsed_s=v[0][2], n=e.size, equal=float((err <= 1e-5 * (1 + np.abs(r))).mean()), err_p50=q(err, .5),
                         err_p99=q(err, .99), err_p999=q(err, .999), err_max=float(err.max()), scale_p50=sc,
                         corr=float(np.corrcoef(e, r)[0, 1]) if e.std() > 0 and r.std() > 0 else float("nan")))
    pl.from_dicts(rows).write_csv(RES / f"validation_features_{split}.csv")
    if fails:
        pl.from_dicts(fails).sort("n_bad", descending=True).write_csv(RES / f"validation_failures_{split}.csv")
    print(split, "features", len(rows), "runs with failures", len(fails))


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["runs", "age", "policy", "validate", "validate_runs"])
    ap.add_argument("--split", default="calibration")
    a = ap.parse_args()
    if a.cmd == "runs":
        runs(a.split)
    elif a.cmd == "age":
        age_tables(a.split)
    elif a.cmd == "policy":
        policy()
    elif a.cmd == "validate_runs":
        validate_runs(a.split)
    else:
        validate()


if __name__ == "__main__":
    main()
