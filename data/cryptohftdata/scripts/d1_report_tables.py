"""D1: generates docs/D1_TABLES.md from derived/d1/results (nothing is typed by hand). Run after every analysis step."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402

R = P.DERIVED / "d1" / "results"
OUT = Path("docs/D1_TABLES.md")
L = []


def h(t, lvl=2):
    L.append("\n" + "#" * lvl + " " + t + "\n")


def p(t):
    L.append(t + "\n")


def fmt(v, nd=4):
    if v is None:
        return "–"
    if isinstance(v, float):
        if np.isnan(v):
            return "–"
        if v == int(v) and abs(v) < 1e7:
            return f"{int(v)}"
        return f"{v:.{nd}g}"
    return str(v)


def table(df, nd=4):
    L.append("| " + " | ".join(df.columns) + " |")
    L.append("|" + "---|" * len(df.columns))
    for r in df.iter_rows():
        L.append("| " + " | ".join(fmt(x, nd) for x in r) + " |")
    L.append("")


def rd(name):
    return pl.read_csv(R / name)


def cert_availability():
    plan = json.loads(Path("configs/d1_intervals.json").read_text())
    fr = []
    for iv in plan["intervals"]:
        for s in iv["starts"]:
            z = pl.read_parquet(P.DERIVED / "d1" / "runs" / iv["id"] / f"{s['name']}_cert_summary.parquet")
            if z.height:
                fr.append(z)
    a = pl.concat(fr).group_by("feature", "el_lo").agg(pl.col("n_both").sum(), pl.col("n_r_only").sum())
    a = a.with_columns(av=pl.col("n_both") / (pl.col("n_both") + pl.col("n_r_only")))
    a.write_csv(R / "certificate_availability.csv")
    return a


def main():
    plan = json.loads(Path("configs/d1_intervals.json").read_text())
    pol = json.loads((R / "policy.json").read_text())
    p("# D1 — tabele generate automat\n\nGenerat de `scripts/d1_report_tables.py` din `derived/d1/results/`. Interpretarea este în `docs/D1_REPORT.md`.")

    h("1. Efectul corecțiilor de rezolver")
    p("Timp ancorat (fracție din lună), ancore, verificări, goluri reale, ancore fără conexiune: rezolver vechi vs. corectat. Doar lanțul, tot istoricul.")
    table(rd("correction_effect.csv"), 3)
    p("Validitate reală a book-ului strict (%): proaspăt ≤ 1 s și în domeniul certificat, eșantion de 1 s. „legacy” din replay-ul strict existent; „fixed” din pasul independent.")
    table(rd("strict_valid_by_month.csv"), 4)

    h("2. Eșantionul pre-declarat")
    iv = pl.DataFrame([{k: i[k] for k in ("regime", "split", "cell", "vol", "liq")} | {"starts": len(i["starts"])} for i in plan["intervals"]])
    table(iv.group_by("regime", "split").agg(pl.len().alias("intervale"), pl.col("starts").sum().alias("starturi")).sort("regime", "split"))
    table(iv.group_by("cell", "split").agg(pl.len().alias("intervale"), pl.col("starts").sum().alias("starturi")).sort("cell", "split"))
    table(iv.filter(pl.col("cell") == "grid").group_by("vol", "liq").len().sort("vol", "liq"))
    p(f"Sămânță {plan['seed']}; terțile rv_1h {plan['terciles']['vol']}, depth5 {plan['terciles']['liq']}; {len(plan['split_days'])} zile împărțite.")

    h("3. Integritate: rulări explicite")
    rows = []
    for s in ("calibration", "validation"):
        i = rd(f"runs_integrity_{s}.csv")
        c = rd(f"runs_certificate_{s}.csv")
        f = rd(f"runs_certificate_features_{s}.csv").filter(~pl.col("feature").str.contains("coverage|cert_"))
        real = f.filter(~pl.col("feature").str.starts_with("std_basis"))
        rows.append(dict(set=s, rulari=i.height, comparatii=i["n"].sum(), cantitati_gresite=i["wrong"].sum(), niveluri_false=i["extra"].sum(),
                         incrucisari=i["crossings"].sum(), varf_certificat=c["n_top_cert"].sum() / c["n"].sum(), varf_certificat_gresit=c["top_wrong"].sum(),
                         perechi_valori=f["n_both"].sum(), diferente_reale=real["n_diff"].sum(),
                         diferente_std_basis=f["n_diff"].sum() - real["n_diff"].sum(), max_std_basis=f.filter(pl.col("feature").str.starts_with("std_basis"))["max_abs"].max()))
    table(pl.from_dicts(rows), 6)

    h("4. Convergență pe rulări explicite (s, orizont ≥ 600 s, elapsed din primul mesaj)")
    c = pl.concat([rd(f"runs_convergence_{s}.csv") for s in ("calibration", "validation")]).filter(pl.col("horizon") >= 600)
    rows = []
    for sp in ("calibration", "validation"):
        for K in (1, 5, 10, 50):
            v = c.filter((pl.col("split") == sp) & (pl.col(f"n_ver{K}") > 100))
            f = v[f"first{K}"].fill_nan(None).drop_nulls().to_numpy()
            l = v[f"last_fail{K}"].to_numpy()
            rows.append(dict(set=sp, K=K, rulari=v.height, prim_p50=np.median(f), prim_p99=np.quantile(f, .99), prim_max=f.max(), stabil_p50=np.median(l),
                             stabil_p90=np.quantile(l, .9), stabil_p99=np.quantile(l, .99), stabil_max=l.max(), recaderi=float(v[f"relapse{K}"].mean())))
    table(pl.from_dicts(rows), 3)
    p("Pe regim (ambele seturi):")
    g = c.group_by("regime").agg(pl.len().alias("rulari"), *[pl.col(f"last_fail{K}").median().alias(f"top{K}_p50") for K in (1, 5, 10, 50)],
                                 *[pl.col(f"last_fail{K}").max().alias(f"top{K}_max") for K in (1, 5, 10, 50)]).sort("regime")
    table(g, 3)
    p("Pe volatilitate și lichiditate (terțile, ambele seturi), mediana și maximul stabilizării top 10:")
    table(c.group_by("vol", "liq").agg(pl.len().alias("rulari"), pl.col("last_fail10").median().alias("p50"), pl.col("last_fail10").quantile(.9).alias("p90"),
                                       pl.col("last_fail10").max().alias("max")).sort("vol", "liq"), 3)
    p("Starturi după un gol real:")
    table(c.filter(pl.col("cell").str.starts_with("post_gap")).group_by("cell").agg(pl.len().alias("rulari"), *[pl.col(f"last_fail{K}").median().alias(f"top{K}_p50") for K in (1, 10, 50)],
                                                                                    *[pl.col(f"last_fail{K}").max().alias(f"top{K}_max") for K in (1, 10, 50)]).sort("cell"), 3)

    h("5. Eșantion mare: rata de nepotrivire la momentul t pentru un book pornit acum δ secunde")
    for s in ("calibration", "validation"):
        a = rd(f"age_pointwise_{s}.csv").filter(pl.col("group") == "ALL")
        p(f"**{s}** ({a['n'].max():,} eșantioane de 1 s). Agregat:")
        table(a.pivot(on="K", index="delta", values="fail", sort_columns=True).sort("delta").rename({str(K): f"top{K}" for K in (1, 5, 10, 50)}), 3)
        p("Percentila 99 a ratei pe oră:")
        table(a.pivot(on="K", index="delta", values="hour_p99", sort_columns=True).sort("delta").rename({str(K): f"top{K}" for K in (1, 5, 10, 50)}), 3)
    h("Persistență: pentru fiecare start, timpul după care top-K rămâne exact pe orizontul de 1 h", 3)
    for s in ("calibration", "validation"):
        p(f"**{s}**")
        table(rd(f"age_persistence_{s}.csv").filter(pl.col("group") == "ALL").select("K", "starts", "never_exact", "first_p50", "first_p99", "first_max", "stable_p50",
                                                                                     "stable_p90", "stable_p99", "stable_p999", "stable_max", "relapse"), 4)
    p("Pe regim (validare):")
    table(rd("age_persistence_validation.csv").filter(pl.col("group") != "ALL").select("K", "group", "starts", "stable_p50", "stable_p99", "stable_p999", "stable_max").sort("K", "group"), 4)
    h("Deficit de adâncime pe benzi (fracție din bandă lipsă), partea bid", 3)
    for s in ("calibration", "validation"):
        p(f"**{s}**")
        table(rd(f"age_bands_{s}.csv").filter(pl.col("side") == "b").select("band", "delta", "n", "exact", "short_p50", "short_p99", "short_p999", "short_max"), 3)

    h("6. Features pe rulări forțate (emise fără certificat), după timpul scurs")
    for s in ("calibration", "validation"):
        f = rd(f"runs_features_{s}.csv")
        keep = ["spread_bps", "qi_1", "qi_top5", "qi_top10", "qi_top20", "qi_1bps", "qi_5bps", "qi_10bps", "qi_25bps", "depth_bid_top10_btc", "depth_bid_5bps_btc",
                "depth_bid_10bps_btc", "depth_bid_25bps_btc", "cost_buy_5btc_bps"]
        p(f"**{s}**: cotă de valori egale cu referința")
        table(f.filter(pl.col("feature").is_in(keep)).pivot(on="el_lo", index="feature", values="equal", sort_columns=True), 4)
        p(f"**{s}**: eroare relativă p99")
        table(f.filter(pl.col("feature").is_in(keep)).pivot(on="el_lo", index="feature", values="rel_p99", sort_columns=True), 3)

    h("7. Features pe ferestre, rulări forțate, după (timp scurs − fereastră)")
    w = pl.concat([rd(f"runs_windows_{s}.csv") for s in ("validation",)])
    for fam in ("ofi_{w}_btc", "tw_spread_bps_{w}", "tw_depth1_{w}_btc", "tw_depth_bid_5bps_{w}_btc", "tw_depth_bid_25bps_{w}_btc", "add_bid_10bps_{w}_btc"):
        p(f"**{fam}** (validare): cotă egală")
        table(w.filter(pl.col("feature") == fam).pivot(on="lag_lo", index="window", values="equal", sort_columns=True), 4)
    p("**ofi_{w}_btc** (validare): corelație cu referința și acord de semn")
    table(w.filter(pl.col("feature") == "ofi_{w}_btc").select("window", "lag_lo", "n", "equal", "err_p99", "err_max", "scale_p50", "corr", "sign_agree"), 4)

    h("8. Disponibilitatea sub certificat, după timpul scurs (cotă din momentele în care referința are valoarea)")
    a = cert_availability()
    keep = ["spread_bps", "qi_1", "depth_bid_top5_btc", "depth_bid_top10_btc", "depth_bid_top20_btc", "depth_bid_1bps_btc", "depth_bid_5bps_btc", "depth_bid_10bps_btc", "depth_bid_25bps_btc",
            "depth_bid_50bps_btc", "cost_buy_1btc_bps", "cost_buy_5btc_bps", "ofi_5s_btc", "ofi_2m_btc", "ofi_15m_btc", "mlofi_l5_30s_btc", "tw_spread_bps_2m",
            "tw_depth_bid_5bps_2m_btc", "tw_depth_bid_25bps_15m_btc", "add_bid_10bps_2m_btc"]
    x = a.filter(pl.col("feature").is_in(keep)).pivot(on="el_lo", index="feature", values="av", sort_columns=True)
    table(x.select(["feature"] + sorted([c for c in x.columns if c != "feature"], key=float)).sort("feature"), 3)

    h("9. Politica de timp fixată pe calibrare și verificarea ei")
    p("```\n" + json.dumps(pol, indent=1) + "\n```")
    table(rd("validation_top.csv").select("family", "delta", "group", "n", "fail", "hour_p99", "hour_max", "passes").sort("family", "group"), 3)
    table(rd("validation_bands.csv").select("band", "side", "delta", "n", "exact", "short_p50", "short_p99", "short_p999", "short_max", "passes"), 3)
    p("Features după politică, rulări explicite de validare, cele mai slabe 25:")
    table(rd("validation_features_validation.csv").select("feature", "min_elapsed_s", "n", "equal", "err_p99", "err_p999", "err_max", "scale_p50").sort("equal").head(25), 4)
    p("Toate eșecurile pe rulări explicite de validare:")
    if (R / "validation_failures_validation.csv").exists():
        table(rd("validation_failures_validation.csv").select("interval", "start", "regime", "cell", "vol", "liq", "family", "delta", "n", "n_bad", "first_bad_el", "last_bad_el", "missing_mean"), 4)
    p("Cele mai slabe ore (eșantion mare, validare):")
    table(rd("validation_worst_hours.csv").head(20), 4)

    h("10. Generalizare")
    p("Rata de nepotrivire pe lună: `ref` exact, cu referință; `visible` fără referință (nivelurile vizibile, ≥ 1 h după repornire; zile eșantion, una din cinci).")
    a = rd("age_by_month.csv")
    for K, d in ((5, 120), (10, 180), (50, 300), (5, 10), (50, 10)):
        p(f"top {K}, δ = {d} s")
        table(a.filter((pl.col("K") == K) & (pl.col("delta") == d)).pivot(on="source", index="month", values="fail", sort_columns=True).sort("month"), 3)
    p("Condiții de piață, ore cu vs. fără referință:")
    table(rd("conditions.csv"), 4)
    p("Flux pe lună:")
    table(rd("feed_by_month.csv"), 4)

    h("11. Acoperire și durate utilizabile")
    p("Segmente vii ale unui book repornit gol după fiecare gol real (fără snapshot-uri), din lanț, tot istoricul:")
    table(rd("segment_stats.csv"), 3)
    p("Cotă din lună: strict corectat (ancorat), book viu, book viu după un warm-up de w secunde de la fiecare repornire:")
    table(rd("coverage_by_month.csv").select("month", "strict_fixed", "chain_live", "rebuilt_60s", "rebuilt_300s", "rebuilt_900s", "rebuilt_1800s", "union_300s"), 3)
    p("Disponibilitate măsurată direct în modul opțional, zile eșantion, grilă de 5 s:")
    table(rd("no_ref_feature_availability.csv"), 3)
    table(rd("no_ref_days.csv"), 3)

    p("Estimare de acoperire pe istoric (cotă din timp): valid strict corectat + disponibilitatea măsurată în modul opțional pe orele cu fișier din perioada fără ancore:")
    hrs = pl.read_parquet(P.DERIVED / "d1" / "chain_pass" / "hours.parquet").filter(pl.col("variant") == "strict")
    tot = hrs.height
    un = hrs.filter((pl.col("day") >= "2025-07-01") & (pl.col("day") < "2026-04-11") & pl.col("file")).height
    miss = hrs.filter((pl.col("day") >= "2025-07-01") & (pl.col("day") < "2026-04-11") & ~pl.col("file")).height
    sv = rd("strict_valid_by_month.csv")
    secs = dict(hrs.with_columns(m=pl.col("day").str.slice(0, 7)).group_by("m").len().rows())
    Hs = sum(r["fixed_valid_pct"] / 100 * secs[r["month"]] for r in sv.iter_rows(named=True))
    Hl = sum(r["legacy_valid_pct"] / 100 * secs[r["month"]] for r in sv.iter_rows(named=True))
    av = rd("no_ref_feature_availability.csv")
    rows = [dict(feature="(toate, strict vechi)", ore=Hl, cota_istoric=Hl / tot), dict(feature="(toate, strict corectat)", ore=Hs, cota_istoric=Hs / tot)]
    for r in av.iter_rows(named=True):
        hh = Hs + r["unanchored"] * un
        rows.append(dict(feature=r["feature"], ore=hh, cota_istoric=hh / tot, cota_perioada_fara_ancore=r["unanchored"] * un / (un + miss)))
    e = pl.from_dicts(rows, infer_schema_length=None)
    e.write_csv(R / "coverage_estimate.csv")
    p(f"Ore în istoric {tot}; ore cu fișier în perioada fără ancore {un}, ore fără fișier {miss}.")
    table(e, 4)
    if (R / "demo_comparison.csv").exists():
        p("Demonstrație: cotă de bare cu valoare, mod opțional (dataseturile strict au 0 pe aceste zile):")
        table(rd("demo_comparison.csv").select("subset", "bars", "n", "state9", "stale", "strict_state0", "spread_bps", "depth_bid_5bps_btc", "depth_bid_25bps_btc", "ofi_30s_btc",
                                               "ofi_2m_btc", "ofi_15m_btc"), 3)

    h("12. Stări încrucișate în flux (scanare exhaustivă)")
    if (R / "cross_days.csv").exists():
        table(rd("cross_days.csv"), 4)
    OUT.write_text("\n".join(L))
    print("written", OUT, len(L))


if __name__ == "__main__":
    main()
