"""Builds multiasset_predictive_diagnostic.ipynb (text in Romanian). Conclusions are read from
notebook_texts.py so that they can be written after the results are known."""
import nbformat as nbf
from pathlib import Path
import importlib.util

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("texts", HERE / "notebook_texts.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(T.TITLE)
md(T.SUMMARY)

code(r'''
import warnings; warnings.filterwarnings("ignore")
import json, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib.pyplot as plt
import mad_lib as M
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
pd.set_option("display.float_format", lambda x: f"{x:.4g}")
T0 = time.time()
print("cache:", M.CACHE)
''')

# ------------------------------------------------------------------ 1. data map
md(T.S1)
code(r'''
inv = []
for p in sorted(M.DSROOT.iterdir()):
    m = p / "manifest.json"
    if not m.exists():
        continue
    j = json.load(open(m)); c = j["config"]; bt = c["bars"]["types"][0]
    md_ = pq.ParquetFile(p / f"{bt}.parquet").metadata
    inv.append(dict(asset=c["instrument"]["base"], bar=bt, threshold=float(c["bars"][bt]["threshold"]),
                    bars_per_day=round(md_.num_rows / 1826), rows=md_.num_rows, cols=md_.num_columns,
                    code=j["code_sha256"][:8], named=not p.name[:16].isalnum() or "_" in p.name,
                    start=c["period"]["start"][:10], end=c["period"]["end"][:10]))
inv = pd.DataFrame(inv).sort_values(["asset", "bar", "threshold"])
print(f"{len(inv)} dataseturi, perioadă {inv.start.min()} .. {inv.end.max()} (capăt exclus), un singur tip de bară per director")
inv.pivot_table(index="asset", columns="bar", values="bars_per_day", aggfunc=lambda s: sorted(s))
''')
code(r'''
# versiuni de cod: ETH există doar în build-urile vechi (cod 6a1d922b, directoare cu nume-hash);
# BTC are în plus build-uri vechi de calibrare la alte praguri.
inv.groupby(["asset", "code"]).agg(n=("rows", "size"), thresholds=("threshold", lambda s: sorted(s)))
''')
md(T.S1_CHOSEN)
code(r'''
d = M.build_panel()                     # cache: features + ținte + randamente cross-asset (vezi mad_lib.build_panel)
meta = json.load(open(M.CACHE / "panel_meta.json"))
yr = pd.to_datetime(d["t_us"], unit="us").dt.year
bpd = d.groupby(["asset", yr], observed=True).size().unstack() / d.groupby(yr).apply(lambda g: g["t_us"].agg(lambda s: (s.max() - s.min()) / 86400e6)).values
print("decizii (bare închise, după deduplicarea timestamp-urilor) pe zi, pe ani:")
bpd.round(0)
''')
md(T.S1_BARS)
code(r'''
bd = M.bar_duration_table(d)
bd.pivot_table(index=["asset", "k"], columns="year", values="med_min").round(0)
''')
code(r'''
# lungimea efectivă a unei bare variază mult: p10–p90 al duratei unei singure bare, în minute
bd[bd.k == 1].pivot_table(index="asset", columns="year", values=["p10", "p90"]).round(0)
''')
md(T.S1_FAM)
code(r'''
def fam_of(f):
    rules = [("ret_", "randamente"), ("rv", "volatilitate realizată"), ("bar_duration", "durată/intensitate"),
             ("trade_", "durată/intensitate"), ("avg_trade", "durată/intensitate"), ("volume_per", "durată/intensitate"),
             ("notional_per", "durată/intensitate"), ("aggr_", "flux agresor"), ("tick_", "flux agresor"),
             ("kyle", "lambda Kyle/Hasbrouck"), ("hasbrouck", "lambda Kyle/Hasbrouck"),
             ("sadf", "structurale"), ("qadf", "structurale"), ("cadf", "structurale"), ("sdfc", "structurale"),
             ("csw", "structurale"), ("bde", "structurale"),
             ("premium_raw_coinbase_btc", "premium Coinbase BTC"), ("premium_fx_coinbase_btc", "premium Coinbase BTC"),
             ("premium_raw_coinbase_eth", "premium Coinbase ETH"), ("premium_fx_coinbase_eth", "premium Coinbase ETH"),
             ("premium_raw_coinbase", "premium Coinbase activ propriu"), ("premium_fx_coinbase", "premium Coinbase activ propriu"),
             ("trend_", "trend 30z"), ("basis", "derivate: basis/premium index/funding"),
             ("premium_index", "derivate: basis/premium index/funding"), ("funding", "derivate: basis/premium index/funding"),
             ("etf_", "ETF flows"), ("fear_greed", "Fear & Greed"), ("dvol", "DVOL (Deribit)"),
             ("skew", "skew opțiuni"), ("iv_", "skew opțiuni"), ("binance_reported", "lichidări"), ("bybit_liq", "lichidări")]
    for pre, fam in rules:
        if f.startswith(pre):
            return fam
    return "L2 / cost / quotes"
cov = []
for a in M.ASSETS:
    c = pd.read_parquet(M.DSROOT / M.DATASETS[a] / "notional_coverage.parquet")
    c = c[~c.feature.str.endswith(("_age_s", "_economic_age_d", "_coverage", "_n_funds_reported"))]
    c["family"] = c.feature.map(fam_of)
    cov.append(c.groupby("family").valid_share.mean().rename(a))
cov = pd.concat(cov, axis=1)
cov.sort_values("BTC", ascending=False).round(2)
''')
md(T.S1_PREMIUM)

# ------------------------------------------------------------------ 2. validity
md(T.S2)
code(r'''
chk = []
for a in M.ASSETS:
    j = json.load(open(M.DSROOT / M.DATASETS[a] / "manifest.json"))
    br = j["methodology_breaks"]["notional"]; kl = j["kline_crosscheck_1m"]
    chk.append(dict(asset=a, segments=len(br["segments"]), gaps=len(br["gaps"]), missing_trade_days=len(j["missing_trade_days"]),
                    kline_minutes=kl["minutes_compared"], kline_mismatch_close=kl["mismatch_close"],
                    kline_mismatch_volume=kl["mismatch_volume"], minutes_only_in_klines=kl["minutes_only_in_klines"],
                    dup_timestamps_dropped=meta["dups_dropped"][a],
                    decision_minus_end_us=j["config"]["timing"]["decision_latency"]))
pd.DataFrame(chk).set_index("asset")
''')
code(r'''
# (a) prețul de referință: ret_5min reconstruit din grila px1s (ultima tranzacție strict înainte de floor_1s(t-5min))
ok = np.isclose(d.ret_5min_ds, d.ret_5min_px, atol=1e-12) | (d.ret_5min_ds.isna() & d.ret_5min_px.isna())
print("ret_5min identic cu cel din dataset:", ok.groupby(d.asset, observed=True).mean().round(6).to_dict())
# (b) premium-ul de 1 minut la decizie = minutul închis strict înainte (nu minutul încă deschis)
pd.DataFrame([M.premium_availability_check(a, 2000) for a in ["BTC", "ETH", "SOL", "XRP"]]).set_index("asset")
''')
code(r'''
# (c) Coinbase vs Binance: lumânarea Coinbase e etichetată prin început? vârful corelației trebuie să fie la lag 0
lag = pd.concat({(a, w): M.coinbase_lag_check(a, *w, lags=range(-2, 3))
                 for a in ["BTC", "ETH", "SOL", "XRP"] for w in [("2022-06-01", "2022-06-08"), ("2025-06-01", "2025-06-08")]}, axis=1).T
lag.round(3)
''')
code(r'''
# (d) pasul de preț în bps (sursa bounce-ului bid/ask în prețurile de tranzacție)
rows = []
for a in M.ASSETS:
    for day in ["2021-10-15", "2022-11-15", "2023-06-14", "2024-06-14", "2025-06-14"]:
        x = pd.read_parquet(M.PX1S / f"binance_spot_{a}USDT_trades" / f"{day}.parquet", columns=["price"])
        dp = np.abs(np.diff(x.price.values)); dp = dp[dp > 0]
        rows.append(dict(asset=a, day=day, tick=dp.min(), tick_bps=dp.min() / x.price.median() * 1e4))
pd.DataFrame(rows).pivot(index="asset", columns="day", values="tick_bps").round(2)
''')
md(T.S2_VERDICT)

# ------------------------------------------------------------------ 3. design
md(T.S3)
code(r'''
fr = pd.DataFrame([dict(fold=k, train_from=pd.Timestamp(a, unit="us").date(), test_from=pd.Timestamp(b, unit="us").date(),
                        test_to=pd.Timestamp(c, unit="us").date()) for k, a, b, c in M.folds()])
fr["rows_test"] = [((d.t_us >= M.us(pd.Timestamp(r.test_from, tz="UTC"))) & (d.t_us < M.us(pd.Timestamp(r.test_to, tz="UTC")))).sum() for r in fr.itertuples()]
print("rezervat (nefolosit în acest notebook):", M.RESERVE_START.date(), "..", M.DATA_END.date())
fr
''')
code(r'''
pd.DataFrame({s: [", ".join(M.SETS[s]), len(M.set_features(s))] for s in M.SETS}, index=["familii", "n_features"]).T
''')
code(r'''
pd.Series({k: ", ".join(v) for k, v in M.FAMILIES.items()}, name="features").to_frame()
''')
code(r'''
# volatilitatea țintelor (bps) — scara la care trebuie citite R² și efectele în bps
ys = d[d.t_us < M.us(M.RESERVE_START)].groupby("asset", observed=True)[["y_15m", "y_15m_d60", "y_2h", "y_1d"]]
pd.concat({"sd_bps": ys.std() * 1e4, "kurtosis": ys.apply(lambda g: g.kurt())}, axis=1).round(1)
''')

# ------------------------------------------------------------------ 4. results
md(T.S4)
code(r'''
files = sorted((M.CACHE / "preds_main").glob("*.parquet"))
P = M.attach(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True), d)
Ps = P.assign(y_bps=P.y_bps / P.sig_bps, pred_bps=P.pred_bps / P.sig_bps)      # versiunea scalată la volatilitate ex-ante
print(len(files), "joburi;", P.groupby(["set", "model", "h", "ycol"]).ngroups, "configurații;", f"{len(P):,} predicții OOS")
t = time.time()
E = M.evaluate(P, n_boot=1000)
Es = M.evaluate(Ps, n_boot=1000, by_asset=True)
print(f"evaluare {time.time() - t:.0f}s")
''')
code(r'''
SETS_ORDER = list(M.SETS)
def piv(E, val, ycol_filter=None, scale=100):
    e = E[E.ycol.isin(ycol_filter)] if ycol_filter else E[E.ycol == "y_" + E.h]
    t = e.pivot_table(index=["h", "scope"], columns=["model", "set"], values=val) * scale
    t = t.reindex(columns=pd.MultiIndex.from_product([["ridge", "lgbm"], SETS_ORDER]))
    return t.reindex(pd.MultiIndex.from_product([["15m", "2h", "1d"], ["ALL"] + M.ASSETS]))
print("R² OOS față de predicția zero, randamente brute în bps, în % (ALL = toate activele împreună)")
piv(E, "r2").round(3)
''')
code(r'''
print("R² OOS scalat la volatilitatea ex-ante (fiecare decizie are aceeași scară), în %")
piv(Es, "r2").round(3)
''')
code(r'''
print("R² OOS (brut) după eliminarea celor 10 zile cu cea mai mare contribuție absolută, în %")
piv(E, "r2_drop10d").round(3)
''')
code(r'''
print("Ponderea câștigului total de SSE adusă de cele 10 zile dominante (1 = tot câștigul)")
piv(E, "top10d_share", scale=1).round(2)
''')
code(r'''
print("Corelația Pearson predicție–randament / Spearman")
pd.concat({"pearson": piv(E, "corr", scale=1), "spearman": piv(E, "spear", scale=1)}, axis=1).round(3)[
    [("pearson", m, s) for m in ["ridge", "lgbm"] for s in ["PV", "OWN", "ALL"]] + [("spearman", m, s) for m in ["ridge", "lgbm"] for s in ["PV", "OWN", "ALL"]]]
''')
code(r'''
print("Folduri (din 6) cu R² brut > 0")
piv(E, "folds_pos", scale=1).astype("Int64")
''')
code(r'''
print("Față de media istorică a ferestrei de antrenare (R² relativ la acel reper, brut, %)")
piv(E, "r2_vs_mean").round(3)[[("ridge", "PV"), ("ridge", "ALL"), ("lgbm", "PV"), ("lgbm", "ALL")]]
''')

# ------------------------------------------------------------------ 5. tests / families
md(T.S5)
code(r'''
PAIRS = [("PV", "OWN"), ("OWN", "ALL"), ("PV", "PV+FLOW"), ("PV", "PV+STRUCT"), ("PV", "PV+DERIV"), ("PV", "PV+CBP"), ("PV", "PV+XRET")]
t = time.time()
Dn = M.nested_delta(P, PAIRS, n_boot=1000)
Dns = M.nested_delta(Ps, PAIRS, n_boot=1000)
print(f"{time.time() - t:.0f}s")
# familia de teste principale (pooled, ținta principală): PV vs zero + 7 comparații, x 3 orizonturi x 2 modele = 48
main = lambda x: x[(x.scope == "ALL") & (x.ycol == "y_" + x.h)]
t0 = main(E)[main(E).set == "PV"][["set", "model", "h", "r2", "r2_lo", "r2_hi", "p"]].rename(columns={"r2": "delta", "r2_lo": "lo", "r2_hi": "hi"})
t0["test"] = "PV vs 0"
t1 = main(Dn)[["base", "cand", "model", "h", "d_r2", "lo", "hi", "p"]].rename(columns={"d_r2": "delta"})
t1["test"] = t1.cand + " vs " + t1.base
TT = pd.concat([t0, t1])[["test", "model", "h", "delta", "lo", "hi", "p"]]
TT["p_holm"] = M.holm(TT.p.values)
# același test pe scara scalată la volatilitate
s0 = main(Es)[main(Es).set == "PV"].set_index(["model", "h"])["r2"]
s1 = main(Dns).assign(test=lambda x: x.cand + " vs " + x.base).set_index(["test", "model", "h"])["d_r2"]
TT["delta_sig"] = [s0.get((r.model, r.h)) if r.test == "PV vs 0" else s1.get((r.test, r.model, r.h)) for r in TT.itertuples()]
dd = main(Dn).assign(test=lambda x: x.cand + " vs " + x.base).set_index(["test", "model", "h"])
TT["delta_drop10d"] = [main(E).set_index(["set", "model", "h"])["r2_drop10d"].get(("PV", r.model, r.h)) if r.test == "PV vs 0"
                       else dd["d_r2_drop10d"].get((r.test, r.model, r.h)) for r in TT.itertuples()]
TT["folds_better"] = [main(E).set_index(["set", "model", "h"])["folds_pos"].get(("PV", r.model, r.h)) if r.test == "PV vs 0"
                      else dd["folds_better"].get((r.test, r.model, r.h)) for r in TT.itertuples()]
out = TT.copy()
for c in ["delta", "lo", "hi", "delta_sig", "delta_drop10d"]:
    out[c] = out[c] * 100
print("ΔR² în puncte procentuale (pp); p = bootstrap pe blocuri de 20 de zile, unilateral; Holm pe toate cele 48 de teste")
out.sort_values(["h", "model", "test"]).round(4).reset_index(drop=True)
''')
code(r'''
print("Contribuția fiecărei familii peste PV (ΔR² brut, pp), pe active — lgbm | ridge")
fam = Dn[Dn.base == "PV"].assign(fam=lambda x: x.cand.str.replace("PV+", ""))
fam = fam[fam.ycol == "y_" + fam.h]
tab = fam.pivot_table(index=["h", "scope"], columns=["model", "fam"], values="d_r2") * 100
tab.reindex(pd.MultiIndex.from_product([["15m", "2h", "1d"], ["ALL"] + M.ASSETS])).round(3)
''')
code(r'''
print("Aceeași contribuție pe scara scalată la volatilitate (ΔR²_σ, pp)")
fs = Dns[Dns.base == "PV"].assign(fam=lambda x: x.cand.str.replace("PV+", ""))
fs = fs[fs.ycol == "y_" + fs.h]
(fs.pivot_table(index=["h", "scope"], columns=["model", "fam"], values="d_r2") * 100).reindex(
    pd.MultiIndex.from_product([["15m", "2h", "1d"], ["ALL"] + M.ASSETS])).round(3)
''')
code(r'''
# BH pe toate comparațiile per activ (descriptiv): câte rămân la q < 0.10?
pa = Dn[(Dn.scope != "ALL") & (Dn.ycol == "y_" + Dn.h)].copy()
pa["q_bh"] = M.bh(pa.p.values)
print("comparații per activ:", len(pa), "| cu q_BH < 0.10:", int((pa.q_bh < 0.10).sum()))
pa[pa.q_bh < 0.10][["base", "cand", "model", "h", "scope", "d_r2", "lo", "hi", "p", "q_bh", "folds_better", "d_r2_drop10d"]].assign(
    d_r2=lambda x: x.d_r2 * 100, lo=lambda x: x.lo * 100, hi=lambda x: x.hi * 100, d_r2_drop10d=lambda x: x.d_r2_drop10d * 100).round(4)
''')
code(r'''
# ce features folosește modelul ALL (importanță gain medie pe folduri, lgbm; coeficienți ridge standardizați) — doar orientativ
imp = {}
for h in ["15m", "2h", "1d"]:
    mt = M.job_meta("ALL", "lgbm", h, f"y_{h}")
    imp[h] = pd.DataFrame([m["coef"] for m in mt]).mean()
imp = pd.DataFrame(imp)
fam_map = {f: k for k, v in M.FAMILIES.items() for f in v}
print("ponderea gain pe familii (lgbm ALL):")
display(imp.groupby(imp.index.map(fam_map)).sum().round(3))
imp.sort_values("1d", ascending=False).head(12).round(3)
''')

# ------------------------------------------------------------------ 6. robustness
md(T.S6)
code(r'''
FR = M.fold_r2(P)
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
for ax, h in zip(axes, ["15m", "2h", "1d"]):
    m = FR[(FR.set == "ALL") & (FR.model == "lgbm") & (FR.h == h) & (FR.ycol == f"y_{h}")].pivot(index="asset", columns="fold", values="r2") * 100
    m = m.reindex(M.ASSETS)
    v = max(np.nanmax(np.abs(m.values)), 0.05)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    im = ax.imshow(m.values, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#0b0b0b")
    ax.set_xticks(range(6)); ax.set_xticklabels([f"{p.year}H{1 if p.month == 1 else 2}" for p in M.TEST_STARTS], fontsize=8)
    ax.set_yticks(range(5)); ax.set_yticklabels(M.ASSETS)
    ax.set_title(f"ALL / lgbm, {h}: R² brut pe fold (%)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout(); plt.show()
''')
code(r'''
print("R² pe fold (%), pooled, ținta principală — stabilitatea în timp")
f2 = M.fold_r2(P.assign(asset="ALL"))
f2 = f2[f2.ycol == "y_" + f2.h]
(f2.pivot_table(index=["h", "model", "set"], columns="fold", values="r2") * 100).round(3).loc[(slice(None), slice(None), ["PV", "OWN", "ALL"]), :]
''')
code(r'''
print("Control bounce la 15 min: țintă de la t+1s (principală) vs de la t+60s; R² brut și scalat (%)")
b = pd.concat({"r2_brut": E.set_index(["h", "ycol", "model", "set", "scope"])["r2"],
               "r2_sigma": Es.set_index(["h", "ycol", "model", "set", "scope"])["r2"],
               "r2_drop10d": E.set_index(["h", "ycol", "model", "set", "scope"])["r2_drop10d"]}, axis=1).loc["15m"] * 100
b = b.loc[(slice(None), slice(None), ["PV", "OWN", "ALL"]), :].unstack("ycol")
b.round(3)
''')

# ------------------------------------------------------------------ 7. economics
md(T.S7)
code(r'''
fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
for ax, h in zip(axes, ["15m", "2h", "1d"]):
    for i, (mdl, col) in enumerate([("ridge", "#2a78d6"), ("lgbm", "#eb6834")]):
        s = P[(P.set == "ALL") & (P.model == mdl) & (P.h == h) & (P.ycol == f"y_{h}")].copy()
        s["q"] = s.groupby(["fold", "asset"])["pred_bps"].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
        g = s.groupby("q")["y_bps"].mean()
        ax.bar(np.arange(5) + (i - 0.5) * 0.38, g.values, width=0.36, color=col, label=mdl)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(range(5)); ax.set_xticklabels(["Q1\n(cele mai mici)", "Q2", "Q3", "Q4", "Q5\n(cele mai mari)"], fontsize=8)
    ax.set_title(f"ALL, {h}: randament realizat mediu pe chintile de predicție (bps)", fontsize=9)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
axes[0].legend(frameon=False)
plt.tight_layout(); plt.show()
''')
code(r'''
print("Randament mediu pe chintile de predicție (bps), pe active — set ALL, lgbm | ridge")
qq = []
for (mdl, h), s in P[(P.set == "ALL") & (P.ycol == "y_" + P.h)].groupby(["model", "h"]):
    s = s.copy()
    s["q"] = s.groupby(["fold", "asset"])["pred_bps"].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False)) + 1
    qq.append(s.groupby(["asset", "q"])["y_bps"].mean().unstack().assign(model=mdl, h=h))
pd.concat(qq).reset_index().set_index(["h", "model", "asset"]).sort_index().round(1)
''')
code(r'''
EC = M.economics(P)
cols = ["trade_frac", "mean_abs_pred", "gross_all", "gross_sel", "gross_sel_se_daily", "gross_sel_drop10d", "days_pos",
        "net_2", "net_5", "net_10", "net_20", "net_10_drop10d"]
print("Regula semn(predicție), activă doar când |predicție| > cuantila 0.9 a |predicție| din antrenare (prag ex-ante).")
print("bps per decizie (pozițiile se suprapun; nu e backtest de portofoliu). net_c = brut − cost dus-întors c bps")
e = EC[(EC.ycol == "y_" + EC.h) & EC.set.isin(["PV", "OWN", "ALL"])].set_index(["h", "model", "set", "scope"])[cols]
e.reindex(pd.MultiIndex.from_product([["15m", "2h", "1d"], ["ridge", "lgbm"], ["PV", "OWN", "ALL"], ["ALL"] + M.ASSETS])).round(2)
''')
code(r'''
print("Același calcul pentru controlul bounce (15m de la t+60s)")
EC[EC.ycol == "y_15m_d60"].set_index(["model", "set", "scope"])[cols].round(2)
''')
md(T.S8)
code(r'''
def blockboot_mean(day, x, n=1000, block=20, seed=0):
    g = pd.DataFrame({"d": day, "x": x}).groupby("d")["x"].agg(["sum", "size"])
    S, N = g["sum"].values, g["size"].values; k = len(S); rng = np.random.default_rng(seed)
    nb = int(np.ceil(k / block)); st = rng.integers(0, k - block + 1, (n, nb))
    idx = (st[:, :, None] + np.arange(block)).reshape(n, -1)[:, :k]
    return np.percentile(S[idx].sum(1) / N[idx].sum(1), [2.5, 97.5])
rows = []
for (s, m, h) in [("PV", "ridge", "15m"), ("PV", "lgbm", "15m"), ("PV", "ridge", "2h"), ("PV", "lgbm", "2h"), ("PV", "ridge", "1d"), ("PV", "lgbm", "1d")]:
    S = P[(P.set == s) & (P.model == m) & (P.h == h) & (P.ycol == f"y_{h}")].copy()
    thr = {x["fold"]: x["train_abs_q"]["0.9"] for x in M.job_meta(s, m, h, f"y_{h}")}
    S = S[S.pred_bps.abs() > S.fold.map(thr)]
    S["pnl"] = np.sign(S.pred_bps) * S.y_bps
    lo, hi = blockboot_mean(S.day.values, S.pnl.values)
    r = dict(set=s, model=m, h=h, n_signals=len(S), days=S.day.nunique(), gross=S.pnl.mean(), ci_lo=lo, ci_hi=hi)
    for side, ss in S.groupby(S.pred_bps > 0):
        nm = "long" if side else "short"
        r[f"{nm}_share"] = len(ss) / len(S); r[f"{nm}_pnl"] = ss.pnl.mean()
        r[f"{nm}_lrv1h"] = d.lrv_1h.values[ss.row].mean(); r[f"{nm}_zr5m"] = d.zr_5m.values[ss.row].mean()
    r["worst_fold"] = S.groupby("fold").pnl.mean().min()
    rows.append(r)
print("Regula decilei extreme (prag ex-ante), descompusă pe direcție; CI = bootstrap pe blocuri de 20 de zile.")
print("lrv_1h mediu necondiționat:", round(float(d.lrv_1h.mean()), 2), "| zr_5m = randamentul pe 5 min în unități de volatilitate")
pd.DataFrame(rows).round(2)
''')
code(r'''
# Descriptiv (fără model, EXPLORATOR: pragurile sunt alese după ce am văzut tabelul de mai sus).
# hv = decila superioară a lrv_1h în anul și activul respectiv; drop = zr_5m < -2. Include 2021–2022 (doar antrenare).
x = d[d.t_us < M.us(M.RESERVE_START)].copy()
x["year"] = pd.to_datetime(x.t_us, unit="us").dt.year
x["hv"] = x.groupby(["asset", "year"], observed=True).lrv_1h.transform(lambda s: s > s.quantile(0.9))
x["state"] = np.where(~x.hv, "vol normală", np.where(x.zr_5m < -2, "vol mare & scădere bruscă", "vol mare, fără scădere bruscă"))
tab = {yc: x.groupby(["year", "state"])[yc].mean().unstack() * 1e4 for yc in ["y_15m", "y_15m_d60", "y_2h"]}
print("randament mediu ulterior (bps) pe stări; BTC: 2022 = -64%, 2023 +156%, 2024 +120%, 2025 -6%")
pd.concat(tab, axis=1).round(1)
''')
code(r'''
print(f"timp total notebook: {(time.time() - T0) / 60:.1f} min")
''')

# ------------------------------------------------------------------ 8. conclusions
md(T.CONCL)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, HERE / "multiasset_predictive_diagnostic.ipynb")
print("written", len(cells), "cells")
