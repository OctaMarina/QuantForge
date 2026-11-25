"""Multi-asset predictive diagnostic (mad) — helpers for the notebook
`multiasset_predictive_diagnostic.ipynb`.

Reads the cryptods datasets read-only, builds causal features, price-grid targets and cross-asset
returns, runs a purged expanding walk-forward with ridge + LightGBM, and evaluates the forecasts.
Everything heavy is cached under MAD_CACHE (default ~/.cache/quantforge/multiasset_diag/v1).
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/octa/Desktop/QuantForge/data/crypto_dataset_pipeline/data")
DSROOT = ROOT / "datasets"
PX1S = ROOT / "normalized" / "px1s"
CB = ROOT / "normalized" / "coinbase"
CACHE = Path(os.environ.get("MAD_CACHE", Path.home() / ".cache/quantforge/multiasset_diag/v1"))
CACHE.mkdir(parents=True, exist_ok=True)

ASSETS = ["BTC", "ETH", "SOL", "XRP", "BNB"]
# Primary configuration (chosen before looking at any performance): quote-notional bars, ~100 bars/day.
DATASETS = {
    "BTC": "BTCUSDT_spot_notional_23900000_100pd_20210901-20260901_ccbcf1e7",
    "ETH": "60a5a92295b2e1cb",  # ETHUSDT notional 12,075,000 (older code 6a1d922b; only naming/overflow-check differ)
    "SOL": "SOLUSDT_spot_notional_4380000_100pd_20210901-20260901_e0752b18",
    "XRP": "XRPUSDT_spot_notional_3030000_100pd_20210901-20260901_c82cd27e",
    "BNB": "BNBUSDT_spot_notional_2290000_100pd_20210901-20260901_9848aac9",
}

US = 1_000_000
SEC, MIN, HOUR, DAY = US, 60 * US, 3600 * US, 86400 * US

# ---- temporal design (fixed before modelling) ----
T_START = pd.Timestamp("2021-09-08", tz="UTC")          # one week after bar origin (rolling-200-bar features warm)
TEST_STARTS = [pd.Timestamp(s, tz="UTC") for s in
               ["2023-01-01", "2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01"]]
RESERVE_START = pd.Timestamp("2026-01-01", tz="UTC")     # 2026-01-01 .. 2026-09-01: not evaluated here
DATA_END = pd.Timestamp("2026-09-01", tz="UTC")
INNER_VAL = pd.DateOffset(months=6)                       # last 6 months of each training window

HORIZONS = {"15m": 15 * MIN, "2h": 2 * HOUR, "1d": DAY}
ENTRY_DELAY = SEC                                         # main target: log P(t+h) - log P(t+1s)
ALT_ENTRY = 60 * SEC                                      # bounce control for 15m: from t+60s
MAX_AGE = 120 * SEC                                       # price-grid staleness limit

PAST_OFF = {"m1d": -DAY, "m4h": -4 * HOUR, "m1h": -HOUR, "m5m": -5 * MIN, "p0": 0}
FUT_OFF = {"f1s": SEC, "f60s": 60 * SEC, "f15m": 15 * MIN, "f2h": 2 * HOUR, "f1d": DAY}


# =====================================================================================
# 1. bars and own-asset features
# =====================================================================================
BASE_COLS = ["bar_seq", "bar_end_us", "decision_time_us", "bar_available_us", "open", "high", "low", "close",
             "notional_usdt", "n_trades", "bar_duration_s", "avg_trade_size",
             "ret_bars_1", "ret_bars_5", "ret_bars_20", "ret_5min", "ret_1h", "ret_4h", "ret_1d", "ret_7d", "ret_30d",
             "rvol_1h", "rvol_1d", "rvol_7d", "rv_bars_100", "trend_30d_volscaled",
             "aggr_imbalance_notional", "tick_imbalance_notional", "aggr_signed_notional",
             "kyle_tstat_trades_aggr", "hasbrouck_tstat_trades_aggr", "kyle_tstat_bars50_aggr", "hasbrouck_tstat_bars50_aggr",
             "sadf_w200_tau30_p1_c", "qadf_q95_w200_tau30_p1_c", "cadf_q95_w200_tau30_p1_c", "sdfc_w200_trim15",
             "sdfc_argmax_bars_ago_w200", "csw_up_w200", "csw_down_w200", "csw_two_w200", "bde_cusum_w200_p1",
             "basis_last_perp_spot_1m", "basis_mark_index_1m", "premium_index_binance_1m", "funding_rate_last"]


def prem_cols(x: str) -> list[str]:
    x = x.lower()
    return [f"premium_raw_coinbase_{x}_1m", f"premium_fx_coinbase_{x}_1m", f"premium_raw_coinbase_{x}_1d",
            f"premium_raw_coinbase_{x}_1d_z90"]


FAMILIES = {
    "PV": ["zr_b1", "zr_b5", "zr_b20", "zr_5m", "zr_1h", "zr_4h", "zr_1d", "zr_7d", "zr_30d",
           "lrv_1h", "lrv_1d", "lrv_7d", "vr_1h1d", "vr_1d7d", "ldur_rel", "lntr_rel", "lats_rel",
           "trend30", "range_b", "clv"],
    "FLOW": ["aggr_imb", "tick_imb", "flow5", "flow20", "kyle_t", "hasb_t", "kyle_t50", "hasb_t50"],
    "STRUCT": ["sadf", "qadf", "cadf", "sdfc", "sdfc_ago", "csw_up", "csw_down", "csw_two", "bde"],
    "DERIV": ["basis_ps", "basis_mi", "prem_idx", "funding"],
    "CBP_OWN": ["cbp_own_raw1m", "cbp_own_fx1m", "cbp_own_raw1d", "cbp_own_z90", "cbp_own_dev"],
    "CBP_OTH": ["cbp_btc_raw1m", "cbp_btc_fx1m", "cbp_btc_z90", "cbp_btc_dev",
                "cbp_eth_raw1m", "cbp_eth_fx1m", "cbp_eth_z90", "cbp_eth_dev"],
    "XRET": [f"{p}_{L}" for p in ("xmkt", "xlead", "rel") for L in ("5m", "1h", "4h", "1d")],
}
SETS = {
    "PV": ["PV"],
    "PV+FLOW": ["PV", "FLOW"],
    "PV+STRUCT": ["PV", "STRUCT"],
    "PV+DERIV": ["PV", "DERIV"],
    "PV+CBP": ["PV", "CBP_OWN", "CBP_OTH"],
    "PV+XRET": ["PV", "XRET"],
    "OWN": ["PV", "FLOW", "STRUCT", "DERIV", "CBP_OWN"],
    "ALL": ["PV", "FLOW", "STRUCT", "DERIV", "CBP_OWN", "CBP_OTH", "XRET"],
}


def set_features(name: str) -> list[str]:
    return [f for fam in SETS[name] for f in FAMILIES[fam]]


def _roll_prev_mean(x: pd.Series, n: int = 200) -> pd.Series:
    """mean of the previous n values (current excluded) - causal."""
    return x.shift(1).rolling(n, min_periods=n // 2).mean()


def load_bars(asset: str) -> pd.DataFrame:
    path = DSROOT / DATASETS[asset] / "notional.parquet"
    cols = BASE_COLS + sum((prem_cols(x) for x in dict.fromkeys(["btc", "eth", asset.lower()])), [])
    df = pd.read_parquet(path, columns=cols)
    df = df.sort_values("bar_seq").reset_index(drop=True)
    n0 = len(df)
    # several bars can close on the same (large) trade: keep the last one per decision timestamp
    df = df.drop_duplicates("decision_time_us", keep="last").reset_index(drop=True)
    df.attrs["n_dup_dropped"] = n0 - len(df)
    return df


def own_features(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    s1d, s7d = df["rvol_1d"], df["rvol_7d"]
    sb = np.sqrt(df["rv_bars_100"] / 100.0)
    for k in (1, 5, 20):
        f[f"zr_b{k}"] = df[f"ret_bars_{k}"] / (sb * np.sqrt(k))
    for col, lab, h in (("ret_5min", "5m", 5 / 1440), ("ret_1h", "1h", 1 / 24), ("ret_4h", "4h", 4 / 24),
                        ("ret_1d", "1d", 1.0)):
        f[f"zr_{lab}"] = df[col] / (s1d * np.sqrt(h))
    f["zr_7d"] = df["ret_7d"] / s7d
    f["zr_30d"] = df["ret_30d"] / (s7d * np.sqrt(30 / 7))
    f["lrv_1h"], f["lrv_1d"], f["lrv_7d"] = np.log(df["rvol_1h"]), np.log(s1d), np.log(s7d)
    f["vr_1h1d"] = np.log(df["rvol_1h"] * np.sqrt(24) / s1d)
    f["vr_1d7d"] = np.log(s1d * np.sqrt(7) / s7d)
    ld = np.log1p(df["bar_duration_s"])
    f["ldur_rel"] = ld - _roll_prev_mean(ld)
    ln = np.log(df["n_trades"].astype(float))
    f["lntr_rel"] = ln - _roll_prev_mean(ln)
    la = np.log(df["avg_trade_size"])
    f["lats_rel"] = la - _roll_prev_mean(la)
    f["trend30"] = df["trend_30d_volscaled"]
    f["range_b"] = np.log(df["high"] / df["low"]) / sb
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    f["clv"] = (df["close"] - df["low"]) / rng
    # flow
    f["aggr_imb"] = df["aggr_imbalance_notional"]
    f["tick_imb"] = df["tick_imbalance_notional"]
    for k in (5, 20):
        f[f"flow{k}"] = (df["aggr_signed_notional"].rolling(k, min_periods=k).sum()
                         / df["notional_usdt"].rolling(k, min_periods=k).sum())
    f["kyle_t"], f["hasb_t"] = df["kyle_tstat_trades_aggr"], df["hasbrouck_tstat_trades_aggr"]
    f["kyle_t50"], f["hasb_t50"] = df["kyle_tstat_bars50_aggr"], df["hasbrouck_tstat_bars50_aggr"]
    # structural
    for a, b in (("sadf", "sadf_w200_tau30_p1_c"), ("qadf", "qadf_q95_w200_tau30_p1_c"), ("cadf", "cadf_q95_w200_tau30_p1_c"),
                 ("sdfc", "sdfc_w200_trim15"), ("sdfc_ago", "sdfc_argmax_bars_ago_w200"), ("csw_up", "csw_up_w200"),
                 ("csw_down", "csw_down_w200"), ("csw_two", "csw_two_w200"), ("bde", "bde_cusum_w200_p1")):
        f[a] = df[b]
    # derivatives premium (own asset's USD-M perpetual)
    f["basis_ps"], f["basis_mi"] = df["basis_last_perp_spot_1m"], df["basis_mark_index_1m"]
    f["prem_idx"], f["funding"] = df["premium_index_binance_1m"], df["funding_rate_last"]
    # Coinbase USD vs Binance USDT premium
    def cbp(x, pre):
        r1, fx, r1d, z = prem_cols(x)
        f[f"{pre}_raw1m"], f[f"{pre}_fx1m"] = df[r1], df[fx]
        if pre == "cbp_own":
            f[f"{pre}_raw1d"] = df[r1d]
        f[f"{pre}_z90"] = df[z]
        f[f"{pre}_dev"] = df[r1] - df[r1d]
    cbp(asset, "cbp_own")
    for x in ("btc", "eth"):
        cbp(x, f"cbp_{x}")
        if x.upper() == asset:                         # own premium already in CBP_OWN
            for c in [c for c in f.columns if c.startswith(f"cbp_{x}_")]:
                f[c] = np.nan
    return f.astype("float32")


# =====================================================================================
# 2. 1-second price grid queries (targets and cross-asset returns)
# =====================================================================================
def _px_files(asset: str) -> dict:
    d = PX1S / f"binance_spot_{asset}USDT_trades"
    return {pd.Timestamp(p.stem, tz="UTC"): p for p in sorted(d.glob("*.parquet"))}


def query_prices(asset: str, q: np.ndarray, max_age: int = MAX_AGE) -> tuple[np.ndarray, np.ndarray]:
    """Price of the last trade strictly before each query time q (µs), and its age (µs).
    NaN when older than max_age.  Processes month by month to bound memory."""
    files = _px_files(asset)
    days = np.array(sorted(files))
    order = np.argsort(q, kind="stable")
    qs = q[order]
    price = np.full(len(q), np.nan)
    age = np.full(len(q), -1, dtype=np.int64)
    if len(qs) == 0:
        return price, age
    t0 = pd.Timestamp(int(qs[0]), unit="us", tz="UTC").floor("D")
    t1 = pd.Timestamp(int(qs[-1]), unit="us", tz="UTC").ceil("D")
    months = pd.date_range(pd.Timestamp(t0.year, t0.month, 1, tz="UTC"), t1, freq="MS")
    for m in months:
        a, b = int(m.value // 1000), int((m + pd.DateOffset(months=1)).value // 1000)
        lo, hi = np.searchsorted(qs, a), np.searchsorted(qs, b)
        if lo == hi:
            continue
        need = [d for d in days if (m - pd.Timedelta(days=2)) <= d <= (m + pd.DateOffset(months=1) + pd.Timedelta(days=1))]
        parts = [pd.read_parquet(files[d], columns=["last_ts_us", "price"]) for d in need]
        px = pd.concat(parts, ignore_index=True)
        ts = px["last_ts_us"].to_numpy(np.int64)
        pv = px["price"].to_numpy(np.float64)
        assert np.all(np.diff(ts) > 0)
        qq = qs[lo:hi]
        idx = np.searchsorted(ts, qq, side="left") - 1      # last trade with ts < q
        ok = idx >= 0
        j = np.where(ok, idx, 0)
        ag = qq - ts[j]
        val = np.where(ok & (ag <= max_age), pv[j], np.nan)
        price[order[lo:hi]] = val
        age[order[lo:hi]] = np.where(ok, ag, -1)
        del px, ts, pv, parts
    return price, age


def asof_from_bars(src: pd.DataFrame, col: str, t: np.ndarray) -> np.ndarray:
    """value of `col` at the last bar of `src` with bar_available_us < t (strict)."""
    av = src["bar_available_us"].to_numpy(np.int64)
    idx = np.searchsorted(av, t, side="left") - 1
    v = src[col].to_numpy(np.float64)
    return np.where(idx >= 0, v[np.maximum(idx, 0)], np.nan)


# =====================================================================================
# 3. panel assembly (cached)
# =====================================================================================
def build_panel(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    out = CACHE / "panel.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    t_start = time.time()
    bars = {a: load_bars(a) for a in ASSETS}
    dups = {a: bars[a].attrs["n_dup_dropped"] for a in ASSETS}
    # decision times of all assets
    T = {a: bars[a]["decision_time_us"].to_numpy(np.int64) for a in ASSETS}
    allT = np.concatenate([T[a] for a in ASSETS])
    seg = np.cumsum([0] + [len(T[a]) for a in ASSETS])
    # one pass over each asset's price grid: past offsets for every asset's decisions (cross-asset),
    # future offsets for its own decisions (targets) and the ret_5min consistency check
    res_us = int(RESERVE_START.value // 1000)
    P, fut, chk = {}, {}, {}
    for b in ASSETS:
        qs, keys = [], []
        for k, off in PAST_OFF.items():
            qs.append(allT + off); keys.append(("P", k))
        for k, off in FUT_OFF.items():
            qs.append(T[b] + off); keys.append(("F", k))
        qs.append(((T[b] - 5 * MIN) // SEC) * SEC); keys.append(("C", "chk"))
        lens = np.cumsum([0] + [len(x) for x in qs])
        p, _ = query_prices(b, np.concatenate(qs))
        for j, (kind, k) in enumerate(keys):
            v = p[lens[j]:lens[j + 1]]
            if kind == "P":
                P[(b, k)] = v
            elif kind == "F":
                fut[(b, k)] = v
            else:
                chk[b] = v
        if verbose:
            print(f"  px {b} done  {time.time() - t_start:.0f}s", flush=True)
    frames = []
    for i, a in enumerate(ASSETS):
        df = bars[a]
        t = T[a]
        f = own_features(df, a)
        sl = slice(seg[i], seg[i + 1])
        # cross-asset returns, normalised by the other asset's ex-ante daily vol (as-of its last closed bar)
        xr = {L: [] for L in ("5m", "1h", "4h", "1d")}
        hfrac = {"5m": 5 / 1440, "1h": 1 / 24, "4h": 4 / 24, "1d": 1.0}
        offk = {"5m": "m5m", "1h": "m1h", "4h": "m4h", "1d": "m1d"}
        zb = {}
        for b in ASSETS:
            if b == a:
                continue
            sig = asof_from_bars(bars[b], "rvol_1d", t)
            for L in xr:
                r = np.log(P[(b, "p0")][sl] / P[(b, offk[L])][sl])
                z = r / (sig * np.sqrt(hfrac[L]))
                xr[L].append(z)
                zb[(b, L)] = z
        lead = "ETH" if a == "BTC" else "BTC"
        with np.errstate(all="ignore"):
            for L in xr:
                f[f"xmkt_{L}"] = np.nanmean(np.vstack(xr[L]), axis=0).astype("float32")
                f[f"xlead_{L}"] = zb[(lead, L)].astype("float32")
                f[f"rel_{L}"] = (f[f"zr_{L}"] - f[f"xmkt_{L}"]).astype("float32")
        # targets (main: from t+1s)
        p1 = fut[(a, "f1s")]
        y = {}
        for h, k in (("15m", "f15m"), ("2h", "f2h"), ("1d", "f1d")):
            y[f"y_{h}"] = np.log(fut[(a, k)] / p1)
        y["y_15m_d60"] = np.log(fut[(a, "f15m")] / fut[(a, "f60s")])
        hide = t >= res_us
        for k in y:
            y[k][hide] = np.nan
        meta = pd.DataFrame({"asset": a, "t_us": t, "close": df["close"].to_numpy(),
                             "rvol_1d": df["rvol_1d"].to_numpy(), "ret_5min_ds": df["ret_5min"].to_numpy(),
                             "ret_5min_px": np.log(df["close"].to_numpy() / chk[a]),
                             "p_entry": p1, "bar_duration_s": df["bar_duration_s"].to_numpy()})
        meta.loc[hide, "p_entry"] = np.nan
        for k, v in y.items():
            meta[k] = v.astype("float64")
        frames.append(pd.concat([meta, f.reset_index(drop=True)], axis=1))
    panel = pd.concat(frames, ignore_index=True)
    panel["asset"] = panel["asset"].astype("category")
    panel = panel[panel["t_us"] >= int(T_START.value // 1000)].reset_index(drop=True)
    panel.to_parquet(out)
    json.dump({"dups_dropped": dups, "build_s": time.time() - t_start}, open(CACHE / "panel_meta.json", "w"))
    if verbose:
        print(f"panel {panel.shape} in {time.time() - t_start:.0f}s")
    return panel


# =====================================================================================
# 4. walk-forward modelling
# =====================================================================================
def us(ts: pd.Timestamp) -> int:
    return int(ts.value // 1000)


def folds():
    """(fold, train_lo, test_lo, test_hi) in µs; expanding window, same boundaries for all assets."""
    ends = TEST_STARTS[1:] + [RESERVE_START]
    return [(k, us(T_START), us(s), us(e)) for k, (s, e) in enumerate(zip(TEST_STARTS, ends))]


class Scaler:
    """per-asset robust scaling learned on training rows only; NaN -> 0 (= training median)."""

    def fit(self, X: np.ndarray, asset: np.ndarray):
        self.p = {}
        for a in np.unique(asset):
            Xa = X[asset == a]
            med = np.nanmedian(Xa, axis=0)
            q75, q25 = np.nanpercentile(Xa, 75, axis=0), np.nanpercentile(Xa, 25, axis=0)
            sc = (q75 - q25) / 1.349
            sd = np.nanstd(Xa, axis=0)
            sc = np.where(np.isfinite(sc) & (sc > 0), sc, np.where(np.isfinite(sd) & (sd > 0), sd, 1.0))
            med = np.where(np.isfinite(med), med, 0.0)
            self.p[a] = (med, sc)
        return self

    def transform(self, X: np.ndarray, asset: np.ndarray, fill: bool = True) -> np.ndarray:
        Z = np.empty_like(X, dtype=np.float32)
        for a, (med, sc) in self.p.items():
            m = asset == a
            Z[m] = np.clip((X[m] - med) / sc, -8, 8)
        unseen = ~np.isin(asset, list(self.p))
        Z[unseen] = np.nan
        if fill:
            Z = np.where(np.isfinite(Z), Z, 0.0).astype(np.float32)
        return Z


def ridge_path(Z, y, alphas):
    """closed-form ridge with unpenalised intercept for several alphas."""
    mu_x, mu_y = Z.mean(0), y.mean()
    Zc = Z - mu_x
    G = Zc.T @ Zc
    b = Zc.T @ (y - mu_y)
    out = {}
    for al in alphas:
        w = np.linalg.solve(G + al * np.eye(G.shape[0]), b)
        out[al] = (w, mu_y - mu_x @ w)
    return out


RIDGE_ALPHAS = [1e0, 1e2, 1e4, 1e5, 1e6, 1e7, 1e8]
LGB_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000,
                  feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, max_bin=63,
                  num_threads=8, verbose=-1, seed=7, deterministic=True, force_col_wise=True)
LGB_MAX_ROUNDS, LGB_PATIENCE = 600, 50
YCLIP = 6.0


def scaled_target(d: pd.DataFrame, h: str, ycol: str) -> tuple[np.ndarray, np.ndarray]:
    sig = d["rvol_1d"].to_numpy(np.float64) * np.sqrt(HORIZONS[h] / DAY)
    return d[ycol].to_numpy(np.float64) / sig, sig


def fit_predict(d: pd.DataFrame, feats: list[str], h: str, ycol: str, model: str, fold) -> dict:
    """Train on the purged expanding window, choose alpha / n_rounds on the purged inner validation
    (last 6 months of the window), refit on the whole window, predict the test window."""
    import lightgbm as lgb
    k, tr_lo, te_lo, te_hi = fold
    H = HORIZONS[h]
    t = d["t_us"].to_numpy(np.int64)
    ok = np.isfinite(d[ycol].to_numpy()) & np.isfinite(d["rvol_1d"].to_numpy()) & (d["rvol_1d"].to_numpy() > 0)
    tr = ok & (t >= tr_lo) & (t + H + ENTRY_DELAY < te_lo)                       # purge: label must end before test
    te = ok & (t >= te_lo) & (t < te_hi) & (t + H + ENTRY_DELAY < us(RESERVE_START))
    va_lo = us(pd.Timestamp(te_lo, unit="us", tz="UTC") - INNER_VAL)
    itr = tr & (t + H + ENTRY_DELAY < va_lo)
    iva = tr & (t >= va_lo)
    ys, sig = scaled_target(d, h, ycol)
    ysc = np.clip(ys, -YCLIP, YCLIP)
    A = d["asset"].astype(str).to_numpy()
    X = d[feats].to_numpy(np.float32)
    res = {"fold": k, "idx": np.flatnonzero(te)}
    if model == "ridge":
        sc_i = Scaler().fit(X[itr], A[itr])
        path = ridge_path(sc_i.transform(X[itr], A[itr]).astype(np.float64), ysc[itr], RIDGE_ALPHAS)
        Zv = sc_i.transform(X[iva], A[iva]).astype(np.float64)
        mse = {al: np.mean((ysc[iva] - (Zv @ w + c)) ** 2) for al, (w, c) in path.items()}
        best = min(mse, key=mse.get)
        sc = Scaler().fit(X[tr], A[tr])
        w, c = ridge_path(sc.transform(X[tr], A[tr]).astype(np.float64), ysc[tr], [best])[best]
        pz = sc.transform(X[te], A[te]).astype(np.float64) @ w + c
        ptr = sc.transform(X[tr], A[tr]).astype(np.float64) @ w + c
        res.update(param=best, coef=dict(zip(feats, w)))
    elif model == "lgbm":
        sc_i = Scaler().fit(X[itr], A[itr])
        dtr = lgb.Dataset(sc_i.transform(X[itr], A[itr], fill=False), ysc[itr], free_raw_data=True)
        dva = lgb.Dataset(sc_i.transform(X[iva], A[iva], fill=False), ysc[iva], reference=dtr)
        m = lgb.train(LGB_PARAMS, dtr, LGB_MAX_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(LGB_PATIENCE, verbose=False)])
        best = max(int(m.best_iteration or 1), 10)
        del dtr, dva, m
        sc = Scaler().fit(X[tr], A[tr])
        Ztr = sc.transform(X[tr], A[tr], fill=False)
        m = lgb.train(LGB_PARAMS, lgb.Dataset(Ztr, ysc[tr]), best)
        pz = m.predict(sc.transform(X[te], A[te], fill=False), num_threads=8)
        ptr = m.predict(Ztr, num_threads=8)
        gain = m.feature_importance("gain")
        res.update(param=best, coef=dict(zip(feats, gain / max(gain.sum(), 1e-12))))
        del m, Ztr
    elif model == "mean":           # historical-mean benchmark (vol-scaled mean of training window)
        c = ysc[tr].mean()
        pz = np.full(te.sum(), c)
        ptr = np.full(tr.sum(), c)
        res.update(param=None, coef={})
    else:
        raise ValueError(model)
    res["pred_bps"] = (pz * sig[te] * 1e4).astype(np.float32)
    # thresholds for the economic check come from TRAINING-window predictions (no test information)
    ptr_bps = ptr * sig[tr] * 1e4
    res["train_abs_q"] = {q: float(np.quantile(np.abs(ptr_bps), q)) for q in (0.5, 0.8, 0.9, 0.95)}
    res["train_pred_sd"] = float(np.std(ptr_bps))
    res["n_train"] = int(tr.sum())
    gc.collect()
    return res


def run_grid(d: pd.DataFrame, jobs: list[tuple], tag: str = "main", verbose: bool = True) -> pd.DataFrame:
    """jobs = [(set_name, model, h, ycol)]; one cache file per job (all folds)."""
    rows = []
    cdir = CACHE / f"preds_{tag}"
    cdir.mkdir(exist_ok=True)
    for (sname, model, h, ycol) in jobs:
        fn = cdir / f"{sname}__{model}__{h}__{ycol}.parquet"
        if fn.exists():
            rows.append(pd.read_parquet(fn))
            continue
        t0 = time.time()
        feats = set_features(sname) if model != "mean" else []
        parts, meta = [], []
        for fold in folds():
            r = fit_predict(d, feats, h, ycol, model, fold)
            parts.append(pd.DataFrame({"row": r["idx"], "pred_bps": r["pred_bps"], "fold": r["fold"]}))
            meta.append({"fold": r["fold"], "param": r["param"], "train_abs_q": r["train_abs_q"],
                         "train_pred_sd": r["train_pred_sd"], "n_train": r["n_train"], "coef": r["coef"]})
        out = pd.concat(parts, ignore_index=True)
        out["set"], out["model"], out["h"], out["ycol"] = sname, model, h, ycol
        out.to_parquet(fn)
        json.dump(meta, open(fn.with_suffix(".json"), "w"), default=float)
        rows.append(out)
        if verbose:
            print(f"  {sname:10s} {model:5s} {h:3s} {ycol:10s} {time.time() - t0:6.1f}s "
                  f"params={[m['param'] for m in meta]}", flush=True)
    return pd.concat(rows, ignore_index=True)


def job_meta(sname, model, h, ycol, tag="main"):
    return json.load(open(CACHE / f"preds_{tag}" / f"{sname}__{model}__{h}__{ycol}.json"))


# =====================================================================================
# 5. evaluation
# =====================================================================================
def day_index(t_us: np.ndarray) -> np.ndarray:
    return (t_us // DAY).astype(np.int64)


def daily_parts(y_bps, p_bps, day, base_bps=None):
    """per-day sums for a ratio statistic: SSE_model, SSE_base (zero by default), sum y^2, n."""
    b = np.zeros_like(y_bps) if base_bps is None else base_bps
    df = pd.DataFrame({"d": day, "e_m": (y_bps - p_bps) ** 2, "e_b": (y_bps - b) ** 2})
    return df.groupby("d")[["e_m", "e_b"]].sum()


def block_bootstrap_r2(g: pd.DataFrame, n_boot: int = 1000, block: int = 20, seed: int = 0):
    """moving-block bootstrap over days of R2 = 1 - SSE_m / SSE_b. Returns (r2, lo, hi, p_le_0)."""
    em, eb = g["e_m"].to_numpy(), g["e_b"].to_numpy()
    r2 = 1 - em.sum() / eb.sum()
    n = len(em)
    if n < 2 * block:
        return r2, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    bm, bb = em[idx].sum(1), eb[idx].sum(1)
    rb = 1 - bm / bb
    lo, hi = np.percentile(rb, [2.5, 97.5])
    # centred bootstrap p-value for H0: R2 <= 0
    p = float(np.mean(rb - r2 >= r2)) if r2 > 0 else 1.0
    return r2, lo, hi, p


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        j = o[i]
        prev = min(prev, p[j] * m / (i + 1))
        q[j] = prev
    return q


# =====================================================================================
# 6. alignment checks
# =====================================================================================
def _cb_minutes(asset: str) -> pd.DataFrame:
    p = sorted(CB.glob(f"{asset}-USD_60_*.parquet"))[0]
    return pd.read_parquet(p, columns=["period_start_us", "period_end_us", "close"])


def coinbase_lag_check(asset: str, start: str, end: str, lags=range(-3, 4)) -> pd.Series:
    """corr( Coinbase 1m return labelled by its candle, Binance 1m return shifted by `lag` minutes ).
    If Coinbase candles are start-labelled (as assumed), the peak must be at lag 0."""
    cb = _cb_minutes(asset)
    a, b = us(pd.Timestamp(start, tz="UTC")), us(pd.Timestamp(end, tz="UTC"))
    cb = cb[(cb.period_start_us >= a) & (cb.period_start_us < b)].set_index("period_start_us")["close"]
    grid = np.arange(a, b + MIN, MIN, dtype=np.int64)
    pbn, _ = query_prices(asset, grid + MIN)          # Binance close of minute [g, g+1m) = last trade < g+1m
    bn = pd.Series(pbn, index=grid)
    rc = np.log(cb.reindex(grid)).diff()
    rb = np.log(bn).diff()
    return pd.Series({L: rc.corr(rb.shift(-L)) for L in lags}, name=asset)


def premium_availability_check(asset: str, n: int = 3000, seed: int = 1) -> dict:
    """Recompute premium_raw_coinbase_<asset>_1m at sampled decision times from Coinbase candles and the
    Binance price grid, using only the last minute that ended strictly before the decision."""
    df = pd.read_parquet(DSROOT / DATASETS[asset] / "notional.parquet",
                         columns=["decision_time_us", f"premium_raw_coinbase_{asset.lower()}_1m"])
    rng = np.random.default_rng(seed)
    s = df.iloc[np.sort(rng.choice(len(df), n, replace=False))]
    t = s["decision_time_us"].to_numpy(np.int64)
    m_end = ((t - 1) // MIN) * MIN                      # last minute end strictly before t
    cb = _cb_minutes(asset).set_index("period_end_us")["close"]
    pc = cb.reindex(m_end).to_numpy()
    pb, _ = query_prices(asset, m_end)
    mine = np.log(pc / pb)
    ds = s.iloc[:, 1].to_numpy()
    ok = np.isfinite(mine) & np.isfinite(ds)
    # same check with the minute that is still open at t (would be look-ahead): must NOT match
    pc2 = cb.reindex(m_end + MIN).to_numpy()
    pb2, _ = query_prices(asset, m_end + MIN)
    wrong = np.log(pc2 / pb2)
    ok2 = np.isfinite(wrong) & np.isfinite(ds)
    return {"asset": asset, "n": int(ok.sum()), "match_closed_minute": float(np.mean(np.isclose(mine[ok], ds[ok], atol=1e-9))),
            "match_open_minute": float(np.mean(np.isclose(wrong[ok2], ds[ok2], atol=1e-9)))}


# =====================================================================================
# 7. evaluation tables
# =====================================================================================
def attach(preds: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """add asset, time, realised return (bps) and ex-ante sigma to prediction rows."""
    p = preds.copy()
    r = p["row"].to_numpy()
    p["asset"] = d["asset"].astype(str).to_numpy()[r]
    p["t_us"] = d["t_us"].to_numpy()[r]
    p["day"] = day_index(p["t_us"].to_numpy())
    y = np.empty(len(p))
    sig = np.empty(len(p))
    for yc in p["ycol"].unique():
        m = (p["ycol"] == yc).to_numpy()
        y[m] = d[yc].to_numpy()[r[m]] * 1e4
        h = p.loc[m, "h"].iloc[0]
        sig[m] = d["rvol_1d"].to_numpy()[r[m]] * np.sqrt(HORIZONS[h] / DAY) * 1e4
    p["y_bps"], p["sig_bps"] = y, sig
    return p


def boot_delta(e_a, e_b, e_0, day, n_boot=1000, block=20, seed=0):
    """Delta = (SSE_a - SSE_b) / SSE_0 with a moving-block bootstrap over days.
    a = reference forecast, b = candidate, 0 = zero forecast. Returns (delta, lo, hi, p(delta<=0))."""
    g = pd.DataFrame({"d": day, "a": e_a, "b": e_b, "z": e_0}).groupby("d").sum()
    A, B, Z = g["a"].to_numpy(), g["b"].to_numpy(), g["z"].to_numpy()
    est = (A.sum() - B.sum()) / Z.sum()
    n = len(A)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    st = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    bd = (A[idx].sum(1) - B[idx].sum(1)) / Z[idx].sum(1)
    lo, hi = np.percentile(bd, [2.5, 97.5])
    p = float(np.mean(bd - est >= est)) if est > 0 else 1.0
    return est, lo, hi, p


def drop_top_days(e_a, e_b, e_0, day, k=10):
    """Delta after removing the k days with the largest |daily contribution|, and the share of the
    total SSE gain those days account for."""
    g = pd.DataFrame({"d": day, "a": e_a, "b": e_b, "z": e_0}).groupby("d").sum()
    contrib = g["a"] - g["b"]
    top = contrib.abs().nlargest(k).index
    rest = g.drop(top)
    tot = contrib.sum()
    return (rest["a"].sum() - rest["b"].sum()) / rest["z"].sum(), (contrib.loc[top].sum() / tot if tot != 0 else np.nan)


def quintile_spread(p: pd.DataFrame) -> float:
    """mean realised bps in the top minus bottom prediction quintile (quintiles inside fold x asset)."""
    q = p.groupby(["fold", "asset"])["pred_bps"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    m = p.groupby(q)["y_bps"].mean()
    return float(m.get(4, np.nan) - m.get(0, np.nan))


def evaluate(P: pd.DataFrame, ref: dict | None = None, n_boot: int = 1000, by_asset: bool = True) -> pd.DataFrame:
    """One row per (set, model, h, ycol, scope). R2 vs zero, vs historical mean, sigma-scaled R2, correlation,
    block-bootstrap CI and p for R2>0, folds with R2>0, quintile spread."""
    rows = []
    keys = ["set", "model", "h", "ycol"]
    mean_pred = P[P["model"] == "mean"].set_index(["h", "ycol", "row"])["pred_bps"]
    for k, g in P[P["model"] != "mean"].groupby(keys, observed=True):
        scopes = [("ALL", g)] + ([(a, ga) for a, ga in g.groupby("asset")] if by_asset else [])
        for sc, s in scopes:
            y, p, sg = s["y_bps"].to_numpy(), s["pred_bps"].to_numpy(), s["sig_bps"].to_numpy()
            mp = mean_pred.loc[(k[2], k[3])].reindex(s["row"]).to_numpy()
            e0, em, eh = y ** 2, (y - p) ** 2, (y - mp) ** 2
            r2, lo, hi, pv = boot_delta(e0, em, e0, s["day"].to_numpy(), n_boot=n_boot)
            fr = s.assign(e0=e0, em=em).groupby("fold")[["e0", "em"]].sum()
            dtop, share = drop_top_days(e0, em, e0, s["day"].to_numpy())
            rows.append(dict(zip(keys, k), scope=sc, n=len(s),
                             r2=r2, r2_lo=lo, r2_hi=hi, p=pv,
                             r2_vs_mean=1 - em.sum() / eh.sum(),
                             r2_sig=1 - np.sum(((y - p) / sg) ** 2) / np.sum((y / sg) ** 2),
                             corr=np.corrcoef(p, y)[0, 1] if np.std(p) > 0 else np.nan,
                             spear=pd.Series(p).corr(pd.Series(y), method="spearman") if np.std(p) > 0 else np.nan,
                             folds_pos=int(((1 - fr["em"] / fr["e0"]) > 0).sum()), n_folds=len(fr),
                             r2_drop10d=dtop, top10d_share=share,
                             q5_q1_bps=quintile_spread(s), pred_sd_bps=float(np.std(p))))
    return pd.DataFrame(rows)


def nested_delta(P: pd.DataFrame, pairs: list[tuple[str, str]], n_boot: int = 1000, by_asset: bool = True) -> pd.DataFrame:
    """Delta R2 (vs zero) of set b over set a, same model / horizon / target, with bootstrap."""
    rows = []
    idx = P.set_index(["set", "model", "h", "ycol", "row"]).sort_index()
    for (a, b) in pairs:
        for (model, h, yc), _ in P[(P["set"] == b) & (P["model"] != "mean")].groupby(["model", "h", "ycol"]):
            if (a, model, h, yc) not in idx.index.droplevel("row").unique():
                continue
            A = idx.loc[(a, model, h, yc)]
            B = idx.loc[(b, model, h, yc)].reindex(A.index)
            S = A.assign(pb=B["pred_bps"].to_numpy())
            scopes = [("ALL", S)] + ([(x, sx) for x, sx in S.groupby("asset")] if by_asset else [])
            for sc, s in scopes:
                y = s["y_bps"].to_numpy()
                ea, eb, e0 = (y - s["pred_bps"].to_numpy()) ** 2, (y - s["pb"].to_numpy()) ** 2, y ** 2
                dl, lo, hi, pv = boot_delta(ea, eb, e0, s["day"].to_numpy(), n_boot=n_boot)
                fr = s.assign(ea=ea, eb=eb).groupby("fold")[["ea", "eb"]].sum()
                dtop, share = drop_top_days(ea, eb, e0, s["day"].to_numpy())
                rows.append(dict(base=a, cand=b, model=model, h=h, ycol=yc, scope=sc, d_r2=dl, lo=lo, hi=hi, p=pv,
                                 folds_better=int((fr["eb"] < fr["ea"]).sum()), d_r2_drop10d=dtop, top10d_share=share))
    return pd.DataFrame(rows)


def fold_r2(P: pd.DataFrame) -> pd.DataFrame:
    s = P.assign(e0=P["y_bps"] ** 2, em=(P["y_bps"] - P["pred_bps"]) ** 2)
    g = s.groupby(["set", "model", "h", "ycol", "asset", "fold"])[["e0", "em"]].sum()
    return (1 - g["em"] / g["e0"]).rename("r2").reset_index()


def economics(P: pd.DataFrame, costs=(0, 2, 5, 10, 20), q: float = 0.9, tag: str = "main") -> pd.DataFrame:
    """Gross and net bps per decision for a sign rule that trades only when |pred| exceeds the q-quantile of
    |pred| on the TRAINING window of the same fold (threshold known ex ante). Overlapping positions are not
    netted: this is a per-signal average, not a portfolio backtest."""
    rows = []
    for (sname, model, h, yc), g in P[P["model"] != "mean"].groupby(["set", "model", "h", "ycol"]):
        meta = job_meta(sname, model, h, yc, tag)
        thr = {m["fold"]: m["train_abs_q"][str(q)] if str(q) in m["train_abs_q"] else m["train_abs_q"][q] for m in meta}
        for sc, s in [("ALL", g)] + list(g.groupby("asset")):
            sel = np.abs(s["pred_bps"].to_numpy()) > s["fold"].map(thr).to_numpy()
            pnl = np.sign(s["pred_bps"].to_numpy()) * s["y_bps"].to_numpy()
            gross_sel = pnl[sel].mean() if sel.any() else np.nan
            dd = pd.DataFrame({"d": s["day"].to_numpy()[sel], "x": pnl[sel]}).groupby("d")["x"].mean()
            se = dd.std() / np.sqrt(len(dd)) if len(dd) > 2 else np.nan
            dsum = pd.DataFrame({"d": s["day"].to_numpy()[sel], "x": pnl[sel]}).groupby("d")["x"].agg(["sum", "size"])
            keep = dsum.drop(dsum["sum"].abs().nlargest(10).index) if len(dsum) > 20 else dsum
            gross_trim = keep["sum"].sum() / keep["size"].sum() if len(keep) else np.nan
            r = dict(set=sname, model=model, h=h, ycol=yc, scope=sc, trade_frac=sel.mean(),
                     gross_all=pnl.mean(), gross_sel=gross_sel, gross_sel_se_daily=se,
                     gross_sel_drop10d=gross_trim, days_pos=float((dd > 0).mean()) if len(dd) else np.nan,
                     mean_abs_pred=np.abs(s["pred_bps"]).mean())
            for c in costs:
                r[f"net_{c}"] = gross_sel - c
                r[f"net_{c}_drop10d"] = gross_trim - c
            rows.append(r)
    return pd.DataFrame(rows)


def bar_duration_table(d: pd.DataFrame) -> pd.DataFrame:
    """effective calendar duration of k-bar horizons (k = 1, 5, 20) - median and IQR in minutes."""
    out = []
    for a, g in d.groupby("asset", observed=True):
        t = g["t_us"].to_numpy()
        yr = pd.to_datetime(t, unit="us").year
        for k in (1, 5, 20):
            dur = (t[k:] - t[:-k]) / MIN
            for y in np.unique(yr[:-k]):
                v = dur[yr[:-k] == y]
                out.append(dict(asset=a, k=k, year=y, med_min=np.median(v), p10=np.percentile(v, 10), p90=np.percentile(v, 90)))
    return pd.DataFrame(out)
