"""Worker code for multi_config_meta_labeling.ipynb.

One configuration = (dataset, strategy, strategy parameters). For each configuration:

1. run the QuantForge strategy (uniform costs: fee_bps per fill, no slippage, tick size from the dataset manifest);
2. meta-label y = 1[trade_return > 0] (net of the simulator's fees);
3. development = trades whose label interval ends before HOLDOUT_START; holdout = trades decided at/after HOLDOUT_START;
4. development: PurgedKFold (5 folds, embargo 2 %, no shuffling) out-of-fold probabilities of a fixed random forest with training-only
   preprocessing (coverage filter, constant filter, median imputation) on the eligible decision-time features;
5. holdout: the same pipeline refitted on all development trades, scored once on the holdout trades.

Development and holdout results are written to SEPARATE files so that the notebook can apply its selection rule to the development
file before it reads the holdout file.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import quant_forge.strategies as qs
from quant_forge.validation import PurgedKFold

HOLDOUT_START = pd.Timestamp("2025-03-01", tz="UTC")
FEE_BPS = 5.0
N_SPLITS = 5
PCT_EMBARGO = 0.02
MIN_COVERAGE = 0.70
MIN_DEV_TRADES = 150          # below this no meta-model is fitted (PnL statistics are still reported)
MIN_HOLDOUT_TRADES = 30
FIXED_NOTIONAL = 10_000.0     # USD per trade, only to express the fixed-notional event PnL
SEED = 42
RF_JOBS = 3

# ---- strategy grid: 1-2 threshold parameters per strategy, fixed a priori around the library defaults -------------
S = qs.Strategies
STRATEGY_GRID = (
    [(S.RSI_CROSS_ABOVE, {"rsi_level": v}) for v in (60.0, 70.0, 80.0)]
    + [(S.DONCHIAN_BREAKOUT_CONTINUATION, {"adx_min": a, "min_break_atr": b}) for a in (15.0, 22.0, 30.0) for b in (0.25, 0.5, 1.0)]
    + [(S.TIS_BTC_MOMENTUM_CONFLUENCE, {"streak_length": v}) for v in (2, 3, 4)]
    + [(S.SUPERTREND_REGIME_CONFLUENCE, {"min_signal_score": v}) for v in (50, 65, 80)]
)

# strategy outputs that are price levels or price-unit distances (non-stationary, AFML ch. 5) -> never model inputs
STRATEGY_PRICE_LEVEL = {"dc_high", "dc_low", "atr_value", "ema_trend", "sma_trend", "st_band", "st_ema", "st_vol_ma",
                        "sl_distance", "tp_distance"}

# registry columns that are raw sizes / currency levels (scale with price or with the bar unit) -> scale-free versions are kept
_SCALED = [r"^aggr_(buy|sell|signed)_(notional|volume)$", r"^tick_signed_(notional|volume)$", r"^(notional|volume)_per_s$",
           r"^avg_trade_size$", r"^quote_spread$", r"^etf_(btc|eth)_(net_flow_total|sum_reported_funds)_usd$"]
_KYLE = r"^kyle_lambda_(trades|bars50)_(tick|aggr)$"


def strategy_label(strategy, params) -> str:
    return strategy.value + "|" + ",".join(f"{k}={v:g}" for k, v in params.items())


def tick_size_of(manifest) -> float:
    dec = manifest["instrument"].get("price_decimals")
    return 10.0 ** -int(dec) if dec is not None else 0.01


def load_dataset(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    ident = list(manifest["series"].values())[0]["identity"]
    bars = pd.read_parquet(path / f"{ident['bar_type']}.parquet")
    registry = pd.DataFrame(json.loads((path / f"{ident['bar_type']}_registry.json").read_text())).set_index("name")
    df = bars.set_index(pd.to_datetime(bars["bar_end_us"], unit="us", utc=True)).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df, registry, manifest, ident


def eligible_registry_columns(df, registry):
    """Point-in-time registry features minus no-data columns, scaled raw sizes and ages of mostly-null feeds."""
    reg = [c for c in registry.index if c in df.columns]
    dev_bars = df.loc[df.index < HOLDOUT_START]
    out, excluded = [], {}
    for c in reg:
        if registry.at[c, "status"] == "unavailable_from_current_source":
            excluded[c] = "no data in build"; continue
        if any(re.match(p, c) for p in _SCALED):
            excluded[c] = "scaled raw size / currency level"; continue
        if re.match(_KYLE, c):
            excluded[c] = "replaced by *_rel"; continue
        stale = False
        for suf in ("_economic_age_d", "_age_s"):
            if c.endswith(suf):
                base = c[: -len(suf)]
                stale = base in df.columns and dev_bars[base].notna().mean() < MIN_COVERAGE <= dev_bars[c].notna().mean()
                break
        if stale:
            excluded[c] = "age of mostly-null feed"; continue
        out.append(c)
    return reg, out, excluded


def rf():
    return RandomForestClassifier(n_estimators=300, criterion="entropy", max_features="sqrt", min_weight_fraction_leaf=0.05,
                                  class_weight="balanced_subsample", n_jobs=RF_JOBS, random_state=SEED)


class Prep:
    """coverage filter -> constant filter -> median imputation, learned from the training rows only"""
    def fit(self, X):
        cov = X.notna().mean()
        self.kept = [c for c in X.columns if cov[c] >= MIN_COVERAGE and X[c].nunique(dropna=True) > 1]
        self.med = X[self.kept].median()
        return self

    def transform(self, X):
        return X[self.kept].fillna(self.med)


def p1(est, X):
    return est.predict_proba(X)[:, list(est.classes_).index(1)]


def clf_scores(y, p, prior):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    out = {"log_loss": log_loss(y, p, labels=[0, 1]), "prior_log_loss": log_loss(y, np.full(len(y), np.clip(prior, 1e-6, 1 - 1e-6)), labels=[0, 1]),
           "brier": brier_score_loss(y, p), "auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan}
    out["dlogloss_vs_prior"] = out["log_loss"] - out["prior_log_loss"]
    return out


def ret_stats(r, prefix):
    r = np.asarray(r, float)
    n = len(r)
    if n == 0:
        return {f"{prefix}n": 0}
    sd = r.std(ddof=1) if n > 1 else np.nan
    z = (r - r.mean()) / sd if n > 2 and sd > 0 else np.full(n, np.nan)
    return {f"{prefix}n": n, f"{prefix}mean_bps": 1e4 * r.mean(), f"{prefix}sd_bps": 1e4 * sd,
            f"{prefix}sr": r.mean() / sd if sd and sd > 0 else np.nan, f"{prefix}t": r.mean() / sd * np.sqrt(n) if sd and sd > 0 else np.nan,
            f"{prefix}skew": float(np.mean(z ** 3)) if n > 2 else np.nan, f"{prefix}kurt": float(np.mean(z ** 4)) if n > 3 else np.nan,
            f"{prefix}pnl_usd": FIXED_NOTIONAL * r.sum(), f"{prefix}win": float((r > 0).mean())}


def evaluate_config(df, reg_all, eligible, tick, strategy, params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = qs.run_strategy(df, strategy, fee_bps=FEE_BPS, slippage_ticks=0, tick_size=tick,
                              context_columns=list(dict.fromkeys(eligible + [c for c in reg_all if re.match(_KYLE, c)] + ["close"])),
                              **params)
    tr = res.closed_trades_frame().sort_values("decision_time").reset_index(drop=True)
    for c in [c for c in reg_all if re.match(_KYLE, c)]:
        tr[f"{c}_rel"] = 1e4 * tr[c] / tr["close"]
    strat_feats = [f for f in res.feature_columns if f not in STRATEGY_PRICE_LEVEL]
    feats = strat_feats + eligible + [f"{c}_rel" for c in reg_all if re.match(_KYLE, c)]
    assert not set(feats) & set(res.post_decision_columns)

    dt, et = pd.DatetimeIndex(tr["decision_time"]), pd.DatetimeIndex(tr["exit_time"])
    dev = tr.loc[et < HOLDOUT_START].copy()
    hold = tr.loc[dt >= HOLDOUT_START].copy()
    base = {"strategy": strategy.value, "params": json.dumps(params), "config": strategy_label(strategy, params),
            "n_trades_total": len(tr), "n_features_offered": len(feats),
            "first_trade": dt.min() if len(tr) else pd.NaT}
    dev_row = {**base, **ret_stats(dev["trade_return"], "dev_"), **ret_stats(dev["trade_return_gross"], "dev_gross_")}
    hold_row = {**base, **ret_stats(hold["trade_return"], "hold_"), **ret_stats(hold["trade_return_gross"], "hold_gross_")}
    if len(dev):
        s, e = pd.DatetimeIndex(dev["decision_time"]).asi8, pd.DatetimeIndex(dev["exit_time"]).asi8
        dev_row["n_overlapping"] = int((s[1:] <= np.maximum.accumulate(e)[:-1]).sum()) if len(dev) > 1 else 0
    if len(dev) < MIN_DEV_TRADES or dev["trade_return"].gt(0).nunique() < 2:
        dev_row["modelled"] = False
        return dev_row, hold_row

    dev.index = pd.DatetimeIndex(dev["decision_time"], name="event_start")
    dev = dev[~dev.index.duplicated(keep="first")]
    X, y = dev[feats], dev["trade_return"].gt(0).astype(int)
    t1 = pd.Series(pd.DatetimeIndex(dev["exit_time"]), index=dev.index)
    cv = PurgedKFold(n_splits=N_SPLITS, samples_info_sets=t1, pct_embargo=PCT_EMBARGO)
    oof, prior_oof, fold_dll = np.full(len(X), np.nan), np.full(len(X), np.nan), []
    for tr_i, te_i in cv.split(X):
        pp = Prep().fit(X.iloc[tr_i])
        est = rf().fit(pp.transform(X.iloc[tr_i]), y.iloc[tr_i])
        oof[te_i] = p1(est, pp.transform(X.iloc[te_i]))
        prior_oof[te_i] = y.iloc[tr_i].mean()
        yt = y.iloc[te_i]
        fold_dll.append(log_loss(yt, np.clip(oof[te_i], 1e-6, 1 - 1e-6), labels=[0, 1])
                        - log_loss(yt, np.full(len(yt), np.clip(prior_oof[te_i][0], 1e-6, 1 - 1e-6)), labels=[0, 1]))
    sc = clf_scores(y, oof, 0.5)
    sc["prior_log_loss"] = log_loss(y, np.clip(prior_oof, 1e-6, 1 - 1e-6), labels=[0, 1])     # fold-train prevalence
    sc["dlogloss_vs_prior"] = sc["log_loss"] - sc["prior_log_loss"]
    acc = oof >= 0.5
    dev_row.update({"modelled": True, "n_dev_modelled": len(X), "base_rate": float(y.mean()), "n_features_used": len(Prep().fit(X).kept),
                    **{f"oof_{k}": v for k, v in sc.items()}, "folds_better_than_prior": int(np.sum(np.array(fold_dll) < 0)),
                    **ret_stats(dev.loc[acc, "trade_return"], "dev_filt_")})
    # holdout: refit on all development trades, score once
    if len(hold) >= MIN_HOLDOUT_TRADES:
        pp = Prep().fit(X)
        est = rf().fit(pp.transform(X), y)
        ph = p1(est, pp.transform(hold[feats]))
        yh = hold["trade_return"].gt(0).astype(int)
        hs = clf_scores(yh, ph, y.mean())
        hold_row.update({"modelled": True, **{f"hold_{k}": v for k, v in hs.items()}, "hold_base_rate": float(yh.mean()),
                         **ret_stats(hold.loc[ph >= 0.5, "trade_return"], "hold_filt_")})
    return dev_row, hold_row


def run_dataset(args):
    """Worker: all strategy configurations on one dataset. Returns (dev_rows, hold_rows, info)."""
    path, meta = args
    df, registry, manifest, ident = load_dataset(Path(path))
    reg_all, eligible, excluded = eligible_registry_columns(df, registry)
    tick = tick_size_of(manifest)
    dev_rows, hold_rows = [], []
    for strategy, params in STRATEGY_GRID:
        try:
            d, h = evaluate_config(df, reg_all, eligible, tick, strategy, params)
        except Exception as exc:          # recorded, never silently dropped
            d = h = {"strategy": strategy.value, "config": strategy_label(strategy, params), "error": repr(exc)[:300]}
        for row in (d, h):
            row.update(meta)
        dev_rows.append(d); hold_rows.append(h)
    info = {**meta, "n_bars": len(df), "tick_size": tick, "n_eligible_registry": len(eligible), "n_excluded": len(excluded)}
    return dev_rows, hold_rows, info


# ---- dataset grid: 5 assets x {volume, notional} bars x ~{25, 50, 100} bars per day ------------------------------
# BTC/BNB/SOL/XRP: the named builds; ETH exists only as hash-named builds (older pipeline code, same features).
DATASETS = {
    ("BTCUSDT", "volume", 25): "BTCUSDT_spot_volume_2650_25pd_20210901-20260901_99bf72d5",
    ("BTCUSDT", "volume", 50): "BTCUSDT_spot_volume_1320_50pd_20210901-20260901_cfb15bd1",
    ("BTCUSDT", "volume", 100): "BTCUSDT_spot_volume_662_100pd_20210901-20260901_d005f485",
    ("BTCUSDT", "notional", 25): "BTCUSDT_spot_notional_95500000_25pd_20210901-20260901_5e690aa9",
    ("BTCUSDT", "notional", 50): "BTCUSDT_spot_notional_47800000_50pd_20210901-20260901_424c3b60",
    ("BTCUSDT", "notional", 100): "BTCUSDT_spot_notional_23900000_100pd_20210901-20260901_ccbcf1e7",
    ("ETHUSDT", "volume", 25): "070d0f262f4dcae8",
    ("ETHUSDT", "volume", 50): "b9ed3b3d7f393876",
    ("ETHUSDT", "volume", 100): "6a915d3a25d574ce",
    ("ETHUSDT", "notional", 25): "87faf03be2f5d6e3",
    ("ETHUSDT", "notional", 50): "89f3d9afc037aa8c",
    ("ETHUSDT", "notional", 100): "60a5a92295b2e1cb",
    ("SOLUSDT", "volume", 25): "SOLUSDT_spot_volume_168000_25pd_20210901-20260901_8d585be4",
    ("SOLUSDT", "volume", 50): "SOLUSDT_spot_volume_84200_50pd_20210901-20260901_52d81b48",
    ("SOLUSDT", "volume", 100): "SOLUSDT_spot_volume_42100_100pd_20210901-20260901_b2c6c83b",
    ("SOLUSDT", "notional", 25): "SOLUSDT_spot_notional_17500000_25pd_20210901-20260901_3279d065",
    ("SOLUSDT", "notional", 50): "SOLUSDT_spot_notional_8750000_50pd_20210901-20260901_4b789114",
    ("SOLUSDT", "notional", 100): "SOLUSDT_spot_notional_4380000_100pd_20210901-20260901_e0752b18",
    ("XRPUSDT", "volume", 25): "XRPUSDT_spot_volume_12500000_25pd_20210901-20260901_ebdeca09",
    ("XRPUSDT", "volume", 50): "XRPUSDT_spot_volume_6270000_50pd_20210901-20260901_c115702e",
    ("XRPUSDT", "volume", 100): "XRPUSDT_spot_volume_3130000_100pd_20210901-20260901_b4c42b9e",
    ("XRPUSDT", "notional", 25): "XRPUSDT_spot_notional_12100000_25pd_20210901-20260901_a2b455ce",
    ("XRPUSDT", "notional", 50): "XRPUSDT_spot_notional_6050000_50pd_20210901-20260901_b9acfe61",
    ("XRPUSDT", "notional", 100): "XRPUSDT_spot_notional_3030000_100pd_20210901-20260901_c82cd27e",
    ("BNBUSDT", "volume", 25): "BNBUSDT_spot_volume_19900_25pd_20210901-20260901_7f7ed2f1",
    ("BNBUSDT", "volume", 50): "BNBUSDT_spot_volume_9970_50pd_20210901-20260901_a988f491",
    ("BNBUSDT", "volume", 100): "BNBUSDT_spot_volume_4980_100pd_20210901-20260901_6cae7188",
    ("BNBUSDT", "notional", 25): "BNBUSDT_spot_notional_9160000_25pd_20210901-20260901_dc28e758",
    ("BNBUSDT", "notional", 50): "BNBUSDT_spot_notional_4580000_50pd_20210901-20260901_4e8e8a35",
    ("BNBUSDT", "notional", 100): "BNBUSDT_spot_notional_2290000_100pd_20210901-20260901_9848aac9",
}


def run_grid(data_root: Path, cache_dir: Path, workers: int = 6):
    """Runs every dataset x strategy configuration; writes dev.parquet, holdout.parquet and datasets.parquet."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(data_root / d), {"asset": a, "bar_type": b, "bars_per_day": n, "dataset": d}) for (a, b, n), d in DATASETS.items()]
    dev, hold, info = [], [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
        futs = {ex.submit(run_dataset, j): j[1]["dataset"] for j in jobs}
        for f in as_completed(futs):
            d, h, i = f.result()
            dev += d; hold += h; info.append(i)
            print(f"done {futs[f]} ({len(info)}/{len(jobs)})", flush=True)
    pd.DataFrame(dev).to_parquet(cache_dir / "dev.parquet")
    pd.DataFrame(hold).to_parquet(cache_dir / "holdout.parquet")
    pd.DataFrame(info).to_parquet(cache_dir / "datasets.parquet")
