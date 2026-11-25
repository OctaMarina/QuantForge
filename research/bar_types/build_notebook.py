"""Write research/bar_types/bar_type_frequency_meta_labeling.ipynb from the cell sources below.

    python build_notebook.py            # then execute with run_notebook.py (nbclient)
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip("\n")))


def code(s: str) -> None:
    CELLS.append(("code", s.strip("\n")))


# =====================================================================================================================
md(r"""
# Bar type and sampling frequency: what happens to the four primary strategies, and what is meta-labeling worth?

**Question.** For the four primary strategies of `quant_forge.strategies` (RSI cross-above, Donchian breakout continuation,
TIS momentum confluence, SuperTrend regime confluence) with their default entry and exit rules, on the 40-asset Binance spot
universe of the reference notebook (`meta_labeling_economic_value_quant_forge.ipynb`, 20 `SEEN` assets for training and 20
`NEW` assets for transfer): **how do the bar type (time, volume, dollar) and the sampling frequency (48, 24, 6 and 1 bars per
day) change the primary strategies, and how much economic value does meta-labeling add in each configuration?**

**Design in one paragraph.** Twelve bar configurations: time bars of 30 minutes, 1 hour, 4 hours and 1 day (Binance klines),
and volume and dollar bars targeting the same average frequencies, built from the complete Binance aggTrades archive
(≈ 16 billion trades) with a threshold estimated **per asset from the trailing 30 complete days of traded volume** (known before
any bar of the day opens), so that the realised frequency rises and falls with market activity. On every configuration the
strategies regenerate their signals and exits, the trades are meta-labeled (`y = 1[net trade return > 0]`, AFML §3.6), and one
simple meta-model specification — an L2 logistic regression on a fixed set of point-in-time features measured in elapsed time,
`C` chosen by purged calendar K-fold log loss, Platt calibration — is refitted walk-forward on 12-month windows using `SEEN`
trades only. Four portfolio rules are compared under one capital budget per strategy (20 notionals, one per asset), the same
execution (next-bar open, 5 bps per fill) and the same daily mark-to-market calendar (2019-03-01 → 2026-11-19): **U** fixed
size, **D** inverse volatility, **D×F** inverse volatility plus the meta-label filter, **D×P** the same filter plus
probability-based sizing. Differences come with paired monthly block-bootstrap intervals; `NEW` is the main result.

**Sources** (AFML = López de Prado, *Advances in Financial Machine Learning*, 2018): information-driven bars, AFML §2.3; the
volume clock, Easley, López de Prado & O'Hara, *The Volume Clock: Insights into the High-Frequency Paradigm*, JPM 2012;
meta-labeling, AFML §3.6 and Joubert, *Meta-Labeling: Theory and Framework*, JFDS 2022; purged K-fold, AFML §7.4; bet sizing
from probabilities, AFML §10.3 (centred at the break-even win rate as in the reference notebook, AFML §15.2).
""")

md(r"""
### What changes with respect to the reference notebook, and why

| | Reference (`meta_labeling_economic_value_quant_forge`) | This notebook |
|---|---|---|
| Bars | 4h klines only | 12 configurations (`quant_forge.data.load_bars` + `BarConfig`), event bars from aggTrades |
| Features | 168 candidates, horizons in bars (`bars_per_day = 6`) | ≈ 70 candidates, **horizons in elapsed time** and panel features aligned **as-of** across asynchronous bars (`quant_forge.features.elapsed_bar_features`) |
| Meta-model | LR / RF / HGB × full / selected / PCA, clustered-MDA selection, chosen per window | **one** spec for every configuration: logistic regression on all usable candidates, `C` by purged calendar K-fold, Platt calibration (`MetaLabelingConfig(model_names=("LR",), feature_selection=False)`) |
| Rules | 12 sizing rules | 4 portfolio rules: U, D, D×F (`DxF_be`), D×P (`DxP`) |
| Economics | realised PnL by exit month, budget = p95 of U's exposure (differs by scope) | **daily mark-to-market** PnL, **fixed budget** of 20 notionals per strategy portfolio for every configuration (`quant_forge.evaluation.portfolio`) |

The simpler model is deliberate: with twelve configurations, per-window model and feature selection would add selection noise
that differs between configurations, and the reference already showed that no model beat the win-rate prior in log loss. A
fixed specification makes the comparison across bar types about the bars, not about which model happened to win a window.

### What has been seen before (evidence quality)

The 4h configuration, the strategies, the universe and the periods were all used by earlier notebooks (see §0 of the reference):
the `NEW` assets were used once for model fitting decisions and the design of this study was informed by the earlier results.
The other eleven configurations were never examined, but on the same calendar. The transfer experiment on `NEW` remains clean
with respect to model fitting (no model ever trains on a `NEW` asset).
""")

md("## 1. Configuration")

code(r"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per process: the walk-forward runs many processes
import gc, re, sys, json, time, warnings, importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import display, Markdown


def find_project_root():
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / "quant_forge/strategies").exists():
            return path
    raise FileNotFoundError("QuantForge project root not found")


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import quant_forge
import quant_forge.strategies as qs
from quant_forge import data as qd, features as qf, metalabeling as ml, evaluation as qe, sizing as qz
from quant_forge.validation import calendar_windows
from quant_forge.utils import DEFAULT_CACHE_ROOT, DiskCache, frame_fingerprint

SEEN, NEW = qd.SEEN_ASSETS, qd.NEW_ASSETS
ALL_ASSETS = SEEN + NEW
GROUPS = ["NEW", "SEEN"]                      # NEW first: the transfer result is the main one
DATA_END = pd.Timestamp("2026-11-19", tz="UTC")
EVAL_START = pd.Timestamp("2019-03-01", tz="UTC")
WINDOWS = calendar_windows(EVAL_START, end=DATA_END, years=1)       # 12-month walk-forward windows (as the reference)
CAL = pd.date_range(EVAL_START, DATA_END - pd.Timedelta(days=1), freq="D")   # common daily calendar of all PnL series
MONTHS = qe.month_labels(CAL)
FEE_BPS, FEE_STRESS_BPS = 5.0, 10.0
NOTIONAL, SLOTS = 10_000.0, 20
BUDGET = SLOTS * NOTIONAL                     # capital of one strategy portfolio: one notional per asset of a group
SIZE_CAP = 10.0                               # no trade larger than 10 notionals after normalisation (as the reference)

FREQS = [48, 24, 6, 1]                        # bars per day: 30m, 1h, 4h, 1d
BAR_TYPES = os.environ.get("QF_BAR_TYPES", "time,volume,dollar").split(",")   # all three for the study
CONFIGS = [qd.BarConfig(bt, n) for bt in BAR_TYPES for n in FREQS]
KEYS = [c.key for c in CONFIGS]
CFG = {c.key: c for c in CONFIGS}
BASELINE = "time_4h"
FREQ_LABEL = {48: "48/d (30m)", 24: "24/d (1h)", 6: "6/d (4h)", 1: "1/d (1d)"}

S = qs.Strategies
STRATEGIES = {S.RSI_CROSS_ABOVE: "RSI", S.DONCHIAN_BREAKOUT_CONTINUATION: "DONCHIAN",
              S.TIS_BTC_MOMENTUM_CONFLUENCE: "TIS", S.SUPERTREND_REGIME_CONFLUENCE: "SUPERTREND"}
SS = list(STRATEGIES.values())
PORTFOLIOS = SS + ["ALL"]                     # ALL = the four strategy portfolios together (budget 4 x 20 notionals)

# one meta-model specification for every configuration (MIN_TRAIN 200, coverage 0.70, 4 calendar folds, 1 % embargo)
MLCFG = ml.MetaLabelingConfig(train_assets=tuple(SEEN), model_names=("LR",), pca_models=(), feature_selection=False, seed=0)
RULES = {"U": "U", "D": "D", "DxF": "DxF_be", "DxP": "DxP"}       # study name -> quant_forge.sizing rule
RULE_NAMES = {"U": "fixed size", "D": "inverse volatility", "DxF": "inv. vol + meta-label filter",
              "DxP": "inv. vol + filter + probability sizing"}
# diagnostic (not one of the four rules): inverse volatility with the whole strategy switched off in a window when its
# training win rate is below its break-even win rate, i.e. a model-free, strategy-level on/off decision
EVAL_RULES = list(RULES) + ["DxS"]
N_BOOT, SEED = 2000, 0

# flags (a configuration-strategy cell is flagged when one of these holds; see section 8)
MIN_TRADES_NEW = 200                          # trades of the strategy on NEW assets in the evaluation period
MIN_MODEL_COVERAGE = 0.75                     # share of evaluated trades that had a fitted meta-model
FREQ_RATIO_BAND = (0.8, 1.25)                 # median achieved / target bars per day across assets
MIN_HTF_COMPLETE = 0.90                       # TIS: share of days whose daily candle passes the strategy's coverage rule

CPUS = os.cpu_count() or 1
N_JOBS = min(8, CPUS)
WF_JOBS = min(16, CPUS)
CACHE_ROOT = DEFAULT_CACHE_ROOT / "bar_types"

# chart colours (validated categorical slots 1-3; diverging blue <-> red around a neutral grey)
COL = {"time": "#2a78d6", "volume": "#eb6834", "dollar": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
DIVERGING = mcolors.LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
plt.rcParams.update({"axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 10,
                     "axes.titleweight": "normal", "font.size": 9, "legend.frameon": False, "lines.linewidth": 1.6})
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 300)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
T0 = time.time()
print("quant_forge", quant_forge.__version__, "from", Path(quant_forge.__file__).parent, "| pandas", pd.__version__,
      "| sklearn", importlib.metadata.version("scikit-learn"), "| cpus", CPUS)
print("configurations:", ", ".join(c.label for c in CONFIGS))
print("walk-forward windows:", [str(w0.date()) for w0, _ in WINDOWS])
""")

# =====================================================================================================================
md(r"""
## 2. Data: twelve bar configurations through one loading interface

Every configuration is loaded with the same call, `qd.load_bars(assets, qd.BarConfig(bar_type, bars_per_day), end=DATA_END)`,
which returns the bar frame of the strategy and feature libraries (index = bar end, microsecond timestamps, OHLCV, taker flow).

* **Time bars** are Binance klines of 30m / 1h / 4h / 1d (`quant_forge.data.binance`, incremental parquet cache); the first 30
  days after listing are dropped, a missing kline or a kline without trades breaks the strategies' indicator segments.
* **Volume and dollar bars** (`quant_forge.data.event_bars`) are built from every aggTrade of the public archive
  (`data.binance.vision`, monthly and daily files streamed and inflated on the fly, SHA-256 checked). A bar closes when the base
  volume (volume bars) or quote volume (dollar bars) traded since it opened reaches its threshold; the threshold is fixed when
  the bar opens: **mean daily traded volume of the asset over the 30 complete UTC days before that day / target bars per day**.
  Nothing after the bar's opening day enters the threshold, so a day busier than the previous month produces more bars and a
  quiet day fewer. No bar opens during the first 30 days of an asset (the same warm-up as the time bars). Trades with the
  same timestamp stay in one bar; the decision time of a bar is its last trade, and the strategies' `next_open` orders fill at
  the first trade of the next bar (the analogue of the next kline's open). A pause of more than 4 hours between two trades
  (outage, halt) closes the bar and starts a new segment, like a missing kline.

The build of the event-bar cache (≈ 230 GB of zipped archive read once, 8 series per asset in one pass) is done by
`research/bar_types/scripts/build_event_bars.py`; the cell below shows its state and checks that the trade stream is complete:
the daily totals accumulated by the builder must equal the daily volumes of the 1-day klines. The archive has two kinds of
defects, both handled by the builder: zero-price aggTrades of BTC, ETH, BNB and LTC on 2017-12-04 and in January 2018 that the
klines do not contain (dropped: a single zero price would give its bar a low of 0 and an infinite log return), and repeated
records (the ETHUSDT 2018-01 file holds parts of the month four times — 14 million duplicate trades —, BTC and XRP repeat some
trades of 2026-02-11), which are dropped by aggTrade id.
""")

code(r"""
EVENT_TYPES = [bt for bt in BAR_TYPES if bt != "time"]
status = {a: qd.event_cache_status(f"{a}USDT") for a in ALL_ASSETS} if EVENT_TYPES else {}
missing_assets = [a for a, s in status.items() if s is None]
build = pd.DataFrame({a: {"group": "SEEN" if a in SEEN else "NEW", "first trade day": s["first_day"].date(),
                          "built through": (s["next_day"] - pd.Timedelta(days=1)).date(), "aggTrades (M)": s["trades"] / 1e6,
                          "archive GB": s["bytes"] / 1e9, "files": s["files"], "missing files": len(s["missing"]),
                          "invalid records dropped": s["dropped"], "duplicates dropped": s["duplicates"],
                          "out-of-order ts": s["reordered"]}
                      for a, s in status.items() if s is not None}).T
if len(build):
    print(f"event-bar cache: {len(build)} of {len(ALL_ASSETS)} assets; total {build['aggTrades (M)'].sum() / 1e3:,.1f} billion "
          f"aggTrades from {build['archive GB'].sum():,.0f} GB of archive; missing assets: {missing_assets or 'none'}")
assert not missing_assets, "build the event bars first (research/bar_types/scripts/build_event_bars.py)"
assert all(pd.Timestamp(s["next_day"]) >= DATA_END.normalize() for s in status.values()), "event bars do not reach DATA_END"
if len(build):
    display(build.sort_values("aggTrades (M)", ascending=False).head(12).round(2))
print("files missing from the archive after listing (any asset):", {a: s["missing"] for a, s in status.items() if s["missing"]} or "none")
print("invalid aggTrades dropped (non-positive price or quantity):", {a: s["dropped"] for a, s in status.items() if s["dropped"]} or "none")
print("repeated aggTrades dropped (same id twice):", {a: s["duplicates"] for a, s in status.items() if s["duplicates"]} or "none")

# completeness: the builder's daily totals vs the 1-day klines (base and quote volume), every asset and day
import pickle
DAILY = qd.load_bars(ALL_ASSETS, qd.BarConfig("time", 1), end=DATA_END, warmup_days=0, update=False)
rows = []
for a in (ALL_ASSETS if EVENT_TYPES else []):
    st = pickle.loads((qd.event_bars.default_event_bars_dir() / f"{a}USDT" / "state.pkl").read_bytes())["builder"]
    k = DAILY[a]
    day = (k["bar_start_us"] // (86_400 * 10 ** 6)).to_numpy()
    rel_b = np.abs(st.day_base[day] - k["volume"].to_numpy()) / k["volume"].clip(lower=1e-12).to_numpy()
    rel_q = np.abs(st.day_quote[day] - k["quote_volume"].to_numpy()) / k["quote_volume"].clip(lower=1e-12).to_numpy()
    rows.append({"asset": a, "days": len(day), "max rel. diff base": rel_b.max(), "max rel. diff quote": rel_q.max(),
                 "days off by > 1e-5": int(((rel_b > 1e-5) | (rel_q > 1e-5)).sum())})
chk = pd.DataFrame(rows, columns=["asset", "days", "max rel. diff base", "max rel. diff quote", "days off by > 1e-5"]).set_index("asset")
print(f"aggTrades daily totals vs 1-day klines: {int(chk['days'].sum()):,} asset-days, worst relative difference "
      f"{chk[['max rel. diff base', 'max rel. diff quote']].max().max():.1e}, days off by more than 1e-5: {int(chk['days off by > 1e-5'].sum())}")
if chk["days off by > 1e-5"].sum():
    display(chk.loc[chk["days off by > 1e-5"] > 0])
""")

code(r"""
# the threshold rule at work: BTC dollar bars at 6/day — threshold, daily dollar volume and bars per day
if "dollar" in BAR_TYPES:
    b = qd.load_bars(["BTC"], qd.BarConfig("dollar", 6), end=DATA_END, update=False)["BTC"]
    dd = DAILY["BTC"].set_index(pd.DatetimeIndex(DAILY["BTC"]["bar_start"]))
    thr = b["threshold"].resample("D").first()
    npd = b["close"].resample("D").size()
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.4))
    ax[0].plot(dd.index, dd["quote_volume"] / 1e9, color="#b7d3f6", lw=0.8, label="daily dollar volume")
    ax[0].plot(thr.index, thr * 6 / 1e9, color=COL["dollar"], lw=1.6, label="threshold x 6 (trailing 30-day mean)")
    ax[0].set_yscale("log"); ax[0].set_ylabel("USDT bn"); ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title("BTC: the dollar-bar threshold follows the trailing 30-day volume")
    ax[1].plot(npd.index, npd.rolling(30).mean(), color=COL["dollar"], label="dollar bars per day (30-day mean)")
    ax[1].axhline(6, color=INK2, ls="--", lw=1, label="target 6")
    ax[1].set_ylabel("bars per day"); ax[1].legend(fontsize=8, loc="upper left")
    ax[1].set_title("realised frequency varies with activity around the target")
    fig.tight_layout(); plt.show()
""")

# =====================================================================================================================
md(r"""
### The per-configuration pipeline

For every configuration the cell below (cached on disk, keyed by a fingerprint of the bars) does the same steps with library
calls: load the bars; record sampling diagnostics (achieved bars per day, bar durations, segments, empty klines, bars closed by
an outage gap, two strategy-specific diagnostics explained in §4); compute the elapsed-time features
(`qf.elapsed_bar_features`: per-asset families V, T, F, CAL in calendar time, and M, R, X from the other assets' latest bars at
or before each decision time); run the four strategies on the 40 assets and keep the closed trades with the features of their
signal bar (`ml.build_trade_panel`, 5 bps per fill, no slippage).
""")

code(r"""
DAY_US = 86_400 * 10 ** 6
CONFIG_CACHE = DiskCache(CACHE_ROOT / "configs", key={"data_end": str(DATA_END), "fee": FEE_BPS, "strategies": SS,
                                                       "features": "elapsed-v1", "quant_forge": quant_forge.__version__})


def tis_daily_candle_coverage(bars, tolerance_h=1.5):
    # share of UTC days (first to last bar) whose daily candle passes the coverage rule of quant_forge.strategies.htf:
    # bars grouped by their END; the first one started <= day start + tolerance, the last one ended >= day end - tolerance
    tol = int(tolerance_h * 3600 * 10 ** 6)
    out = {}
    for a, b in bars.items():
        s, e = b["bar_start_us"].to_numpy(np.int64), b["bar_end_us"].to_numpy(np.int64)
        day = (e - 1) // DAY_US
        first, last = np.r_[True, day[1:] != day[:-1]], np.r_[day[1:] != day[:-1], True]
        d0 = day[first] * DAY_US
        out[a] = float(((s[first] <= d0 + tol) & (e[last] >= d0 + DAY_US - tol)).sum() / (day[-1] - day[0] + 1))
    return pd.Series(out)


def build_config(cfg):
    t = time.time()
    bars = qd.load_bars(ALL_ASSETS, cfg, end=DATA_END, update=False)
    name = f"{cfg.key}_{frame_fingerprint(bars, columns=['open', 'high', 'low', 'close', 'volume'])}"
    if name in CONFIG_CACHE:
        return CONFIG_CACHE.load(name)
    durs = np.concatenate([qd.bar_durations(b).dropna().to_numpy() for b in bars.values()])
    info = {"summary": qd.sampling_summary(bars, cfg.bars_per_day),
            "yearly": qd.sampling_by_period(bars, "YS"), "monthly": qd.sampling_by_period(bars, "MS"),
            "dur_hist": np.histogram(np.log2(durs[durs > 0] / cfg.nominal_hours), bins=np.arange(-10, 8.01, 0.125))[0],
            "tis_htf_complete": tis_daily_candle_coverage(bars),
            "vol_filter_pass": pd.Series({a: float((b["volume"] > b["volume"].rolling(20).mean()).mean()) for a, b in bars.items()}),
            "empty_bars": pd.Series({a: int((b["complete"] == 0).sum()) for a, b in bars.items()}),
            "gap_closed": pd.Series({a: int((b["closed_by"] == "gap").sum()) if "closed_by" in b else 0 for a, b in bars.items()}),
            "n_bars": pd.Series({a: len(b) for a, b in bars.items()})}
    feats = qf.elapsed_bar_features(bars, n_jobs=N_JOBS)
    feats = {a: f.astype({c: np.float32 for c in f.columns if c != "raw_close"}) for a, f in feats.items()}
    cols = qf.model_columns(feats["BTC"])
    trades = ml.build_trade_panel(bars, feats, STRATEGIES, feature_columns=cols, tick_sizes=qd.TICK_SIZES,
                                  fee_bps=FEE_BPS, slippage_ticks=0, n_jobs=N_JOBS)
    del bars, feats
    gc.collect()
    trades["group"] = ml.assign_groups(trades, {"SEEN": SEEN, "NEW": NEW})
    trades["month"] = trades["exit_time"].dt.strftime("%Y-%m")
    out = {"trades": trades, "info": info, "seconds": time.time() - t}
    CONFIG_CACHE.save(name, out)
    return out


TRADES, INFO = {}, {}
for cfg in CONFIGS:
    t = time.time()
    res_cfg = build_config(cfg)
    TRADES[cfg.key], INFO[cfg.key] = res_cfg["trades"], res_cfg["info"]
    print(f"{cfg.label:12s} bars {int(INFO[cfg.key]['n_bars'].sum()):>10,}  closed trades {len(TRADES[cfg.key]):>7,}  "
          f"({time.time() - t:5.0f} s now, {res_cfg['seconds']:5.0f} s when computed)")
    del res_cfg
    gc.collect()
""")

# =====================================================================================================================
md(r"""
## 3. Sampling frequency: how closely does each configuration match its target?

Achieved bars per day are counted per asset over the days on which the asset had bars; a bar's duration is the time since the
previous bar's end in the same segment (the sampling interval). For time bars both are fixed by construction, except for
missing klines and klines without trades (which become segment breaks). For volume and dollar bars both follow activity: the
target is met only on average, and only to the extent that the trailing 30-day volume predicts the next day's volume.
""")

code(r"""
rows = []
for k in KEYS:
    cfg, inf = CFG[k], INFO[k]
    s, y = inf["summary"], inf["yearly"]
    yr = y.loc[y["active days"] > 60, "bars/day"] / cfg.bars_per_day
    rows.append({"config": cfg.label, "bar type": cfg.bar_type, "target /day": cfg.bars_per_day,
                 "achieved /day (median asset)": s["bars/day"].median(),
                 "ratio median": s["ratio to target"].median(), "ratio min asset": s["ratio to target"].min(),
                 "ratio max asset": s["ratio to target"].max(),
                 "asset-year ratio p10": yr.quantile(0.1), "asset-year ratio p90": yr.quantile(0.9),
                 "nominal h": cfg.nominal_hours, "dur p05 h": s["dur p05 (h)"].median(), "dur p50 h": s["dur p50 (h)"].median(),
                 "dur p95 h": s["dur p95 (h)"].median(), "share > 4x nominal": s["share > 4x nominal"].median(),
                 "segments / asset": s["segments"].median(), "empty klines (all)": int(inf["empty_bars"].sum()),
                 "gap-closed bars (all)": int(inf["gap_closed"].sum())})
SAMPLING = pd.DataFrame(rows).set_index("config")
SAMPLING["freq flag"] = ~SAMPLING["ratio median"].between(*FREQ_RATIO_BAND)
print("Achieved frequency and bar durations (medians across the 40 assets unless stated):")
display(SAMPLING.round(3))
""")

code(r"""
fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), sharex=True)
for j, n in enumerate(FREQS):
    for bt in BAR_TYPES:
        cfg = qd.BarConfig(bt, n)
        m = INFO[cfg.key]["monthly"]
        m = m.loc[m["active days"] >= 20].copy()
        m["ratio"] = m["bars/day"] / n
        m["dur_rel"] = m["dur p50 (h)"] / cfg.nominal_hours
        for i, col in enumerate(["ratio", "dur_rel"]):
            g = m.groupby("period")[col]
            med, lo, hi = g.median(), g.quantile(0.1), g.quantile(0.9)
            ax = axes[i, j]
            ax.plot(med.index, med, color=COL[bt], label=bt, lw=1.5)
            if bt != "time":
                ax.fill_between(med.index, lo, hi, color=COL[bt], alpha=0.15, lw=0)
    for i in range(2):
        axes[i, j].axhline(1, color=INK2, ls="--", lw=0.8)
        axes[i, j].set_yscale("log", base=2)
    axes[0, j].set_title(f"target {FREQ_LABEL[n]}")
import matplotlib.dates as mdates
for ax in axes.ravel():
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[0, 0].set_ylabel("achieved / target bars per day\n(median asset, band p10-p90)")
axes[1, 0].set_ylabel("median bar duration / nominal\n(median asset, band p10-p90)")
axes[0, 0].legend(fontsize=8, loc="upper left")
fig.suptitle("Realised sampling frequency over time: time bars are fixed, event bars follow activity", y=1.0)
fig.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
centers = np.arange(-10, 8, 0.125) + 0.0625
for j, n in enumerate(FREQS):
    ax = axes[j]
    for bt in [b_ for b_ in ["volume", "dollar"] if b_ in BAR_TYPES]:
        h = INFO[qd.BarConfig(bt, n).key]["dur_hist"]
        ax.plot(2.0 ** centers, h / h.sum() / 0.125, color=COL[bt], label=bt)
    ax.axvline(1, color=COL["time"], lw=1.6, label="time (all bars)")
    ax.set_xscale("log", base=2); ax.set_xlim(2 ** -8, 2 ** 5)
    ax.set_title(f"target {FREQ_LABEL[n]}"); ax.set_xlabel("bar duration / nominal")
axes[0].set_ylabel("density (per log2 unit)"); axes[0].legend(fontsize=8)
fig.suptitle("Distribution of bar durations, pooled over assets and years", y=1.03)
fig.tight_layout(); plt.show()

per_asset = pd.DataFrame({CFG[k].label: INFO[k]["summary"]["ratio to target"] for k in KEYS if CFG[k].bar_type != "time"})
if not per_asset.empty:
    print("Achieved / target bars per day by asset (event bars), whole history of each asset:")
    display(per_asset.describe(percentiles=[0.1, 0.5, 0.9]).T.round(3))
    print("Assets furthest from the target (dollar 6/d):",
          per_asset["dollar 6/d"].sort_values().iloc[[0, 1, 2, -3, -2, -1]].round(3).to_dict())
""")

# =====================================================================================================================
md(r"""
## 4. The primary strategies on every configuration

Same rules and default parameters everywhere; only the bars change. Indicator lengths are in bars, so a 200-bar EMA spans
100 hours on 30-minute bars and 200 days on daily bars, and Donchian's 20-bar maximum hold is 10 hours or 20 days: that is what
"changing the sampling frequency of a strategy" means. None of the four strategies reads another asset's bars in its rules
(TIS is named after BTC but trades each asset on its own bars and its own daily candles); BTC and cross-asset information enter
only through the meta-model's M, R and X features, which are aligned as-of across asynchronous bars (§5). Two rules interact
with the bar type beyond their length:

* **TIS** confirms with green *daily* candles rebuilt from its input bars (`htf = "1D"`), and a daily candle only counts when
  the bars ending in that day cover it within 1.5 hours of both boundaries. Event bars do not respect day boundaries; the
  share of days that pass this rule is shown below (`TIS daily candle complete`), and on daily event bars it collapses.
* **SuperTrend** requires the flip bar's volume to exceed its 20-bar mean. On volume bars every bar carries about the threshold
  volume, so the filter passes about half of the bars, decided by the overshoot of the closing trade rather than by activity;
  on dollar bars it reacts to price moves (base volume per dollar falls when the price rises). Shown as `volume filter pass`.

Counts below are closed trades decided in the evaluation period (2019-03-01 → 2026-11-19).
""")

code(r"""
def eval_trades(k):
    tr = TRADES[k]
    return tr.loc[(tr["decision_time"] >= EVAL_START) & (tr["decision_time"] < DATA_END)]


rows = []
for k in KEYS:
    ev = eval_trades(k)
    for (s, g), x in ev.groupby(["strategy", "group"]):
        asset_years = sum(((DATA_END - max(EVAL_START, INFO[k]["summary"].loc[a, "first bar"])).days / 365.25)
                          for a in (SEEN if g == "SEEN" else NEW))
        rows.append({"config": CFG[k].label, "bar type": CFG[k].bar_type, "freq": CFG[k].bars_per_day, "strategy": s,
                     "group": g, "trades": len(x), "trades / asset-year": len(x) / asset_years,
                     "win rate": x["y"].mean(), "mean net bps": 1e4 * x["trade_return"].mean(),
                     "mean gross bps": 1e4 * x["trade_return_gross"].mean(), "median hold h": x["holding_hours"].median(),
                     "p90 hold h": x["holding_hours"].quantile(0.9), "short share": (x["side"] < 0).mean()})
PRIMARY = pd.DataFrame(rows)
cnt = PRIMARY.pivot_table(index=["group", "strategy"], columns="config", values="trades", sort=False)[[CFG[k].label for k in KEYS]]
print("Closed trades in the evaluation period by group, strategy and configuration:")
display(cnt.astype(int))
print("Per-trade statistics on NEW assets (net of 5 bps per fill):")
display(PRIMARY.loc[PRIMARY["group"].eq("NEW")].set_index(["strategy", "bar type", "freq"]).sort_index()
        [["trades", "trades / asset-year", "win rate", "mean gross bps", "mean net bps", "median hold h", "p90 hold h"]].round(3))

diag = pd.DataFrame({CFG[k].label: {"TIS daily candle complete (median asset)": INFO[k]["tis_htf_complete"].median(),
                                    "TIS daily candle complete (worst asset)": INFO[k]["tis_htf_complete"].min(),
                                    "volume filter pass (median asset)": INFO[k]["vol_filter_pass"].median()} for k in KEYS})
print("Strategy-specific consequences of the bar type:")
display(diag.round(3))
""")

md(r"""
### Reproduction check: the 4h time-bar baseline gives the reference signals

The reference notebook built its trades with `qf.bar_features(bars, bars_per_day=6)` (bar-count features) and
`ml.build_trade_panel`. Trades depend only on the bars and the strategy rules, so the 4h time-bar configuration of this
notebook — loaded through the new `qd.load_bars` interface and carrying the new elapsed-time features — must produce exactly the
same signals, entries, exits and returns. The cell rebuilds the reference trade table with the reference calls on the common
history (the whole reference period, first bar 2017-09-16 → 2026-11-19), compares it trade by trade, and compares the counts
with the output stored in the reference notebook.
""")

code(r"""
bars4 = qd.load_binance_bars(ALL_ASSETS, end=DATA_END, interval="4h", warmup_days=30)
ref_feats = qf.add_raw_levels(qf.bar_features(bars4, bars_per_day=6, btc="BTC"), bars4)
ref_trades = ml.build_trade_panel(bars4, ref_feats, STRATEGIES, feature_columns=["raw_close", "v_rvol_30d"],
                                  tick_sizes=qd.TICK_SIZES, fee_bps=FEE_BPS, slippage_ticks=0, n_jobs=N_JOBS)
del ref_feats, bars4
gc.collect()
KEYCOLS = ["strategy", "asset", "side", "decision_time", "entry_time", "exit_time", "entry_price", "trade_return", "exit_reason"]
a = ref_trades[KEYCOLS].sort_values(KEYCOLS[:4]).reset_index(drop=True)
bb = TRADES[BASELINE].sort_values(KEYCOLS[:4]).reset_index(drop=True)
b = bb[KEYCOLS]
identical = a.equals(b)
print(f"reference pipeline: {len(a):,} closed trades; this notebook's time 4h: {len(b):,}; identical trade by trade: {identical}")

ref_nb = json.loads((ROOT / "research/meta-labeling/meta_labeling_economic_value_quant_forge.ipynb").read_text())
stored = None
for cell in ref_nb["cells"]:
    for o in cell.get("outputs", []):
        txt = "".join(o.get("text", ""))
        m_ = re.search(r"([\d,]+) closed trades", txt)
        if m_:
            stored = int(m_.group(1).replace(",", ""))
print(f"closed trades printed by the reference notebook: {stored:,} -> match: {stored == len(b)}")
period = ml.assign_periods(bb["decision_time"], [("A", pd.Timestamp("2021-09-01", tz="UTC")),
                                                  ("B", pd.Timestamp("2026-09-01", tz="UTC"))], last="C")
print("Same layout as the reference notebook's count table (group x strategy x period):")
display(pd.crosstab([bb["group"], bb["strategy"]], period, margins=True, margins_name="total"))
assert identical and stored == len(b)
""")

# =====================================================================================================================
md(r"""
## 5. Meta-labeling with one specification for every configuration

* **Label**: `y = 1[net trade return > 0]` of the primary trade (AFML §3.6); the primary strategy fixes side and exit.
* **Candidates** (fixed list, the same for every configuration): the elapsed-time families V, T, F, CAL, M, R, X of the signal
  bar, the strategy's own features (LIB; price levels as ratios to the signal close), the track record S (trades closed
  strictly before the decision; open positions and signals in the last 24 hours), with direction-carrying features multiplied
  by the trade side (`ml.prepare_meta_dataset`). Features missing on more than 30 % of a window's training trades are dropped.
* **Model**: logistic regression on median-imputed, normal-quantile-transformed features; `C ∈ {0.003, 0.01, 0.03, 0.1, 0.3}`
  chosen by the inner purged K-fold on common calendar folds (4 folds, purge by decision-to-exit overlap, 1 % embargo, log-loss
  scoring); Platt calibration on the inner out-of-fold probabilities; sizing constants (break-even win rate `p_be` from the
  training payouts, median training volatility, rule normalisations) from the same training trades
  (`ml.fit_window` with `MetaLabelingConfig(model_names=("LR",), pca_models=(), feature_selection=False)`).
* **Walk-forward**: 12-month windows from 2019-03-01; training = `SEEN` trades that **exited** before the window start (at least
  200); predictions for the window's trades on both groups.
* **Windows without a model** (fewer than 200 training trades): their trades stay in the evaluation and the rules fall back to
  inverse volatility (no filter, no probability sizing) — a practitioner cannot meta-label before having trades. The share of
  evaluated trades covered by a model is reported and flagged.
""")

code(r"""
from quant_forge.validation.calendar_kfold import to_int64_ns

DS, RES, PRED = {}, {}, {}
for k in KEYS:
    t = time.time()
    ds = ml.prepare_meta_dataset(TRADES[k])
    cands = {s: ds.candidates_for(s) for s in SS}
    cache = DiskCache(CACHE_ROOT / "walk_forward", key={
        "bar": k, "config": MLCFG.cache_key(), "windows": [str(w) for w in WINDOWS], "candidates": cands,
        "trades": frame_fingerprint(ds.trades[["strategy", "asset", "decision_time", "exit_time", "trade_return"]])})
    res = ml.run_walk_forward(ds.trades, SS, WINDOWS, cands, MLCFG, cache=cache, n_jobs=WF_JOBS, inner_n_jobs=1, verbose=False)
    PRED[k] = ml.collect_predictions(res, ds.trades, ["strategy", "asset", "group", "decision_time", "entry_time", "exit_time",
                                                      "trade_return", "y", "v_rvol_30d", "month"]) if res else pd.DataFrame()
    DS[k], RES[k] = ds, res
    TRADES[k] = ds.trades
    print(f"{CFG[k].label:12s} {len(res):2d} of {len(SS) * len(WINDOWS)} (strategy, window) fits, "
          f"{len(PRED[k]):>7,} outer predictions, candidates per strategy {[len(c) for c in cands.values()]} ({time.time() - t:4.0f} s)")
FAMILY = DS[BASELINE].family
print("candidate features by family (time 4h, RSI):",
      pd.Series([FAMILY[c] for c in DS[BASELINE].candidates_for("RSI")]).value_counts().reindex(list(qf.FAMILY_ORDER)).fillna(0).astype(int).to_dict())
""")

code(r"""
rows = []
for k in KEYS:
    tr = TRADES[k]
    ev = tr.loc[(tr["decision_time"] >= EVAL_START) & (tr["decision_time"] < DATA_END)]
    covered = set(PRED[k]["row"]) if len(PRED[k]) else set()
    for s in SS:
        fits = {w0: r.diagnostics for (ss, w0), r in RES[k].items() if ss == s}
        e = ev.loc[ev["strategy"].eq(s)]
        rows.append({"config": CFG[k].label, "strategy": s, "windows fitted": len(fits),
                     "first fitted window": min(fits).date() if fits else None,
                     "median training trades": np.median([d["n_train"] for d in fits.values()]) if fits else 0,
                     "usable features (median)": np.median([d["n_usable"] for d in fits.values()]) if fits else 0,
                     "C (most frequent)": pd.Series([d["tuning"]["LR-FULL"]["params"]["C"] for d in fits.values()]).mode().iloc[0] if fits else None,
                     "inner ll − prior (mean)": np.mean([d["tuning"]["LR-FULL"]["inner_ll"] - d["prior_inner_ll"] for d in fits.values()]) if fits else np.nan,
                     "model coverage (all)": e.index.isin(covered).mean() if len(e) else np.nan,
                     "model coverage (NEW)": e.loc[e["group"].eq("NEW")].index.isin(covered).mean() if len(e) else np.nan})
COVERAGE = pd.DataFrame(rows).set_index(["strategy", "config"]).sort_index(level=0, sort_remaining=False)
print("Walk-forward fits and model coverage of the evaluated trades:")
display(COVERAGE.round(4))
c_all = [d.diagnostics["tuning"]["LR-FULL"]["params"]["C"] for k in KEYS for d in RES[k].values()]
ll_all = [d.diagnostics["tuning"]["LR-FULL"]["inner_ll"] - d.diagnostics["prior_inner_ll"] for k in KEYS for d in RES[k].values()]
from quant_forge import models as qm
grid_min = min(p_["C"] for p_ in qm.lr_spec().grid)
print(f"{len(c_all)} fitted windows: the smallest C of the grid ({grid_min}) was chosen in {np.mean(np.isclose(c_all, grid_min)):.0%}; "
      f"the inner log loss beat the training prior in {np.mean(np.array(ll_all) < 0):.0%} of them")
""")

code(r"""
rows = []
for k in KEYS:
    p = PRED[k]
    if not len(p):
        continue
    for (s, g), x in p.groupby(["strategy", "group"]):
        if len(x) < 50 or x["y"].nunique() < 2:
            continue
        dl, lo, hi = qe.delta_log_loss_ci(x, "p", n_boot=N_BOOT, seed=SEED)
        rows.append({"group": g, "strategy": s, "config": CFG[k].label, "n": len(x), "win rate": x["y"].mean(),
                     "prior (mean)": x["prior"].mean(), "AUC": qe.auc_safe(x["y"], x["p"]), "Δ log loss": dl, "CI low": lo,
                     "CI high": hi, "IC(label)": qe.information_coefficient(x, target="y"),
                     "IC(return)": qe.information_coefficient(x)})
PROBQ = pd.DataFrame(rows).set_index(["group", "strategy", "config"])
print("Outer (walk-forward) probability quality of the meta-model; Δ log loss < 0 = better than the training win rate:")
for g in GROUPS:
    tab = PROBQ.loc[g]
    print(f"\n{g}: AUC by strategy and configuration")
    display(tab["AUC"].unstack("config").reindex(columns=[CFG[k].label for k in KEYS]).round(3))
    print(f"{g}: Δ log loss vs prior (negative = better); cells whose 95 % CI lies below 0: {int((tab['CI high'] < 0).sum())} of {len(tab)}")
    display(tab["Δ log loss"].unstack("config").reindex(columns=[CFG[k].label for k in KEYS]).round(4))
""")

# =====================================================================================================================
md(r"""
## 6. Four portfolio rules under one capital budget

| Rule | Size of a signalled trade (notionals of 10 000 USDT) | Question |
|---|---|---|
| **U** | 1 | the primary strategy as it is |
| **D** | `clip(median training vol / vol, 0.25, 4)`, vol = the asset's 30-day realised volatility at the signal | inverse volatility, no model |
| **D×F** | `D × 1[p ≥ p_be]` | what the meta-label filter adds to inverse volatility |
| **D×P** | `D × (2Φ(z) − 1)`, `z = (p − p_be) / sqrt(p (1 − p))`, 0 when `p ≤ p_be` | the same filter plus probability-based sizing (AFML §10.3 centred at the break-even win rate) |

`p` is the calibrated meta-model probability and `p_be = L / (W + L)` the break-even win rate of the strategy's training payouts,
so `D×P` trades exactly the trades that `D×F` trades and only changes their size: **D×F − D is the value of filtering and
D×P − D×F the value of sizing after filtering**.

A filter at `p_be` can add value in two ways: by ranking trades within a window, or simply because the model's average
probability (≈ the training win rate, since the inner cross-validation keeps the logistic regression strongly shrunk) falls
below `p_be` when the strategy lost money in training, which switches the whole strategy off. The diagnostic rule **D×S** does
only the second: `D × 1[training win rate ≥ p_be]` (equivalently, trade a strategy in a window only if its training trades had a
positive mean net return). **D×F − D×S** is then the part of the filter's value that comes from selecting trades. Every rule is divided by its mean on the window's training trades (equal expected
exposure per signal) and capped at 10 notionals.

**Common budget.** Each (configuration, strategy, group) portfolio has the same capital K = 20 notionals (200 000 USDT): one per
asset, so U never needs more (a strategy holds at most one position per asset). Trades are replayed in entry order and cut to
the capital still free (`qz.capital_constrained_sizes`). `ALL` is the four strategy portfolios side by side (4 K).

**Calendar-based measurement.** Every trade is marked to market at each UTC day end with the asset's daily close
(`qe.expand_trade_days`, entry fee on the entry day, exit fee on the exit day; the daily PnL of a trade sums exactly to its net
return), so a strategy that holds for hours and one that holds for months are measured on the same 2 820 days. Sharpe is the
annualised Sharpe of daily PnL (× √365); drawdown is on cumulative daily PnL in units of K; turnover is entry plus exit notional
per year in units of K. Intervals come from one paired monthly block bootstrap (2 000 draws of the 93 calendar months, the same
draws for every series), so two rules or two configurations are always compared on the same resampled months.
""")

code(r"""
CLOSES = qe.daily_close_panel(DAILY, CAL)
WIN_STARTS = np.array([w0.value for w0, _ in WINDOWS], dtype=np.int64)


def portfolio_sizes(k):
    tr = TRADES[k]
    ev = tr.loc[(tr["decision_time"] >= EVAL_START) & (tr["decision_time"] < DATA_END)].copy()
    wi = np.searchsorted(WIN_STARTS, to_int64_ns(ev["decision_time"]), side="right") - 1
    ev["window"] = [WINDOWS[i][0] for i in wi]
    pred = PRED[k].set_index("row") if len(PRED[k]) else pd.DataFrame()
    has = ev.index.isin(pred.index) if len(pred) else np.zeros(len(ev), bool)
    W = pd.DataFrame(np.nan, index=ev.index, columns=EVAL_RULES)
    if has.any():
        sub = pred.loc[ev.index[has]]
        consts = {key: r.diagnostics["constants"] for key, r in RES[k].items()}
        W.loc[sub.index, list(RULES)] = qz.normalized_sizes(sub, consts, cap=SIZE_CAP, rules=tuple(RULES.values()))[list(RULES.values())].to_numpy()
        on = np.array([consts[(s_, w_)].prior >= consts[(s_, w_)].p_be for s_, w_ in zip(sub["strategy"], sub["window"])])
        W.loc[sub.index, "DxS"] = W.loc[sub.index, "D"].to_numpy() * on
        ev.loc[sub.index, "p"] = sub["p"]
        ev.loc[sub.index, "p_be"] = [consts[(s, w)].p_be for s, w in zip(sub["strategy"], sub["window"])]
    for (s, w0), g in ev.loc[~has].groupby(["strategy", "window"]):          # no model yet: inverse volatility only
        vol_tr = tr.loc[tr["strategy"].eq(s) & tr["asset"].isin(SEEN) & (tr["exit_time"] < w0), "v_rvol_30d"].dropna()
        if len(vol_tr) >= 30:
            vm = float(vol_tr.median())
            d = (np.clip(vm / g["v_rvol_30d"], 0.25, 4).fillna(1.0) / float(np.clip(vm / vol_tr, 0.25, 4).mean()))
            d = d.clip(upper=SIZE_CAP).to_numpy()
        else:
            d = np.ones(len(g))
        W.loc[g.index, "U"] = 1.0
        for r in ("D", "DxF", "DxP", "DxS"):
            W.loc[g.index, r] = d
    ev["model"] = has
    Wc = W.copy()
    for _, idx in ev.groupby(["strategy", "group"]).groups.items():
        for r in EVAL_RULES:
            Wc.loc[idx, r] = qz.capital_constrained_sizes(ev.loc[idx, "entry_time"], ev.loc[idx, "exit_time"],
                                                          W.loc[idx, r].to_numpy(), BUDGET, NOTIONAL)
    return ev, W, Wc


def sharpe(x):
    # annualised Sharpe of daily PnL; a portfolio that never trades (all-zero PnL) counts as 0
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    if not np.any(x):
        return 0.0
    return float(x.mean() / sd * np.sqrt(365)) if sd > 0 else np.nan


PNL, SIZES, stat_rows = {}, {}, []
G2 = {"NEW": 0, "SEEN": 1}
for k in KEYS:
    ev, W, Wc = portfolio_sizes(k)
    SIZES[k] = (ev, W, Wc)
    code_ = ev["strategy"].map({s: i for i, s in enumerate(SS)}).to_numpy() * 2 + ev["group"].map(G2).to_numpy()
    for fee in (FEE_BPS, FEE_STRESS_BPS):
        td = qe.expand_trade_days(ev, CLOSES, fee_bps=fee, calendar=CAL)
        if fee == FEE_BPS:
            assert np.allclose(td.unit_return(), ev["trade_return"].to_numpy(), atol=1e-9), "mark-to-market != net returns"
        D = qe.daily_pnl(td, Wc[EVAL_RULES].to_numpy().T, notional=NOTIONAL, group=code_, n_groups=2 * len(SS))
        for si, s in enumerate(SS):
            for g, gi in G2.items():
                for ri, r in enumerate(EVAL_RULES):
                    PNL[(fee, g, k, s, r)] = D[si * 2 + gi, ri]
        for g in G2:
            for r in EVAL_RULES:
                PNL[(fee, g, k, "ALL", r)] = np.sum([PNL[(fee, g, k, s, r)] for s in SS], axis=0)
    for g in GROUPS:
        for p in PORTFOLIOS:
            m = ev["group"].eq(g).to_numpy() & (ev["strategy"].eq(p).to_numpy() if p != "ALL" else True)
            K = BUDGET * (len(SS) if p == "ALL" else 1)
            tab = qe.portfolio_table({r: PNL[(FEE_BPS, g, k, p, r)] for r in EVAL_RULES}, budget=K,
                                     trades={r: ev.loc[m] for r in EVAL_RULES}, sizes={r: Wc.loc[m, r].to_numpy() for r in EVAL_RULES},
                                     notional=NOTIONAL)
            for r in EVAL_RULES:
                w_, wc_ = W.loc[m, r].to_numpy(), Wc.loc[m, r].to_numpy()
                stat_rows.append({"group": g, "portfolio": p, "config": k, "rule": r, **tab.loc[r].to_dict(),
                                  "Sharpe 10 bps": sharpe(PNL[(FEE_STRESS_BPS, g, k, p, r)]),
                                  "net PnL 10 bps": PNL[(FEE_STRESS_BPS, g, k, p, r)].sum(),
                                  "signals": int(m.sum()), "model coverage": float(ev.loc[m, "model"].mean()) if m.any() else np.nan,
                                  "share cut by budget": float((wc_ < w_ - 1e-12).mean()) if m.any() else np.nan,
                                  "months with trades": int(ev["month"].iloc[np.flatnonzero(m)[wc_ > 0]].nunique()) if m.any() else 0})
STATS = pd.DataFrame(stat_rows).set_index(["group", "portfolio", "config", "rule"])

BOOT = {}
for fee in (FEE_BPS, FEE_STRESS_BPS):
    for g in GROUPS:
        names = [(k, p, r) for k in KEYS for p in PORTFOLIOS for r in EVAL_RULES]
        B = qe.bootstrap_sharpe(np.vstack([PNL[(fee, g, *n)] for n in names]), MONTHS, n_boot=N_BOOT, seed=SEED)
        for n, b in zip(names, B):
            # flat series (a strategy switched off throughout) -> Sharpe 0 in every draw; otherwise a draw whose
            # resampled months are all flat is also a zero-risk, zero-return portfolio
            BOOT[(fee, g, *n)] = np.where(np.isnan(b), 0.0, b) if not np.any(PNL[(fee, g, *n)]) else b


def delta(g, a, b, fee=FEE_BPS):
    # paired Sharpe difference a - b, a and b = (config, portfolio, rule): (estimate, CI low, CI high, P(Δ > 0))
    d = BOOT[(fee, g, *a)] - BOOT[(fee, g, *b)]
    d = d[np.isfinite(d)]
    est = sharpe(PNL[(fee, g, *a)]) - sharpe(PNL[(fee, g, *b)])
    if not len(d):
        return est, np.nan, np.nan, np.nan
    return est, float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), float((d > 0).mean())


def fmt(t):
    est, lo, hi, _ = t
    star = "*" if (lo > 0 or hi < 0) else ""
    return f"{est:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}" if np.isfinite(est) else "n/a"


print(f"{len(PNL) // 2} daily PnL series per fee level on {len(CAL)} days; "
      f"share of trades cut by the budget (NEW, ALL portfolio, median over configurations): "
      + ", ".join(f"{r} {STATS.xs(('NEW', 'ALL', r), level=['group', 'portfolio', 'rule'])['share cut by budget'].median():.3f}" for r in RULES))
""")

# =====================================================================================================================
md(r"""
## 7. Results

### 7.1 The primary strategy itself (rule U): which bars and frequencies improve it?

Annualised Sharpe of daily mark-to-market PnL, net of 5 bps per fill, fixed size, budget K = 20 notionals per strategy. Cells
with `*` in the difference tables have a 95 % paired-bootstrap interval that excludes zero.
""")

code(r"""
LABELS = [CFG[k].label for k in KEYS]


def heat(ax, M, title, vmax=None, stars=None, fmt_="{:+.2f}"):
    vals = M.to_numpy(float)
    v = vmax or np.nanmax(np.abs(vals)) or 1.0
    ax.imshow(vals, cmap=DIVERGING, vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=45, ha="right")
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(vals[i, j]):
                star = "*" if stars is not None and bool(stars.iat[i, j]) else ""
                ax.text(j, i, fmt_.format(vals[i, j]) + star, ha="center", va="center", fontsize=7.5,
                        color="white" if abs(vals[i, j]) >= 0.65 * v else INK)
    for x in np.arange(len(FREQS), M.shape[1], len(FREQS)) - 0.5:
        ax.axvline(x, color="white", lw=3)
    ax.set_title(title)


def stat_matrix(g, rule, col):
    x = STATS.xs((g, rule), level=["group", "rule"])[col].unstack("config")
    return x.reindex(index=PORTFOLIOS, columns=KEYS).set_axis(LABELS, axis=1)


fig, axes = plt.subplots(1, 2, figsize=(15, 3.6))
for ax, g in zip(axes, GROUPS):
    heat(ax, stat_matrix(g, "U", "Sharpe"), f"{g}: Sharpe of the primary strategy (U), daily MTM, net of 5 bps", vmax=1.5)
fig.tight_layout(); plt.show()

cols = ["Sharpe", "net PnL", "return on K (ann.)", "max DD / K", "turnover (x K / yr)", "trades", "exposure / K", "Sharpe 10 bps"]
for g in GROUPS:
    t = STATS.xs((g, "U"), level=["group", "rule"])[cols].copy()
    t["net PnL"] = t["net PnL"] / 1e3
    t = t.rename(columns={"net PnL": "net PnL (kUSD)"}).reindex(pd.MultiIndex.from_product([PORTFOLIOS, KEYS]))
    t.index = pd.MultiIndex.from_tuples([(p, CFG[k].label) for p, k in t.index], names=["portfolio", "config"])
    print(f"{g}: primary strategy (U) — economics by strategy and configuration")
    display(t.round(3))

for g in GROUPS:
    tab = pd.DataFrame({CFG[k].label: {p: fmt(delta(g, (k, p, "U"), (BASELINE, p, "U"))) for p in PORTFOLIOS}
                        for k in KEYS if k != BASELINE})
    print(f"{g}: Δ Sharpe of U against the time 4h baseline [95 % paired CI]")
    display(tab)
""")

md(r"""
### 7.2 What meta-labeling adds in each configuration

`D − U`: inverse volatility (no model). `D×F − D`: the meta-label filter on top of inverse volatility. `D×P − D×F`: probability
sizing of the trades the filter accepted. `D×P − D`: the whole meta-labeling contribution over inverse volatility. Diagnostic:
`D×S − D` switching a strategy off when its training trades lost money, and `D×F − D×S` what the filter adds beyond that switch
(trade selection). `filter accepts` = share of model-covered trades with `p ≥ p_be`; `switch on` = share traded by `D×S`.
""")

code(r"""
PAIRS_META = [("D", "U"), ("DxF", "D"), ("DxP", "DxF"), ("DxP", "D"), ("DxS", "D"), ("DxF", "DxS")]


def meta_table(g, fee=FEE_BPS):
    rows = []
    for p in PORTFOLIOS:
        for k in KEYS:
            ev = SIZES[k][0]
            m = ev["group"].eq(g) & (ev["strategy"].eq(p) if p != "ALL" else True) & ev["model"]
            row = {"portfolio": p, "config": CFG[k].label}
            for r in EVAL_RULES:
                row[f"Sharpe {r}"] = sharpe(PNL[(fee, g, k, p, r)])
            for a, b in PAIRS_META:
                row[f"{a} − {b}"] = fmt(delta(g, (k, p, a), (k, p, b), fee))
            row["filter accepts"] = float((ev.loc[m, "p"] >= ev.loc[m, "p_be"]).mean()) if m.any() else np.nan
            row["switch on"] = float((SIZES[k][1].loc[m, "DxS"] > 0).mean()) if m.any() else np.nan
            row["model coverage"] = STATS.loc[(g, p, k, "U"), "model coverage"]
            rows.append(row)
    return pd.DataFrame(rows).set_index(["portfolio", "config"])


META = {g: meta_table(g) for g in GROUPS}
print("NEW assets (transfer): value of inverse volatility, filtering and sizing [95 % paired CI]")
display(META["NEW"].round(3))

fig, axes = plt.subplots(2, 2, figsize=(16, 7))
for ax, (a, b) in zip(axes.ravel(), [("DxF", "D"), ("DxF", "DxS"), ("DxP", "DxF"), ("DxP", "D")]):
    est = pd.DataFrame({CFG[k].label: {p: delta("NEW", (k, p, a), (k, p, b))[0] for p in PORTFOLIOS} for k in KEYS})
    sig = pd.DataFrame({CFG[k].label: {p: (lambda t: t[1] > 0 or t[2] < 0)(delta("NEW", (k, p, a), (k, p, b))) for p in PORTFOLIOS} for k in KEYS})
    heat(ax, est, f"NEW: Δ Sharpe {a} − {b}  (* = CI excludes 0)", vmax=0.6, stars=sig)
fig.tight_layout(); plt.show()
""")

md(r"""
### 7.3 The four rules side by side (all four strategies together)

`ALL` = the four strategy portfolios run together, capital 4 K = 80 notionals. Net PnL in thousands of USDT over the 93 months.
""")

code(r"""
for g in GROUPS:
    t = STATS.xs((g, "ALL"), level=["group", "portfolio"])
    out = {}
    for col, scale in [("Sharpe", 1), ("net PnL", 1e3), ("max DD / K", 1), ("turnover (x K / yr)", 1), ("trades", 1), ("Sharpe 10 bps", 1)]:
        out[col] = (t[col] / scale).unstack("rule").reindex(index=KEYS, columns=list(RULES)).set_axis(LABELS, axis=0)
    tab = pd.concat(out, axis=1)
    print(f"{g}: ALL portfolio by configuration and rule (net PnL in kUSD)")
    display(tab.round(3))

fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharey="row")
for i, r in enumerate(["U", "DxP"]):
    for j, n in enumerate(FREQS):
        ax = axes[i, j]
        for bt in BAR_TYPES:
            k = qd.BarConfig(bt, n).key
            ax.plot(CAL, np.cumsum(PNL[(FEE_BPS, "NEW", k, "ALL", r)]) / 1e3, color=COL[bt], label=bt)
        ax.set_title(f"NEW, ALL, rule {r}: {FREQ_LABEL[n]}")
        ax.axhline(0, color="#c3c2b7", lw=0.8)
    axes[i, 0].set_ylabel("cumulative PnL (kUSD)")
axes[0, 0].legend(fontsize=8, loc="upper left")
fig.tight_layout(); plt.show()
""")

md(r"""
### 7.3b Concentration: how much of the result comes from a few days?

Daily mark-to-market makes it possible to check whether a configuration's result rests on a handful of days. For the `ALL` book
under U and D×P: the Sharpe without the 10 best days of the 2 820, and the share of the net PnL earned on those 10 days.
""")

code(r"""
rows = []
for g in GROUPS:
    for k in KEYS:
        for r in ("U", "DxP"):
            x = PNL[(FEE_BPS, g, k, "ALL", r)]
            top = np.argsort(x)[-10:]
            rows.append({"group": g, "config": CFG[k].label, "rule": r, "Sharpe": sharpe(x),
                         "Sharpe without 10 best days": sharpe(np.delete(x, top)), "net PnL (kUSD)": x.sum() / 1e3,
                         "share of PnL from 10 best days": x[top].sum() / x.sum() if x.sum() > 0 else np.nan})
CONC = pd.DataFrame(rows).set_index(["group", "config", "rule"])
for g in GROUPS:
    print(f"{g}: ALL book, concentration of the result in its 10 best days")
    display(CONC.loc[g].unstack("rule").reindex([CFG[k].label for k in KEYS]).round(3))
""")

md(r"""
### 7.4 After higher costs

The same trades, sizes and probabilities with 10 bps instead of 5 bps per fill (the meta-models are not refitted: a cost stress,
not a new design). High-frequency configurations trade more and lose more to costs.
""")

code(r"""
for g in GROUPS:
    rows = {}
    for p in PORTFOLIOS:
        for k in KEYS:
            rows[(p, CFG[k].label)] = {"Sharpe U 5 bps": sharpe(PNL[(FEE_BPS, g, k, p, "U")]),
                                       "Sharpe U 10 bps": sharpe(PNL[(FEE_STRESS_BPS, g, k, p, "U")]),
                                       "Sharpe D×P 10 bps": sharpe(PNL[(FEE_STRESS_BPS, g, k, p, "DxP")]),
                                       "D×F − D, 10 bps": fmt(delta(g, (k, p, "DxF"), (k, p, "D"), FEE_STRESS_BPS)),
                                       "D×P − D×F, 10 bps": fmt(delta(g, (k, p, "DxP"), (k, p, "DxF"), FEE_STRESS_BPS)),
                                       "U − U(time 4h), 10 bps": fmt(delta(g, (k, p, "U"), (BASELINE, p, "U"), FEE_STRESS_BPS)) if k != BASELINE else ""}
    print(f"{g}: cost stress, 10 bps per fill")
    display(pd.DataFrame(rows).T.round(3))
""")

md(r"""
### 7.5 Training assets (`SEEN`)

The same decomposition on the 20 assets the meta-models train on (in-sample with respect to the asset universe, out-of-sample in
time).
""")

code(r"""
display(META["SEEN"].round(3))
""")

# =====================================================================================================================
md(r"""
## 8. Where the comparison is weak: flags

A configuration × strategy cell is flagged when its conclusion rests on thin ground:

* **sparse trades** — fewer than 200 trades of the strategy on `NEW` assets in 93 months;
* **low model coverage** — less than 75 % of the evaluated trades had a fitted meta-model (the rest fall back to `D`, so the
  filter and sizing effects are diluted);
* **realised frequency off target** — the median asset's achieved bars per day is outside 0.8–1.25 × the target, so the
  configuration does not sample at the frequency it is compared at;
* **TIS daily candles incomplete** — fewer than 90 % of days pass TIS's own coverage rule for its daily confirmation;
* **volume filter uninformative** — SuperTrend's volume filter on volume bars (every bar carries the threshold volume).
""")

code(r"""
rows = []
for k in KEYS:
    for s in SS:
        st = STATS.loc[("NEW", s, k, "U")]
        cov = COVERAGE.loc[(s, CFG[k].label), "model coverage (all)"]
        ratio = SAMPLING.loc[CFG[k].label, "ratio median"]
        htf = INFO[k]["tis_htf_complete"].median()
        flags = []
        if st["signals"] < MIN_TRADES_NEW:
            flags.append("sparse trades")
        if not cov >= MIN_MODEL_COVERAGE:
            flags.append("low model coverage")
        if not FREQ_RATIO_BAND[0] <= ratio <= FREQ_RATIO_BAND[1]:
            flags.append("realised frequency off target")
        if s == "TIS" and htf < MIN_HTF_COMPLETE:
            flags.append("TIS daily candles incomplete")
        if s == "SUPERTREND" and CFG[k].bar_type == "volume":
            flags.append("volume filter uninformative")
        rows.append({"config": CFG[k].label, "strategy": s, "NEW trades": int(st["signals"]),
                     "months with trades (NEW)": int(st["months with trades"]), "model coverage": cov,
                     "achieved / target": ratio, "TIS daily candle complete": htf if s == "TIS" else np.nan,
                     "flags": "; ".join(flags)})
FLAGS = pd.DataFrame(rows).set_index(["strategy", "config"]).sort_index(level=0, sort_remaining=False)
display(FLAGS.round(3))
print(f"flagged cells: {int((FLAGS['flags'] != '').sum())} of {len(FLAGS)}")
""")

# =====================================================================================================================
md(r"""
## 9. Summary of the comparisons (computed)

For every portfolio on `NEW`: the baseline (time 4h) Sharpe of U and of D×P, the configurations whose U or D×P Sharpe is
distinguishable from the baseline's, and the configurations where filtering, sizing or inverse volatility add value with an
interval that excludes zero. "flagged" marks cells listed in §8.
""")

code(r"""
def flagged(p, k):
    return p != "ALL" and FLAGS.loc[(p, CFG[k].label), "flags"] != ""


rows = []
for g in GROUPS:
    for p in PORTFOLIOS:
        row = {"group": g, "portfolio": p, "baseline U": sharpe(PNL[(FEE_BPS, g, BASELINE, p, "U")]),
               "baseline D×P": sharpe(PNL[(FEE_BPS, g, BASELINE, p, "DxP")])}
        for rule in ("U", "DxP"):
            better, worse = [], []
            for k in KEYS:
                if k == BASELINE:
                    continue
                est, lo, hi, _ = delta(g, (k, p, rule), (BASELINE, p, rule))
                tag = CFG[k].label + (" (flagged)" if flagged(p, k) else "")
                if lo > 0:
                    better.append(f"{tag} {est:+.2f}")
                elif hi < 0:
                    worse.append(f"{tag} {est:+.2f}")
            row[f"{rule} better than baseline"] = ", ".join(better) or "none"
            row[f"{rule} worse than baseline"] = ", ".join(worse) or "none"
        for a, b in PAIRS_META:
            hits = []
            for k in KEYS:
                est, lo, hi, _ = delta(g, (k, p, a), (k, p, b))
                if lo > 0:
                    hits.append(f"{CFG[k].label} {est:+.2f}")
                elif hi < 0:
                    hits.append(f"{CFG[k].label} {est:+.2f} (worse)")
            row[f"{a} − {b} ≠ 0"] = ", ".join(hits) or "none"
        rows.append(row)
SUMMARY = pd.DataFrame(rows).set_index(["group", "portfolio"])
with pd.option_context("display.max_colwidth", 400):
    display(SUMMARY.loc["NEW"])
    display(SUMMARY.loc["SEEN"])

best = STATS.xs(("NEW", "U"), level=["group", "rule"])["Sharpe"].unstack("config")[KEYS]
print("Best configuration by U Sharpe on NEW, per portfolio:", {p: CFG[best.loc[p].idxmax()].label for p in PORTFOLIOS})
best2 = STATS.xs(("NEW", "DxP"), level=["group", "rule"])["Sharpe"].unstack("config")[KEYS]
print("Best configuration by D×P Sharpe on NEW, per portfolio:", {p: CFG[best2.loc[p].idxmax()].label for p in PORTFOLIOS})
print(f"total runtime {(time.time() - T0) / 60:.1f} min")
""")

md(r"""
## 10. Conclusions

All numbers are copied from the outputs above. `NEW` = the 20 assets no model ever trained on (main result); Sharpe = annualised
Sharpe of daily mark-to-market PnL, net of 5 bps per fill, fixed budget of 20 notionals per strategy, 2019-03-01 → 2026-11-19;
brackets are 95 % paired monthly block-bootstrap intervals; `ALL` = the four strategies together.

### 10.1 Data and sampling

* **Event bars from every trade.** 15.2 billion aggTrades (232 GB of archive) were turned into volume and dollar bars for all 40
  assets, with no missing archive file. Two archive defects were found and removed: 138 zero-price records (BTC, ETH, BNB, LTC in
  2017-12 / 2018-01) and 14.0 million repeated records (the ETHUSDT 2018-01 file holds parts of the month four times; BTC and XRP
  repeat some trades of 2026-02-11). After that, the daily traded volume of every asset-day equals the 1-day klines to 4e-6.
* **Realised frequency.** Time bars hit their targets exactly (30m bars lose 901 bars without trades, which become segment
  breaks). Event bars run **4–9 % above target** (median asset 1.04–1.09; worst asset 1.24, DOGE dollar bars): volume grew
  over the sample and a trailing 30-day threshold lags growth. Per asset-year, 80 % of the ratios lie within 0.92–1.25, and no
  configuration's median asset is outside the flag band. Durations of event bars are wide: at 6 bars per day (nominal 4 h) the
  median bar lasts ~3 h, about 5 % last under a quarter of an hour and 5 % over 9.5 h; the realised frequency follows activity
  over time (figure in §3).
* **The 4h baseline reproduces the reference exactly**: 22 389 closed trades, identical trade by trade over the whole reference
  history, through the new `load_bars` interface.

### 10.2 Where the primary strategy itself improves (rule U)

| `ALL`, NEW | 48/d (30m) | 24/d (1h) | 6/d (4h) | 1/d (1d) |
|---|---|---|---|---|
| time | −1.84 | −0.59 | **+0.87** | **+0.94** |
| volume | −2.69 | −1.20 | −0.11 | +0.71 |
| dollar | −2.10 | −1.26 | −0.55 | +0.02 |

* **Frequency dominates bar type, and higher frequency is decisively worse after costs.** Every 30m / 1h / 48-per-day /
  24-per-day configuration is significantly worse than time 4h for the four-strategy book (Δ −1.5 to −3.6). Per-trade gross
  edges shrink faster than costs (Donchian on time bars: +1 828 bps net per trade at 1d, +184 at 4h, +26 at 1h, −13 at 30m) while
  turnover rises from 7 × K per year (1d) to 33 (4h) and 237 (30m).
* **Event bars do not improve these strategies at any frequency.** Their Sharpe for the book is below that of time bars of the
  same frequency at every frequency; at 6 per day volume bars are −0.98 [−1.63, −0.27] and dollar bars −1.42 [−2.64, −0.11] below
  time 4h. Two of the four rules depend on time-bar semantics: **TIS** needs day-aligned candles for its daily confirmation (on
  event bars only 93 / 76 / 29 / 5 % of days pass at 48 / 24 / 6 / 1 bars per day, and it all but stops trading), and
  **SuperTrend**'s volume filter is uninformative on volume bars (it is negative on every event-bar configuration). Donchian is
  the only strategy for which event bars are competitive (volume 1/d +1.20, dollar 1/d +1.08 vs time 4h +0.95, not significant).
* **Daily time bars are the only statistically indistinguishable alternative to 4h** for the book: +0.94 vs +0.87,
  Δ +0.06 [−0.45, +0.59] on NEW, −0.25 [−0.80, +0.30] on SEEN, with larger net PnL per unit of capital (return on K 0.44 vs 0.16 per
  year) but a four times deeper drawdown (−0.76 K vs −0.20 K), and it rests on fewer days: without its 10 best days (of 2 820)
  the daily book's Sharpe falls from 0.94 to 0.41 (66 % of its net PnL was earned on those days, in two bursts in early 2021 and
  late 2024), the 4h book's from 0.87 to 0.42. By strategy, RSI turns from a loser at 4h (−0.17) into a winner on daily bars
  (+0.90, Δ +1.07 [+0.55, +1.69]); Donchian is +0.24 (n.s.) better daily; TIS and SuperTrend are best at time 4h.

### 10.3 Where meta-labeling adds value (filter D×F, sizing D×P)

* **At the 4h baseline meta-labeling adds a modest, significant amount**: D×P − D = **+0.22 [+0.06, +0.41]** (U 0.87 → D 0.97 →
  D×F 1.07 → D×P 1.19), of which the filter is +0.09 [+0.02, +0.17] and probability sizing +0.12 [−0.02, +0.29] (not
  significant). This is the same size as the reference notebook's +0.23 with a much simpler model, and at 4h the filter's value is
  trade selection (the strategy-level switch never turns a strategy off there).
* **The large "gains" at high frequency are mostly a strategy-level off switch, not trade selection.** For the book the filter
  adds +0.74 to +2.55 in six configurations (time 30m and 1h, volume 48 / 24 / 6 per day, dollar 24 per day; +1.02 on dollar 48/d
  with an interval touching zero), but the model-free diagnostic D×S — switch a strategy off when its training trades lost money —
  captures most of it: D×S − D is significant in five of the six (all but dollar 24/d), and D×F − D×S is not distinguishable from
  zero in any of them. The rescued books also stay below the 4h baseline: the best are volume 6/d D×P (+0.79) and time 1h D×F
  (+0.58); the others remain near or below zero.
* **Probability sizing beyond the filter** (D×P − D×F) is significant for the book only on dollar 48/d (+0.42) and for Donchian on
  24-per-day event bars; it is slightly harmful on dollar 1/d (−0.05).
* **On daily bars meta-labeling adds nothing** (the filter accepts 92–98 % of trades; models exist for only 54–64 % of the
  book's trades, 0–74 % by strategy, because the strategies trade too rarely to reach 200 training trades), and inverse
  volatility even costs Sharpe on daily event bars (D − U −0.27 / −0.29, significant).
* **The model's information is thin everywhere.** Outer AUC on NEW between 0.50 and 0.59, except in thin daily cells
  (0.41–0.63); the probability beats the training win rate in log
  loss in 4 of 44 strategy × configuration cells; the inner purged CV chose the strongest shrinkage (C = 0.003) in 99 % of 305
  windows and beat the prior in only 29 % of them. What the model mostly delivers is whether the strategy has been profitable,
  plus a weak ranking.

### 10.4 After higher costs and on the training assets

At 10 bps per fill the ranking is unchanged and the gap widens: `ALL` time 4h U 0.78 / D×P 1.11, time 1d 0.93 / 0.78, volume 1/d
U 0.68, and U below −2.7 for every 30m / 48-per-day configuration. On `SEEN` the same picture holds at higher levels: time 4h is
the best configuration for U (1.33) and for D×P (1.71); the nearest are volume 1/d and time 1d for U (1.10, 1.08) and time 1h and
time 1d for D×P (1.28, 1.11); no configuration is significantly better than time 4h and most are significantly worse.

### 10.5 What the evidence supports about the 4h setup

**Keep 4-hour time bars.** For the four-strategy book no configuration beats them on `NEW` or `SEEN`, either as primary strategies
(U) or with meta-labeling (D×P); 4h is also where meta-labeling has its one clean, significant trade-selection effect, with high
model coverage (95 %) and moderate turnover, and its meta-labeled book is the least dependent on a few days (D×P Sharpe 1.19, still
0.77 without its 10 best days, against 0.79 → 0.19 for the daily book). **Do not replace them with volume or dollar bars** for these rules: event bars are
worse at every frequency, partly because two of the four rules are written for calendar bars. **Do not go faster**: 30m and 1h
(and their event-bar counterparts) lose after costs, and meta-labeling there mostly just switches strategies off. **Daily time bars
are the only credible alternative**, equal for the book, better for RSI (and possibly Donchian) individually, but with deeper
drawdowns, too few trades to train a meta-model, and no meta-labeling value. A mixed setup — RSI (and perhaps Donchian) on daily
bars, TIS and SuperTrend on 4h bars — is the one change worth testing, but it was chosen after looking at these results, so it
needs fresh data (after 2026-11-19) before it can replace the current setup.

### 10.6 Limitations

* **Evidence quality.** Same calendar as the earlier studies; the `NEW` assets were used before and informed the design. The
  eleven new configurations are fresh, the period is not.
* **Multiplicity.** 12 configurations × 5 portfolios × several contrasts; intervals that barely exclude zero would not survive a
  correction. The robust statements are the large, consistent ones: high frequency loses, event bars do not help, the 4h
  meta-labeling gain is modest.
* **Coverage.** Daily configurations have low model coverage (flagged); TIS on event bars at 24 per day or less and SuperTrend on
  volume bars measure the interaction of those rules with event bars more than the bars themselves (flagged).
* **Simplifications.** One logistic-regression specification (by design); 5 bps per fill without slippage, no funding or
  borrowing cost for SuperTrend's short trades; 40 assets that all still exist (survivorship); event-bar thresholds from a trailing
  30-day mean (realised frequency 4–9 % above target).
""")


def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == "markdown" else nbf.v4.new_code_cell(s) for t, s in CELLS]
    path = HERE / "bar_type_frequency_meta_labeling.ipynb"
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    print(build(), len(CELLS), "cells")
