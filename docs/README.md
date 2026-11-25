# QuantForge documentation

QuantForge (import name `quant_forge`) is a research library for systematic crypto trading built around one workflow:
**primary strategy → meta-labeling → bet sizing → economic evaluation**, with every learned quantity estimated
point-in-time. This folder documents each sub-package; the notebooks
`../research/meta-labeling` (4h bars) and
`research/bar_types/bar_type_frequency_meta_labeling.ipynb` (time / volume / dollar bars at four frequencies) use all of them end
to end.

## Packages

| Package | What it does | Needs | Guide |
|---|---|---|---|
| `quant_forge.data` | Binance spot klines (REST + incremental parquet cache) and the public archive → **time, volume and dollar bars** through one interface (`load_bars` + `BarConfig`; event bars from aggTrades with point-in-time thresholds); sampling diagnostics; research asset universes | numpy, pandas, pyarrow (+ numba for event bars) | [data.md](data.md) |
| `quant_forge.features` | point-in-time bar features (volatility, trend, flow, calendar, market state, relative strength, cross-sectional ranks) in bar counts or **in elapsed time with as-of panels for asynchronous bars**, fractional differentiation, trade-level features | numpy, pandas (+ numba optional, statsmodels for ADF) | [features.md](features.md) |
| `quant_forge.strategies` | the four primary strategies: specification, signals, execution, closed trades with decision-time features | numpy, pandas | [strategies.md](strategies.md) |
| `quant_forge.validation` | purged K-fold with embargo (single series and panel / calendar folds), walk-forward windows | numpy, pandas | [validation.md](validation.md) |
| `quant_forge.feature_selection` | redundancy and ONC clustering, placebo features, clustered MDA / MDI, cluster selection, PCA eigen-rank check | scikit-learn, scipy | [feature_selection.md](feature_selection.md) |
| `quant_forge.models` | model specifications (LR / RF / HGB), purged hyper-parameter search, Platt calibration | scikit-learn | [models.md](models.md) |
| `quant_forge.sizing` | AFML bet sizing, the rule set (uniform, inverse volatility, filters, rank, probability), training constants, capital-budget replay | scipy | [sizing.md](sizing.md) |
| `quant_forge.evaluation` | probability quality (Δ log loss vs prior, AUC, IC, calibration) and economics (Sharpe, bootstrap, permutation, exposure, drawdown; daily mark-to-market portfolios on a common calendar with paired bootstrap across trade lists) | numpy, pandas, scikit-learn | [evaluation.md](evaluation.md) |
| `quant_forge.metalabeling` | trade panels from strategies, the per-window meta-labeling worker, parallel cached walk-forward, run reports | all of the above | [metalabeling.md](metalabeling.md) |
| `quant_forge.utils` | `DiskCache`, `stable_hash`, `frame_fingerprint` | pandas | below |

Dependency direction (no cycles): `utils` ← `data`; `features`, `strategies`, `validation` are independent;
`evaluation.metrics` ← `models` ← `feature_selection`; `sizing`, `evaluation.economics` use `validation` helpers;
`metalabeling` sits on top. `import quant_forge.strategies` never imports scikit-learn.

## Installation

```bash
pip install -e ".[dev]"        # library + pytest + scikit-learn, scipy, statsmodels, joblib (all tests)
pip install -e ".[research]"   # + pyarrow, matplotlib, catboost, notebook tooling, numba
pytest                         # tests/ (synthetic data; a few tests compare with mlfinpy when it is installed)
```

The distribution name is `quantforge`; if another package with that name is already installed in an environment, the
notebooks fall back to putting the repository root on `sys.path` (they look for `quant_forge/strategies`).

## Conventions shared by all packages

* **Timestamps.** Bar frames carry microsecond integers (`bar_start_us`, `bar_end_us`) for the strategy library and are
  indexed by the tz-aware UTC bar end. Event tables use tz-aware `decision_time`, `entry_time`, `exit_time`.
* **Point-in-time.** A feature at bar `t` uses bars `≤ t`; a trade feature uses only trades that **exited strictly before**
  its decision; every constant of a model, of a filter or of a sizing rule is fitted on training data of the window that
  uses it. Tests in `tests/features` truncate the data and check that nothing before the cut changes.
* **Auxiliary columns** start with `_` (e.g. `_log_close`): inputs of later steps, never model inputs
  (`features.model_columns` drops them).
* **Families** are encoded in column prefixes: `v_` V (volatility), `t_` T (trend), `f_` F (flow), `cal_` CAL, `m_`/`b_` M
  (market / BTC), `x_` R (relative strength), `xr_` X (cross-sectional ranks), `ffd_` FFD, `s_` S (track record),
  `lib_` LIB (the strategy's own features), `raw_` RAW (raw levels, never pooled inputs).
* **Reproducibility.** Random draws come from explicit `numpy.random.Generator`s seeded by the caller; parallel execution
  (`n_jobs`) never changes results except for floating-point rounding of multi-threaded forest averaging (~1e-12).

## End-to-end sketch

```python
import pandas as pd
import quant_forge.strategies as qs
from quant_forge import data as qd, features as qf, metalabeling as ml, sizing as qz, evaluation as qe
from quant_forge.validation import calendar_windows
from quant_forge.utils import DiskCache

end = pd.Timestamp("2026-11-19", tz="UTC")
bars = qd.load_binance_bars(qd.ALL_ASSETS, end=end)                       # {asset: 4h bars}
feats = qf.bar_features(bars, bars_per_day=6)                             # V T F CAL R X M (+ aux)
sel = qf.select_ffd_order({a: f["_log_close"] for a, f in feats.items()}, train_end=pd.Timestamp("2019-03-01", tz="UTC"))
feats = qf.add_raw_levels(qf.add_ffd_features(feats, sel.pooled), bars)

S = qs.Strategies
trades = ml.build_trade_panel(bars, feats, {S.RSI_CROSS_ABOVE: "RSI"}, feature_columns=qf.model_columns(feats["BTC"]),
                              tick_sizes=qd.TICK_SIZES, fee_bps=5, n_jobs=8)
trades["group"] = ml.assign_groups(trades, {"SEEN": qd.SEEN_ASSETS, "NEW": qd.NEW_ASSETS})
trades["month"] = trades["exit_time"].dt.strftime("%Y-%m")
ds = ml.prepare_meta_dataset(trades)                                      # + S family, side orientation, candidates

cfg = ml.MetaLabelingConfig(train_assets=tuple(qd.SEEN_ASSETS))
windows = calendar_windows("2019-03-01", end=end, years=1)
res = ml.run_walk_forward(ds.trades, ["RSI"], windows, {"RSI": ds.candidates_for("RSI")}, cfg,
                          cache=DiskCache("~/.cache/quantforge/my_run", key=cfg.cache_key()), n_jobs=8)
pred = ml.collect_predictions(res, ds.trades, ["strategy", "group", "entry_time", "exit_time", "trade_return", "y",
                                               "v_rvol_30d", "month"])
sizes = qz.normalized_sizes(pred, {k: r.diagnostics["constants"] for k, r in res.items()}, cap=10)
ev = qe.evaluate_sizing(pred, sizes)
print(ev.table[["Sharpe", "Δ vs U", "CI U lo", "CI U hi", "p perm"]])
print(qe.paired_sharpe_table(ev, [("P", "U"), ("DxP", "D")]))
```

## `quant_forge.utils`

```python
from quant_forge.utils import DiskCache, stable_hash, frame_fingerprint, DEFAULT_CACHE_ROOT

cache = DiskCache(DEFAULT_CACHE_ROOT / "study", key={"params": {...}, "data": frame_fingerprint(bars, columns=["close"])})
value = cache.get_or_compute("name", expensive_function, arg1, arg2)    # pickled under <root>/<hash(key)>/name.pkl
cache.clear("name")   # or cache.clear() for all entries
```

* `stable_hash(obj)` – SHA-256 of `json.dumps(obj, sort_keys=True, default=str)` (first 12 hex digits).
* `frame_fingerprint(frames, columns=None)` – content hash of a DataFrame or of a `{name: DataFrame}` mapping.
* Writes are atomic (temporary file + rename). The cache never decides staleness by itself: put everything the result
  depends on (configuration, data fingerprint, library version) in `key`.
