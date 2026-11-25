# `quant_forge.metalabeling` — meta-labeling on top of the primary strategies

Meta-labeling (AFML §3.6; Joubert 2022): the primary strategy decides side and exit; a secondary model predicts whether a
signalled trade will be profitable after costs, `y = 1[trade_return > 0]`, and its probability is used to filter and size
trades. This package builds the dataset, fits everything that is learned inside walk-forward windows, and summarises the run.

## 1. Dataset

```python
from quant_forge import data as qd, features as qf, metalabeling as ml
import quant_forge.strategies as qs

S = qs.Strategies
STRATEGIES = {S.RSI_CROSS_ABOVE: "RSI", S.DONCHIAN_BREAKOUT_CONTINUATION: "DONCHIAN",
              S.TIS_BTC_MOMENTUM_CONFLUENCE: "TIS", S.SUPERTREND_REGIME_CONFLUENCE: "SUPERTREND"}
trades = ml.build_trade_panel(bars, feats, STRATEGIES, feature_columns=qf.model_columns(feats["BTC"]),
                              tick_sizes=qd.TICK_SIZES, fee_bps=5, slippage_ticks=0, n_jobs=8)
trades["group"] = ml.assign_groups(trades, {"SEEN": qd.SEEN_ASSETS, "NEW": qd.NEW_ASSETS})
trades["period"] = ml.assign_periods(trades["decision_time"], [("A", t_a), ("B", t_b)], last="C")
trades["month"] = trades["exit_time"].dt.strftime("%Y-%m")       # needed by the evaluation functions
ds = ml.prepare_meta_dataset(trades)        # MetaDataset(trades, family, candidates)
ds.candidates_for("RSI")                    # drops the lib_* features of the other strategies
```

* `build_trade_panel` runs every strategy on every asset (assets in parallel processes), attaches the bar features as
  `context_columns` (values of the signal bar), keeps closed trades, converts the strategy's level features to ratios
  (`features.relative_levels`) and labels `y`. Rows are sorted by `(decision_time, strategy, asset)`; the row index is the
  trade id used everywhere downstream.
* `prepare_meta_dataset` adds the track record (family S, from trades closed strictly before the decision), orients
  direction-carrying features by side, and lists the candidate inputs (every family except RAW and `*_level` diagnostics).

## 2. One window — `fit_window(g, w0, w1, candidates, config, name=..., n_jobs=...)`

`g` = trades of one strategy (all assets). Training = trades exited before `w0` (restricted to `config.train_assets`);
test = trades decided in `[w0, w1)` on every asset. Steps, all on training data only:

1. coverage filter (`min_coverage`) and constant-feature removal;
2. inner folds: `CalendarPurgedKFold(inner_splits, embargo_frac, cover_all_events)`;
3. `redundancy_clusters` → `make_placebos` → jointly permuted `clustered_mda` → `select_clusters` (above the placebo
   maximum and positive on `inner_splits − 1` folds); `clustered_mdi` for comparison;
4. `tune_purged` for each model family on the full and on the selected set (+ PCA variant for `pca_models`); the lowest
   inner log loss is `META`;
5. refit on all training trades, Platt calibration on the inner OOF probabilities, predictions for the window;
6. `fit_sizing_constants` from META's calibrated OOF probabilities.

Returns `WindowFit(predictions, diagnostics)` or `None` (fewer than `min_train` trades, one class, empty window).
`predictions`: `row`, `p_<combo>_raw`, `p_<combo>` (calibrated), `p` (META), `prior`, `window`, `meta`.
`diagnostics`: sizes, fold sizes, `groups`, `mda`, `mdi`, `selected`, `placebo_max`, `tuning`, `prior_inner_ll`, `meta`,
`meta_beats_prior`, `constants` (`SizingConstants`), `calib_oof_ll`, `raw_oof_ll`, `platt`.

### `MetaLabelingConfig`

| Field | Default | Meaning |
|---|---|---|
| `min_train` | 200 | minimum training trades |
| `min_coverage` | 0.70 | minimum non-missing share of a feature |
| `inner_splits`, `embargo_frac` | 4, 0.01 | inner calendar purged K-fold |
| `cover_all_events` | True | False reproduces the research notebook's fold edges |
| `cluster_max_dist` | 0.30 | redundancy clusters |
| `n_placebo`, `n_perm` | 5, 3 | placebos, permutations per cluster and fold |
| `seed` | 0 | window RNG = `seed + (w0 seconds mod 1000)`; model seeds and grids |
| `train_assets` | None | assets the model may train on (None = all) |
| `model_names`, `pca_models` | ("LR", "RF", "HGB"), ("LR",) | model families and PCA variants |
| `vol_clip`, `vol_column` | (0.25, 4), `v_rvol_30d` | inverse-volatility input |
| column names | `y`, `trade_return`, `decision_time`, `exit_time`, `asset` | |
| `feature_selection` | True | False skips clusters, placebos, clustered MDA / MDI: every model family is tuned on all usable features only (no `-SEL` combos). With `model_names=("LR",)`, `pca_models=()` this is the one-model specification of the bar-type study. The field is left out of `cache_key()` at its default, so existing caches keep their keys |

## 3. All windows — `run_walk_forward`

```python
from quant_forge.validation import calendar_windows
from quant_forge.utils import DiskCache, frame_fingerprint

cfg = ml.MetaLabelingConfig(train_assets=tuple(qd.SEEN_ASSETS))
windows = calendar_windows("2019-03-01", end=end, years=1)
cache = DiskCache("~/.cache/quantforge/meta_run", key={"cfg": cfg.cache_key(), "windows": [str(w) for w in windows],
                                                      "data": frame_fingerprint(bars, columns=["close"])})
res = ml.run_walk_forward(ds.trades, list(STRATEGIES.values()), windows,
                          {s: ds.candidates_for(s) for s in STRATEGIES.values()}, cfg,
                          cache=cache, n_jobs=8, inner_n_jobs=2)
pred = ml.collect_predictions(res, ds.trades, ["strategy", "asset", "group", "period", "entry_time", "exit_time",
                                               "trade_return", "y", "v_rvol_30d", "month"])
```

Pairs run in parallel processes (largest training sets first), each receiving only the columns it needs; every pair is
cached as `<STRATEGY>_<YYYYMMDD>.pkl`. Results do not depend on `n_jobs`. Cost is dominated by the random forests: about
9 minutes for 4 strategies × 8 windows × 40 assets on 22 cores, seconds when cached.

## 4. Reports

* `window_summary(diagnostics)` – per window: sizes, clusters, selected clusters, inner log losses, META choice.
* `feature_selection_records(diagnostics, family)` – per (window, feature): cluster MDA, positive folds, selected, size, MDI.
* `family_selection_table(records)` – selection share / mean MDA per family and strategy.
* `sizing_constants_table(diagnostics)` – training-side sizing constants.
* `probability_columns(pred)` – `{combo: column}` (+ `META → p`) for `evaluation.probability_quality_table`.

Then size and evaluate:

```python
from quant_forge import sizing as qz, evaluation as qe
sizes = qz.normalized_sizes(pred, {k: r.diagnostics["constants"] for k, r in res.items()}, cap=10)
ev = qe.evaluate_sizing(pred.loc[pred["group"].eq("NEW")], sizes.loc[pred["group"].eq("NEW")])
qe.paired_sharpe_table(ev, [("F_be", "U"), ("P", "F_be"), ("P", "U"), ("DxP", "D")])
```

The complete study (features, selection, calibration, sizing decomposition, capital budget, robustness) is
`../research/meta-labeling`.
