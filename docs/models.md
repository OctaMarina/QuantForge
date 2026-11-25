# `quant_forge.models` — model specifications, purged tuning, calibration

```python
from quant_forge.models import default_model_specs, tune_purged, fit_predict, oof_prior, PlattCalibrator

specs = default_model_specs(seed=0)                        # {"LR": ModelSpec, "RF": ..., "HGB": ...}
best = tune_purged(specs["RF"], X, y, folds, columns=selected_columns, n_jobs=4)
best.score, best.params, best.oof, best.scores             # OOF log loss, winner, its OOF probabilities, all grid points
p_raw, pipeline, _ = fit_predict(specs["RF"], best.params, X[selected_columns], y, X_test[selected_columns])
cal = PlattCalibrator().fit(best.oof[~np.isnan(best.oof)], y[~np.isnan(best.oof)])
p = cal.predict(p_raw)
```

## `ModelSpec`

A model family split into a parameter-free **preprocessor** and a parametrised **estimator**:

| Spec | Preprocessor | Estimator / grid |
|---|---|---|
| `lr_spec()` | median imputation + normal quantile transform (200 quantiles) | `LogisticRegression(C, max_iter=3000)`, `C ∈ {0.003, 0.01, 0.03, 0.1, 0.3}` |
| `rf_spec(grid)` | median imputation | `RandomForestClassifier` (150 trees, entropy, balanced subsample) with `max_features`, `mwfl` (min weight fraction leaf), `max_depth` |
| `hgb_spec(grid)` | median imputation | `HistGradientBoostingClassifier` (150 iterations, no early stopping) with `lr`, `depth`, `msl`, `l2`, `mf` |

`default_model_specs(seed, n_random=4)` draws the RF and HGB grids from one `default_rng(seed)` stream (RF first), so they
are reproducible. `spec.pipeline(params)` returns the equivalent ordinary scikit-learn `Pipeline`; `make_lr`, `make_rf`,
`make_hgb` build pipelines directly. `pca_preprocessor()` = imputation + standardisation + PCA keeping 95 % of variance.

## Tuning — `tune_purged(spec, X, y, folds, columns=None, pca=False, n_jobs=None)`

Grid search scored by **out-of-fold log loss** (AFML §9.4: the proper scoring rule for probabilities) on the folds you
pass (typically `CalendarPurgedKFold`). The preprocessor (and PCA) is fitted once per fold and reused for every grid
point – the result is identical to refitting the full pipeline (tested), at a fraction of the cost for the logistic
regression. Ties keep the first grid point. Returns `TuningResult(score, params, oof, scores)`.

`oof_prior(y, folds)` gives the OOF predictions of the constant model (training-fold win rate): the reference every
model must beat. Choosing the best of many `(model, feature set)` pairs by inner log loss is optimistic; expect worse
out of sample.

## Calibration — `PlattCalibrator`

Fits `P(y=1) = sigmoid(a · logit(p) + b)` on OOF probabilities (practically unregularised logistic regression);
`coefficients` returns `(a, b)`. Calibrate before any rule that reads the probability literally (bet sizing, break-even
filters).
