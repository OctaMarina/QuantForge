# `quant_forge.feature_selection` — clustered importance with placebos

Feature importance on financial data is distorted by substitution effects (correlated features share importance) and by
in-sample overfitting. The package implements the remedies of AFML ch. 8 and MLAM ch. 4/6: cluster features first,
measure importance out of sample on purged folds by permuting whole clusters, and compare with placebo features.

```python
import numpy as np, pandas as pd
from quant_forge.feature_selection import redundancy_clusters, make_placebos, clustered_mda, clustered_mdi, select_clusters
from quant_forge.validation import CalendarPurgedKFold

Z = X.fillna(X.median())                                  # training features (no NaN for permutations)
folds = list(CalendarPurgedKFold(4, 0.01).split(train["decision_time"], train["exit_time"]))
rng = np.random.default_rng(0)
groups = redundancy_clusters(Z, max_dist=0.3)             # {"c00": [...], ...}
P = make_placebos(Z, 5, rng)                              # null features
mda = clustered_mda(pd.concat([Z, P], axis=1), y, folds, {**groups, **{c: [c] for c in P}}, n_perm=3, rng=rng)
sel = select_clusters(mda, groups, list(P.columns), min_positive_folds=3)
sel.selected, sel.columns, sel.placebo_max
mdi = clustered_mdi(Z, y, groups)                         # in-sample counterpart
```

## Clustering — `clustering.py`

* `redundancy_clusters(Z, max_dist=0.30, method="spearman")` – average-linkage clustering on `1 − |ρ|`, cut at
  `max_dist` (members correlate at `|ρ| ≥ 0.7` on average). Deterministic; names `c00, c01, …`.
* `onc_clusters(Z, max_clusters=30, n_init=5, seed=0)` – ONC (MLAM §4.4, Snippets 4.1–4.2): K-means on
  `sqrt((1 − ρ)/2)`, number of clusters by the silhouette t-statistic, recursive re-clustering of weak clusters
  (`cluster_kmeans_base`, `cluster_kmeans_top`). On large heterogeneous feature sets it can return very few clusters, which
  is why the meta-labeling pipeline uses redundancy clusters for selection and ONC as a diagnostic.

## Importance — `importance.py`

| Function | What it computes |
|---|---|
| `make_placebos(Z, n, rng)` | `n` circularly shifted copies of random real columns (shift 20–80 % of the sample): same distribution and autocorrelation, no relation to the label |
| `clustered_mda(Z, y, folds, groups, n_perm, rng, estimator=None, n_jobs)` | per fold: validation log loss of the fitted model (`_base`), of the training prior (`_prior`), and for each cluster the mean increase in log loss when its columns are permuted **jointly** (one row permutation for the whole cluster). Default model: random forest (150 trees, entropy, `max_features="sqrt"`, `min_weight_fraction_leaf=0.02`, balanced subsample). Permuted copies are predicted in memory-bounded chunks |
| `clustered_mdi(Z, y, groups)` | MDI of a forest with `max_features=1` (zeros of unused features ignored per tree, AFML Snippet 8.2), normalised, summed per cluster |
| `select_clusters(mda, groups, placebo_columns, min_positive_folds)` | clusters whose mean MDA exceeds `max(0, largest placebo MDA)` **and** is positive on at least `min_positive_folds` folds → `ClusterSelection(selected, columns, placebo_max, mean_importance, positive_folds)` |
| `pca_eigen_rank_check(Z, y, n_null=10)` | AFML §8.4.2: weighted Kendall τ between the MDI of a forest on the principal components and their inverse eigen-rank, with a label-permutation null |

## Reading the results

* The fold-consistency rule alone passes a placebo roughly a quarter of the time; selection shares must be compared
  with that chance level, not with zero.
* MDI and out-of-sample MDA often disagree; trust MDA.
* Random numbers: `clustered_mda` draws permutations from the `rng` you pass, in cluster order, so results are reproducible.
