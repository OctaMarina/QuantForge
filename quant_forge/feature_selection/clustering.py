"""Feature clustering before importance analysis (MLAM ch. 4 and §6.5).

* :func:`redundancy_clusters` - average-linkage hierarchical clustering on ``1 - |Spearman rho|`` cut at
  ``max_dist`` (members correlate at ``|rho| >= 1 - max_dist`` on average).  Deterministic, cheap, and the
  grouping used for clustered importance in the research notebooks.
* :func:`onc_clusters` - ONC, "optimal number of clusters" (MLAM §4.4, Snippets 4.1-4.2): K-means on the
  correlation distance ``sqrt((1 - rho) / 2)`` with the number of clusters chosen by the silhouette t-statistic
  and recursive re-clustering of below-average clusters.

Both return ``{cluster name: sorted member columns}``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["redundancy_clusters", "onc_clusters", "cluster_kmeans_base", "cluster_kmeans_top"]


def redundancy_clusters(Z: pd.DataFrame, max_dist: float = 0.30, method: str = "spearman") -> dict[str, list[str]]:
    """Clusters ``c00, c01, ...`` of redundant columns of ``Z`` (NaN correlations count as 0)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    cols = list(Z.columns)
    if len(cols) < 2:
        return {"c00": cols}
    dist = 1.0 - np.abs(Z.corr(method=method).fillna(0.0).to_numpy())
    np.fill_diagonal(dist, 0.0)
    lab = fcluster(linkage(squareform((dist + dist.T) / 2, checks=False), method="average"), t=max_dist,
                   criterion="distance")
    groups: dict[int, list[str]] = {}
    for c, l in zip(cols, lab):
        groups.setdefault(int(l), []).append(c)
    return {f"c{i:02d}": sorted(m) for i, (_, m) in enumerate(sorted(groups.items()))}


def cluster_kmeans_base(corr0: pd.DataFrame, max_clusters: int = 30, n_init: int = 5, seed: int = 0):
    """MLAM Snippet 4.1: best K-means partition by the silhouette t-statistic (mean / sd)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples

    x = np.sqrt(((1 - corr0.fillna(0)) / 2.0).clip(lower=0))
    best_silh, best_km, best_stat = None, None, -np.inf
    for init in range(n_init):
        for k in range(2, min(max_clusters, len(x) - 1) + 1):
            km = KMeans(n_clusters=k, n_init=1, random_state=seed + init * 1000 + k).fit(x)
            if len(np.unique(km.labels_)) < 2:
                continue
            silh = silhouette_samples(x, km.labels_)
            stat = silh.mean() / silh.std() if silh.std() > 0 else -np.inf
            if stat > best_stat:
                best_silh, best_km, best_stat = silh, km, stat
    clstrs = {i: corr0.columns[np.flatnonzero(best_km.labels_ == i)].tolist() for i in np.unique(best_km.labels_)}
    return clstrs, pd.Series(best_silh, index=corr0.index)


def cluster_kmeans_top(corr0: pd.DataFrame, max_clusters: int = 30, n_init: int = 5, seed: int = 0, depth: int = 0):
    """MLAM Snippet 4.2: re-cluster the clusters whose quality is below average while that improves it."""
    if corr0.shape[1] < 4:
        return {0: list(corr0.columns)}, pd.Series(1.0, index=corr0.index)
    clstrs, silh = cluster_kmeans_base(corr0, max_clusters, n_init, seed)
    tstat = {i: silh[m].mean() / silh[m].std() if len(m) > 1 and silh[m].std() > 0 else 0.0 for i, m in clstrs.items()}
    tmean = np.mean(list(tstat.values()))
    redo = [i for i in clstrs if tstat[i] < tmean]
    if len(redo) <= 1 or depth > 3:
        return clstrs, silh
    keys = [j for i in redo for j in clstrs[i]]
    if len(keys) < 4:
        return clstrs, silh
    clstrs2, silh2 = cluster_kmeans_top(corr0.loc[keys, keys], max_clusters, n_init, seed + 1, depth + 1)
    keep = {i: m for i, m in clstrs.items() if i not in redo}
    new = {}
    for m in list(keep.values()) + list(clstrs2.values()):
        new[len(new)] = m
    new_silh = pd.concat([silh[[c for m in keep.values() for c in m]], silh2]) if keep else silh2
    new_t = np.mean([new_silh[m].mean() / new_silh[m].std() if len(m) > 1 and new_silh[m].std() > 0 else 0.0
                     for m in new.values()])
    old_t = np.mean([tstat[i] for i in redo])
    return (new, new_silh) if new_t > old_t else (clstrs, silh)


def onc_clusters(Z: pd.DataFrame, max_clusters: int = 30, n_init: int = 5, seed: int = 0) -> dict[str, list[str]]:
    """ONC clusters ``onc00, onc01, ...`` of the Spearman correlation matrix of ``Z``."""
    c = Z.corr(method="spearman").fillna(0.0)
    a = c.to_numpy(copy=True)
    np.fill_diagonal(a, 1.0)
    corr = pd.DataFrame(a, index=c.index, columns=c.columns)
    clstrs, _ = cluster_kmeans_top(corr, max_clusters, n_init, seed)
    return {f"onc{i:02d}": sorted(m) for i, m in clstrs.items()}
