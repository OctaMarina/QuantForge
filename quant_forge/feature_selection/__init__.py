"""Feature clustering, clustered importance (MDA / MDI) with placebos and cluster selection (needs scikit-learn, scipy).

    from quant_forge.feature_selection import redundancy_clusters, make_placebos, clustered_mda, select_clusters

    groups = redundancy_clusters(Z, max_dist=0.3)
    P = make_placebos(Z, 5, rng)
    mda = clustered_mda(pd.concat([Z, P], axis=1), y, folds, {**groups, **{c: [c] for c in P}}, rng=rng)
    sel = select_clusters(mda, groups, list(P), min_positive_folds=3)
"""

from .clustering import cluster_kmeans_base, cluster_kmeans_top, onc_clusters, redundancy_clusters
from .importance import (ClusterSelection, clustered_mda, clustered_mdi, make_placebos, pca_eigen_rank_check,
                         select_clusters)

__all__ = ["redundancy_clusters", "onc_clusters", "cluster_kmeans_base", "cluster_kmeans_top", "make_placebos",
           "clustered_mda", "clustered_mdi", "ClusterSelection", "select_clusters", "pca_eigen_rank_check"]
