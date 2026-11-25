import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from quant_forge.feature_selection import (clustered_mda, clustered_mdi, make_placebos, onc_clusters,
                                           pca_eigen_rank_check, redundancy_clusters, select_clusters)
from quant_forge.validation import CalendarPurgedKFold
from tests.synthetic import make_trade_panel


def _data(n=900, seed=0):
    tr = make_trade_panel(n, seed=seed, signal=1.5)
    Z = tr[[c for c in tr.columns if c.startswith(("f_", "t_"))]]
    folds = list(CalendarPurgedKFold(4, 0.01).split(tr["decision_time"], tr["exit_time"]))
    return tr, Z, tr["y"].to_numpy(), folds


def test_redundancy_clusters_group_correlated_columns():
    _, Z, _, _ = _data()
    groups = redundancy_clusters(Z, 0.3)
    together = [m for m in groups.values() if "f_inf" in m][0]
    assert together == ["f_inf", "f_inf_copy"]
    assert sum(len(m) for m in groups.values()) == Z.shape[1]
    assert list(groups) == [f"c{i:02d}" for i in range(len(groups))]


def test_onc_recovers_blocks():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(500, 3))
    Z = pd.DataFrame({f"b{b}_{j}": base[:, b] + rng.normal(0, 0.3, 500) for b in range(3) for j in range(4)})
    groups = onc_clusters(Z, max_clusters=6, n_init=2)
    assert sorted(sorted(m) for m in groups.values()) == [[f"b{b}_{j}" for j in range(4)] for b in range(3)]


def test_placebos_are_rolled_copies():
    _, Z, _, _ = _data(300)
    P = make_placebos(Z, 3, np.random.default_rng(1))
    assert P.shape == (300, 3) and list(P.columns) == ["placebo_0", "placebo_1", "placebo_2"]
    for c in P:
        assert any(np.array_equal(np.sort(P[c]), np.sort(Z[s])) for s in Z)


def test_clustered_mda_selects_the_informative_cluster():
    tr, Z, y, folds = _data()
    rng = np.random.default_rng(0)
    groups = redundancy_clusters(Z, 0.3)
    P = make_placebos(Z, 3, rng)
    mda = clustered_mda(pd.concat([Z, P], axis=1), y, folds, {**groups, **{c: [c] for c in P}}, n_perm=2, rng=rng,
                        n_jobs=1)
    assert list(mda.columns[:2]) == ["_base", "_prior"] and len(mda) == len(folds)
    sel = select_clusters(mda, groups, list(P.columns), min_positive_folds=3)
    inf = [g for g, m in groups.items() if "f_inf" in m][0]
    assert inf in sel.selected and {"f_inf", "f_inf_copy"} <= set(sel.columns)
    assert sel.mean_importance.drop(list(P.columns)).idxmax() == inf
    assert sel.mean_importance[inf] > 10 * max(sel.placebo_max, 1e-4)
    # chunked prediction gives the same numbers
    rng2 = np.random.default_rng(0)
    P2 = make_placebos(Z, 3, rng2)
    mda2 = clustered_mda(pd.concat([Z, P2], axis=1), y, folds, {**groups, **{c: [c] for c in P2}}, n_perm=2, rng=rng2,
                         n_jobs=1, max_rows_per_predict=100)
    pd.testing.assert_frame_equal(mda, mda2)


def test_clustered_mdi_and_pca_check():
    _, Z, y, _ = _data(500)
    groups = redundancy_clusters(Z, 0.3)
    mdi = clustered_mdi(Z, y, groups, n_jobs=1)
    assert sum(mdi.values()) == pytest.approx(1.0)
    assert max(mdi, key=mdi.get) == [g for g, m in groups.items() if "f_inf" in m][0]
    res = pca_eigen_rank_check(Z, y, n_null=2, n_jobs=1)
    assert set(res) == {"n_components", "pc1_variance", "tau", "null_mean", "null_p95"}
