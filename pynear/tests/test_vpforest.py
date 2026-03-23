"""
Tests for VPForest (IVF-style) indices.

Correctness strategy: for each query we check that every result returned by
the forest is also in the brute-force top-k.  We allow a small recall gap
because n_probe < n_clusters is intentionally approximate; at n_probe ==
n_clusters the result must be exact.
"""

import pickle

import numpy as np
import pytest

import pynear
from pynear import VPForestL2Index, VPForestL1Index, VPForestChebyshevIndex
from pynear import VPTreeL2Index


# ── helpers ────────────────────────────────────────────────────────────────────


def brute_knn_l2(data, queries, k):
    """Exact L2 KNN via brute force."""
    results_idx, results_dist = [], []
    for q in queries:
        dists = np.sum((data - q) ** 2, axis=1)
        idx = np.argsort(dists)[:k]
        results_idx.append(idx.tolist())
        results_dist.append(dists[idx].tolist())
    return results_idx, results_dist


def recall(forest_indices, exact_indices):
    """Mean per-query recall@k."""
    total = sum(
        len(set(fi) & set(ei)) / max(len(ei), 1)
        for fi, ei in zip(forest_indices, exact_indices)
    )
    return total / len(forest_indices)


# ── basic correctness ──────────────────────────────────────────────────────────


def test_exact_when_n_probe_equals_n_clusters():
    """n_probe == n_clusters must give exact results."""
    rng = np.random.default_rng(0)
    data = rng.random((500, 32)).astype(np.float32)
    queries = rng.random((20, 32)).astype(np.float32)
    k = 5

    index = VPForestL2Index(n_clusters=20, n_probe=20)
    index.set(data)

    forest_idx, forest_dists = index.searchKNN(queries, k)
    exact_idx, exact_dists = brute_knn_l2(data, queries, k)

    assert recall(forest_idx, exact_idx) == pytest.approx(1.0)


def test_high_recall_with_reasonable_n_probe():
    """n_probe=15 out of 20 clusters on low-dim data should give ≥ 90 % recall."""
    rng = np.random.default_rng(1)
    data = rng.random((2000, 8)).astype(np.float32)
    queries = rng.random((50, 8)).astype(np.float32)
    k = 10

    index = VPForestL2Index(n_clusters=20, n_probe=15)
    index.set(data)

    forest_idx, _ = index.searchKNN(queries, k)
    exact_idx, _ = brute_knn_l2(data, queries, k)

    assert recall(forest_idx, exact_idx) >= 0.90


def test_search1nn_matches_searchknn():
    rng = np.random.default_rng(2)
    data = rng.random((500, 16)).astype(np.float32)
    queries = rng.random((10, 16)).astype(np.float32)

    index = VPForestL2Index(n_clusters=10, n_probe=10)
    index.set(data)

    idx1, dist1 = index.search1NN(queries)
    idxk, distk = index.searchKNN(queries, 1)

    assert idx1 == idxk
    assert dist1 == distk


def test_single_query_vector():
    """1-D query (not batched) should not crash."""
    rng = np.random.default_rng(3)
    data = rng.random((200, 8)).astype(np.float32)
    index = VPForestL2Index(n_clusters=10, n_probe=5)
    index.set(data)

    q = rng.random(8).astype(np.float32)
    idx, dist = index.searchKNN(q, 3)
    assert len(idx) == 1
    assert len(idx[0]) == 3


def test_k_larger_than_cluster_size():
    """k can exceed a single cluster's size — results come from multiple clusters."""
    rng = np.random.default_rng(4)
    data = rng.random((50, 4)).astype(np.float32)
    index = VPForestL2Index(n_clusters=20, n_probe=20)
    index.set(data)

    idx, dist = index.searchKNN(rng.random((1, 4)).astype(np.float32), k=10)
    assert len(idx[0]) == 10


def test_fewer_points_than_clusters():
    """n_clusters is silently capped at N."""
    rng = np.random.default_rng(5)
    data = rng.random((10, 4)).astype(np.float32)
    index = VPForestL2Index(n_clusters=100, n_probe=100)
    index.set(data)
    assert index.n_clusters <= 10


# ── high-dimensional smoke test ────────────────────────────────────────────────


def test_high_dim_1024():
    """Smoke test for 1024-D image-embedding-like data."""
    rng = np.random.default_rng(6)
    data = rng.random((2000, 1024)).astype(np.float32)
    queries = rng.random((5, 1024)).astype(np.float32)

    index = VPForestL2Index(n_clusters=40, n_probe=10)
    index.set(data)

    idx, dist = index.searchKNN(queries, k=5)
    assert len(idx) == 5
    assert all(len(r) == 5 for r in idx)


# ── metric variants ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("IndexClass", [VPForestL1Index, VPForestChebyshevIndex])
def test_metric_variants_run(IndexClass):
    rng = np.random.default_rng(7)
    data = rng.random((300, 16)).astype(np.float32)
    queries = rng.random((5, 16)).astype(np.float32)

    index = IndexClass(n_clusters=10, n_probe=10)
    index.set(data)
    idx, dist = index.searchKNN(queries, k=3)
    assert len(idx) == 5


# ── pickle ─────────────────────────────────────────────────────────────────────


def test_pickle_roundtrip():
    rng = np.random.default_rng(8)
    data = rng.random((500, 32)).astype(np.float32)
    queries = rng.random((10, 32)).astype(np.float32)
    k = 5

    index = VPForestL2Index(n_clusters=20, n_probe=20)
    index.set(data)

    idx_before, dist_before = index.searchKNN(queries, k)

    blob = pickle.dumps(index)
    index2 = pickle.loads(blob)

    idx_after, dist_after = index2.searchKNN(queries, k)

    assert idx_before == idx_after
    np.testing.assert_allclose(dist_before, dist_after)
