"""Tests for VPTreeCosineIndex and IVFFlatCosineIndex.

Both indices implement cosine distance via L2 normalisation. Returned distances
lie in [0, 2] (0 = identical direction, 1 = orthogonal, 2 = antiparallel).
The order convention matches VPTreeL2Index: results within the top-k are
sorted farthest → nearest (caller reverses if they need nearest-first).
"""

import numpy as np
import pickle
import pytest

import pynear


def _reference_topk_cosine(db: np.ndarray, queries: np.ndarray, k: int):
    """Brute-force cosine top-k. Returns (indices, distances) nearest-first."""
    db_n = db / np.linalg.norm(db, axis=1, keepdims=True)
    q_n = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    cos_dists = 1.0 - q_n @ db_n.T  # (Q, N), values in [0, 2]
    order = np.argsort(cos_dists, axis=1)[:, :k]
    sorted_dists = np.take_along_axis(cos_dists, order, axis=1)
    return order, sorted_dists


def _nearest_first(indices, distances):
    """Reverse pynear's farthest-first-within-top-k convention."""
    indices = np.array(indices, dtype=np.int64)[:, ::-1]
    distances = np.array(distances, dtype=np.float64)[:, ::-1]
    return indices, distances


@pytest.mark.parametrize("cls", [pynear.VPTreeCosineIndex])
def test_vptree_cosine_matches_brute_force(cls):
    rng = np.random.default_rng(42)
    db = rng.standard_normal((2000, 64)).astype(np.float32)
    queries = rng.standard_normal((20, 64)).astype(np.float32)
    k = 10

    idx = cls()
    idx.set(db)
    py_idx, py_dist = idx.searchKNN(queries, k=k)
    py_idx, py_dist = _nearest_first(py_idx, py_dist)

    ref_idx, ref_dist = _reference_topk_cosine(db, queries, k)
    np.testing.assert_array_equal(py_idx, ref_idx)
    np.testing.assert_allclose(py_dist, ref_dist, rtol=1e-5, atol=1e-6)


def test_vptree_cosine_distance_scale():
    """d_cos(u, u) == 0; d_cos(u, -u) == 2; d_cos(u, v⊥u) == 1."""
    db = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],   # same direction as query
            [-1.0, 0.0, 0.0, 0.0],  # antiparallel
            [0.0, 1.0, 0.0, 0.0],   # orthogonal
        ],
        dtype=np.float32,
    )
    q = np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # parallel to row 0

    idx = pynear.VPTreeCosineIndex()
    idx.set(db)
    indices, distances = idx.searchKNN(q, k=3)
    py_idx, py_dist = _nearest_first(indices, distances)

    assert py_idx.tolist() == [[0, 2, 1]]
    np.testing.assert_allclose(py_dist[0], [0.0, 1.0, 2.0], atol=1e-5)


def test_vptree_cosine_search1NN():
    rng = np.random.default_rng(7)
    db = rng.standard_normal((500, 32)).astype(np.float32)
    queries = rng.standard_normal((10, 32)).astype(np.float32)

    idx = pynear.VPTreeCosineIndex()
    idx.set(db)
    nn_idx, nn_dist = idx.search1NN(queries)

    ref_idx, ref_dist = _reference_topk_cosine(db, queries, 1)
    assert list(nn_idx) == [int(x) for x in ref_idx[:, 0]]
    np.testing.assert_allclose(np.asarray(nn_dist), ref_dist[:, 0], rtol=1e-5, atol=1e-6)


def test_vptree_cosine_pickle_round_trip():
    rng = np.random.default_rng(123)
    db = rng.standard_normal((300, 24)).astype(np.float32)
    queries = rng.standard_normal((5, 24)).astype(np.float32)

    idx = pynear.VPTreeCosineIndex()
    idx.set(db)
    i1, d1 = idx.searchKNN(queries, k=5)

    blob = pickle.dumps(idx)
    restored = pickle.loads(blob)
    i2, d2 = restored.searchKNN(queries, k=5)

    np.testing.assert_array_equal(np.array(i1), np.array(i2))
    np.testing.assert_allclose(np.array(d1), np.array(d2), rtol=1e-6)


def test_vptree_cosine_handles_zero_rows():
    """Zero-norm rows must not produce NaN; query returns finite distances."""
    db = np.vstack(
        [
            np.zeros((1, 8), dtype=np.float32),
            np.eye(8, dtype=np.float32),
        ]
    )
    q = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    idx = pynear.VPTreeCosineIndex()
    idx.set(db)
    _, distances = idx.searchKNN(q, k=3)
    dist_arr = np.asarray(distances[0])
    assert np.all(np.isfinite(dist_arr))


def test_ivfflat_cosine_exact_matches_brute_force():
    """With n_probe == n_clusters, IVFFlatCosineIndex must equal brute force."""
    rng = np.random.default_rng(7)
    db = rng.standard_normal((1000, 32)).astype(np.float32)
    queries = rng.standard_normal((10, 32)).astype(np.float32)
    k = 5

    ivf = pynear.IVFFlatCosineIndex(n_clusters=10, n_probe=10)
    ivf.set(db)
    py_idx, py_dist = ivf.searchKNN(queries, k=k)
    py_idx = np.array(py_idx, dtype=np.int64)
    py_dist = np.array(py_dist, dtype=np.float64)

    # IVFFlat returns nearest-first (different convention from VPTree).
    ref_idx, ref_dist = _reference_topk_cosine(db, queries, k)
    np.testing.assert_array_equal(py_idx, ref_idx)
    np.testing.assert_allclose(py_dist, ref_dist, rtol=1e-5, atol=1e-6)


def test_ivfflat_cosine_approximate_recall():
    """Approximate IVF should achieve ≥ 90% recall@10 on uniform Gaussian data."""
    rng = np.random.default_rng(99)
    db = rng.standard_normal((5000, 48)).astype(np.float32)
    queries = rng.standard_normal((30, 48)).astype(np.float32)
    k = 10

    # n_probe=32 of 64 = 50% probing rate. On uniform Gaussian data
    # (worst case — no natural cluster structure), this gives ~0.91 recall.
    # Real text-embedding data clusters more cleanly and reaches higher recall
    # at lower probing rates.
    ivf = pynear.IVFFlatCosineIndex(n_clusters=64, n_probe=32)
    ivf.set(db)
    py_idx, _ = ivf.searchKNN(queries, k=k)
    py_idx = np.array(py_idx, dtype=np.int64)

    ref_idx, _ = _reference_topk_cosine(db, queries, k)

    recall = 0.0
    for q in range(len(queries)):
        recall += len(set(py_idx[q].tolist()) & set(ref_idx[q].tolist())) / k
    recall /= len(queries)
    assert recall >= 0.90, f"recall@10 = {recall:.3f}"


def test_sklearn_adapter_cosine():
    """metric='cosine' must route to VPTreeCosineIndex and rank correctly."""
    pytest.importorskip("sklearn", reason="scikit-learn not installed")
    from pynear.sklearn_adapter import PyNearNearestNeighbors

    rng = np.random.default_rng(0)
    db = rng.standard_normal((500, 16)).astype(np.float32)
    queries = rng.standard_normal((5, 16)).astype(np.float32)
    k = 5

    nn = PyNearNearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(db)
    distances, indices = nn.kneighbors(queries)

    ref_idx, ref_dist = _reference_topk_cosine(db, queries, k)
    # Adapter is expected to expose nearest-first (it reverses internally).
    np.testing.assert_array_equal(np.asarray(indices), ref_idx)
    np.testing.assert_allclose(np.asarray(distances), ref_dist, rtol=1e-5, atol=1e-6)
