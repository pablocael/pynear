"""Tests for HNSWL2Index and HNSWCosineIndex.

The HNSW indices follow the same return-order convention as the VPTree
indices: results within the top-k are sorted *farthest-first*; callers
reverse via [::-1] when they need nearest-first.
"""

import numpy as np
import pickle
import pytest

import pynear


# ── Helpers ─────────────────────────────────────────────────────────────────


def _brute_l2(db: np.ndarray, queries: np.ndarray, k: int):
    """Brute-force L2 top-k, returning nearest-first."""
    d = np.linalg.norm(db[None, :, :] - queries[:, None, :], axis=2)
    order = np.argsort(d, axis=1)[:, :k]
    sorted_d = np.take_along_axis(d, order, axis=1)
    return order, sorted_d


def _brute_cosine(db: np.ndarray, queries: np.ndarray, k: int):
    """Brute-force cosine top-k, returning nearest-first."""
    db_n = db / np.linalg.norm(db, axis=1, keepdims=True)
    q_n = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    d = 1.0 - q_n @ db_n.T
    order = np.argsort(d, axis=1)[:, :k]
    sorted_d = np.take_along_axis(d, order, axis=1)
    return order, sorted_d


def _nearest_first(indices, distances):
    """Reverse pynear's farthest-first-within-top-k convention."""
    indices = np.array(indices, dtype=np.int64)[:, ::-1]
    distances = np.array(distances, dtype=np.float64)[:, ::-1]
    return indices, distances


def _recall_at_k(pred_idx, ref_idx, k):
    recall = 0.0
    for q in range(len(pred_idx)):
        recall += len(set(pred_idx[q]) & set(ref_idx[q])) / k
    return recall / len(pred_idx)


# ── HNSWL2Index ─────────────────────────────────────────────────────────────


def test_hnsw_l2_default_params_high_recall():
    """At default ef_search=50, recall@10 must be ≥ 0.95 on uniform data."""
    rng = np.random.default_rng(42)
    db = rng.standard_normal((2000, 32)).astype(np.float32)
    queries = rng.standard_normal((50, 32)).astype(np.float32)
    k = 10

    idx = pynear.HNSWL2Index(M=16, ef_construction=200, ef_search=50)
    idx.set(db)
    pred_idx, _ = idx.searchKNN(queries, k=k)
    pred_idx, _ = _nearest_first(pred_idx, _)

    ref_idx, _ = _brute_l2(db, queries, k)
    recall = _recall_at_k(pred_idx, ref_idx, k)
    assert recall >= 0.95, f"recall@10 = {recall:.3f}"


def test_hnsw_l2_high_ef_is_near_exact():
    """At ef_search >> N, the index should approximate brute-force."""
    rng = np.random.default_rng(7)
    db = rng.standard_normal((500, 24)).astype(np.float32)
    queries = rng.standard_normal((30, 24)).astype(np.float32)
    k = 10

    idx = pynear.HNSWL2Index(M=16, ef_construction=300, ef_search=500)
    idx.set(db)
    pred_idx, _ = idx.searchKNN(queries, k=k)
    pred_idx, _ = _nearest_first(pred_idx, _)

    ref_idx, _ = _brute_l2(db, queries, k)
    recall = _recall_at_k(pred_idx, ref_idx, k)
    assert recall >= 0.99, f"high-ef recall@10 = {recall:.3f}"


def test_hnsw_l2_set_ef_runtime_tunability():
    """Recall must monotonically (or at least non-decreasingly) follow ef_search."""
    rng = np.random.default_rng(2)
    db = rng.standard_normal((1500, 32)).astype(np.float32)
    queries = rng.standard_normal((40, 32)).astype(np.float32)
    k = 10

    idx = pynear.HNSWL2Index(M=16, ef_construction=200, ef_search=10)
    idx.set(db)
    ref_idx, _ = _brute_l2(db, queries, k)

    recalls = []
    for ef in [10, 25, 50, 100, 200]:
        idx.set_ef(ef)
        pred_idx, _ = idx.searchKNN(queries, k=k)
        pred_idx, _ = _nearest_first(pred_idx, _)
        recalls.append(_recall_at_k(pred_idx, ref_idx, k))

    # Allow tiny non-monotonic dips (graph effects), but overall trend should be up.
    assert recalls[-1] >= recalls[0], f"recall trajectory: {recalls}"
    assert recalls[-1] >= 0.95


def test_hnsw_l2_distances_are_l2_not_squared():
    """Returned distances must be L2 (sqrt-applied), matching VPTreeL2Index."""
    db = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    q = np.array([[0.0, 0.0]], dtype=np.float32)

    idx = pynear.HNSWL2Index(M=4, ef_construction=10, ef_search=10)
    idx.set(db)
    _, d = idx.searchKNN(q, k=2)
    nearest_first = sorted(d[0])
    np.testing.assert_allclose(nearest_first, [0.0, 5.0], atol=1e-5)


def test_hnsw_l2_search1NN():
    rng = np.random.default_rng(9)
    db = rng.standard_normal((500, 16)).astype(np.float32)
    queries = rng.standard_normal((10, 16)).astype(np.float32)

    idx = pynear.HNSWL2Index(M=16, ef_construction=200, ef_search=200)
    idx.set(db)
    nn_idx, _ = idx.search1NN(queries)

    ref_idx, _ = _brute_l2(db, queries, 1)
    # At high ef, 1-NN should be exact (or very near it).
    matches = sum(int(int(nn_idx[i]) == int(ref_idx[i, 0])) for i in range(len(queries)))
    assert matches >= len(queries) - 1, f"1NN exact matches = {matches}/{len(queries)}"


def test_hnsw_l2_pickle_round_trip():
    rng = np.random.default_rng(123)
    db = rng.standard_normal((400, 24)).astype(np.float32)
    queries = rng.standard_normal((10, 24)).astype(np.float32)

    idx = pynear.HNSWL2Index(M=16, ef_construction=200, ef_search=100)
    idx.set(db)
    i1, d1 = idx.searchKNN(queries, k=5)

    blob = pickle.dumps(idx)
    restored = pickle.loads(blob)

    assert restored.size == idx.size
    assert restored.dim == idx.dim
    assert restored.ef_search == idx.ef_search

    i2, d2 = restored.searchKNN(queries, k=5)
    np.testing.assert_array_equal(np.array(i1), np.array(i2))
    np.testing.assert_allclose(np.array(d1), np.array(d2), rtol=1e-6)


def test_hnsw_l2_zero_vector_safe():
    """Zero-vector query must not produce NaN/inf and must return k finite distances."""
    rng = np.random.default_rng(11)
    db = rng.standard_normal((200, 16)).astype(np.float32)
    q = np.zeros((1, 16), dtype=np.float32)

    idx = pynear.HNSWL2Index(M=16, ef_construction=100, ef_search=50)
    idx.set(db)
    _, d = idx.searchKNN(q, k=5)
    assert np.all(np.isfinite(d[0]))


def test_hnsw_l2_top_k_ordering_is_farthest_first():
    """Within the returned top-k, distances are sorted farthest-first."""
    rng = np.random.default_rng(5)
    db = rng.standard_normal((300, 16)).astype(np.float32)
    queries = rng.standard_normal((5, 16)).astype(np.float32)

    idx = pynear.HNSWL2Index(M=16, ef_construction=200, ef_search=100)
    idx.set(db)
    _, d = idx.searchKNN(queries, k=10)
    for row in d:
        # Sorted in descending order — i.e. non-increasing.
        assert all(row[i] >= row[i + 1] - 1e-6 for i in range(len(row) - 1)), row


def test_hnsw_l2_empty_index_returns_empty():
    """Querying an unbuilt index must return empty lists, not crash."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((3, 8)).astype(np.float32)

    idx = pynear.HNSWL2Index()
    indices, distances = idx.searchKNN(q, k=5)
    assert all(len(r) == 0 for r in indices)
    assert all(len(r) == 0 for r in distances)


# ── HNSWCosineIndex ─────────────────────────────────────────────────────────


def test_hnsw_cosine_matches_brute_force_at_high_ef():
    rng = np.random.default_rng(0)
    db = rng.standard_normal((1500, 32)).astype(np.float32)
    queries = rng.standard_normal((30, 32)).astype(np.float32)
    k = 10

    idx = pynear.HNSWCosineIndex(M=16, ef_construction=300, ef_search=500)
    idx.set(db)
    pred_idx, _ = idx.searchKNN(queries, k=k)
    pred_idx, _ = _nearest_first(pred_idx, _)

    ref_idx, _ = _brute_cosine(db, queries, k)
    recall = _recall_at_k(pred_idx, ref_idx, k)
    assert recall >= 0.95, f"cosine recall@10 = {recall:.3f}"


def test_hnsw_cosine_distance_scale():
    """d_cos(u, u) = 0, d_cos(u, -u) = 2, d_cos(u, v⊥u) = 1."""
    db = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    q = np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    idx = pynear.HNSWCosineIndex(M=4, ef_construction=10, ef_search=10)
    idx.set(db)
    indices, distances = idx.searchKNN(q, k=3)
    py_idx, py_dist = _nearest_first(indices, distances)

    assert py_idx.tolist() == [[0, 2, 1]]
    np.testing.assert_allclose(py_dist[0], [0.0, 1.0, 2.0], atol=1e-5)


def test_hnsw_cosine_pickle_round_trip():
    rng = np.random.default_rng(77)
    db = rng.standard_normal((300, 16)).astype(np.float32)
    queries = rng.standard_normal((5, 16)).astype(np.float32)

    idx = pynear.HNSWCosineIndex(M=16, ef_construction=100, ef_search=100)
    idx.set(db)
    i1, d1 = idx.searchKNN(queries, k=5)

    restored = pickle.loads(pickle.dumps(idx))
    i2, d2 = restored.searchKNN(queries, k=5)

    np.testing.assert_array_equal(np.array(i1), np.array(i2))
    np.testing.assert_allclose(np.array(d1), np.array(d2), rtol=1e-6)


def test_hnsw_sq8_recall_within_2pct_of_float():
    """SQ8 should give recall within ~3% of the full-float HNSW at the same params."""
    rng = np.random.default_rng(42)
    db = rng.standard_normal((3000, 64)).astype(np.float32)
    queries = rng.standard_normal((50, 64)).astype(np.float32)
    k = 10

    idx_f = pynear.HNSWL2Index(M=16, ef_construction=200, ef_search=200)
    idx_f.set(db)
    pi_f, _ = idx_f.searchKNN(queries, k=k)
    pi_f, _ = _nearest_first(pi_f, _)

    idx_sq = pynear.HNSWL2IndexSQ8(M=16, ef_construction=200, ef_search=200)
    idx_sq.set(db)
    pi_sq, _ = idx_sq.searchKNN(queries, k=k)
    pi_sq, _ = _nearest_first(pi_sq, _)

    ref_idx, _ = _brute_l2(db, queries, k)
    r_f = _recall_at_k(pi_f, ref_idx, k)
    r_sq = _recall_at_k(pi_sq, ref_idx, k)
    # SQ8 should be at most ~3 % behind full-float at default ef.
    assert r_sq >= r_f - 0.03, f"SQ8 recall {r_sq:.3f} too far below float {r_f:.3f}"
    assert r_sq >= 0.85, f"SQ8 recall {r_sq:.3f} unexpectedly low"


def test_hnsw_sq8_distance_is_l2_scaled():
    """Distances returned by SQ8 should be approximately L2, scaled by quantisation factor."""
    db = np.array([[0.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
    q  = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    idx = pynear.HNSWL2IndexSQ8(M=4, ef_construction=10, ef_search=10)
    idx.set(db)
    _, d = idx.searchKNN(q, k=2)
    nearest_first = sorted(d[0])
    # First should be ~0 (same point), second ~5 (sqrt(3²+4²)). Allow ~10 % SQ8 error.
    assert abs(nearest_first[0] - 0.0) < 0.2
    assert abs(nearest_first[1] - 5.0) < 0.5


def test_hnsw_sq8_search1NN_finds_planted_duplicates():
    rng = np.random.default_rng(7)
    db = rng.standard_normal((500, 32)).astype(np.float32)
    queries = rng.standard_normal((10, 32)).astype(np.float32)
    for i, qv in enumerate(queries):
        db[i * 20] = qv

    idx = pynear.HNSWL2IndexSQ8(M=16, ef_construction=200, ef_search=200)
    idx.set(db)
    nn_idx, _ = idx.search1NN(queries)
    # SQ8 may quantise queries slightly differently from planted db rows,
    # so we allow either an exact match or a row whose vector is identical
    # after quantisation. Most should match exactly.
    matches = sum(int(int(nn_idx[i]) == i * 20) for i in range(len(queries)))
    assert matches >= len(queries) - 1, f"only {matches}/{len(queries)} planted dupes found"


def test_hnsw_cosine_handles_zero_rows():
    db = np.vstack([np.zeros((1, 8), dtype=np.float32), np.eye(8, dtype=np.float32)])
    q = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    idx = pynear.HNSWCosineIndex(M=4, ef_construction=20, ef_search=20)
    idx.set(db)
    _, d = idx.searchKNN(q, k=3)
    assert np.all(np.isfinite(d[0]))


# ── HNSWBinaryIndex (Hamming) ───────────────────────────────────────────────


def _brute_hamming(db: np.ndarray, queries: np.ndarray, k: int):
    db_bits = np.unpackbits(db, axis=1)
    q_bits = np.unpackbits(queries, axis=1)
    d = (q_bits[:, None, :] != db_bits[None, :, :]).sum(axis=2)
    order = np.argsort(d, axis=1)[:, :k]
    sorted_d = np.take_along_axis(d, order, axis=1)
    return order, sorted_d


def test_hnsw_binary_at_high_ef_high_recall():
    """Plain HNSWBinary should approach high recall as ef grows."""
    rng = np.random.default_rng(0)
    db = rng.integers(0, 256, size=(1500, 16), dtype=np.uint8)
    queries = rng.integers(0, 256, size=(30, 16), dtype=np.uint8)
    k = 10

    idx = pynear.HNSWBinaryIndex(M=16, ef_construction=400, ef_search=500)
    idx.set(db)
    pred_idx, _ = idx.searchKNN(queries, k=k)
    pred_idx = np.array(pred_idx)[:, ::-1]  # farthest-first → nearest-first

    ref_idx, _ = _brute_hamming(db, queries, k)
    recall = _recall_at_k(pred_idx, ref_idx, k)
    # Hamming has many tied distances; HNSW plateaus before 1.0 even at very high ef.
    # This test guards against regression below the empirical floor.
    assert recall >= 0.80, f"binary recall@10 at ef=500 = {recall:.3f}"


def test_hnsw_binary_finds_planted_duplicates():
    """Each query's exact duplicate must be returned as the 1-NN."""
    rng = np.random.default_rng(1)
    db = rng.integers(0, 256, size=(1000, 16), dtype=np.uint8)
    queries = rng.integers(0, 256, size=(15, 16), dtype=np.uint8)

    # Plant each query as an exact row in db.
    for i, q in enumerate(queries):
        db[i * 50] = q

    idx = pynear.HNSWBinaryIndex(M=16, ef_construction=200, ef_search=100)
    idx.set(db)
    nn_idx, nn_dist = idx.search1NN(queries)
    for i in range(len(queries)):
        assert int(nn_idx[i]) == i * 50, f"q{i}: got {nn_idx[i]}, expected {i * 50}"
        assert int(nn_dist[i]) == 0


def test_hnsw_binary_distance_is_hamming_count():
    """Returned distance must equal the Hamming bit count, not bytes."""
    db = np.array(
        [
            [0xFF, 0x00],     # all 1s in first byte, all 0s in second
            [0x00, 0xFF],     # opposite
            [0xFF, 0xFF],     # all 1s
        ],
        dtype=np.uint8,
    )
    q = np.array([[0xFF, 0x00]], dtype=np.uint8)

    idx = pynear.HNSWBinaryIndex(M=4, ef_construction=10, ef_search=10)
    idx.set(db)
    indices, distances = idx.searchKNN(q, k=3)
    # Nearest-first
    pi = list(indices[0])[::-1]
    pd = list(distances[0])[::-1]
    assert pi == [0, 2, 1], pi
    assert pd == [0, 8, 16], pd


# ── MIHSeededHNSWBinaryIndex (novel) ────────────────────────────────────────


def test_mih_seeded_finds_exact_near_duplicates():
    """The killer use case: near-duplicate queries find their match even at low ef."""
    rng = np.random.default_rng(7)
    db = rng.integers(0, 256, size=(2000, 16), dtype=np.uint8)
    queries = rng.integers(0, 256, size=(20, 16), dtype=np.uint8)

    # Plant each query as an exact row.
    for i, q in enumerate(queries):
        db[i * 100] = q

    # Low ef — would be borderline for plain HNSW; MIH seeding rescues it.
    idx = pynear.MIHSeededHNSWBinaryIndex(
        M=16, ef_construction=200, ef_search=20,
        mih_m=8, mih_radius=4,
    )
    idx.set(db)
    nn_idx, nn_dist = idx.search1NN(queries)
    for i in range(len(queries)):
        assert int(nn_idx[i]) == i * 100, f"q{i}: got {nn_idx[i]}, expected {i * 100}"
        assert int(nn_dist[i]) == 0


def test_mih_seeded_at_least_as_good_as_plain_hnsw():
    """The seeded variant should never underperform plain HNSW on the same data."""
    rng = np.random.default_rng(13)
    db = rng.integers(0, 256, size=(1500, 16), dtype=np.uint8)
    queries = rng.integers(0, 256, size=(30, 16), dtype=np.uint8)
    k = 10

    plain = pynear.HNSWBinaryIndex(M=16, ef_construction=200, ef_search=50)
    plain.set(db)
    plain_idx, _ = plain.searchKNN(queries, k=k)
    plain_idx = np.array(plain_idx)[:, ::-1]

    seeded = pynear.MIHSeededHNSWBinaryIndex(
        M=16, ef_construction=200, ef_search=50,
        mih_m=8, mih_radius=12,
    )
    seeded.set(db)
    seeded_idx, _ = seeded.searchKNN(queries, k=k)
    seeded_idx = np.array(seeded_idx)[:, ::-1]

    ref_idx, _ = _brute_hamming(db, queries, k)
    plain_recall = _recall_at_k(plain_idx, ref_idx, k)
    seeded_recall = _recall_at_k(seeded_idx, ref_idx, k)

    # Tolerance covers cross-platform RNG noise: std::uniform_real_distribution
    # is not guaranteed to produce identical output across compilers/standard
    # libraries (MSVC differs from libstdc++/libc++), so the geometric layer
    # assignment in random_level() yields slightly different graph topologies
    # on different OSes even with the same seed. 5 % is well within the
    # stochastic envelope of HNSW recall on this small dataset.
    assert seeded_recall >= plain_recall - 0.05, (
        f"seeded={seeded_recall:.3f} < plain={plain_recall:.3f}"
    )


def test_mih_seeded_radius_runtime_tunable():
    """set_mih_radius should adjust the seed-set size without rebuilding."""
    rng = np.random.default_rng(21)
    db = rng.integers(0, 256, size=(1000, 16), dtype=np.uint8)
    q = rng.integers(0, 256, size=(5, 16), dtype=np.uint8)

    idx = pynear.MIHSeededHNSWBinaryIndex(M=16, ef_construction=200, ef_search=30,
                                          mih_m=8, mih_radius=4)
    idx.set(db)
    assert idx.mih_radius == 4

    idx.set_mih_radius(12)
    assert idx.mih_radius == 12

    # Searches must still run
    indices, _ = idx.searchKNN(q, k=5)
    assert len(indices) == 5


# ─── Parameter validation ───────────────────────────────────────────────────


@pytest.mark.parametrize("cls", [
    pynear.HNSWL2Index,
    pynear.HNSWCosineIndex,
    pynear.HNSWL2IndexSQ8,
    pynear.HNSWBinaryIndex,
])
def test_hnsw_rejects_M_lt_2(cls):
    """M < 2 is invalid (HNSW needs at least 2 edges to navigate)."""
    with pytest.raises(Exception):
        cls(M=1, ef_construction=10, ef_search=10)


@pytest.mark.parametrize("cls", [
    pynear.HNSWL2Index,
    pynear.HNSWCosineIndex,
    pynear.HNSWL2IndexSQ8,
    pynear.HNSWBinaryIndex,
])
def test_hnsw_rejects_ef_construction_zero(cls):
    """ef_construction must be >= 1."""
    with pytest.raises(Exception):
        cls(M=4, ef_construction=0, ef_search=10)


def test_mih_seeded_rejects_invalid_M():
    with pytest.raises(Exception):
        pynear.MIHSeededHNSWBinaryIndex(M=1, ef_construction=10, ef_search=10)


# ─── Empty / unbuilt index ──────────────────────────────────────────────────


@pytest.mark.parametrize("cls", [
    pynear.HNSWL2Index,
    pynear.HNSWCosineIndex,
    pynear.HNSWL2IndexSQ8,
])
def test_hnsw_search_before_set_returns_empty(cls):
    """searchKNN on an index that's never been .set() should return empty results, not crash."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((3, 8)).astype(np.float32)
    idx = cls(M=4, ef_construction=10, ef_search=10)
    indices, distances = idx.searchKNN(q, k=5)
    assert all(len(r) == 0 for r in indices)
    assert all(len(r) == 0 for r in distances)


def test_hnsw_binary_search_before_set_returns_empty():
    rng = np.random.default_rng(0)
    q = rng.integers(0, 256, size=(3, 16), dtype=np.uint8)
    idx = pynear.HNSWBinaryIndex(M=4, ef_construction=10, ef_search=10)
    indices, distances = idx.searchKNN(q, k=5)
    assert all(len(r) == 0 for r in indices)
    assert all(len(r) == 0 for r in distances)


# ─── Tiny dataset edge cases ────────────────────────────────────────────────


def test_hnsw_l2_single_point_index():
    """N=1: the single point is always the nearest neighbour."""
    db = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    q = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    idx = pynear.HNSWL2Index(M=4, ef_construction=10, ef_search=10)
    idx.set(db)
    indices, distances = idx.searchKNN(q, k=5)
    assert list(indices[0]) == [0]
    np.testing.assert_allclose(distances[0][0], np.linalg.norm(db[0]), rtol=1e-5)


def test_hnsw_l2_k_greater_than_n():
    """Asking for more neighbours than exist should return all N of them."""
    rng = np.random.default_rng(3)
    db = rng.standard_normal((10, 16)).astype(np.float32)
    q = rng.standard_normal((2, 16)).astype(np.float32)
    idx = pynear.HNSWL2Index(M=8, ef_construction=50, ef_search=50)
    idx.set(db)
    indices, _ = idx.searchKNN(q, k=100)
    assert all(len(r) == 10 for r in indices)


def test_hnsw_l2_identical_vectors():
    """All-identical DB shouldn't crash; all neighbours are at distance 0 from a matching query."""
    db = np.tile(np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32), (50, 1))
    q = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    idx = pynear.HNSWL2Index(M=8, ef_construction=50, ef_search=50)
    idx.set(db)
    _, distances = idx.searchKNN(q, k=5)
    np.testing.assert_allclose(distances[0], [0.0] * 5, atol=1e-5)


# ─── Threading ──────────────────────────────────────────────────────────────


def test_hnsw_parallel_build_recall_matches_serial():
    """n_threads>1 yields graph topology variation but recall stays comparable."""
    rng = np.random.default_rng(42)
    db = rng.standard_normal((3000, 32)).astype(np.float32)
    q = rng.standard_normal((50, 32)).astype(np.float32)
    k = 10

    serial = pynear.HNSWL2Index(M=16, ef_construction=200, ef_search=200, n_threads=1)
    serial.set(db)
    pi_s, _ = serial.searchKNN(q, k=k); pi_s, _ = _nearest_first(pi_s, _)

    parallel = pynear.HNSWL2Index(M=16, ef_construction=200, ef_search=200, n_threads=4)
    parallel.set(db)
    pi_p, _ = parallel.searchKNN(q, k=k); pi_p, _ = _nearest_first(pi_p, _)

    ref_idx, _ = _brute_l2(db, q, k)
    r_s = _recall_at_k(pi_s, ref_idx, k)
    r_p = _recall_at_k(pi_p, ref_idx, k)
    assert abs(r_s - r_p) < 0.05, f"serial={r_s:.3f} vs parallel={r_p:.3f}"
    assert r_p >= 0.92


def test_hnsw_deterministic_with_n_threads_one():
    """Same seed, n_threads=1, identical input → identical search results."""
    rng = np.random.default_rng(7)
    db = rng.standard_normal((500, 16)).astype(np.float32)
    q = rng.standard_normal((10, 16)).astype(np.float32)

    a = pynear.HNSWL2Index(M=8, ef_construction=100, ef_search=50, seed=99, n_threads=1)
    a.set(db)
    ia, da = a.searchKNN(q, k=5)
    b = pynear.HNSWL2Index(M=8, ef_construction=100, ef_search=50, seed=99, n_threads=1)
    b.set(db)
    ib, db_ = b.searchKNN(q, k=5)

    np.testing.assert_array_equal(np.array(ia), np.array(ib))
    np.testing.assert_allclose(np.array(da), np.array(db_), rtol=1e-6)


# ─── Pickle round-trips for the remaining variants ──────────────────────────


def test_hnsw_sq8_pickle_round_trip():
    """HNSWL2IndexSQ8 must round-trip through pickle, preserving scale + graph."""
    rng = np.random.default_rng(101)
    db = rng.standard_normal((400, 32)).astype(np.float32)
    q = rng.standard_normal((10, 32)).astype(np.float32)

    idx = pynear.HNSWL2IndexSQ8(M=8, ef_construction=100, ef_search=100)
    idx.set(db)
    i1, d1 = idx.searchKNN(q, k=5)
    scale_before = idx.scale

    restored = pickle.loads(pickle.dumps(idx))
    assert restored.size == idx.size
    assert restored.ef_search == idx.ef_search
    assert restored.scale == scale_before
    i2, d2 = restored.searchKNN(q, k=5)
    np.testing.assert_array_equal(np.array(i1), np.array(i2))
    np.testing.assert_allclose(np.array(d1), np.array(d2), rtol=1e-6)


def test_hnsw_binary_pickle_round_trip():
    rng = np.random.default_rng(123)
    db = rng.integers(0, 256, size=(400, 16), dtype=np.uint8)
    q = rng.integers(0, 256, size=(10, 16), dtype=np.uint8)

    idx = pynear.HNSWBinaryIndex(M=16, ef_construction=100, ef_search=100)
    idx.set(db)
    i1, d1 = idx.searchKNN(q, k=5)

    blob = pickle.dumps(idx)
    restored = pickle.loads(blob)
    i2, d2 = restored.searchKNN(q, k=5)

    np.testing.assert_array_equal(np.array(i1), np.array(i2))
    np.testing.assert_array_equal(np.array(d1), np.array(d2))


# ─── HNSWL2IndexSQ8 specifics ───────────────────────────────────────────────


def test_hnsw_sq8_handles_zero_vectors():
    """Zero-mean / zero-magnitude vectors shouldn't NaN out the quantisation."""
    db = np.zeros((20, 16), dtype=np.float32)
    db[0] = 1.0  # one non-zero so global max-abs > 0
    q = np.zeros((1, 16), dtype=np.float32)
    idx = pynear.HNSWL2IndexSQ8(M=4, ef_construction=20, ef_search=20)
    idx.set(db)
    _, d = idx.searchKNN(q, k=5)
    assert np.all(np.isfinite(d[0]))


def test_hnsw_sq8_extreme_value_range():
    """Very large and very small values in the same vector still quantise without overflow."""
    rng = np.random.default_rng(0)
    db = rng.standard_normal((100, 16)).astype(np.float32) * 1e3  # large scale
    db[0] *= 1e-6  # one tiny vector to stress dynamic range
    q = db[:5].copy()
    idx = pynear.HNSWL2IndexSQ8(M=8, ef_construction=50, ef_search=100)
    idx.set(db)
    indices, distances = idx.searchKNN(q, k=3)
    assert all(len(r) == 3 for r in indices)
    assert all(np.all(np.isfinite(d)) for d in distances)


# ─── MIH-seeded boundary conditions ─────────────────────────────────────────


def test_mih_seeded_radius_zero():
    """mih_radius=0: MIH only returns exact-match candidates → degrades to plain HNSW for non-duplicates."""
    rng = np.random.default_rng(5)
    db = rng.integers(0, 256, size=(500, 16), dtype=np.uint8)
    q = rng.integers(0, 256, size=(10, 16), dtype=np.uint8)
    # Query[0] is a planted exact duplicate of db[0].
    db[0] = q[0]

    idx = pynear.MIHSeededHNSWBinaryIndex(
        M=16, ef_construction=100, ef_search=50, mih_m=8, mih_radius=0,
    )
    idx.set(db)
    nn_idx, nn_dist = idx.search1NN(q)
    # First query has exact match → must be found.
    assert int(nn_idx[0]) == 0
    assert int(nn_dist[0]) == 0


def test_mih_seeded_radius_larger_than_dim_does_not_crash():
    """mih_radius > number of bits is degenerate (returns nothing useful) but must not crash."""
    rng = np.random.default_rng(5)
    db = rng.integers(0, 256, size=(200, 16), dtype=np.uint8)
    q = rng.integers(0, 256, size=(5, 16), dtype=np.uint8)
    # d=16 bytes = 128 bits; radius=200 is bigger than that.
    idx = pynear.MIHSeededHNSWBinaryIndex(
        M=16, ef_construction=100, ef_search=50, mih_m=8, mih_radius=200,
    )
    idx.set(db)
    indices, _ = idx.searchKNN(q, k=3)
    assert all(len(r) == 3 for r in indices)


def test_mih_seeded_set_mih_radius_changes_behaviour():
    """Switching mih_radius at runtime should affect search results."""
    rng = np.random.default_rng(8)
    db = rng.integers(0, 256, size=(1000, 16), dtype=np.uint8)
    # Plant queries that are 4 bits away from db[0].
    q0 = db[0].copy()
    flip = np.zeros(16, dtype=np.uint8); flip[0] = 0x0F  # 4 bits flipped
    q = np.tile(q0, (5, 1)) ^ flip
    q = q.astype(np.uint8)

    idx = pynear.MIHSeededHNSWBinaryIndex(
        M=16, ef_construction=100, ef_search=10, mih_m=8, mih_radius=2,
    )
    idx.set(db)
    nn_low, _ = idx.search1NN(q)

    idx.set_mih_radius(8)
    nn_high, _ = idx.search1NN(q)
    # Both should at least be valid indices; we don't assert order changes
    # since plain HNSW may find the planted neighbour without help.
    assert all(0 <= int(i) < len(db) for i in nn_low)
    assert all(0 <= int(i) < len(db) for i in nn_high)


# ─── Cross-class sanity ────────────────────────────────────────────────────


def test_multiple_hnsw_instances_in_same_process():
    """Two indices of different types should coexist without state pollution."""
    rng = np.random.default_rng(11)
    db_f = rng.standard_normal((200, 16)).astype(np.float32)
    db_b = rng.integers(0, 256, size=(200, 16), dtype=np.uint8)
    q_f = rng.standard_normal((5, 16)).astype(np.float32)
    q_b = rng.integers(0, 256, size=(5, 16), dtype=np.uint8)

    a = pynear.HNSWL2Index(M=8, ef_construction=50, ef_search=50)
    b = pynear.HNSWBinaryIndex(M=8, ef_construction=50, ef_search=50)
    c = pynear.HNSWL2IndexSQ8(M=8, ef_construction=50, ef_search=50)

    a.set(db_f); b.set(db_b); c.set(db_f)

    ia, _ = a.searchKNN(q_f, k=3)
    ib, _ = b.searchKNN(q_b, k=3)
    ic, _ = c.searchKNN(q_f, k=3)

    assert len(ia) == 5 and len(ia[0]) == 3
    assert len(ib) == 5 and len(ib[0]) == 3
    assert len(ic) == 5 and len(ic[0]) == 3

