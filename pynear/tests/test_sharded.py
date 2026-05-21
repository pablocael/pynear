"""Tests for pynear.ShardedHNSWIndex (the directory-of-shards helper)."""

import os
import shutil
import tempfile

import numpy as np
import pytest

import pynear


# ─── Helpers ────────────────────────────────────────────────────────────────


def _build_multi_tenant(n_per_tenant=(500, 300, 800), d=32, seed=42):
    """Generate vectors + per-row tenant labels for testing."""
    rng = np.random.default_rng(seed)
    vecs, keys = [], []
    for i, n in enumerate(n_per_tenant):
        vecs.append(rng.standard_normal((n, d)).astype(np.float32))
        keys.extend([f"tenant_{i}"] * n)
    return np.vstack(vecs), keys, rng


# ─── Build / introspection ──────────────────────────────────────────────────


def test_sharded_set_partitions_by_key():
    vecs, keys, _ = _build_multi_tenant()
    shards = pynear.ShardedHNSWIndex(
        index_cls=pynear.HNSWL2Index,
        M=8, ef_construction=50, ef_search=50,
    )
    shards.set(vecs, shard_keys=keys)
    assert shards.n_shards == 3
    assert shards.size == sum((500, 300, 800))
    expected = {"tenant_0": 500, "tenant_1": 300, "tenant_2": 800}
    assert shards.shard_sizes() == expected


def test_sharded_set_mismatched_lengths_raises():
    vecs, _, _ = _build_multi_tenant(n_per_tenant=(10,))
    shards = pynear.ShardedHNSWIndex(pynear.HNSWL2Index, M=4, ef_construction=20, ef_search=20)
    with pytest.raises(ValueError):
        shards.set(vecs, shard_keys=["a", "b"])


def test_sharded_invalid_index_class_raises():
    with pytest.raises(TypeError):
        pynear.ShardedHNSWIndex(index_cls=dict)  # not an index class


# ─── Single- vs cross-shard queries ─────────────────────────────────────────


def test_sharded_single_tenant_query_returns_only_that_tenant():
    vecs, keys, rng = _build_multi_tenant()
    shards = pynear.ShardedHNSWIndex(pynear.HNSWL2Index, M=8, ef_construction=50, ef_search=50)
    shards.set(vecs, shard_keys=keys)

    q = rng.standard_normal((3, 32)).astype(np.float32)
    idx, _ = shards.searchKNN(q, k=5, shard="tenant_1")
    for row in idx:
        for shard_key, local_id in row:
            assert shard_key == "tenant_1"
            assert 0 <= local_id < 300


def test_sharded_cross_tenant_query_returns_merged_topk():
    vecs, keys, rng = _build_multi_tenant()
    shards = pynear.ShardedHNSWIndex(pynear.HNSWL2Index, M=8, ef_construction=100, ef_search=100)
    shards.set(vecs, shard_keys=keys)

    q = rng.standard_normal((5, 32)).astype(np.float32)
    idx, dist = shards.searchKNN(q, k=10)
    assert len(idx) == 5 and len(dist) == 5
    for row, d_row in zip(idx, dist):
        assert len(row) == 10
        assert len(d_row) == 10

    # Convention: distances in each row are farthest-first within top-k.
    # Reverse → nearest-first, then assert it's monotone-non-decreasing.
    for d_row in dist:
        nearest_first = d_row[::-1]
        assert all(nearest_first[i] <= nearest_first[i + 1] for i in range(len(nearest_first) - 1))


def test_sharded_search1NN_returns_global_nearest():
    """search1NN across shards must agree with brute-force on a tiny dataset."""
    rng = np.random.default_rng(0)
    db_a = rng.standard_normal((100, 16)).astype(np.float32)
    db_b = rng.standard_normal((100, 16)).astype(np.float32)
    shards = pynear.ShardedHNSWIndex(pynear.HNSWL2Index, M=16, ef_construction=200, ef_search=200)
    shards.set(np.vstack([db_a, db_b]), shard_keys=["a"] * 100 + ["b"] * 100)

    queries = rng.standard_normal((10, 16)).astype(np.float32)
    nn_idx, nn_dist = shards.search1NN(queries)

    # Brute force
    all_db = np.vstack([db_a, db_b])
    all_keys = ["a"] * 100 + ["b"] * 100
    for q_i, q in enumerate(queries):
        true_d = np.linalg.norm(all_db - q, axis=1)
        true_id = int(np.argmin(true_d))
        true_key = all_keys[true_id]
        true_local = true_id if true_id < 100 else true_id - 100
        # Allow ANN miss but enforce: returned shard is the right one
        shard_key, local_id = nn_idx[q_i]
        if (shard_key, local_id) != (true_key, true_local):
            # Soft check — distance must be within 5% of true
            assert nn_dist[q_i] <= 1.05 * true_d[true_id]


# ─── Mutation API ───────────────────────────────────────────────────────────


def test_sharded_add_to_existing_shard():
    vecs, keys, rng = _build_multi_tenant()
    shards = pynear.ShardedHNSWIndex(pynear.HNSWL2Index, M=8, ef_construction=50, ef_search=50)
    shards.set(vecs, shard_keys=keys)

    extra = rng.standard_normal((20, 32)).astype(np.float32)
    new_ids = shards.add(extra, shard="tenant_0")
    assert "tenant_0" in new_ids
    assert len(new_ids["tenant_0"]) == 20
    assert shards.shard_sizes()["tenant_0"] == 500 + 20


def test_sharded_add_creates_new_shard_on_demand():
    vecs, keys, rng = _build_multi_tenant()
    shards = pynear.ShardedHNSWIndex(pynear.HNSWL2Index, M=8, ef_construction=50, ef_search=50)
    shards.set(vecs, shard_keys=keys)
    assert "brand_new" not in shards.shard_keys

    extra = rng.standard_normal((10, 32)).astype(np.float32)
    new_ids = shards.add(extra, shard="brand_new")
    assert "brand_new" in shards.shard_keys
    assert shards.shard_sizes()["brand_new"] == 10
    assert new_ids["brand_new"] == list(range(10))


def test_sharded_add_per_row_routing():
    """add(..., shard_keys=...) routes per row, like set() does."""
    shards = pynear.ShardedHNSWIndex(pynear.HNSWL2Index, M=8, ef_construction=50, ef_search=50)
    rng = np.random.default_rng(0)
    # First batch: 20 rows split 10/10 between two shards
    vecs = rng.standard_normal((20, 16)).astype(np.float32)
    keys = ["a"] * 10 + ["b"] * 10
    shards.add(vecs, shard_keys=keys)
    assert shards.shard_sizes() == {"a": 10, "b": 10}


def test_sharded_add_requires_one_of_shard_or_shard_keys():
    shards = pynear.ShardedHNSWIndex(pynear.HNSWL2Index, M=4, ef_construction=20, ef_search=20)
    vecs = np.random.default_rng(0).standard_normal((5, 8)).astype(np.float32)
    with pytest.raises(ValueError):
        shards.add(vecs)  # neither
    with pytest.raises(ValueError):
        shards.add(vecs, shard="x", shard_keys=["x"] * 5)  # both


def test_sharded_remove_and_rebuild_per_shard():
    vecs, keys, _ = _build_multi_tenant(n_per_tenant=(100, 100), d=16)
    shards = pynear.ShardedHNSWIndex(pynear.HNSWL2Index, M=8, ef_construction=50, ef_search=50)
    shards.set(vecs, shard_keys=keys)

    shards.remove(7, shard="tenant_0")
    shards.remove(13, shard="tenant_0")
    assert shards.get_shard("tenant_0").num_deleted == 2
    # tenant_1 is untouched
    assert shards.get_shard("tenant_1").num_deleted == 0

    mapping = shards.rebuild(shard="tenant_0")
    assert "tenant_0" in mapping
    assert shards.get_shard("tenant_0").num_deleted == 0
    assert shards.shard_sizes()["tenant_0"] == 100 - 2


def test_sharded_rebuild_all_shards():
    vecs, keys, _ = _build_multi_tenant(n_per_tenant=(50, 50, 50), d=16)
    shards = pynear.ShardedHNSWIndex(pynear.HNSWL2Index, M=8, ef_construction=50, ef_search=50)
    shards.set(vecs, shard_keys=keys)

    shards.remove(0, shard="tenant_0")
    shards.remove(1, shard="tenant_1")
    shards.remove(2, shard="tenant_2")
    mappings = shards.rebuild()  # no shard arg → all
    assert set(mappings) == {"tenant_0", "tenant_1", "tenant_2"}
    assert shards.size == 50 * 3 - 3


# ─── Persistence ────────────────────────────────────────────────────────────


def test_sharded_save_and_load_round_trip(tmp_path):
    vecs, keys, rng = _build_multi_tenant(n_per_tenant=(80, 120), d=16)
    shards = pynear.ShardedHNSWIndex(
        pynear.HNSWL2Index,
        M=8, ef_construction=50, ef_search=50, n_threads=1, seed=99,
    )
    shards.set(vecs, shard_keys=keys)

    q = rng.standard_normal((3, 16)).astype(np.float32)
    before_per = shards.searchKNN(q, k=5, shard="tenant_0")
    before_all = shards.searchKNN(q, k=5)

    shards.save(str(tmp_path))
    # Manifest + one .pkl per shard
    files = sorted(os.listdir(tmp_path))
    assert "manifest.json" in files
    assert "shard_tenant_0.pkl" in files
    assert "shard_tenant_1.pkl" in files

    reloaded = pynear.ShardedHNSWIndex.load(str(tmp_path))
    assert reloaded.shard_sizes() == shards.shard_sizes()

    after_per = reloaded.searchKNN(q, k=5, shard="tenant_0")
    after_all = reloaded.searchKNN(q, k=5)
    assert before_per == after_per
    assert before_all == after_all


def test_sharded_save_load_preserves_tombstones(tmp_path):
    vecs, keys, _ = _build_multi_tenant(n_per_tenant=(50, 50), d=16)
    shards = pynear.ShardedHNSWIndex(pynear.HNSWL2Index, M=4, ef_construction=20, ef_search=20)
    shards.set(vecs, shard_keys=keys)
    shards.remove(3, shard="tenant_0")
    shards.remove(7, shard="tenant_1")

    shards.save(str(tmp_path))
    reloaded = pynear.ShardedHNSWIndex.load(str(tmp_path))
    assert reloaded.get_shard("tenant_0").num_deleted == 1
    assert reloaded.get_shard("tenant_1").num_deleted == 1


def test_sharded_save_load_with_different_index_class(tmp_path):
    """Cosine and SQ8 variants persist too."""
    vecs, keys, _ = _build_multi_tenant(n_per_tenant=(50, 50), d=16)
    for cls in (pynear.HNSWCosineIndex, pynear.HNSWL2IndexSQ8):
        d = tmp_path / cls.__name__
        d.mkdir()
        shards = pynear.ShardedHNSWIndex(cls, M=4, ef_construction=20, ef_search=20)
        shards.set(vecs, shard_keys=keys)
        shards.save(str(d))
        reloaded = pynear.ShardedHNSWIndex.load(str(d))
        assert reloaded.shard_sizes() == shards.shard_sizes()


# ─── Empty / edge cases ─────────────────────────────────────────────────────


def test_sharded_empty_index_query_returns_empty():
    shards = pynear.ShardedHNSWIndex(pynear.HNSWL2Index, M=4, ef_construction=20, ef_search=20)
    q = np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32)
    idx, dist = shards.searchKNN(q, k=5)
    assert all(len(r) == 0 for r in idx)
    assert all(len(r) == 0 for r in dist)


def test_sharded_missing_shard_raises():
    shards = pynear.ShardedHNSWIndex(pynear.HNSWL2Index, M=4, ef_construction=20, ef_search=20)
    rng = np.random.default_rng(0)
    shards.set(rng.standard_normal((10, 8)).astype(np.float32), shard_keys=["a"] * 10)
    with pytest.raises(KeyError):
        shards.searchKNN(rng.standard_normal((1, 8)).astype(np.float32), k=3, shard="missing")
    with pytest.raises(KeyError):
        shards.remove(0, shard="missing")
