"""Tests for the searchKNN_arrays dense-array API.

searchKNN_arrays returns (ids, distances) numpy arrays of shape (n_queries, k),
ordered NEAREST-FIRST along axis 1. The list API (searchKNN) orders
VPTree/HNSW rows farthest-first, while MIH/IVF list rows are already
ascending; the arrays variant normalises all of them to nearest-first.
Rows with fewer than k hits are padded at the tail: id == -1, distance ==
+inf (float indexes) or INT64_MAX (hamming indexes).
"""

import numpy as np
import pytest

pynear = pytest.importorskip("pynear")


INT64_MAX = np.iinfo(np.int64).max

# (class name, ctor kwargs, data kind, dim (floats) / row bytes (binary),
#  dist dtype, list rows farthest-first?, supports filter kwarg, search kwargs)
SPECS = [
    ("VPTreeL2Index", {}, "float", 8, np.float32, True, False, {}),
    ("VPTreeL1Index", {}, "float", 8, np.float32, True, False, {}),
    ("VPTreeChebyshevIndex", {}, "float", 8, np.float32, True, False, {}),
    ("VPTreeCosineIndex", {}, "float", 8, np.float32, True, False, {}),
    ("HNSWL2Index", dict(M=8, ef_construction=100, ef_search=200), "float", 8, np.float32, True, True, {}),
    ("HNSWCosineIndex", dict(M=8, ef_construction=100, ef_search=200), "float", 8, np.float32, True, True, {}),
    ("HNSWL2IndexSQ8", dict(M=8, ef_construction=100, ef_search=200), "float", 8, np.float32, True, True, {}),
    ("HNSWBinaryIndex", dict(M=8, ef_construction=100, ef_search=200), "binary", 16, np.int64, True, True, {}),
    ("MIHSeededHNSWBinaryIndex", dict(M=8, ef_construction=100, ef_search=200, mih_m=8, mih_radius=8), "binary", 16, np.int64, True, False, {}),
    ("VPTreeBinaryIndexN", {}, "binary", 16, np.int64, True, False, {}),
    ("VPTreeBinaryIndex64", {}, "binary", 8, np.int64, True, False, {}),
    ("VPTreeBinaryIndex128", {}, "binary", 16, np.int64, True, False, {}),
    ("VPTreeBinaryIndex256", {}, "binary", 32, np.int64, True, False, {}),
    ("VPTreeBinaryIndex512", {}, "binary", 64, np.int64, True, False, {}),
    ("IVFFlatBinaryIndex", dict(nlist=2, nprobe=2), "binary", 16, np.int64, False, False, {}),
    ("MIHBinaryIndex", dict(m=8), "binary", 16, np.int64, False, False, dict(radius=8)),
]

SPEC_IDS = [s[0] for s in SPECS]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _get_cls(name):
    cls = getattr(pynear, name, None)
    if cls is None:
        pytest.skip(f"pynear.{name} not available in this build")
    if not hasattr(cls, "searchKNN_arrays"):
        pytest.skip(f"pynear.{name} has no searchKNN_arrays in this build")
    return cls


def _make_data(kind, rng, n, width):
    if kind == "float":
        return rng.standard_normal((n, width)).astype(np.float32)
    return rng.integers(0, 256, size=(n, width), dtype=np.uint8)


def _make_queries(kind, rng, data, nq):
    """Queries near stored points so approximate binary indexes get hits."""
    if kind == "float":
        return rng.standard_normal((nq, data.shape[1])).astype(np.float32)
    q = data[np.arange(nq) % len(data)].copy()
    for i in range(nq):
        byte = int(rng.integers(0, q.shape[1]))
        q[i, byte] ^= np.uint8(1 << int(rng.integers(0, 8)))
    return q


def _pad_dist(dist_dtype):
    return np.inf if dist_dtype == np.float32 else INT64_MAX


# ── (a)+(b)+(c): arrays variant vs list variant, dtypes/shapes, padding ─────


@pytest.mark.parametrize("n,k", [(60, 5), (4, 9)], ids=["k_lt_n", "k_gt_n"])
@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_arrays_variant_matches_list_variant(spec, n, k):
    name, ctor, kind, width, dist_dtype, far_first, _has_filter, skw = spec
    cls = _get_cls(name)
    rng = np.random.default_rng(123)
    data = _make_data(kind, rng, n, width)
    queries = _make_queries(kind, rng, data, 10)

    index = cls(**ctor)
    index.set(data)

    list_ids, list_dists = index.searchKNN(queries, k, **skw)
    ids, dists = index.searchKNN_arrays(queries, k, **skw)

    # (b) dtypes and shapes
    assert ids.shape == (len(queries), k)
    assert dists.shape == (len(queries), k)
    assert ids.dtype == np.int64
    assert dists.dtype == dist_dtype

    for qi in range(len(queries)):
        row_ids = [int(v) for v in list_ids[qi]]
        row_dists = [float(v) for v in list_dists[qi]]
        if far_first:
            # VPTree/HNSW list rows are farthest-first; arrays are nearest-first.
            row_ids = row_ids[::-1]
            row_dists = row_dists[::-1]
        m = len(row_ids)
        assert m <= k

        # (a) arrays row == list row, reversed to nearest-first
        assert ids[qi, :m].tolist() == row_ids
        np.testing.assert_allclose(
            dists[qi, :m].astype(np.float64),
            np.asarray(row_dists, dtype=np.float64),
            rtol=1e-6,
            atol=1e-6,
        )

        # (c) padding sentinels appear exactly where the list row is short
        assert np.all(ids[qi, m:] == -1)
        if dist_dtype == np.float32:
            assert np.all(np.isinf(dists[qi, m:]))
        else:
            assert np.all(dists[qi, m:] == INT64_MAX)


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_arrays_variant_k_gt_n_pads_every_row(spec):
    """k > n: every row must contain at least k - n padding entries."""
    name, ctor, kind, width, dist_dtype, _ff, _hf, skw = spec
    cls = _get_cls(name)
    rng = np.random.default_rng(7)
    n, k = 5, 8
    data = _make_data(kind, rng, n, width)
    queries = _make_queries(kind, rng, data, 4)

    index = cls(**ctor)
    index.set(data)
    ids, dists = index.searchKNN_arrays(queries, k, **skw)

    pad = ids == -1
    assert np.all(pad.sum(axis=1) >= k - n)
    # Padding is a tail block: once padding starts in a row it never stops.
    for qi in range(len(queries)):
        row = pad[qi]
        first = int(np.argmax(row)) if row.any() else k
        assert np.all(row[first:])
        # Sentinel distances accompany sentinel ids.
        if dist_dtype == np.float32:
            assert np.all(np.isinf(np.asarray(dists[qi])[row]))
        else:
            assert np.all(np.asarray(dists[qi])[row] == INT64_MAX)


def test_mih_radius_limited_rows_are_padded():
    """MIH with radius=0 returns (near-)empty rows → tail padding + radius kwarg."""
    cls = _get_cls("MIHBinaryIndex")
    rng = np.random.default_rng(5)
    db = rng.integers(0, 256, size=(64, 16), dtype=np.uint8)
    index = cls(m=8)
    index.set(db)

    q = np.vstack([db[0], db[1] ^ np.uint8(0xFF)])
    k = 3
    list_ids, list_dists = index.searchKNN(q, k, radius=0)
    ids, dists = index.searchKNN_arrays(q, k, radius=0)

    for qi in range(len(q)):
        m = len(list_ids[qi])
        # MIH list rows are already ascending — no reversal expected.
        assert ids[qi, :m].tolist() == [int(v) for v in list_ids[qi]]
        assert dists[qi, :m].tolist() == [int(v) for v in list_dists[qi]]
        assert np.all(ids[qi, m:] == -1)
        assert np.all(dists[qi, m:] == INT64_MAX)

    # The exact duplicate is retrieved nearest-first at distance 0...
    assert ids[0, 0] == 0
    assert dists[0, 0] == 0
    # ...and the radius-0 search leaves at least one padded slot somewhere.
    assert np.any(ids == -1)


# ── (d): HNSW filter kwarg through the arrays variant ───────────────────────


FILTER_SPECS = [s for s in SPECS if s[6]]


@pytest.mark.parametrize("spec", FILTER_SPECS, ids=[s[0] for s in FILTER_SPECS])
def test_filter_kwarg_via_arrays(spec):
    name, ctor, kind, width, _dist_dtype, far_first, _hf, skw = spec
    cls = _get_cls(name)
    rng = np.random.default_rng(21)
    data = _make_data(kind, rng, 100, width)
    queries = _make_queries(kind, rng, data, 5)

    index = cls(**ctor)
    index.set(data)

    mask = np.zeros(100, dtype=np.uint8)
    mask[:50] = 1  # only ids < 50 allowed

    ids, _dists = index.searchKNN_arrays(queries, 5, filter=mask, **skw)
    real = ids[ids != -1]
    assert real.size > 0
    assert np.all(real < 50)

    # Same filter through the list API gives the same rows (reversed).
    list_ids, _ = index.searchKNN(queries, 5, filter=mask, **skw)
    for qi in range(len(queries)):
        row = [int(v) for v in list_ids[qi]]
        if far_first:
            row = row[::-1]
        assert ids[qi, : len(row)].tolist() == row


# ── (e): sharded cross-shard merge ───────────────────────────────────────────


class _ListOnlyShard:
    """Wraps a pynear index but hides searchKNN_arrays, forcing the sharded
    merge onto its legacy list-API fallback path."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        if name == "searchKNN_arrays":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_inner"), name)


def test_sharded_searchknn_matches_bruteforce_merge():
    """Cross-shard merge (arrays fast path) must reproduce an exact global
    top-k. VPTreeL2Index shards are exact, so the check is deterministic."""
    if not hasattr(pynear, "ShardedHNSWIndex"):
        pytest.skip("ShardedHNSWIndex not available")
    _get_cls("VPTreeL2Index")

    rng = np.random.default_rng(11)
    d, n, k = 12, 240, 7
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    keys = [f"t{i % 3}" for i in range(n)]

    shards = pynear.ShardedHNSWIndex(pynear.VPTreeL2Index)
    shards.set(vecs, shard_keys=keys)

    queries = rng.standard_normal((12, d)).astype(np.float32)
    idx, dist = shards.searchKNN(queries, k)

    rows_by_key = {}
    for i, key in enumerate(keys):
        rows_by_key.setdefault(key, []).append(i)

    dmat = np.linalg.norm(vecs[None, :, :] - queries[:, None, :], axis=2)
    for qi in range(len(queries)):
        got = idx[qi][::-1]  # output stays farthest-first → reverse
        got_d = [float(v) for v in dist[qi]][::-1]
        assert len(got) == k
        assert got_d == sorted(got_d), "merged row not nearest-first after reversal"

        got_global = [rows_by_key[s][local] for s, local in got]
        ref = np.argsort(dmat[qi], kind="stable")[:k]
        assert set(got_global) == set(int(r) for r in ref)
        np.testing.assert_allclose(got_d, np.sort(dmat[qi])[:k], rtol=1e-5, atol=1e-5)


def test_sharded_arrays_merge_identical_to_list_fallback():
    """The searchKNN_arrays fast path and the legacy list fallback must produce
    byte-for-byte identical merged output (ids, shard keys, distances, order)."""
    if not hasattr(pynear, "ShardedHNSWIndex"):
        pytest.skip("ShardedHNSWIndex not available")
    if not hasattr(pynear.HNSWL2Index, "searchKNN_arrays"):
        pytest.skip("HNSWL2Index has no searchKNN_arrays in this build")

    rng = np.random.default_rng(3)
    d, n, k = 16, 300, 6
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    keys = [f"tenant_{i % 4}" for i in range(n)]

    shards = pynear.ShardedHNSWIndex(
        pynear.HNSWL2Index, M=8, ef_construction=100, ef_search=200
    )
    shards.set(vecs, shard_keys=keys)
    queries = rng.standard_normal((10, d)).astype(np.float32)

    idx_fast, dist_fast = shards.searchKNN(queries, k)

    # Hide searchKNN_arrays on every shard → same query runs the list path.
    original = dict(shards._shards)
    try:
        shards._shards = {key: _ListOnlyShard(s) for key, s in original.items()}
        idx_slow, dist_slow = shards.searchKNN(queries, k)
    finally:
        shards._shards = original

    assert idx_fast == idx_slow
    assert dist_fast == dist_slow
