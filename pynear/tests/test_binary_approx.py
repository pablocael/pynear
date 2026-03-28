"""Tests for IVFFlatBinaryIndex and MIHBinaryIndex."""

import numpy as np
import pytest

import pynear
from pynear import IVFFlatBinaryIndex, MIHBinaryIndex


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_db(n, nbytes, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, nbytes), dtype=np.uint8)


def _flip_bits(vec, n_flips, rng):
    """Return a copy of vec with exactly n_flips bits flipped."""
    out = vec.copy()
    bits = rng.choice(len(out) * 8, size=n_flips, replace=False)
    for b in bits:
        out[b // 8] ^= np.uint8(1 << (b % 8))
    return out


def _make_near_queries(db, true_indices, n_flips, seed=1):
    """Build queries by flipping n_flips bits from db[true_indices]."""
    rng = np.random.default_rng(seed)
    queries = np.array(
        [_flip_bits(db[i], n_flips, rng) for i in true_indices], dtype=np.uint8
    )
    return queries


# ── IVFFlatBinaryIndex ────────────────────────────────────────────────────────

class TestIVFFlatBinaryIndex:
    def test_basic_512bit(self):
        db = _make_db(500, 64)
        idx = IVFFlatBinaryIndex(nlist=16, nprobe=4)
        idx.set(db)
        ti = list(range(0, 50, 5))
        q = _make_near_queries(db, ti, n_flips=3)
        res_idx, res_dist = idx.searchKNN(q, k=5)
        assert len(res_idx) == len(ti)
        assert all(len(r) <= 5 for r in res_idx)
        # Distances should be non-negative integers
        for dv in res_dist:
            assert all(d >= 0 for d in dv)

    def test_recall_high_nprobe(self):
        """With nprobe ≥ nlist, every query's true neighbour should be found."""
        N = 1000
        db = _make_db(N, 64)
        ti = list(range(0, 100, 10))  # 10 true neighbours
        q = _make_near_queries(db, ti, n_flips=4)

        idx = IVFFlatBinaryIndex(nlist=32, nprobe=32)  # probe all clusters
        idx.set(db)
        res_idx, _ = idx.searchKNN(q, k=10)

        for i, true_i in enumerate(ti):
            assert true_i in res_idx[i], (
                f"True neighbour {true_i} not found in results {res_idx[i]}"
            )

    def test_set_nprobe(self):
        db = _make_db(200, 64)
        idx = IVFFlatBinaryIndex(nlist=8, nprobe=2)
        idx.set(db)
        assert idx.nprobe() == 2
        idx.set_nprobe(4)
        assert idx.nprobe() == 4

    def test_nlist_property(self):
        idx = IVFFlatBinaryIndex(nlist=64)
        idx.set(_make_db(200, 64))
        assert idx.nlist() == 64

    def test_256bit(self):
        db = _make_db(400, 32)  # 256-bit
        ti = list(range(0, 40, 4))
        q = _make_near_queries(db, ti, n_flips=3)
        idx = IVFFlatBinaryIndex(nlist=16, nprobe=8)
        idx.set(db)
        res_idx, res_dist = idx.searchKNN(q, k=3)
        assert len(res_idx) == len(ti)

    def test_128bit(self):
        db = _make_db(400, 16)  # 128-bit
        ti = [0, 10, 20]
        q = _make_near_queries(db, ti, n_flips=2)
        idx = IVFFlatBinaryIndex(nlist=8, nprobe=4)
        idx.set(db)
        res_idx, _ = idx.searchKNN(q, k=3)
        assert len(res_idx) == 3

    def test_results_sorted_by_distance(self):
        db = _make_db(500, 64)
        idx = IVFFlatBinaryIndex(nlist=16, nprobe=8)
        idx.set(db)
        q = _make_near_queries(db, [0], n_flips=4)
        _, dist = idx.searchKNN(q, k=10)
        d = dist[0]
        assert d == sorted(d), "Distances not sorted in ascending order"

    def test_k_greater_than_cluster(self):
        """Requesting k > cluster size should return fewer results without error."""
        db = _make_db(50, 64)
        idx = IVFFlatBinaryIndex(nlist=10, nprobe=1)
        idx.set(db)
        q = db[:3]
        res_idx, res_dist = idx.searchKNN(q, k=100)
        assert len(res_idx) == 3
        # Each result has ≤ 100 entries; no crash expected
        for r in res_idx:
            assert len(r) <= 100


# ── MIHBinaryIndex ────────────────────────────────────────────────────────────

class TestMIHBinaryIndex:
    def test_basic_512bit(self):
        db = _make_db(500, 64)  # 512-bit, m=8 → sub_nbytes=8
        ti = list(range(0, 50, 5))
        q = _make_near_queries(db, ti, n_flips=3)
        idx = MIHBinaryIndex(m=8)
        idx.set(db)
        res_idx, res_dist = idx.searchKNN(q, k=5, radius=8)
        assert len(res_idx) == len(ti)
        for dv in res_dist:
            assert all(d >= 0 for d in dv)

    def test_exact_recall_small_flip(self):
        """Flipping 5 bits; pigeonhole with m=8, r_sub=floor(8/8)=1 guarantees recall."""
        N = 2000
        db = _make_db(N, 64)
        ti = list(range(0, 200, 20))  # 10 true neighbours
        q = _make_near_queries(db, ti, n_flips=5)

        idx = MIHBinaryIndex(m=8)
        idx.set(db)
        res_idx, res_dist = idx.searchKNN(q, k=10, radius=8)

        for i, true_i in enumerate(ti):
            assert true_i in res_idx[i], (
                f"True neighbour {true_i} not found; got {res_idx[i]}"
            )

    def test_256bit_m4(self):
        db = _make_db(500, 32)  # 256-bit, m=4 → sub_nbytes=8
        ti = list(range(0, 50, 5))
        q = _make_near_queries(db, ti, n_flips=3)
        idx = MIHBinaryIndex(m=4)
        idx.set(db)
        res_idx, _ = idx.searchKNN(q, k=5, radius=8)
        assert len(res_idx) == len(ti)

    def test_128bit_m4(self):
        db = _make_db(400, 16)  # 128-bit, m=4 → sub_nbytes=4
        ti = [0, 10, 20]
        q = _make_near_queries(db, ti, n_flips=2)
        idx = MIHBinaryIndex(m=4)
        idx.set(db)
        res_idx, _ = idx.searchKNN(q, k=3, radius=4)
        assert len(res_idx) == 3

    def test_properties(self):
        db = _make_db(100, 64)
        idx = MIHBinaryIndex(m=8)
        idx.set(db)
        assert idx.m() == 8
        assert idx.n() == 100
        assert idx.nbytes() == 64

    def test_invalid_m_not_divisible(self):
        db = _make_db(100, 64)  # 64 bytes / m=5 = 12.8 — not integer
        idx = MIHBinaryIndex(m=5)
        with pytest.raises(Exception):
            idx.set(db)

    def test_invalid_sub_nbytes_too_large(self):
        db = _make_db(100, 64)  # 64/4 = 16 bytes > 8 bytes limit
        idx = MIHBinaryIndex(m=4)
        with pytest.raises(Exception):
            idx.set(db)

    def test_radius_zero(self):
        """radius=0: only exact sub-string matches; a database member should find itself."""
        db = _make_db(500, 64)
        idx = MIHBinaryIndex(m=8)
        idx.set(db)
        # Query is identical to db[0]; should always be found with radius=0
        q = db[:1]
        res_idx, res_dist = idx.searchKNN(q, k=1, radius=0)
        assert 0 in res_idx[0]
        assert res_dist[0][0] == 0

    def test_results_sorted_by_distance(self):
        N = 1000
        db = _make_db(N, 64)
        ti = [42]
        q = _make_near_queries(db, ti, n_flips=3)
        idx = MIHBinaryIndex(m=8)
        idx.set(db)
        _, dist = idx.searchKNN(q, k=20, radius=8)
        d = dist[0]
        assert d == sorted(d), "Distances not sorted in ascending order"

    def test_empty_set(self):
        idx = MIHBinaryIndex(m=8)
        idx.set(_make_db(0, 64))
        assert idx.n() == 0
        db = _make_db(10, 64)
        idx2 = MIHBinaryIndex(m=8)
        idx2.set(db)
        # Querying with no matches should return empty lists
        res_idx, res_dist = idx2.searchKNN(db[:2], k=5, radius=0)
        # At least the exact copies (distance 0) are found
        assert 0 in res_idx[0]
        assert 1 in res_idx[1]
