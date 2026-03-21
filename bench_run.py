"""
Standalone benchmark for pynear — compares build + search performance
across dimensions, dataset sizes, k values, and index types.
"""
import time
import numpy as np
from sklearn.neighbors import KDTree

import pynear

NUM_RUNS = 6  # runs per config; drop min+max, average the rest
DATASET_SIZE = 100_000
NUM_QUERIES = 100


def make_data(n, d, dtype=np.float32, seed=42):
    rng = np.random.default_rng(seed)
    return rng.random((n, d), dtype=dtype) if dtype == np.float32 else rng.random((n, d)).astype(dtype)


def make_binary(n, d, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, d), dtype=np.uint8)


def timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def avg_runs(fn, n=NUM_RUNS):
    times = sorted(timed(fn) for _ in range(n))
    # drop best and worst
    trimmed = times[1:-1] if len(times) > 2 else times
    return sum(trimmed) / len(trimmed)


def print_row(label, build_s, search_s, k, n_queries, n_data, dim):
    qps = n_queries / search_s
    print(f"  {label:<28s}  build={build_s*1000:7.1f}ms  search={search_s*1000:7.2f}ms  {qps:8.0f} q/s")


def section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ── Float L2 index across dimensions ────────────────────────────────────────
section(f"VPTreeL2Index  |  n={DATASET_SIZE:,}  nq={NUM_QUERIES}  k=1")
print(f"  {'index':<28s}  {'build':>13s}  {'search':>13s}  {'q/s':>8s}")
print("  " + "-" * 66)

dims = [4, 8, 16, 32, 64, 128]
for d in dims:
    data = make_data(DATASET_SIZE, d)
    query = make_data(NUM_QUERIES, d, seed=99)

    idx = pynear.VPTreeL2Index()
    build_t = avg_runs(lambda: idx.set(data))
    search_t = avg_runs(lambda: idx.searchKNN(query, 1))
    print_row(f"L2 d={d}", build_t, search_t, 1, NUM_QUERIES, DATASET_SIZE, d)

# ── Float L2: k sweep ────────────────────────────────────────────────────────
section(f"VPTreeL2Index  |  n={DATASET_SIZE:,}  nq={NUM_QUERIES}  d=32  k sweep")
print(f"  {'index':<28s}  {'build':>13s}  {'search':>13s}  {'q/s':>8s}")
print("  " + "-" * 66)

d = 32
data = make_data(DATASET_SIZE, d)
query = make_data(NUM_QUERIES, d, seed=99)
idx = pynear.VPTreeL2Index()
idx.set(data)
build_t = avg_runs(lambda: idx.set(data))  # re-time build once
for k in [1, 4, 8, 16, 32]:
    search_t = avg_runs(lambda: idx.searchKNN(query, k))
    print_row(f"L2 d={d} k={k}", build_t, search_t, k, NUM_QUERIES, DATASET_SIZE, d)

# ── search1NN vs searchKNN(k=1) ──────────────────────────────────────────────
section(f"search1NN vs searchKNN(k=1)  |  n={DATASET_SIZE:,}  nq={NUM_QUERIES}  d=32")
print(f"  {'index':<28s}  {'build':>13s}  {'search':>13s}  {'q/s':>8s}")
print("  " + "-" * 66)

idx = pynear.VPTreeL2Index()
idx.set(data)
search_knn1 = avg_runs(lambda: idx.searchKNN(query, 1))
search_1nn = avg_runs(lambda: idx.search1NN(query))
print_row("searchKNN(k=1)", 0, search_knn1, 1, NUM_QUERIES, DATASET_SIZE, 32)
print_row("search1NN     ", 0, search_1nn,  1, NUM_QUERIES, DATASET_SIZE, 32)

# ── L1 and Chebyshev ─────────────────────────────────────────────────────────
section(f"Index type comparison  |  n={DATASET_SIZE:,}  nq={NUM_QUERIES}  d=32  k=8")
print(f"  {'index':<28s}  {'build':>13s}  {'search':>13s}  {'q/s':>8s}")
print("  " + "-" * 66)

for name, cls in [("VPTreeL2Index", pynear.VPTreeL2Index),
                   ("VPTreeL1Index", pynear.VPTreeL1Index),
                   ("VPTreeChebyshevIndex", pynear.VPTreeChebyshevIndex)]:
    idx = cls()
    build_t = avg_runs(lambda: idx.set(data))
    search_t = avg_runs(lambda: idx.searchKNN(query, 8))
    print_row(name, build_t, search_t, 8, NUM_QUERIES, DATASET_SIZE, 32)

# ── vs sklearn KDTree (brute-force exact) ────────────────────────────────────
section(f"VPTreeL2 vs sklearn KDTree  |  n={DATASET_SIZE:,}  nq={NUM_QUERIES}  k=8")
print(f"  {'index':<28s}  {'build':>13s}  {'search':>13s}  {'q/s':>8s}")
print("  " + "-" * 66)

for d in [4, 16, 64]:
    data_d = make_data(DATASET_SIZE, d)
    query_d = make_data(NUM_QUERIES, d, seed=99)

    vp = pynear.VPTreeL2Index()
    vp_build = avg_runs(lambda: vp.set(data_d))
    vp_search = avg_runs(lambda: vp.searchKNN(query_d, 8))
    print_row(f"VPTreeL2 d={d}", vp_build, vp_search, 8, NUM_QUERIES, DATASET_SIZE, d)

    def sk_run():
        t = KDTree(data_d)
        t.query(query_d, k=8)

    sk_build_search = avg_runs(sk_run)
    print_row(f"SKLearn KDTree d={d}", 0, sk_build_search, 8, NUM_QUERIES, DATASET_SIZE, d)
    print()

# ── Binary index ─────────────────────────────────────────────────────────────
section(f"VPTreeBinaryIndex  |  n={DATASET_SIZE:,}  nq={NUM_QUERIES}  k=8")
print(f"  {'index':<28s}  {'build':>13s}  {'search':>13s}  {'q/s':>8s}")
print("  " + "-" * 66)

for d in [8, 16, 32, 64]:
    bdata = make_binary(DATASET_SIZE, d)
    bquery = make_binary(NUM_QUERIES, d, seed=99)
    idx = pynear.VPTreeBinaryIndex()
    build_t = avg_runs(lambda: idx.set(bdata))
    search_t = avg_runs(lambda: idx.searchKNN(bquery, 8))
    print_row(f"Binary d={d}B ({d*8}bit)", build_t, search_t, 8, NUM_QUERIES, DATASET_SIZE, d)

# ── Query-count scaling ──────────────────────────────────────────────────────
section(f"Query count scaling  |  n={DATASET_SIZE:,}  d=32  k=8  L2")
print(f"  {'nq':<28s}  {'build':>13s}  {'search':>13s}  {'q/s':>8s}")
print("  " + "-" * 66)

d = 32
data = make_data(DATASET_SIZE, d)
idx = pynear.VPTreeL2Index()
idx.set(data)
for nq in [1, 4, 16, 64, 256, 1024]:
    q = make_data(nq, d, seed=99)
    search_t = avg_runs(lambda: idx.searchKNN(q, 8))
    print_row(f"nq={nq}", 0, search_t, 8, nq, DATASET_SIZE, d)

print()
print("Done.")
