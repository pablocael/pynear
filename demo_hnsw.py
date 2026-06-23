"""HNSW demo for pynear.

Three quick scenarios:

1. HNSWL2Index   — float vector ANN, the standard HNSW workload.
2. HNSWCosineIndex — text-embedding-style cosine search.
3. MIHSeededHNSWBinaryIndex — the novel variant for binary descriptors;
   shows that even at low ef_search, near-duplicate queries are recovered
   exactly thanks to the MIH seeding.

Run:   python demo_hnsw.py
"""

import time

import numpy as np

import pynear


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def hnsw_l2_demo() -> None:
    import os
    section("HNSWL2Index — float L2 ANN")
    rng = np.random.default_rng(42)
    db = rng.standard_normal((50_000, 128)).astype(np.float32)
    queries = rng.standard_normal((100, 128)).astype(np.float32)

    # Sequential build (default, deterministic)
    idx = pynear.HNSWL2Index(M=16, ef_construction=200, ef_search=50)
    t0 = time.perf_counter()
    idx.set(db)
    seq_build = time.perf_counter() - t0

    # Parallel build (opt-in via n_threads)
    nt = max(1, os.cpu_count() or 1)
    idx_par = pynear.HNSWL2Index(M=16, ef_construction=200, ef_search=50, n_threads=nt)
    t0 = time.perf_counter()
    idx_par.set(db)
    par_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    indices, _ = idx.searchKNN(queries, k=10)
    query_s = time.perf_counter() - t0

    print(f"  N=50_000  D=128")
    print(f"  build seq (nt=1):  {seq_build:.2f}s")
    print(f"  build par (nt={nt}): {par_build:.2f}s   speedup={seq_build/par_build:.1f}x")
    print(f"  query: {query_s*1000/len(queries):.3f}ms/query")
    print(f"  top-3 NN of query 0 (nearest-first): {list(indices[0])[::-1][:3]}")


def hnsw_cosine_demo() -> None:
    section("HNSWCosineIndex — text-embedding-style cosine")
    rng = np.random.default_rng(0)
    # Pretend these are sentence-embedding vectors
    db = rng.standard_normal((20_000, 384)).astype(np.float32)
    queries = rng.standard_normal((20, 384)).astype(np.float32)

    idx = pynear.HNSWCosineIndex(M=16, ef_construction=200, ef_search=64)
    idx.set(db)
    indices, distances = idx.searchKNN(queries, k=5)
    # Cosine distances are in [0, 2]; smaller = more similar.
    print(f"  N=20_000  D=384  top-5 cosine distances of query 0 (nearest-first):")
    print(f"  {[round(d, 4) for d in list(distances[0])[::-1]]}")


def mih_seeded_hnsw_demo() -> None:
    section("MIHSeededHNSWBinaryIndex — novel: HNSW + MIH-seeded for binary")
    rng = np.random.default_rng(7)
    # 50_000 × 128-bit (16-byte) descriptors, e.g. perceptual hashes
    db = rng.integers(0, 256, size=(50_000, 16), dtype=np.uint8)
    queries = rng.integers(0, 256, size=(20, 16), dtype=np.uint8)

    # Plant 20 known near-duplicates: each query exists exactly at row q*500 in db.
    for i, q in enumerate(queries):
        db[i * 500] = q

    seeded = pynear.MIHSeededHNSWBinaryIndex(
        M=16, ef_construction=200, ef_search=20,
        mih_m=8, mih_radius=4,
    )
    seeded.set(db)
    nn_idx, nn_dist = seeded.search1NN(queries)

    perfect = sum(int(nn_idx[i]) == i * 500 for i in range(len(queries)))
    print(f"  N=50_000  D=128bit  planted exact duplicates: {len(queries)}")
    print(f"  recovered as 1-NN: {perfect}/{len(queries)} at ef_search=20")
    print(f"  (Plain HNSW often misses these at low ef on dense binary data;")
    print(f"   the MIH seed set guarantees exact recovery within radius={seeded.mih_radius}.)")


def main() -> None:
    hnsw_l2_demo()
    hnsw_cosine_demo()
    mih_seeded_hnsw_demo()


if __name__ == "__main__":
    main()
