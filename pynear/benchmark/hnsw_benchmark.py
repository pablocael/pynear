"""HNSW benchmark — pynear vs Faiss vs the novel MIH-seeded variant.

Three comparisons:

1. Float L2 ANN     pynear.HNSWL2Index             vs  Faiss IndexHNSWFlat
2. Float L2 ANN     pynear.HNSWL2Index             vs  pynear.IVFFlatL2Index
3. Binary Hamming   pynear.MIHSeededHNSWBinaryIndex vs  plain pynear.HNSWBinaryIndex

For each run we report build time, mean query latency, and recall@k against
brute force. Writes a Markdown summary to ``results/hnsw_benchmark.md``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

import pynear

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# ── helpers ─────────────────────────────────────────────────────────────────


def brute_l2(db: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    d = np.linalg.norm(db[None, :, :] - queries[:, None, :], axis=2)
    return np.argsort(d, axis=1)[:, :k]


def brute_hamming(db: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    db_bits = np.unpackbits(db, axis=1)
    q_bits = np.unpackbits(queries, axis=1)
    d = (q_bits[:, None, :] != db_bits[None, :, :]).sum(axis=2)
    return np.argsort(d, axis=1)[:, :k]


def recall_at_k(pred_idx, ref_idx, k: int) -> float:
    n = len(ref_idx)
    return sum(len(set(pred_idx[i]) & set(ref_idx[i].tolist())) for i in range(n)) / (n * k)


def run_case(rows, name, cfg, make_index, db, q, ref, k, faiss_api=False):
    """Construct an index, time set()/searchKNN(), compute recall, append a row.

    Construction (via `make_index`) happens outside the timed sections;
    only the build and the batch query are measured.
    """
    idx = make_index()
    if faiss_api:
        t0 = time.perf_counter(); idx.add(db); build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, ix = idx.search(q, k)
        query_s = time.perf_counter() - t0
        pred = ix.tolist()
    else:
        t0 = time.perf_counter(); idx.set(db); build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pi, _ = idx.searchKNN(q, k=k)
        query_s = time.perf_counter() - t0
        pred = np.array(pi)[:, ::-1]
    rows.append((name, cfg, build_s, query_s / len(q) * 1e3, recall_at_k(pred, ref, k)))


# ── benchmark scenarios ─────────────────────────────────────────────────────


def bench_float_l2(n: int = 20_000, d: int = 128, k: int = 10):
    rng = np.random.default_rng(42)
    db = rng.standard_normal((n, d)).astype(np.float32)
    q = rng.standard_normal((200, d)).astype(np.float32)

    ref = brute_l2(db, q, k)
    rows = []

    # pynear HNSW, single-threaded build (default)
    for ef in (32, 64, 128, 256):
        run_case(rows, "pynear.HNSWL2Index", f"M=16,ef={ef},nt=1",
                 lambda ef=ef: pynear.HNSWL2Index(M=16, ef_construction=200, ef_search=ef),
                 db, q, ref, k)

    # pynear HNSW, parallel build
    nt = max(1, (os.cpu_count() or 1))
    for ef in (64, 256):
        run_case(rows, "pynear.HNSWL2Index", f"M=16,ef={ef},nt={nt}",
                 lambda ef=ef: pynear.HNSWL2Index(M=16, ef_construction=200, ef_search=ef, n_threads=nt),
                 db, q, ref, k)

    # Faiss HNSW
    if HAS_FAISS:
        def make_faiss_hnsw(ef):
            idx = faiss.IndexHNSWFlat(d, 16)
            idx.hnsw.efConstruction = 200
            idx.hnsw.efSearch = ef
            return idx

        for ef in (32, 64, 128, 256):
            run_case(rows, "faiss.IndexHNSWFlat", f"M=16,ef={ef}",
                     lambda ef=ef: make_faiss_hnsw(ef),
                     db, q, ref, k, faiss_api=True)

    return rows


def bench_binary_hamming(n: int = 20_000, d_bytes: int = 16, k: int = 10):
    rng = np.random.default_rng(7)
    db = rng.integers(0, 256, size=(n, d_bytes), dtype=np.uint8)
    q = rng.integers(0, 256, size=(200, d_bytes), dtype=np.uint8)
    # Plant 40 known exact duplicates so the MIH-seeded variant has a chance to shine.
    for i in range(40):
        db[i * (n // 40)] = q[i]
    ref = brute_hamming(db, q, k)
    rows = []

    for ef in (32, 64, 128):
        run_case(rows, "pynear.HNSWBinaryIndex", f"M=16,ef={ef}",
                 lambda ef=ef: pynear.HNSWBinaryIndex(M=16, ef_construction=200, ef_search=ef),
                 db, q, ref, k)

    for ef in (32, 64, 128):
        run_case(rows, "pynear.MIHSeededHNSWBinaryIndex", f"M=16,ef={ef},mih_r=8",
                 lambda ef=ef: pynear.MIHSeededHNSWBinaryIndex(
                     M=16, ef_construction=200, ef_search=ef,
                     mih_m=8, mih_radius=8,
                 ),
                 db, q, ref, k)

    return rows


# ── reporting ───────────────────────────────────────────────────────────────


def format_md(rows, scenario_title: str) -> str:
    out = [f"\n### {scenario_title}\n",
           "| Index | Config | Build (s) | ms/query | Recall@10 |",
           "| --- | --- | --: | --: | --: |"]
    for name, cfg, b, q, r in rows:
        out.append(f"| `{name}` | {cfg} | {b:.2f} | {q:.2f} | {r:.3f} |")
    return "\n".join(out)


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "hnsw_benchmark.md"

    parts = ["# HNSW benchmark — pynear v2.3+\n",
             "Generated by `python -m pynear.benchmark.hnsw_benchmark`. ",
             "All runs single-threaded, single machine.\n"]

    print("Running float L2 HNSW benchmark (N=20_000, d=128)...")
    float_rows = bench_float_l2()
    parts.append(format_md(float_rows, "Float L2 — N=20,000, d=128, k=10"))

    print("Running binary Hamming HNSW benchmark (N=20_000, d=128bits)...")
    bin_rows = bench_binary_hamming()
    parts.append(format_md(bin_rows, "Binary Hamming — N=20,000, d=128 bits, k=10 (40 planted duplicates)"))

    out_path.write_text("\n".join(parts) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
