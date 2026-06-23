# PyNear vs Faiss, Annoy, scikit-learn

Full feature matrix. The README carries an abridged 4-row version; this page is the source of truth.

| | PyNear | Faiss | Annoy | scikit-learn |
|---|---|---|---|---|
| **Exact results** | ✅ VPTree always | ✅ flat index | ❌ approximate | ✅ |
| **Approximate (fast, tunable)** | ✅ IVFFlatL2Index | ✅ IVF | ✅ | ❌ |
| **Metric agnostic** | ✅ L2, L1, L∞, cosine, Hamming | L2 / inner product / cosine | L2 / cosine / Hamming | L2 / others |
| **Low-dim sweet spot** | ✅ | ❌ | ❌ | ❌ |
| **High-dim (512-D – 1024-D)** | ✅ IVFFlatL2Index | ✅ | ✅ | ❌ |
| **Binary / Hamming exact** | ✅ hardware popcount | ✅ | ✅ | ❌ |
| **Binary / Hamming approx** | ✅ MIH + IVFFlat (faster than Faiss MIH) | ✅ MIH + IVF | ❌ | ❌ |
| **Threshold / range search** | ✅ BKTree | ❌ | ❌ | ❌ |
| **Pickle serialization** | ✅ | ❌ | ✅ | ✅ |
| **No extra native deps** | ✅ NumPy only | ❌ compiled lib + optional GPU | ❌ | ❌ |
| **scikit-learn compatible API** | ✅ drop-in adapters | ❌ | ❌ | — |

## When PyNear is the right choice

- You want **exact** answers without giving up speed (VP-Tree pruning + SIMD beats brute-force well past where naïve kd-trees collapse).
- You're using **binary descriptors** (ORB / BRIEF / AKAZE / perceptual hashes / SimHash). On 512-bit near-duplicate retrieval, `MIHBinaryIndex` is **~40× faster than Faiss's brute-force `IndexBinaryFlat`** at 100% Recall@10, and **1.3–1.6× faster than Faiss's own `IndexBinaryMultiHash`** at matched recall on SIFT1M (128-bit). See [results/faiss_comparison.md](../results/faiss_comparison.md). (On *narrow* descriptors at high recall, an optimised brute-force scan can still win — match the index to the workload.)
- You need **threshold / range queries** (Hamming radius search) — `BKTreeBinaryIndex` is the only option here outside of pynear.
- You want a **scikit-learn drop-in** that's faster than `sklearn.neighbors` without rewriting your training pipeline.
- You want a **wheel-only install** with no system-level BLAS/Faiss dependency.

## When to reach for Faiss instead

- Database is >10M vectors and memory matters — Faiss has PQ / OPQ compression; pynear doesn't (yet).
- You need GPU inference — Faiss has CUDA backends; pynear is CPU-only.
- You're inside a heavy ML stack that already pulls Faiss in transitively.
