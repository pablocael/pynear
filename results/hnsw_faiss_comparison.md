# pynear HNSW vs Faiss HNSW

N=100,000 x 128-D float32 (clustered, queries drawn from the same cluster
distribution), 1000 queries, k=10, M=16, ef_construction=200, 24 threads for
both systems, best of 5 batch runs per point. Faiss measured in a faiss-only
subprocess (OpenMP-runtime isolation; see faiss_comparison.md).
Reproduce: `python -m pynear.benchmark.hnsw_faiss_benchmark`.

Sourcing note: each system's row is taken from a clean run (all sweeps
monotone: recall non-decreasing and QPS non-increasing in ef). Ambient
desktop load prevented one single combined clean run; the Faiss rows are
cross-validated by two independent clean runs agreeing within ~3%.

| Index | Build (s) | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 | ef=512 | ef=1024 |
|---|---|---|---|---|---|---|---|---|
| pynear HNSWL2Index | 1.22 | 359,084 QPS @ 0.717 | 217,249 QPS @ 0.868 | 131,038 QPS @ 0.964 | 85,711 QPS @ 0.991 | 58,235 QPS @ 0.996 | 39,090 QPS @ 0.996 | 30,712 QPS @ 0.996 |
| pynear HNSWL2IndexSQ8 | 0.67 | 568,756 QPS @ 0.693 | 401,026 QPS @ 0.838 | 291,061 QPS @ 0.911 | 174,511 QPS @ 0.936 | 100,730 QPS @ 0.940 | 75,773 QPS @ 0.940 | 60,775 QPS @ 0.940 |
| faiss IndexHNSWFlat | 1.02 | 553,821 QPS @ 0.726 | 335,004 QPS @ 0.875 | 203,156 QPS @ 0.967 | 129,842 QPS @ 0.995 | 90,634 QPS @ 1.000 | 61,006 QPS @ 1.000 | 28,928 QPS @ 1.000 |
| faiss IndexHNSWSQ(8bit) | 0.84 | 1,184,604 QPS @ 0.715 | 731,426 QPS @ 0.848 | 438,913 QPS @ 0.921 | 277,383 QPS @ 0.942 | 155,581 QPS @ 0.944 | 92,057 QPS @ 0.944 | 38,463 QPS @ 0.944 |

Since v2.5.0: SQ8 uses per-dimension affine quantization with asymmetric
(float-query vs decoded-code) search — recall ceiling 0.889 -> 0.940
(faiss: 0.944) — and the search path is allocation-free (pooled scratch,
direct-to-numpy results), which pays for the decode kernel: SQ8 QPS at
ef>=64 matches the old lower-recall kernel. pynear's SQ8 now tracks or
beats faiss's *float* index up to ~0.91 recall at 4x less vector memory.
