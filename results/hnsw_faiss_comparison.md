# pynear HNSW vs Faiss HNSW

N=100,000 x 128-D float32 (clustered), 1000 queries, k=10, M=16, ef_construction=200, 24 threads, best of 5 batch runs. Faiss measured in a faiss-only subprocess (OpenMP-runtime isolation; see faiss_comparison.md).

| Index | Build (s) | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 | ef=512 | ef=1024 |
|---|---|---|---|---|---|---|---|---|
| pynear HNSWL2Index | 0.92 | 676,194 QPS @ 0.314 | 458,828 QPS @ 0.427 | 261,511 QPS @ 0.551 | 146,987 QPS @ 0.683 | 86,045 QPS @ 0.773 | 55,735 QPS @ 0.819 | 40,248 QPS @ 0.851 |
| pynear HNSWL2IndexSQ8 | 0.95 | 645,550 QPS @ 0.331 | 425,707 QPS @ 0.440 | 254,610 QPS @ 0.547 | 140,701 QPS @ 0.626 | 74,488 QPS @ 0.694 | 43,953 QPS @ 0.739 | 28,213 QPS @ 0.796 |
| faiss IndexHNSWFlat | 0.78 | 1,189,162 QPS @ 0.308 | 729,124 QPS @ 0.429 | 395,743 QPS @ 0.569 | 214,315 QPS @ 0.694 | 122,969 QPS @ 0.772 | 64,446 QPS @ 0.843 | 23,327 QPS @ 0.892 |
| faiss IndexHNSWSQ(8bit) | 0.66 | 1,368,530 QPS @ 0.286 | 863,315 QPS @ 0.396 | 477,410 QPS @ 0.537 | 99,451 QPS @ 0.659 | 47,666 QPS @ 0.742 | 27,989 QPS @ 0.815 | 30,965 QPS @ 0.868 |
