# pynear HNSW vs Faiss HNSW

N=100,000 x 128-D float32 (clustered), 1000 queries, k=10, M=16, ef_construction=200, 24 threads, best of 5 batch runs. Faiss measured in a faiss-only subprocess (OpenMP-runtime isolation; see faiss_comparison.md).

| Index | Build (s) | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 | ef=512 | ef=1024 |
|---|---|---|---|---|---|---|---|---|
| pynear HNSWL2Index | 1.16 | 328,003 QPS @ 0.721 | 218,085 QPS @ 0.873 | 130,347 QPS @ 0.969 | 91,457 QPS @ 0.994 | 58,703 QPS @ 0.998 | 41,107 QPS @ 0.998 | 31,076 QPS @ 0.998 |
| pynear HNSWL2IndexSQ8 | 0.68 | 722,626 QPS @ 0.688 | 483,515 QPS @ 0.811 | 303,362 QPS @ 0.871 | 180,001 QPS @ 0.885 | 103,056 QPS @ 0.889 | 72,326 QPS @ 0.889 | 62,117 QPS @ 0.889 |
| faiss IndexHNSWFlat | 1.02 | 553,821 QPS @ 0.726 | 335,004 QPS @ 0.875 | 203,156 QPS @ 0.967 | 129,842 QPS @ 0.995 | 90,634 QPS @ 1.000 | 61,006 QPS @ 1.000 | 28,928 QPS @ 1.000 |
| faiss IndexHNSWSQ(8bit) | 0.84 | 1,184,604 QPS @ 0.715 | 731,426 QPS @ 0.848 | 438,913 QPS @ 0.921 | 277,383 QPS @ 0.942 | 155,581 QPS @ 0.944 | 92,057 QPS @ 0.944 | 38,463 QPS @ 0.944 |
