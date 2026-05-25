# pynear vs Faiss — binary index comparison

All measurements on **24 threads**, k=10, best of 5 timed runs.
pynear built with OpenMP; `faiss-cpu` set to 24 threads. Reproduce with
`python demo_faiss_comparison.py`.

> **OpenMP gotcha:** pynear links libgomp, `faiss-cpu` links libomp. Loaded in
> one process the two runtimes contend and serialise Faiss's parallel flat
> scan (~78× slower in-process here). MIH/IVF are unaffected. So Faiss
> `IndexBinaryFlat` is timed in a separate faiss-only subprocess for a fair
> number.

## 1. Where MIH shines — 512-bit near-duplicate retrieval

1,000,000 × 512-bit random descriptors; 500 queries are existing
rows with 5 random bits flipped. All configurations reach **100% Recall@10**.

| Index | ms / query | QPS | vs Faiss brute-force |
| --- | --- | --- | --- |
| **pynear `MIHBinaryIndex`** (m=8, radius=4) | **0.0083** | **120,515** | **~39× faster** |
| pynear `IVFFlatBinaryIndex` (nlist=512, nprobe=16) | 1.824 | 548 | 0.18× |
| Faiss `IndexBinaryFlat` (exact brute-force) | 0.3208 | 3,118 | 1× (baseline) |
| Faiss `IndexBinaryMultiHash` | 22.113 | 45 | 0.01× |

On wide descriptors, `MIHBinaryIndex` finds near-duplicates at 100% recall
**~39× faster than Faiss's exact brute-force scan**, while Faiss's
own Multi-Index Hashing is not competitive at this width.

## 2. SIFT1M (128-bit) — pynear MIH vs Faiss MIH at matched recall

1,000,000 × 128-bit sign-quantised SIFT descriptors, 500 queries.
At matched recall, `MIHBinaryIndex` is consistently faster than Faiss's own
`IndexBinaryMultiHash`:

| Recall@10 | pynear MIH (m=4) | Faiss MIH (nhash=4) | speedup |
| --- | --- | --- | --- |
| 0.28 | 86,376 (r=6) | 47,875 (nflip=1) | 1.80× |
| 0.53 | 23,481 (r=8) | 15,500 (nflip=2) | 1.51× |
| 0.73 | 4,690 (r=14) | 3,032 (nflip=3) | 1.55× |
| 0.82 | 809 (r=16) | 630 (nflip=4) | 1.28× |
| 0.84 | 171 (r=20) | 154 (nflip=5) | 1.11× |
| 0.84 | 43 (r=24) | 44 (nflip=6) | 0.97× |

**The honest caveat:** on *narrow* 128-bit descriptors, an optimised
brute-force POPCNT scan is hard to beat — Faiss `IndexBinaryFlat` does
**22,559 QPS** (exact) here, faster than either MIH
implementation above the ~0.5 recall mark. Multi-Index Hashing earns its keep
on **wide descriptors** and **small-radius / near-duplicate** retrieval, as the
512-bit table shows. For high-recall search on narrow descriptors, prefer
brute force or `IVFFlatBinaryIndex`.

> Recall@10 on SIFT1M is the standard `|returned ∩ true| / k` against a fixed
> exact-Hamming ground truth. Because Hamming distances are integers, the
> 10-th-nearest boundary is frequently tied, so the recall ceiling (~0.84
> here) reflects tie-breaking against that reference, not missed neighbours.
