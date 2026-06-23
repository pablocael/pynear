## Real-World Benchmark — SIFT1M Binary

Performance of pynear's approximate Hamming-distance indices on the
[INRIA TEXMEX SIFT1M](http://corpus-texmex.irisa.fr/) dataset:
1,000,000 × 128-dim float SIFT descriptors sign-quantised to **128-bit binary**
(16 bytes/descriptor).  Ground truth computed by exact brute-force Hamming k-NN
over 500 queries, k=10.  Machine: Intel(R) Core(TM) Ultra 9 285K.

The baseline below is a *naive* numpy scan. For the apples-to-apples comparison
against Faiss's optimised brute-force (`IndexBinaryFlat`) and Faiss's own
Multi-Index Hashing, see
[results/faiss_comparison.md](faiss_comparison.md).

![QPS vs Recall@10](binary_benchmark_qps.png)

| Index                     | Configuration         | Build (s) | ms / query | QPS   | Recall@10 |
| ------------------------- | --------------------- | --------- | ---------- | ----- | --------- |
| numpy brute-force (naive) | N=1,000,000           | —         | 50.1       | 20    | 1.000     |
| IVFFlatBinaryIndex        | nlist=500, nprobe=31  | 6.24      | 1.47       | 679   | 0.825     |
| IVFFlatBinaryIndex        | nlist=500, nprobe=62  | 6.24      | 2.85       | 351   | 0.842     |
| IVFFlatBinaryIndex        | nlist=500, nprobe=125 | 6.24      | 5.65       | 177   | 0.845     |
| IVFFlatBinaryIndex        | nlist=500, nprobe=250 | 6.24      | 10.74      | 93    | 0.845     |
| IVFFlatBinaryIndex        | nlist=500, nprobe=500 | 6.24      | 20.95      | 48    | 0.845     |
| MIHBinaryIndex            | m=8, radius=4         | 2.81      | 0.09       | 10825 | 0.585     |
| MIHBinaryIndex            | m=8, radius=8         | 2.81      | 0.97       | 1031  | 0.829     |
| MIHBinaryIndex            | m=8, radius=12        | 2.81      | 0.95       | 1053  | 0.829     |
| MIHBinaryIndex            | m=8, radius=16        | 2.81      | 4.73       | 211   | 0.842     |
| MIHBinaryIndex            | m=8, radius=24        | 2.81      | 12.37      | 81    | 0.844     |
| MIHBinaryIndex            | m=8, radius=32        | 2.81      | 19.79      | 51    | 0.843     |
| MIHBinaryIndex            | m=8, radius=48        | 2.81      | 36.34      | 28    | 0.843     |

> Recall@10 is the standard `|returned ∩ true| / k`, measured against a fixed
> exact-Hamming ground truth. Because Hamming distances are integers, the
> 10-th-nearest boundary is often tied, so even an exact scan can score below
> 1.0 against this reference — the value reflects tie-breaking, not missed
> neighbours.

**Key takeaways:**
- `IVFFlatBinaryIndex` (nprobe=125) reaches Recall@10=0.845 at **177 QPS** (**9× faster than the naive numpy scan**).
- `MIHBinaryIndex` (radius=4) is the lowest-latency single configuration at **10825 QPS** (Recall@10=0.585).
- MIH's real advantage shows on **wide descriptors (256–512-bit)** and
  **small-radius / near-duplicate** retrieval. On narrow 128-bit data at high
  recall, an optimised brute-force scan can outperform it — pick the index to
  the workload.

> **Reproduce:** `python demo_binary.py` · add `--small` for a 10 K quick test · `--n-gt-queries N` to adjust evaluation size.
