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

| Index                     | Configuration         | Build (s) | ms / query | QPS    | Recall@10 |
| ------------------------- | --------------------- | --------- | ---------- | ------ | --------- |
| numpy brute-force (naive) | N=1,000,000           | —         | 47.7       | 21     | 1.000     |
| IVFFlatBinaryIndex        | nlist=500, nprobe=31  | 3.10      | 0.01       | 125776 | 0.825     |
| IVFFlatBinaryIndex        | nlist=500, nprobe=62  | 3.10      | 0.01       | 87783  | 0.842     |
| IVFFlatBinaryIndex        | nlist=500, nprobe=125 | 3.10      | 0.02       | 56859  | 0.845     |
| IVFFlatBinaryIndex        | nlist=500, nprobe=250 | 3.10      | 0.03       | 34433  | 0.845     |
| IVFFlatBinaryIndex        | nlist=500, nprobe=500 | 3.10      | 0.05       | 19100  | 0.845     |
| MIHBinaryIndex            | m=8, radius=4         | 2.64      | 0.03       | 38554  | 0.466     |
| MIHBinaryIndex            | m=8, radius=8         | 2.64      | 0.06       | 18158  | 0.652     |
| MIHBinaryIndex            | m=8, radius=12        | 2.64      | 0.14       | 7326   | 0.799     |
| MIHBinaryIndex            | m=8, radius=16        | 2.64      | 0.24       | 4254   | 0.832     |
| MIHBinaryIndex            | m=8, radius=24        | 2.64      | 0.65       | 1541   | 0.841     |
| MIHBinaryIndex            | m=8, radius=32        | 2.64      | 1.37       | 731    | 0.840     |
| MIHBinaryIndex            | m=8, radius=48        | 2.64      | 3.54       | 282    | 0.840     |

> Recall@10 is the standard `|returned ∩ true| / k`, measured against a fixed
> exact-Hamming ground truth. Because Hamming distances are integers, the
> 10-th-nearest boundary is often tied, so even an exact scan can score below
> 1.0 against this reference — the value reflects tie-breaking, not missed
> neighbours.

**Key takeaways:**
- `IVFFlatBinaryIndex` (nprobe=125) reaches Recall@10=0.845 at **56859 QPS** (**2385× faster than the naive numpy scan**).
- `MIHBinaryIndex` (radius=4) is the lowest-latency single configuration at **38554 QPS** (Recall@10=0.466).
- MIH's real advantage shows on **wide descriptors (256–512-bit)** and
  **small-radius / near-duplicate** retrieval. On narrow 128-bit data at high
  recall, an optimised brute-force scan can outperform it — pick the index to
  the workload.

> **Reproduce:** `python demo_binary.py` · add `--small` for a 10 K quick test · `--n-gt-queries N` to adjust evaluation size.
