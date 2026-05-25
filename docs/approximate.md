# Approximate Search and Recall

IVFFlatL2Index trades **recall** for **speed** by searching only a subset
of the data.  This page explains what recall means, how to measure it, and
how to tune `n_clusters` and `n_probe` for your use case.

---

## What is recall?

**Recall@k** is the fraction of the true k nearest neighbours that your
index actually returns.  For a single query:

```
recall@k = |returned ∩ true_top_k| / k
```

A recall of 1.0 means the index returned exactly the correct k neighbours.
A recall of 0.8 means 2 out of every 10 expected results were missed on
average.

For a batch of Q queries, recall@k is the mean over all queries:

```
recall@k = (1/Q) * Σ  |returned_i ∩ true_top_k_i| / k
```

---

## Why does IVFFlatL2Index lose recall?

IVFFlatL2Index partitions the dataset into `n_clusters` Voronoi cells.  A query
probes only the `n_probe` nearest cells.  If a true nearest neighbour lives
in a cell that was not probed, it is missed.

The risk grows with:
- **Higher dimensionality** — Voronoi boundaries become fuzzier; points that
  are geometrically close can end up in different cells.
- **Fewer probes** — a smaller `n_probe` means fewer cells checked.
- **Smaller clusters** — more cells means each one holds fewer points, so
  missing one is more costly.

---

## How to measure recall

Compare IVFFlatL2Index results against brute-force (or `VPTreeL2Index` at small
scale) on a held-out sample of queries:

```python
import numpy as np
import pynear

rng = np.random.default_rng(42)
data    = rng.random((50_000, 128)).astype(np.float32)
queries = rng.random((200, 128)).astype(np.float32)
k = 10

# Ground truth — exact search on a small sample
exact = pynear.VPTreeL2Index()
exact.set(data)
true_idx, _ = exact.searchKNN(queries, k)

# Approximate index under test
approx = pynear.IVFFlatL2Index(n_clusters=224, n_probe=20)
approx.set(data)
approx_idx, _ = approx.searchKNN(queries, k)

# Recall@k
recall = np.mean([
    len(set(a) & set(t)) / k
    for a, t in zip(approx_idx, true_idx)
])
print(f"Recall@{k}: {recall:.3f}")   # e.g. 0.923
```

---

## Tuning n_clusters and n_probe

### Choosing n_clusters

A good starting point is `n_clusters ≈ sqrt(N)`:

| Dataset size N | Suggested n_clusters |
|---|---|
| 10 000 | 100 |
| 100 000 | 316 |
| 1 000 000 | 1 000 |

More clusters → smaller cells → each cluster scan is faster, but recall
drops faster as `n_probe` decreases.  Fewer clusters → larger cells → each
cluster scan is slower, but recall is more robust to small `n_probe`.

### Choosing n_probe

Use the recall measurement above to sweep `n_probe` and pick the
smallest value that meets your recall target:

```python
for n_probe in [1, 5, 10, 20, 30, 50, 100]:
    approx = pynear.IVFFlatL2Index(n_clusters=224, n_probe=n_probe)
    approx.set(data)
    approx_idx, _ = approx.searchKNN(queries, k)
    recall = np.mean([
        len(set(a) & set(t)) / k
        for a, t in zip(approx_idx, true_idx)
    ])
    print(f"n_probe={n_probe:4d}  recall@{k}={recall:.3f}")
```

Typical output for 128-D data:

```
n_probe=  1  recall@10=0.512
n_probe=  5  recall@10=0.781
n_probe= 10  recall@10=0.873
n_probe= 20  recall@10=0.931
n_probe= 30  recall@10=0.958
n_probe= 50  recall@10=0.981
n_probe=100  recall@10=0.997
```

### Practical guidelines

| Target recall | Typical n_probe / n_clusters |
|---|---|
| ≥ 95 % | ~ 15–25 % |
| ≥ 99 % | ~ 40–60 % |
| 100 % (exact) | 100 % (`n_probe == n_clusters`) |

These ratios are rough — always measure on your own data.

---

## Approximate Hamming search — `MIHBinaryIndex`

For **binary descriptors** (ORB, BRIEF, AKAZE, perceptual hashes, SimHash),
pynear provides `MIHBinaryIndex`, an implementation of **Multi-Index
Hashing**. Unlike `IVFFlatL2Index`, its accuracy is controlled by a single
`radius` parameter that carries an *exact* guarantee.

### How it works

A `d`-bit descriptor is split into `m` equal sub-strings of `d/m` bits, and
`m` hash tables map each sub-string to the points that contain it. At query
time MIH leans on the **pigeonhole principle**:

> If two descriptors are within Hamming distance `r`, then at least one of
> their `m` sub-strings must match within `⌊r / m⌋` bits.

So to find every neighbour within radius `r`, MIH only has to look up each
query sub-string in its table within the much smaller radius
`r_sub = ⌊r / m⌋`, union the candidates, and verify their full Hamming
distance with POPCNT. On wide descriptors that collects a tiny candidate set
instead of scanning all `N` points.

### The `radius` parameter

`radius` is the Hamming radius for candidate enumeration:

```python
mih = pynear.MIHBinaryIndex(m=4)   # m=4 for 128/256-bit, m=8 for 512-bit
mih.set(db)
idx, dist = mih.searchKNN(queries, k=10, radius=12)
```

It comes with an exact guarantee: **any** true neighbour within Hamming
distance ≤ `radius` is returned with probability 1 — there are no false
negatives inside the radius (this is the pigeonhole guarantee, not a
probabilistic bound like `nprobe`). Recall is only lost for true neighbours
that lie *beyond* the chosen radius. Larger `radius` → higher recall, more
candidates, slower:

| `radius` (m=4, 128-bit) | candidate set | recall@10 (SIFT1M) | relative speed |
|---|---|---|---|
| small (4–8)   | tiny     | partial — near-duplicates only | fastest |
| medium (12–16) | moderate | rising | moderate |
| large (20+)   | large    | approaches brute-force | slowest |

The measured recall–throughput curve is in the
[SIFT1M benchmark](../results/binary_benchmark.md).

### Choosing `m`

`m` must divide the descriptor's byte width, and each sub-string must fit in
a `uint64_t` (`d/m ≤ 64` bits):

| Descriptor width | Recommended `m` | Sub-string width |
|---|---|---|
| 128-bit (16 bytes) | 4 | 32 bits |
| 256-bit (32 bytes) | 4 | 64 bits |
| 512-bit (64 bytes) | 8 | 64 bits |

### When MIH wins — and when it doesn't

MIH is strongest when the candidate set stays small:

- **Near-duplicate retrieval** (small Hamming radius) — image/video dedup,
  copy detection, perceptual-hash lookup.
- **Wide descriptors** (256–512 bits) — sub-tables are sparse, so each
  lookup returns few candidates. On 512-bit near-duplicate workloads pynear's
  MIH runs ~40× faster than Faiss's brute-force `IndexBinaryFlat` and is far
  ahead of Faiss's own MIH (see
  [results/faiss_comparison.md](../results/faiss_comparison.md)).

It is *not* the right tool when you need **high recall on narrow, clustered
descriptors**. On 128-bit SIFT1M, pushing recall toward 1.0 forces a large
radius, the candidate set balloons, and an optimised brute-force POPCNT scan
becomes competitive or faster. In that regime prefer `IVFFlatBinaryIndex`
(predictable `nprobe` cost) or an exact `VPTreeBinaryIndex` / brute-force
scan.

---

## When to use exact vs approximate

| Situation | Recommendation |
|---|---|
| Dimensionality ≤ 128-D | `VPTreeL2Index` — exact and fast |
| Dimensionality 256-D – 1024-D, N > 50 K | `IVFFlatL2Index` with `n_probe` tuned to recall target |
| Need guaranteed exact results | `IVFFlatL2Index` with `n_probe = n_clusters` |
| Binary descriptors — near-duplicate / small radius | `MIHBinaryIndex` with `radius` tuned to recall target |
| Binary descriptors — general approximate KNN | `IVFFlatBinaryIndex` with `nprobe` tuned to recall target |
| Binary descriptors — exact / range | `VPTreeBinaryIndex` (exact) or `BKTreeBinaryIndex` (range) |

---

## Making IVFFlatL2Index exact

Setting `n_probe = n_clusters` guarantees exact results regardless of
dimensionality:

```python
index = pynear.IVFFlatL2Index(n_clusters=316, n_probe=316)
index.set(data)  # now fully exact — every cluster is probed
```

At that point it behaves like a partitioned exact search and will be
somewhat slower than a single `VPTreeL2Index` for low-dimensional data (due
to the clustering overhead), but faster for very high-dimensional data
where the VPTree's pruning efficiency degrades.

---

## Why approximate search? The curse of dimensionality

Tree-based exact search relies on pruning: a branch is discarded when its
closest possible point is provably farther than the current best candidate.
This pruning becomes ineffective as dimensionality grows — a phenomenon rooted
in a fundamental geometric property of high-dimensional spaces.

**Volume concentration near the boundary.**
Consider $N$ points drawn uniformly at random inside an $n$-dimensional ball
of radius $R$. A point at distance $r$ from the origin is closer to the
boundary than to the origin whenever $R - r < r$, i.e. $r > R/2$.
The fraction of the ball's volume satisfying this condition is:

$$F(n) = \frac{V_n(R) - V_n\left(\tfrac{R}{2}\right)}{V_n(R)} = 1 - \left(\frac{1}{2}\right)^{n}$$

where $V_n(r) = \dfrac{\pi^{n/2}}{\Gamma\left(\tfrac{n}{2}+1\right)} r^n$ is
the volume of an $n$-ball of radius $r$. Because $V_n$ scales as $r^n$, the
ratio simplifies cleanly to $1 - 2^{-n}$, independent of $R$.

**Median distance from the origin.**
The median distance $r_m$ is the radius such that exactly half the volume lies
within it:

$$\frac{V_n(r_m)}{V_n(R)} = \frac{1}{2}
\;\Longrightarrow\;
\left(\frac{r_m}{R}\right)^n = \frac{1}{2}
\;\Longrightarrow\;
r_m = R \cdot 2^{-1/n}$$

As $n \to \infty$, $r_m \to R$: the typical point is arbitrarily close to
the surface of the ball.

**Numerical illustration:**

| Dimensionality $n$ | Points closer to border than origin $F(n)$ | Median distance $r_m / R$ |
|:------------------:|:-------------------------------------------:|:-------------------------:|
| 1                  | 50.0 %                                      | 0.500                     |
| 2                  | 75.0 %                                      | 0.707                     |
| 5                  | 96.9 %                                      | 0.871                     |
| 10                 | 99.9 %                                      | 0.933                     |
| 100                | ≈ 100 %                                     | 0.993                     |

**Consequence for KNN trees.**
When $n$ is large, nearly all points are concentrated in a thin shell near
the boundary, and the distances between any two points become almost equal.
With no contrast in distances, a tree has nothing to prune — every branch
must be explored — and search degrades to exhaustive linear scan, $O(N)$.
This is the fundamental reason why exact tree search offers diminishing
returns beyond $d \approx 256$, and why approximate methods such as
**IVFFlatL2Index** (probing only a fraction of clusters) or Faiss IVF are
necessary at high dimensionalities.
