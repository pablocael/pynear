# Approximate Search and Recall

VPForest indices trade **recall** for **speed** by searching only a subset
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

## Why does VPForest lose recall?

VPForest partitions the dataset into `n_clusters` Voronoi cells.  A query
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

Compare VPForest results against brute-force (or `VPTreeL2Index` at small
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
approx = pynear.VPForestL2Index(n_clusters=224, n_probe=20)
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

More clusters → smaller cells → each VPTree search is faster, but recall
drops faster as `n_probe` decreases.  Fewer clusters → larger cells → each
VPTree search is slower, but recall is more robust to small `n_probe`.

### Choosing n_probe

Use the recall measurement above to sweep `n_probe` and pick the
smallest value that meets your recall target:

```python
for n_probe in [1, 5, 10, 20, 30, 50, 100]:
    approx = pynear.VPForestL2Index(n_clusters=224, n_probe=n_probe)
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

## When to use exact vs approximate

| Situation | Recommendation |
|---|---|
| Dimensionality ≤ 128-D | `VPTreeL2Index` — exact and fast |
| Dimensionality 256-D – 1024-D, N > 50 K | `VPForestL2Index` with `n_probe` tuned to recall target |
| Need guaranteed exact results at any dimensionality | `VPForestL2Index` with `n_probe = n_clusters` |
| Binary descriptors (ORB, BRIEF) | `VPTreeBinaryIndex` (exact) or `BKTreeBinaryIndex` (range) |

---

## Making VPForest exact

Setting `n_probe = n_clusters` guarantees exact results regardless of
dimensionality:

```python
index = pynear.VPForestL2Index(n_clusters=316, n_probe=316)
index.set(data)  # now fully exact — every cluster is probed
```

At that point VPForest behaves like a partitioned exact search and will be
somewhat slower than a single `VPTreeL2Index` for low-dimensional data (due
to the clustering overhead), but can be faster for very high-dimensional data
where the single VPTree's pruning efficiency degrades.
