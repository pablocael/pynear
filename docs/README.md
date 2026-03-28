# Basic Usage

## Available Indices

PyNear provides three families of indices: **exact** (VPTree / BKTree),
**approximate float** (IVFFlatL2Index), and **approximate binary**
(IVFFlatBinaryIndex / MIHBinaryIndex).

### Exact KNN Indices

Results are always correct — every returned neighbour is truly among the k
closest points in the dataset.

| Index | Distance | Input dtype | Notes |
|---|---|---|---|
| `pynear.VPTreeL2Index` | L2 (Euclidean) | `float32` | SIMD-accelerated on x86-64 |
| `pynear.VPTreeL1Index` | L1 (Manhattan) | `float32` | SIMD-accelerated on x86-64 |
| `pynear.VPTreeChebyshevIndex` | L∞ (Chebyshev) | `float32` | SIMD-accelerated on x86-64 |
| `pynear.VPTreeBinaryIndex` | Hamming | `uint8` | Hardware popcount; any byte-aligned dimension |

All VPTree indices support:
- `set(data)` — build the index from a 2-D NumPy array
- `searchKNN(queries, k)` — return the k nearest neighbours for each query
- `search1NN(queries)` — return the single nearest neighbour (faster than `searchKNN` with `k=1`)
- `to_string()` — print tree structure (slow, for debugging only)

### Approximate KNN Indices — Float (IVFFlatL2Index)

Partitions data into Voronoi clusters via K-Means.  Each query probes only
the `n_probe` nearest clusters and performs a BLAS-backed flat scan inside
each one, trading a small, configurable recall loss for a large speed gain.
Best for **high-dimensional float embeddings** (512-D – 1024-D).

| Index | Distance | Input dtype | Notes |
|---|---|---|---|
| `pynear.IVFFlatL2Index` | L2 (Euclidean) | `float32` | BLAS SGEMV inner scan; best for 512-D – 1024-D |

Setting `n_probe == n_clusters` makes the search exact.

See [Approximate search and recall](./approximate.md) for a full guide on
choosing `n_clusters`, measuring recall, and tuning `n_probe`.

### Approximate KNN Indices — Binary (Hamming)

Two indices for approximate Hamming-distance search on binary descriptors
(ORB, BRIEF, AKAZE, etc.).  Both are dramatically faster than exact
brute-force at high dimensionality.

| Index | Distance | Input dtype | Notes |
|---|---|---|---|
| `pynear.IVFFlatBinaryIndex` | Hamming | `uint8` | Binary K-Means IVF; predictable cost per query |
| `pynear.MIHBinaryIndex` | Hamming | `uint8` | Multi-Index Hashing; **257× faster** than brute-force at N=1M, d=512 |

**`IVFFlatBinaryIndex`** clusters data with binary K-Means (majority-vote
centroids) and scans `nprobe` clusters per query with POPCNT.  Good when the
query radius is unknown or the data distribution is non-uniform.

**`MIHBinaryIndex`** splits each descriptor into `m` sub-strings and builds
`m` hash tables.  At query time it enumerates all sub-strings within Hamming
radius `floor(radius/m)` per table — the *pigeonhole principle* guarantees
that every true neighbour within `radius` bits is found.  This makes it
ideal for near-duplicate image retrieval where true matches lie within a
small Hamming radius.

**Choosing `m`:**

| Descriptor width | Recommended `m` | Sub-string bits | Lookups/query |
|---|---|---|---|
| 128-bit (16 B) | 4 | 32 | ~2 116 |
| 256-bit (32 B) | 4 | 64 | ~8 324 |
| 512-bit (64 B) | 8 | 64 | 520 |

Rule: choose `m` so that `nbytes / m ≤ 8` (sub-string fits in `uint64_t`)
and `floor(radius / m) ≤ 2` to keep the enumeration tractable.

### Threshold-Based Indices

| Index | Distance | Input dtype | Notes |
|---|---|---|---|
| `pynear.BKTreeBinaryIndex` | Hamming | `uint8` | Range queries within a distance threshold |

---

## Usage Examples

### `pynear.VPTreeL2Index`

```python
import numpy as np
import pynear

dimension = 32
num_points = 10_000
data = np.random.rand(num_points, dimension).astype(np.float32)

num_queries = 8
queries = np.random.rand(num_queries, dimension).astype(np.float32)

k = 5

index = pynear.VPTreeL2Index()
index.set(data)

# searchKNN returns a tuple of (indices, distances)
# each is a list of lists, one entry per query
indices, distances = index.searchKNN(queries, k)

# search1NN returns (indices, distances) as flat lists, one entry per query
nn_indices, nn_distances = index.search1NN(queries)
```

Usage is analogous for `VPTreeL1Index` and `VPTreeChebyshevIndex`.

---

### `pynear.VPTreeBinaryIndex`

Binary vectors are stored as `uint8` arrays where each byte holds 8 bits.
A `dimension` of 32 bytes represents a 256-bit descriptor.

```python
import numpy as np
import pynear

dimension = 32   # bytes → 256-bit vectors
num_points = 10_000
data = np.random.randint(0, 256, size=(num_points, dimension), dtype=np.uint8)

num_queries = 8
queries = np.random.randint(0, 256, size=(num_queries, dimension), dtype=np.uint8)

k = 5

index = pynear.VPTreeBinaryIndex()
index.set(data)

indices, distances = index.searchKNN(queries, k)
```

---

### `pynear.IVFFlatL2Index`

```python
import numpy as np
import pynear

# 100 000 image embeddings, 1024-D (e.g. CLIP, ResNet)
num_points = 100_000
dimension = 1024
data = np.random.rand(num_points, dimension).astype(np.float32)

# Rule of thumb: n_clusters ≈ sqrt(N), n_probe controls recall vs speed
index = pynear.IVFFlatL2Index(n_clusters=316, n_probe=20)
index.set(data)  # clusters data with K-Means, stores raw vectors per cluster

queries = np.random.rand(10, dimension).astype(np.float32)
indices, distances = index.searchKNN(queries, k=5)
```

Set `n_probe = n_clusters` for exact results at the cost of speed:

```python
index = pynear.IVFFlatL2Index(n_clusters=316, n_probe=316)  # exact
```

See [Approximate search and recall](./approximate.md) for how to measure
and tune recall for your dataset.

---

### `pynear.IVFFlatBinaryIndex`

Binary IVF for approximate Hamming KNN.  Good general-purpose choice when
the query radius is unknown or the data is non-uniform.

```python
import numpy as np
import pynear

# 500 000 × 512-bit descriptors (64 bytes each)
num_points = 500_000
data = np.random.randint(0, 256, size=(num_points, 64), dtype=np.uint8)

# nlist: number of clusters (rule of thumb: sqrt(N) to N/100)
# nprobe: clusters scanned per query — increase for higher recall
index = pynear.IVFFlatBinaryIndex(nlist=512, nprobe=16)
index.set(data)

queries = np.random.randint(0, 256, size=(10, 64), dtype=np.uint8)
indices, distances = index.searchKNN(queries, k=10)

# Tune nprobe at runtime without rebuilding
index.set_nprobe(32)   # higher recall, slower
index.set_nprobe(4)    # lower recall, faster
```

---

### `pynear.MIHBinaryIndex`

Multi-Index Hashing for near-duplicate binary descriptor retrieval.
Any true neighbour within `radius` Hamming bits is guaranteed to be found.
Extremely fast at d=512 — 257× faster than brute-force at N=1M.

```python
import numpy as np
import pynear

# 1M × 512-bit descriptors (e.g. ORB, BRIEF-512)
num_points = 1_000_000
data = np.random.randint(0, 256, size=(num_points, 64), dtype=np.uint8)

# m=8 for 512-bit (64B / 8 = 8-byte sub-strings)
# m=4 for 256-bit (32B / 4 = 8-byte sub-strings)
# m=4 for 128-bit (16B / 4 = 4-byte sub-strings)
index = pynear.MIHBinaryIndex(m=8)
index.set(data)

queries = np.random.randint(0, 256, size=(100, 64), dtype=np.uint8)

# radius: all true neighbours within this Hamming distance are guaranteed found.
# Increase radius if recall is too low; decrease for fewer candidates (faster).
indices, distances = index.searchKNN(queries, k=10, radius=8)

print(f"Index size: {index.n()} vectors, {index.nbytes()} bytes each, {index.m()} sub-tables")
```

**Performance at N=1M, d=512, k=10 (near-duplicate setting):**

| Method | Query time | Recall@10 |
|---|---|---|
| `MIHBinaryIndex` (m=8, radius=8) | **0.037 ms** | 100% |
| `IVFFlatBinaryIndex` (nlist=512, nprobe=16) | 1.95 ms | 100% |
| Faiss `IndexBinaryFlat` (exact) | 9.5 ms | 100% |

---

### `pynear.BKTreeBinaryIndex`

`BKTreeBinaryIndex` is designed for **range queries**: it returns all points whose Hamming distance to each query is at most `threshold`.

```python
import numpy as np
import pynear

dimension = 32   # bytes → 256-bit vectors
num_points = 10_000
data = np.random.randint(0, 256, size=(num_points, dimension), dtype=np.uint8)

num_queries = 4
queries = np.random.randint(0, 256, size=(num_queries, dimension), dtype=np.uint8)

index = pynear.BKTreeBinaryIndex()
index.set(data)

# find_threshold returns (indices, distances, keys) for each query
# threshold = dimension * 8 finds all matches (max possible Hamming distance)
threshold = dimension * 8
indices, distances, keys = index.find_threshold(queries, threshold)
```
