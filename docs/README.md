# Basic Usage

## Available Indices

PyNear provides two families of indices: **exact** (VPTree / BKTree) and
**approximate** (VPForest).

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

### Approximate KNN Indices (VPForest)

VPForest indices partition data into clusters and build one VPTree per
cluster.  Each query probes only the `n_probe` nearest clusters, trading a
small, configurable recall loss for a large speed gain.  This is the right
choice for **high-dimensional data** (512-D / 1024-D image or text
embeddings) where a single VPTree would explore most of the dataset anyway.

| Index | Distance | Input dtype | Notes |
|---|---|---|---|
| `pynear.VPForestL2Index` | L2 (Euclidean) | `float32` | IVF-style; tunable recall vs speed |
| `pynear.VPForestL1Index` | L1 (Manhattan) | `float32` | IVF-style; tunable recall vs speed |
| `pynear.VPForestChebyshevIndex` | L∞ (Chebyshev) | `float32` | IVF-style; tunable recall vs speed |

VPForest indices share the same `set` / `searchKNN` / `search1NN` API as
VPTree indices.  Setting `n_probe == n_clusters` makes the search exact.

See [Approximate search and recall](./approximate.md) for a full guide on
choosing `n_clusters`, measuring recall, and tuning `n_probe`.

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

### `pynear.VPForestL2Index`

```python
import numpy as np
import pynear

# 100 000 image embeddings, 1024-D (e.g. CLIP, ResNet)
num_points = 100_000
dimension = 1024
data = np.random.rand(num_points, dimension).astype(np.float32)

# Rule of thumb: n_clusters ≈ sqrt(N), n_probe controls recall vs speed
index = pynear.VPForestL2Index(n_clusters=316, n_probe=20)
index.set(data)  # clusters data with K-Means, builds one VPTree per cluster

queries = np.random.rand(10, dimension).astype(np.float32)
indices, distances = index.searchKNN(queries, k=5)
```

Set `n_probe = n_clusters` for exact results at the cost of speed:

```python
index = pynear.VPForestL2Index(n_clusters=316, n_probe=316)  # exact
```

See [Approximate search and recall](./approximate.md) for how to measure
and tune recall for your dataset.

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
