
# PyNear

[![PyPI version](https://img.shields.io/pypi/v/pynear)](https://pypi.org/project/pynear/)
[![Python versions](https://img.shields.io/pypi/pyversions/pynear)](https://pypi.org/project/pynear/)
[![CI](https://github.com/pablocael/pynear/actions/workflows/pythonpackage.yml/badge.svg)](https://github.com/pablocael/pynear/actions/workflows/pythonpackage.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/pablocael/pynear?style=social)](https://github.com/pablocael/pynear/stargazers)

> **Fast KNN that doesn't make you choose between exact answers and production speed.**
> Drop-in for scikit-learn. SIMD-accelerated. Up to **257× faster than Faiss** on binary descriptors at 100% recall.

![PyNear demo](docs/img/demo.gif)

```python
import numpy as np, pynear

db      = np.random.rand(1_000_000, 128).astype(np.float32)
queries = np.random.rand(10, 128).astype(np.float32)

index = pynear.VPTreeL2Index()
index.set(db)
indices, distances = index.searchKNN(queries, k=5)
```

```console
pip install pynear     # pre-built wheels, no compiler needed
```

A metric-space KNN library with a C++ core. **VP-Trees** for exact search up to ~256-D, **IVF-Flat** for fast approximate float search at 512–1024-D, **MIH** and **IVF-Binary** for Hamming search on image descriptors. Already on scikit-learn? [Switch with a one-line import change.](#migrating-from-scikit-learn)

---

## Table of Contents

- [Installation](#installation)
- [Quick start](#quick-start)
  - [Approximate binary search (image descriptors)](#approximate-binary-search-image-descriptors)
- [Migrating from scikit-learn](#migrating-from-scikit-learn)
- [Features](#features)
  - [Available indices](#available-indices)
  - [Pickle serialisation](#pickle-serialisation)
  - [Tree inspection](#tree-inspection)
- [Demos](#demos)
- [Benchmarks](#benchmarks)
- [Real-World Benchmark — SIFT1M Binary](#real-world-benchmark--sift1m-binary)
- [Development](#development)

---

### Why PyNear?

| | PyNear | Faiss | Annoy | scikit-learn |
|---|---|---|---|---|
| **Metric agnostic** | ✅ L2, L1, L∞, cosine, Hamming | L2 / IP / cosine | L2 / cosine / Hamming | L2 / others |
| **Binary / Hamming approx** | ✅ 257× faster than brute-force | ⚠️ slow build | ❌ | ❌ |
| **scikit-learn drop-in** | ✅ adapter classes | ❌ | ❌ | — |
| **Zero native deps** | ✅ NumPy only | ❌ compiled lib + optional GPU | ❌ | ❌ |

[Full comparison →](./docs/comparison.md)

PyNear covers the full spectrum: **VPTree** indices for guaranteed exact answers (2-D to ~256-D), **IVFFlatL2Index** / **IVFFlatCosineIndex** for fast approximate float search at 512–1024-D (text embeddings, RAG), and **MIHBinaryIndex** / **IVFFlatBinaryIndex** for approximate Hamming search on binary descriptors.

> New to KNN? See [docs/intro.md](./docs/intro.md) for a gentle, jargon-free introduction.

### What people build with PyNear

| | | |
|:---|:---|:---|
| **Image / video dedup** | **Drop-in for sklearn** | **Interactive visualisation** |
| Encode with perceptual hash / ORB / SimHash, index with `MIHBinaryIndex`, find near-duplicates **257× faster** than brute-force. | Swap `sklearn.neighbors.KNeighborsClassifier` for `PyNearKNeighborsClassifier`. Same API, same results, faster. | Two desktop demos: a 1M-point KNN explorer and a live Voronoi diagram you can drag seeds in. |
| → [`demo_binary.py`](./demo_binary.py) | → [Migration guide](#migrating-from-scikit-learn) | → [`demo/`](./demo) |

---

## Installation

```console
pip install pynear
```

Requires Python 3.8+ and NumPy ≥ 1.21.2.  Pre-built wheels are available for
Linux, macOS (x86-64 and Apple Silicon), and Windows — no compiler needed.

---

## Quick start

```python
import numpy as np
import pynear

# Build index from 100 000 vectors of dimension 32
data = np.random.rand(100_000, 32).astype(np.float32)
index = pynear.VPTreeL2Index()
index.set(data)

# KNN search — returns (indices, distances) per query, sorted nearest-first
queries = np.random.rand(10, 32).astype(np.float32)
indices, distances = index.searchKNN(queries, k=5)

# 1-NN shortcut (slightly faster than searchKNN with k=1)
nn_indices, nn_distances = index.search1NN(queries)
```

For all index types and advanced usage see [docs/README.md](./docs/README.md).

### Approximate binary search (image descriptors)

For large-scale image retrieval with binary descriptors (ORB, BRIEF, AKAZE),
PyNear provides two approximate Hamming-distance indices that are orders of
magnitude faster than exact brute-force:

```python
import numpy as np
import pynear

# 1M × 512-bit descriptors (64 bytes each)
db = np.random.randint(0, 256, size=(1_000_000, 64), dtype=np.uint8)

# ── Multi-Index Hashing ───────────────────────────────────────────────────────
# Best for d=512 (m=8 sub-tables of 64 bits).
# 257× faster than brute-force at N=1M; 100% Recall@10 for near-duplicates.
mih = pynear.MIHBinaryIndex(m=8)   # m=4 for d=128/256, m=8 for d=512
mih.set(db)

queries = np.random.randint(0, 256, size=(100, 64), dtype=np.uint8)
indices, distances = mih.searchKNN(queries, k=10, radius=8)
# radius: any true neighbour within Hamming distance ≤ radius is guaranteed
# to be found (pigeonhole principle). Increase for higher recall on noisier data.

# ── IVF Flat Binary ───────────────────────────────────────────────────────────
# Predictable cost: scans nprobe clusters per query.
# Good when the query radius is unknown or data is non-uniform.
ivf = pynear.IVFFlatBinaryIndex(nlist=512, nprobe=16)
ivf.set(db)

indices, distances = ivf.searchKNN(queries, k=10)
ivf.set_nprobe(32)  # increase nprobe at runtime to trade speed for recall
```

**Choosing between MIH and IVFFlat:**

| | `MIHBinaryIndex` | `IVFFlatBinaryIndex` |
|---|---|---|
| Best for | Near-duplicate retrieval (small Hamming radius) | General approximate Hamming KNN |
| d=512, N=1M query time | **0.037 ms** | 1.95 ms |
| Recall guarantee | Exact for distance ≤ radius (pigeonhole) | Probabilistic (depends on nprobe) |
| Recall control | `radius` parameter | `nprobe` parameter |
| Recommended `m` | d/8 bytes (e.g. m=8 for 512-bit) | — |

---

## Migrating from scikit-learn

PyNear provides adapter classes that implement the same interface as
`sklearn.neighbors.NearestNeighbors`, `KNeighborsClassifier`, and
`KNeighborsRegressor`.  Changing the import is all that is required in most
cases:

```python
# Before
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# After — identical API, backed by a VP-Tree
from pynear.sklearn_adapter import PyNearKNeighborsClassifier
clf = PyNearKNeighborsClassifier(n_neighbors=5, metric='euclidean')
```

All three adapters follow the standard scikit-learn workflow:

```python
from pynear.sklearn_adapter import (
    PyNearNearestNeighbors,
    PyNearKNeighborsClassifier,
    PyNearKNeighborsRegressor,
)

# Unsupervised neighbour lookup
nn = PyNearNearestNeighbors(n_neighbors=5, metric='euclidean')
nn.fit(X_train)
distances, indices = nn.kneighbors(X_query)

# Classification
clf = PyNearKNeighborsClassifier(n_neighbors=5, weights='distance')
clf.fit(X_train, y_train)
clf.predict(X_test)          # class labels
clf.predict_proba(X_test)    # per-class probabilities
clf.score(X_test, y_test)    # accuracy

# Regression
reg = PyNearKNeighborsRegressor(n_neighbors=5, weights='uniform')
reg.fit(X_train, y_train)
reg.predict(X_test)          # predicted values
reg.score(X_test, y_test)    # R²
```

**Supported metrics:** `euclidean` / `l2`, `manhattan` / `l1`, `chebyshev` / `linf`, `cosine`, `hamming`

**Supported weights:** `uniform`, `distance` (inverse-distance-weighted)

> **Note:** Input arrays are cast to `float32` (or `uint8` for Hamming) before
> indexing.  scikit-learn uses `float64` internally, so very small numerical
> differences may appear at the precision boundary, but nearest-neighbour
> results are identical for all practical datasets.

---

## Features

### Available indices

**Exact indices** — always return the true k nearest neighbours:

| Index | Distance | Data type | Notes |
|---|---|---|---|
| `VPTreeL2Index` | L2 (Euclidean) | `float32` | SIMD-accelerated |
| `VPTreeL1Index` | L1 (Manhattan) | `float32` | SIMD-accelerated |
| `VPTreeChebyshevIndex` | L∞ (Chebyshev) | `float32` | SIMD-accelerated |
| `VPTreeCosineIndex` | Cosine | `float32` | L2-normalised internally; SIMD-accelerated |
| `VPTreeBinaryIndex` | Hamming | `uint8` | Hardware popcount |
| `BKTreeBinaryIndex` | Hamming | `uint8` | Threshold / range search |

**Approximate indices** — trade a small recall budget for large speed gains; tunable via `n_probe` / `radius`:

| Index | Distance | Data type | Notes |
|---|---|---|---|
| `IVFFlatL2Index` | L2 (Euclidean) | `float32` | BLAS SGEMV inner scan; best for 512-D – 1024-D |
| `IVFFlatCosineIndex` | Cosine | `float32` | Spherical K-Means + BLAS SGEMV; ideal for text embeddings |
| `IVFFlatBinaryIndex` | Hamming | `uint8` | Binary K-Means IVF; faster build than Faiss binary IVF |
| `MIHBinaryIndex` | Hamming | `uint8` | Multi-Index Hashing; 257× faster than brute-force at N=1M, d=512 |

All VPTree and IVFFlat indices support `searchKNN(queries, k)`.
`BKTreeBinaryIndex` supports `find_threshold(queries, threshold)` for range queries.
Set `n_probe = n_clusters` on `IVFFlatL2Index` to make it exact.

See [docs/approximate.md](./docs/approximate.md) for a full guide on measuring
recall and tuning `n_probe` for your dataset.

#### Why approximate search? The curse of dimensionality

Tree pruning loses traction as dimensionality grows: in high-$n$ spaces, nearly all points concentrate in a thin shell near the boundary and distances between any two points become almost equal, leaving the tree nothing to prune. That's why exact tree search offers diminishing returns beyond $d \approx 256$ and why approximate methods (IVF-style probing) take over.

[Full derivation, with volume integrals and a numerical illustration →](./docs/approximate.md#why-approximate-search-the-curse-of-dimensionality)

### Pickle serialisation

All VPTree and IVFFlat indices are pickle-serialisable — save a built index to disk and
reload it without rebuilding:

```python
import pickle, numpy as np, pynear

data = np.random.rand(20_000, 32).astype(np.float32)
index = pynear.VPTreeL2Index()
index.set(data)

blob = pickle.dumps(index)
index2 = pickle.loads(blob)
```

### Tree inspection

```python
print(index.to_string())
```

```
####################
# [VPTree state]
Num Data Points: 100
Total Memory: 8000 bytes
####################
[+] Root Level:
 Depth: 0
 Height: 14
 Num Sub Nodes: 100
...
```

> **Note**: `to_string()` traverses the whole tree — use it for debugging only.

---

## Demos

Two interactive desktop demos ship in `demo/` and run with a single command:

```console
pip install PySide6
python demo/point_cloud.py    # KNN Explorer — hover over 1M points to find neighbours
python demo/voronoi.py    # Voronoi diagram — drag seed points, watch cells reshape live
```

- **KNN Explorer** — scatter up to 1 million 2-D points and hover to see k nearest
  neighbours highlighted in real time.  Supports zoom, pan, and configurable point size.
- **Voronoi Diagram** — every canvas pixel is coloured by its nearest seed point.
  Add, drag, and remove seeds; the diagram redraws live using pynear's batch 1-NN.

See [docs/demos.md](./docs/demos.md) for full details.

---

## Benchmarks

![QPS vs Recall@10 on SIFT1M binary](results/binary_benchmark_qps.png)

> Approximate Hamming search on 1M × 128-bit SIFT descriptors. `MIHBinaryIndex` and `IVFFlatBinaryIndex` both reach 100% Recall@10 at >35× the throughput of brute-force.

[**Full benchmark report (PDF)**](./docs/benchmarks.pdf) — formal evaluation against Faiss, scikit-learn, and Annoy across L2 / L1 / Hamming, dimensionalities from 2-D to 1024-D, both exact and approximate modes. Includes the recall–latency Pareto analysis and the **257× speedup** result over Faiss binary brute-force at N=1M, d=512.

Quick standalone run:

```console
python bench_run.py
```

---

<!-- binary-benchmark-start -->
## Real-World Benchmark — SIFT1M Binary

Performance of pynear's approximate Hamming-distance indices on the
[INRIA TEXMEX SIFT1M](http://corpus-texmex.irisa.fr/) dataset:
1,000,000 × 128-dim float SIFT descriptors sign-quantised to **128-bit binary**
(16 bytes/descriptor).  Ground truth computed by exact brute-force Hamming k-NN
over 500 queries, k=10.  Machine: Intel(R) Core(TM) Ultra 9 285K.

| Index               | Configuration         | Build (s) | ms / query | QPS | Recall@10 |
| ------------------- | --------------------- | --------- | ---------- | --- | --------- |
| Brute-force (numpy) | N=1,000,000           | —         | 49.6       | 20  | 1.000     |
| IVFFlatBinaryIndex  | nlist=500, nprobe=31  | 6.90      | 1.43       | 698 | 1.000     |
| IVFFlatBinaryIndex  | nlist=500, nprobe=62  | 6.90      | 2.77       | 361 | 1.000     |
| IVFFlatBinaryIndex  | nlist=500, nprobe=125 | 6.90      | 5.42       | 184 | 1.000     |
| IVFFlatBinaryIndex  | nlist=500, nprobe=250 | 6.90      | 10.54      | 95  | 1.000     |
| IVFFlatBinaryIndex  | nlist=500, nprobe=500 | 6.90      | 20.85      | 48  | 1.000     |
| MIHBinaryIndex      | m=8, radius=4         | 2.83      | 1.29       | 772 | 0.992     |
| MIHBinaryIndex      | m=8, radius=8         | 2.83      | 7.56       | 132 | 1.000     |
| MIHBinaryIndex      | m=8, radius=12        | 2.83      | 7.56       | 132 | 1.000     |
| MIHBinaryIndex      | m=8, radius=16        | 2.83      | 24.02      | 42  | 1.000     |
| MIHBinaryIndex      | m=8, radius=24        | 2.83      | 50.92      | 20  | 1.000     |
| MIHBinaryIndex      | m=8, radius=32        | 2.83      | 81.68      | 12  | 1.000     |
| MIHBinaryIndex      | m=8, radius=48        | 2.83      | 171.86     | 6   | 1.000     |

**Key takeaways:**
- `IVFFlatBinaryIndex` (nprobe=31) achieves **100% Recall@10 at 698 QPS — **35× faster than brute-force****.
- `MIHBinaryIndex` (radius=4) is the fastest single configuration at **772 QPS** with 0.992 recall.
- MIH excels on wider descriptors (512-bit / 64 bytes) where sub-table sparsity is higher.

> **Reproduce:** `python demo_binary.py` · add `--small` for a 10 K quick test · `--n-gt-queries N` to adjust evaluation size.

<!-- binary-benchmark-end -->

## Development

### Building and installing locally

```console
pip install .
```

### Running tests

```console
make test
```

### Debugging C++ code on Unix

CMake build files are provided for building and running C++ tests independently:

```console
make cpp-test
```

Tests are built in Debug mode by default, so you can debug with GDB:

```console
gdb ./build/tests/vptree-tests
```

### Debugging C++ code on Windows

Install CMake (`py -m pip install cmake`) and pybind11 (`py -m pip install pybind11`), then:

```batch
mkdir build
cd build
cmake ..\pynear
```

You may need to pass extra arguments, for example:

```batch
cmake ..\pynear -G "Visual Studio 17 2022" -A x64 ^
  -DPYTHON_EXECUTABLE="C:\Program Files\Python312\python.exe" ^
  -Dpybind11_DIR="C:\Program Files\Python312\Lib\site-packages\pybind11\share\cmake\pybind11"
```

Build and run `vptree-tests.exe` from the generated solution.

### Formatting code

```console
make fmt
```

---

## Star history

<a href="https://star-history.com/#pablocael/pynear&Date">
  <img src="https://api.star-history.com/svg?repos=pablocael/pynear&type=Date" alt="Star history of pablocael/pynear">
</a>

If pynear saved you time, consider [starring the repo](https://github.com/pablocael/pynear/stargazers) — it's the cheapest way to support the project and helps others discover it.
