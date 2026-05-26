
# PyNear

[![PyPI version](https://img.shields.io/pypi/v/pynear)](https://pypi.org/project/pynear/)
[![Python versions](https://img.shields.io/pypi/pyversions/pynear)](https://pypi.org/project/pynear/)
[![CI](https://github.com/pablocael/pynear/actions/workflows/pythonpackage.yml/badge.svg)](https://github.com/pablocael/pynear/actions/workflows/pythonpackage.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/pablocael/pynear?style=social)](https://github.com/pablocael/pynear/stargazers)

> **Fast KNN that doesn't make you choose between exact answers and production speed.**
> Drop-in for scikit-learn. SIMD-accelerated. Binary Multi-Index Hashing that **beats Faiss's own MIH** at matched recall — and finds 512-bit near-duplicates **~40× faster than Faiss's brute-force scan** at 100% recall.

![PyNear demo](docs/img/demo.gif)

**PyNear** is a metric-space nearest-neighbour library with a C++ core: exact **VP-Trees** up to ~256-D, **IVF-Flat** for approximate float search at 512–1024-D (embeddings, RAG), and **Multi-Index Hashing** / **IVF-Binary** for Hamming search on binary descriptors — one small, NumPy-only API with a scikit-learn drop-in and pre-built wheels (`pip install pynear`).

---

## Table of Contents

- [Introduction](#introduction)
- [Why PyNear?](#why-pynear)
- [Installation](#installation)
- [Quick start](#quick-start)
  - [Low-dimensional exact search (VPTreeL2Index)](#low-dimensional-exact-search-vptreel2index)
  - [High-dimensional binary descriptors (MIHBinaryIndex)](#high-dimensional-binary-descriptors-mihbinaryindex)
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

## Introduction

Search, recommendation, deduplication, and retrieval-augmented generation all
reduce to the same primitive: turn an item — an image, an audio clip, a
document, a face — into a **descriptor** (a fixed-length vector or bit-string),
then find the descriptors nearest to it. Similar items map to nearby points, so
*"find similar"* becomes *"find nearest neighbours."*

The right way to search depends on the data, and PyNear gives you one API for
all three regimes instead of forcing every problem through the same tool:

- **Low-to-mid dimensions (a few up to ~256-D)** — exact tree search wins. A
  **VP-Tree** prunes by distance to vantage points and returns the *true*
  nearest neighbours, no recall loss, no tuning.
- **High-dimensional float vectors (512–1024-D embeddings)** — exact pruning
  collapses (the *curse of dimensionality*), so **IVF-Flat** trades a sliver of
  recall for large speed-ups.
- **Binary descriptors (ORB, BRIEF, perceptual hashes, SimHash)** — Hamming
  distance plus **Multi-Index Hashing** uses the pigeonhole principle to find
  near-duplicates without scanning the whole dataset.

**What people build with it:**

- **Image / video deduplication & copy detection** — perceptual-hash / ORB
  descriptors + `MIHBinaryIndex`.
- **Audio fingerprinting** (Shazam-style) — spectrogram-peak descriptors +
  Hamming search.
- **Semantic & RAG retrieval** — text/image embeddings + `IVFFlatCosineIndex`.
- **Classic ML** — drop-in `KNeighborsClassifier` / `Regressor` backed by
  VP-Trees.

> New to nearest-neighbour search? See [docs/intro.md](./docs/intro.md) for a
> gentle, jargon-free introduction — or the deep dive,
> [*The shared recipe behind image search, Shazam, and RAG*](https://medium.com/@pablo.cael/the-shared-recipe-behind-search-images-shazam-and-rag-08fc93a276ac).

---

## Why PyNear?

| | PyNear | Faiss | Annoy | scikit-learn |
|---|---|---|---|---|
| **Metric agnostic** | ✅ L2, L1, L∞, cosine, Hamming | L2 / IP / cosine | L2 / cosine / Hamming | L2 / others |
| **Binary / Hamming approx** | ✅ MIH + IVF, faster than Faiss MIH | ✅ MIH + IVF | ❌ | ❌ |
| **scikit-learn drop-in** | ✅ adapter classes | ❌ | ❌ | — |
| **Zero native deps** | ✅ NumPy only | ❌ compiled lib + optional GPU | ❌ | ❌ |

[Full comparison →](./docs/comparison.md)

---

## Installation

```console
pip install pynear
```

Requires Python 3.8+ and NumPy ≥ 1.21.2.  Pre-built wheels are available for
Linux, macOS (x86-64 and Apple Silicon), and Windows — no compiler needed.

---

## Quick start

PyNear's two headline indices: exact **VP-Trees** for low-to-mid dimensions,
and **Multi-Index Hashing** for binary descriptors.

### Low-dimensional exact search (VPTreeL2Index)

VP-Trees partition points by *distance to a vantage point*, so they prune whole
branches in any metric space and return **exact** neighbours — no recall loss,
no tuning — and stay effective up to ~256-D. The same API backs L2, L1, L∞,
cosine, and Hamming.

```python
import numpy as np
import pynear

# 100,000 vectors in 32-D
data = np.random.rand(100_000, 32).astype(np.float32)
index = pynear.VPTreeL2Index()
index.set(data)

# KNN search — returns (indices, distances) per query, sorted nearest-first
queries = np.random.rand(10, 32).astype(np.float32)
indices, distances = index.searchKNN(queries, k=5)

# 1-NN shortcut (slightly faster than searchKNN with k=1)
nn_indices, nn_distances = index.search1NN(queries)
```

### High-dimensional binary descriptors (MIHBinaryIndex)

`MIHBinaryIndex` is pynear's flagship for **binary** descriptors (ORB, BRIEF,
AKAZE, perceptual hashes, SimHash). Multi-Index Hashing splits each *d*-bit
descriptor into `m` sub-strings and hashes them; by the **pigeonhole
principle**, any neighbour within `radius` Hamming bits is *guaranteed* to be
found. On wide descriptors it retrieves near-duplicates **~40× faster than
Faiss's brute-force scan** at 100% recall — and faster than Faiss's own MIH.

```python
import numpy as np
import pynear

# 1M × 512-bit descriptors (64 bytes each)
db      = np.random.randint(0, 256, size=(1_000_000, 64), dtype=np.uint8)
queries = np.random.randint(0, 256, size=(100, 64), dtype=np.uint8)

mih = pynear.MIHBinaryIndex(m=8)   # 8 sub-tables of 64 bits (m=4 for 128/256-bit)
mih.set(db)
indices, distances = mih.searchKNN(queries, k=10, radius=8)
# radius: any true neighbour within this Hamming distance is guaranteed found
# (pigeonhole). Increase for higher recall on noisier data.
```

When you'd rather cap the cost per query than reason about a radius,
`IVFFlatBinaryIndex` scans a fixed number of clusters instead:

```python
ivf = pynear.IVFFlatBinaryIndex(nlist=512, nprobe=16)
ivf.set(db)
indices, distances = ivf.searchKNN(queries, k=10)
ivf.set_nprobe(32)   # trade speed for recall at runtime
```

**Choosing between MIH and IVFFlat:**

| | `MIHBinaryIndex` | `IVFFlatBinaryIndex` |
|---|---|---|
| Best for | Near-duplicate retrieval (small Hamming radius) | General approximate Hamming KNN |
| d=512, N=1M query time (near-duplicate) | **0.008 ms** | 1.82 ms |
| Recall guarantee | Exact for distance ≤ radius (pigeonhole) | Probabilistic (depends on nprobe) |
| Recall control | `radius` parameter | `nprobe` parameter |
| Recommended `m` | d/8 bytes (e.g. m=8 for 512-bit) | — |

For wide **float** vectors (512-D–1024-D embeddings, e.g. text / RAG) reach for
`IVFFlatL2Index` / `IVFFlatCosineIndex`. Every index type and its tuning knobs
are covered in [docs/README.md](./docs/README.md).

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
| `MIHBinaryIndex` | Hamming | `uint8` | Multi-Index Hashing; ~40× faster than Faiss brute-force on 512-bit near-duplicates, and faster than Faiss's own MIH |

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

> Recall vs throughput for approximate Hamming search on 1M × 128-bit SIFT descriptors (the ~0.84 ceiling is a Hamming-tie artifact, not missed neighbours). For the apples-to-apples comparison against Faiss's brute-force `IndexBinaryFlat` and Faiss's own `IndexBinaryMultiHash` — including the 512-bit workload where MIH is ~40× faster than brute-force — see [results/faiss_comparison.md](./results/faiss_comparison.md).

[**Full benchmark report (PDF)**](./docs/benchmarks.pdf) — formal evaluation against Faiss, scikit-learn, and Annoy across L2 / L1 / Hamming, dimensionalities from 2-D to 1024-D, both exact and approximate modes, including the recall–latency Pareto analysis. (The binary-descriptor numbers are superseded by the reproducible, thread-matched [results/faiss_comparison.md](./results/faiss_comparison.md).)

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

The baseline below is a *naive* numpy scan. For the apples-to-apples comparison
against Faiss's optimised brute-force (`IndexBinaryFlat`) and Faiss's own
Multi-Index Hashing, see
[results/faiss_comparison.md](results/faiss_comparison.md).

![QPS vs Recall@10](results/binary_benchmark_qps.png)

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
