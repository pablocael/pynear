# PyNear

[![PyPI version](https://img.shields.io/pypi/v/pynear)](https://pypi.org/project/pynear/)
[![Python versions](https://img.shields.io/pypi/pyversions/pynear)](https://pypi.org/project/pynear/)
[![CI](https://github.com/pablocael/pynear/actions/workflows/pythonpackage.yml/badge.svg)](https://github.com/pablocael/pynear/actions/workflows/pythonpackage.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Fast, exact and approximate K-nearest-neighbour search for Python**

<video src="https://github.com/user-attachments/assets/4e07e82d-f3a0-4d47-9d34-a5ea4f752e14" autoplay muted loop width="100%"></video>

PyNear is a Python library with a C++ core for exact or approximate (fast) KNN search over metric
spaces.  It is built around [Vantage Point Trees](./docs/vptrees.md), a metric
tree that scales well to higher dimensionalities where kd-trees degrade, and
uses SIMD intrinsics (AVX2 on x86-64, portable fallbacks on arm64/Apple
Silicon) to accelerate the hot distance computation paths.
Already using scikit-learn's KNN? PyNear ships **drop-in adapter classes** that
implement the same `fit` / `predict` / `score` / `kneighbors` API —
[migrate in one line](#migrating-from-scikit-learn).

### Why PyNear?

| | PyNear | Faiss | Annoy | scikit-learn |
|---|---|---|---|---|
| **Exact results** | ✅ VPTree always | ✅ flat index | ❌ approximate | ✅ |
| **Approximate (fast, tunable)** | ✅ IVFFlatL2Index | ✅ IVF | ✅ | ❌ |
| **Metric agnostic** | ✅ L2, L1, L∞, Hamming | L2 / inner product | L2 / cosine / Hamming | L2 / others |
| **Low-dim sweet spot** | ✅ | ❌ | ❌ | ❌ |
| **High-dim (512-D – 1024-D)** | ✅ IVFFlatL2Index | ✅ | ✅ | ❌ |
| **Binary / Hamming exact** | ✅ hardware popcount | ✅ | ✅ | ❌ |
| **Binary / Hamming approx** | ✅ MIH + IVFFlat | ⚠️ slow build | ❌ | ❌ |
| **Threshold / range search** | ✅ BKTree | ❌ | ❌ | ❌ |
| **Pickle serialization** | ✅ | ❌ | ✅ | ✅ |
| **No extra native deps** | ✅ NumPy only | ❌ compiled lib + optional GPU | ❌ | ❌ |
| **scikit-learn compatible API** | ✅ drop-in adapters | ❌ | ❌ | — |

PyNear covers the full spectrum: use **VPTree** indices when you need
guaranteed exact answers (2-D to ~256-D), **IVFFlatL2Index** for fast
approximate float search on high-dimensional data (512-D to 1024-D), or
**MIHBinaryIndex** / **IVFFlatBinaryIndex** for approximate Hamming search
on binary descriptors — achieving up to **257× speedup** over exact
binary brute-force at N=1M, d=512 with 100% Recall@10.

#### For the Layman

 K-Nearest Neighbours (KNN) is simply the idea of finding the k most similar items to a given query in a collection.

  Think of it like asking: "given this song I like, what are the 5 most similar songs in my library?" The algorithm measures the "distance" between items
  (how different they are) and returns the closest ones.

  The two key parameters are:
  - k — how many neighbours to return (e.g. the 5 most similar)
  - distance metric — how "similarity" is measured (e.g. Euclidean, Manhattan, Hamming)

  Everything else — VP-Trees, SIMD, approximate search — is just engineering to make that search fast at scale.

##### Main applications of KNN search 

  1. Image retrieval — finding visually similar images by searching nearest neighbours in an embedding space (e.g. face recognition, reverse image
  search).
  2. Recommendation systems — suggesting similar items (products, songs, articles) by finding the closest user or item embeddings.
  3. Anomaly detection — flagging data points whose nearest neighbours are unusually distant as potential outliers or fraud cases.
  4. Semantic search — retrieving documents or passages whose dense vector representations are closest to a query embedding (e.g. RAG pipelines).
  5. Broad-phase collision detection — quickly finding candidate object pairs that might be colliding by looking up the nearest neighbours of each object's
   bounding volume, before running the expensive narrow-phase test.
  6. Soft body / cloth simulation — finding the nearest mesh vertices or particles to resolve contact constraints and self-collision.
  7. Particle systems (SPH, fluid sim) — each particle needs to know its neighbours within a radius to compute pressure and density forces.


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

**Supported metrics:** `euclidean` / `l2`, `manhattan` / `l1`, `chebyshev` / `linf`, `hamming`

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
| `VPTreeBinaryIndex` | Hamming | `uint8` | Hardware popcount |
| `BKTreeBinaryIndex` | Hamming | `uint8` | Threshold / range search |

**Approximate indices** — trade a small recall budget for large speed gains; tunable via `n_probe` / `radius`:

| Index | Distance | Data type | Notes |
|---|---|---|---|
| `IVFFlatL2Index` | L2 (Euclidean) | `float32` | BLAS SGEMV inner scan; best for 512-D – 1024-D |
| `IVFFlatBinaryIndex` | Hamming | `uint8` | Binary K-Means IVF; faster build than Faiss binary IVF |
| `MIHBinaryIndex` | Hamming | `uint8` | Multi-Index Hashing; 257× faster than brute-force at N=1M, d=512 |

All VPTree and IVFFlat indices support `searchKNN(queries, k)`.
`BKTreeBinaryIndex` supports `find_threshold(queries, threshold)` for range queries.
Set `n_probe = n_clusters` on `IVFFlatL2Index` to make it exact.

See [docs/approximate.md](./docs/approximate.md) for a full guide on measuring
recall and tuning `n_probe` for your dataset.

#### Why approximate search? The curse of dimensionality

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

[**Benchmark Report (PDF)**](./docs/benchmarks.pdf)

A formal evaluation of PyNear against Faiss, scikit-learn, and Annoy across
Euclidean, Manhattan, and Hamming distance metrics, dimensionalities from
2-D to 1024-D, and both exact and approximate search modes. Includes
TikZ-rendered latency charts, a recall–latency Pareto analysis of
IVFFlatL2Index vs Faiss IndexIVFFlat, and approximate binary-descriptor
benchmarks showing MIHBinaryIndex achieving **257× speedup** over
Faiss exact binary brute-force at N=1M, d=512 with 100% Recall@10.

To run a quick standalone benchmark:

```console
python bench_run.py
```

---

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
