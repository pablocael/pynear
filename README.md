# PyNear

[![PyPI version](https://img.shields.io/pypi/v/pynear)](https://pypi.org/project/pynear/)
[![Python versions](https://img.shields.io/pypi/pyversions/pynear)](https://pypi.org/project/pynear/)
[![CI](https://github.com/pablocael/pynear/actions/workflows/pythonpackage.yml/badge.svg)](https://github.com/pablocael/pynear/actions/workflows/pythonpackage.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Fast, exact K-nearest-neighbour search for Python**

<video src="https://github.com/user-attachments/assets/4e07e82d-f3a0-4d47-9d34-a5ea4f752e14" autoplay muted loop width="100%"></video>

PyNear is a Python library with a C++ core for exact or approximate (fast) KNN search over metric
spaces.  It is built around [Vantage Point Trees](./docs/vptrees.md), a metric
tree that scales well to higher dimensionalities where kd-trees degrade, and
uses SIMD intrinsics (AVX2 on x86-64, portable fallbacks on arm64/Apple
Silicon) to accelerate the hot distance computation paths.

### Why PyNear?

| | PyNear | Faiss | Annoy | scikit-learn |
|---|---|---|---|---|
| **Exact results** | ✅ VPTree always | ✅ flat index | ❌ approximate | ✅ |
| **Approximate (fast, tunable)** | ✅ VPForest | ✅ IVF | ✅ | ❌ |
| **Metric agnostic** | ✅ L2, L1, L∞, Hamming | L2 / inner product | L2 / cosine / Hamming | L2 / others |
| **Low-dim sweet spot** | ✅ | ❌ | ❌ | ❌ |
| **High-dim (512-D – 1024-D)** | ✅ VPForest | ✅ | ✅ | ❌ |
| **Binary / Hamming** | ✅ hardware popcount | ✅ | ✅ | ❌ |
| **Threshold / range search** | ✅ BKTree | ❌ | ❌ | ❌ |
| **Pickle serialization** | ✅ | ❌ | ✅ | ✅ |
| **Zero native dependencies** | ✅ | ❌ GPU/BLAS | ❌ | ❌ |

PyNear covers the full spectrum: use **VPTree** indices when you need
guaranteed exact answers (2-D to ~256-D), or **VPForest** when you need fast
approximate search on high-dimensional data (512-D to 1024-D) with a
configurable recall target.

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

**Approximate indices** — partition data into clusters, search `n_probe` of them per query; tunable recall vs speed:

| Index | Distance | Data type | Notes |
|---|---|---|---|
| `VPForestL2Index` | L2 (Euclidean) | `float32` | Best for 512-D – 1024-D embeddings |
| `VPForestL1Index` | L1 (Manhattan) | `float32` | |
| `VPForestChebyshevIndex` | L∞ (Chebyshev) | `float32` | |

All VPTree and VPForest indices support `searchKNN(queries, k)` and `search1NN(queries)`.
`BKTreeBinaryIndex` supports `find_threshold(queries, threshold)` for range queries.
Set `n_probe = n_clusters` on any VPForest index to make it exact.

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
**VPForest** (probing only a fraction of clusters) or Faiss IVF are
necessary at high dimensionalities.

### Pickle serialisation

All VPTree and VPForest indices are pickle-serialisable — save a built index to disk and
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
TikZ-rendered latency charts and a recall–latency Pareto analysis of
VPForestL2Index vs Faiss IndexIVFFlat.

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
