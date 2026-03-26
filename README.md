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

See the [benchmark README](./pynear/benchmark/README.md) for charts comparing
PyNear against scikit-learn, Faiss, and Annoy across different dimensionalities,
dataset sizes, and distance metrics — including an
[approximate search comparison](./pynear/benchmark/README.md#approximate-l2-search--high-dimensionality-vpforest-vs-faiss-ivf)
of VPForestL2Index vs Faiss IndexIVFFlat at 128-D to 1024-D.

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
