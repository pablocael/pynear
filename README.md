# PyNear

[![PyPI version](https://img.shields.io/pypi/v/pynear)](https://pypi.org/project/pynear/)
[![Python versions](https://img.shields.io/pypi/pyversions/pynear)](https://pypi.org/project/pynear/)
[![CI](https://github.com/pablocael/pynear/actions/workflows/pythonpackage.yml/badge.svg)](https://github.com/pablocael/pynear/actions/workflows/pythonpackage.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PyNear is a Python library (C++ core) for fast, **exact** K-nearest-neighbor search using metric distance functions.
It is built around [Vantage Point Trees](./docs/vptrees.md), which remain efficient at higher dimensionalities compared to kd-trees.
SIMD acceleration (AVX2 on x86-64, portable fallbacks on arm64/Apple Silicon) is used for the hot distance paths.

---

## Installation

```console
pip install pynear
```

Requires Python 3.8+ and NumPy ≥ 1.21.2. Pre-built wheels are available for Linux, macOS (x86-64 and Apple Silicon), and Windows.

---

## Quick start

```python
import numpy as np
import pynear

# Build index
data = np.random.rand(100_000, 32).astype(np.float32)
index = pynear.VPTreeL2Index()
index.set(data)

# Search — returns (indices, distances) for each query
queries = np.random.rand(10, 32).astype(np.float32)
indices, distances = index.searchKNN(queries, k=5)

# Nearest-neighbour shortcut (slightly faster than searchKNN with k=1)
nn_indices, nn_distances = index.search1NN(queries)
```

For all available index types and detailed usage see [docs](./docs/README.md).

---

## Features

### Available indices

| Index | Distance | Data type | Notes |
|---|---|---|---|
| `VPTreeL2Index` | L2 (Euclidean) | `float32` | SIMD-accelerated |
| `VPTreeL1Index` | L1 (Manhattan) | `float32` | SIMD-accelerated |
| `VPTreeChebyshevIndex` | L∞ (Chebyshev) | `float32` | SIMD-accelerated |
| `VPTreeBinaryIndex` | Hamming | `uint8` | Exact, hardware popcount |
| `BKTreeBinaryIndex` | Hamming | `uint8` | Threshold search |

All VPTree indices support `searchKNN(queries, k)` and `search1NN(queries)`.
`BKTreeBinaryIndex` supports `find_threshold(queries, threshold)` for range queries.

### Pickle serialization

All VPTree indices are pickle-serializable, so they can be saved to disk and reloaded without rebuilding:

```python
import pickle
import numpy as np
import pynear

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

Output (truncated):

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

> **Note**: `to_string()` traverses the whole tree and is slow — use it for debugging only.

---

## Benchmarks

See the [benchmark README](./pynear/benchmark/README.md) for charts comparing PyNear against scikit-learn, Faiss, and Annoy across different dimensionalities, dataset sizes, and index types.

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

### Running Python tests

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
