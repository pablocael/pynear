# Introduction

PyNear is a python library, internally built in C++, for efficient KNN search using metric distance function such as L2 distance (see VPTreeL2Index) or Hamming distances (VPTreeBinaryIndex) as well as other indices. It uses AVX2 instructions to optimize distance functions so to improve search performance.

PyNear aims providing different efficient algorithms for nearest neighbor search. One of the differentials of PyNear is the adoption of [Vantage Point Tree](./docs/vptrees.md) in order to mitigate course of dimensionality for high dimensional features (see VPTree* indices for more information in [docs](./docs/README.md)).


# Installation

```console
pip install .
```

Performance can dramatically decrease if this library is compiled without support to Open MP and AVX. This library was not tested under windows.

# Requirement

This library needs OpenMP support to be built and installed. The whole compilation proccess occur automatically by performing the installation step above.

# Features

For more features and all available index types, refer to [docs](./docs/README.md).

## Pickle serialization

vptree indices are pickle serializable:
```python
import numpy as np
import pynear

np.random.seed(seed=42)

num_points = 20000
dimension = 32
num_queries = 2
data = np.random.rand(num_points, dimension).astype(dtype=np.uint8)

queries = np.random.rand(num_queries, dimension).astype(dtype=np.uint8)

vptree = pynear.VPTreeBinaryIndex()
vptree.set(data)

data = pickle.dumps(vptree)
recovered = pickle.loads(data)
```
## String serialization

Sometimes to check state of tree is interesting to be able to print the whole tree including information about the size and balancing.
By using `to_string()` method one can print the whole tree to string. **Be aware that this method is really slow and should not be used for any performance demanding tasks**.

```
print(vptree.to_string())
```

Output:
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
 Index Start: 0
 Index End:   99
 Left Subtree Height: 12
 Right Subtree Height: 12
 [+] Left children:
.... Depth: 1
.... Height: 12
.... Num Sub Nodes: 49
.... Index Start: 1
.... Index End:   49
.... Left Subtree Height: 10
.... Right Subtree Height: 10
.... [+] Left children:
........ Depth: 2
........ Height: 10
........ Num Sub Nodes: 24
........ Index Start: 2
........ Index End:   25
........ Left Subtree Height: 8
........ Right Subtree Height: 8
........ [+] Left children:
............ Depth: 3
............ Height: 8
............ Num Sub Nodes: 11
............ Index Start: 3
............ Index End:   13
............ Left Subtree Height: 6
............ Right Subtree Height: 6
............ [+] Left children:

...
```
Notice that this output can be very large.



# Benchmarks

To visualize, customize or regenerate the benchmarks as well as to see benchmark results, see [benchmarks](./pynear/benchmark/README.md) session.

# Development


## Running Python Tests

```
make test
```

## Debugging and Running C++ Code on Unix

For debugging and running C++ code independently from python module, CMake config files are provided in pynear/CMakeLists.txt.
For building and running C++ tests run:

```
make cpp-test

```

Since tests are built in Debug mode (default CMakeLists build mode), one can debug tests with gdb using built test binary:

```
gdb ./build/tests/vptree-tests
```

## Debugging and Running C++ Code on Windows

Install CMake (for example `py -m pip install cmake`) and pybind11 (`py -m pip install pybind11`).

```batch
mkdir build
cd build
cmake ..\pynear
```

You may have to specify some arguments like the correct generator `-G "Visual Studio 15 2017 Win64"`
or paths for Python `-DPYTHON_EXECUTABLE="C:\Program Files\Python38\python.exe"`
and pybind11 `-Dpybind11_DIR="C:\Program Files\Python38\Lib\site-packages\pybind11\share\cmake\pybind11"`
for CMake to work correctly.

Build generated files using Visual Studio (or whichever generator you chose) and run `vptree-tests.exe`.

## Formatting code

```
make fmt
```
