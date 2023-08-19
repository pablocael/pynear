# Introduction

Pyvptree is a python library, internally built in C++, for efficient KNN search using metric distance function such as L2 distance (see VPTreeL2Index) or Hamming distances (VPTreeBinaryIndex).

## How VP-Trees work

VP-Trees are binary trees that successively divide spaces in order to perform different types of tasks, such as Nearest Neighbor Search. It differs from Kd-Trees in the sense that they always partition the whole space, instead of individual dimensional axes, using a specific metric function and a selected "Vantage Point" that will be used as reference to allow splitting the dataset. For more details on how it works please access the following references:

- https://en.wikipedia.org/wiki/Vantage-point_tree
- https://fribbels.github.io/vptree/writeup
- [Probabilistic analysis of vantage point trees](https://www.vmsta.org/journal/VMSTA/article/219/file/pdf)

### Theoretical advantage of Vantage Points Trees compared to Kd-Trees

- Intrinsic Dimensionality: VP-trees perform well in data with high intrinsic dimensionality, where the effective dimensionality is high. In contrast, kd-trees are efficient in low-dimensional spaces but their performance degrades quickly as the number of dimensions increases. This is often referred to as the "curse of dimensionality".

- No Assumption about Axis-Alignment: Unlike kd-trees, which make a specific assumption about axis-alignment, VP-trees do not make such assumptions. This makes VP-trees potentially more robust to different kinds of data, especially when there is no natural way to align the axes with respect to the data.

- Metric Spaces: VP-trees can handle general metric spaces (any space where a distance function is defined that satisfies the triangle inequality), not just Euclidean spaces. This makes them more adaptable to different problem settings where the distance metric might not be the standard Euclidean distance.

- Handling Categorical Data: VP-trees, due to their use of arbitrary distance functions, can handle categorical data more naturally than kd-trees. While there are workarounds to use kd-trees with categorical data, they often require substantial tweaking and may not perform optimally. For categorical data, another reference structure is the [BK-Tree](https://en.wikipedia.org/wiki/BK-tree), which is very efficient when comes to low dimensional data.

- Balanced Tree Structure: VP-trees inherently try to create a balanced tree structure, which is beneficial for efficient searching. kd-trees can become unbalanced in certain situations, particularly with non-uniform data, leading to inefficient search operations.

It's important to notice, however, that practical implementation approaches such as using [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) instructions or using highly optimized distance functions can strongly affect final performance in high dimensions. For instance, [Faiss Library](https://github.com/facebookresearch/faiss) performs very efficiently in very high dimensions, where searches become near linear and exaustive, due to highly optimized code.

Typically spatial search structures tend to perform worse with increasing number of dimensions of dataset.


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

Pyvptree is pickle serializable:
```python
import numpy as np
import pyvptree

np.random.seed(seed=42)

num_points = 20000
dimension = 32
num_queries = 2
data = np.random.rand(num_points, dimension).astype(dtype=np.uint8)

queries = np.random.rand(num_queries, dimension).astype(dtype=np.uint8)

vptree = pyvptree.VPTreeBinaryIndex()
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

To visualize, customize or regenerate the benchmarks as well as to see benchmark results, see [benchmarks](./pyvptree/benchmark/README.md) session.

# Development


## Running Python Tests

```
make test
```

## Debugging and Running C++ Code on Unix

For debugging and running C++ code independently from python module, CMake config files are provided in pyvptree/CMakeLists.txt.
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
cmake ..\pyvptree
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

