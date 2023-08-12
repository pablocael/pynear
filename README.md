# Introduction

Pyvptree is a python library, internally built in C++, for efficient KNN search using metric distance function such as L2 distance (see VPTreeL2Index) or Hamming distances (VPTreeBinaryIndex). 

## How VP-Trees work

VP-Trees are binary trees that successively divide spaces in order to peform different types of tasks, such as Nearest Neighbor Search. It differs from Kd-Trees in the sense that they always partition the whole space, instead of invidiual dimensional axes, using a specific metric function and a selected "Vantage Point" that will be used as reference to allow spliting the dataset. For more details on how it works please access the following references:

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

This library needs OpenMP support to be built and installed. The whole compilation procces occur automatically by performing the installation step above.

# Benchmarks

Several datasets were built and used for time benchmarks using L2 distance functions. Although Pyvptree does support Binary Index (VPTreeBinaryIndex) using hamming distance functions,
the benchmarks focus on L2 distances since Pyvptree's binary index still need more optimizations to be able to compete with faiss in any way.

All benchmarks were generated using 12th Gen Intel(R) Core(TM) i7-1270P with 16 cores.

For each benchmark we use [Faiss Library](https://github.com/facebookresearch/faiss) and [Scikit-Learn](https://scikit-learn.org/stable/install.html) as baseline of comparison.

The below benchmarks are for different values of K (1, 2, 4, 8, 16), comparing Faiss, Sklearn and Pyvptree.

Benchmarks are split into dimensionality ranges for better analysis.


## 2 to 10 dimensions Range

![k=16, L2 index](docs/img/from_2_to_10/VPTreeL2Index_k_16.png "K=16, L2 index")

## 11 to 16 dimensions Range

![k=16, L2 index](docs/img/from_11_to_16/VPTreeL2Index_k_16.png "K=16, L2 index")

## 17 to 32 dimensions Range
![k=16, L2 index](docs/img/from_17_to_32/VPTreeL2Index_k_16.png "K=16, L2 index")

## 33 to 48 dimensions Range

![k=16, L2 index](docs/img/from_33_to_48/VPTreeL2Index_k_16.png "K=16, L2 index")

To customize or regenerate the benchmarks as well as to see other benchmark results, see [benchmarks](./pyvptree/benchmark/README.md) session.

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

## Formating code

```
make fmt
```

