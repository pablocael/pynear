# Introduction

Pyvptree is a python library, internally built in C++, for efficient KNN search using metric distance function such as L2 distance (see VPTreeL2Index) or Hamming distances (VPTreeBinaryIndex). 

## Theoretical advantage of Vantage Points Trees compared to Kd-Trees

Intrinsic Dimensionality: VP-trees perform well in data with high intrinsic dimensionality, where the effective dimensionality is high. In contrast, kd-trees are efficient in low-dimensional spaces but their performance degrades quickly as the number of dimensions increases. This is often referred to as the "curse of dimensionality".

No Assumption about Axis-Alignment: Unlike kd-trees, which make a specific assumption about axis-alignment, VP-trees do not make such assumptions. This makes VP-trees potentially more robust to different kinds of data, especially when there is no natural way to align the axes with respect to the data.

Metric Spaces: VP-trees can handle general metric spaces (any space where a distance function is defined that satisfies the triangle inequality), not just Euclidean spaces. This makes them more adaptable to different problem settings where the distance metric might not be the standard Euclidean distance.

Handling Categorical Data: VP-trees, due to their use of arbitrary distance functions, can handle categorical data more naturally than kd-trees. While there are workarounds to use kd-trees with categorical data, they often require substantial tweaking and may not perform optimally.

Balanced Tree Structure: VP-trees inherently try to create a balanced tree structure, which is beneficial for efficient searching. kd-trees can become unbalanced in certain situations, particularly with un-uniform data, leading to inefficient search operations.

Please note that while these points generally favor VP-trees, the performance may vary significantly depending on the specifics of your dataset and task.

Also, practical implementation constants in the time complexity of the algorithm can strongly affect final performance in high dimensions. For instance, [Faiss Library](https://github.com/facebookresearch/faiss) performs very efficiently even for near-linear-exaustive searches due to highly optimized code.

Tipically spatial search structures tend to perform worse with increasing number of dimensions of dataset. This is because points tend to be far apart and also.

# How this library works

This library implements a [Vantage Point Tree](https://en.wikipedia.org/wiki/Vantage-point_tree) to perform search within multidimensional metric spaces using arbitrary distance functions.

This library still provides no feature compresion strategy (yet), and only sypport raw (uncompressed) feature search.

# Installation

```console
pip install .
```

Performance can dramatically decrease if this library is compiled without support to Open MP and AVX. This library was not tested under windows.

# Requeriments

This library needs OpenMP support to be built and installed. The whole compilation procces occur automatically by performing the installation step above.

# Benchmarks


We build several datasets and perform time benchmarks for L2 and Hamming distance functions. For each benchmark we use [Faiss Library](https://github.com/facebookresearch/faiss) as baseline of comparison.

