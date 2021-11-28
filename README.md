# vptree-cpp
 A C++ efficient Vantage Point Tree Implementation for KNN search using L2 distance (for points in multidimensional space), or Hamming distances (for binary features represented by uint8 type).
 
 This library provides no feature compresion strategy (yet), and only sypport raw (uncompressed) feature search.
 
 This library has python3 bindings.
 

# Installation

```console
python setup.py install
```

# Usage

## Searching L2 Features

```python

import pyvptree
import numpy as np

np.random.seed(seed=42)

data = np.random.rand(num_features, dimension)

num_points = 400000 # can be arbitrary feature size, as long fits your memory
dimension = 3 # can be arbitrary dimension size
data = np.random.rand(num_points, dimension)

num_queries = 2000
queries = np.random.rand(num_queries, dimension)

K = 4 # search 4 nearest neighbors

vptree = pyvptree.VPTreeL2Index()
vptree.set(data)
vptree_indices, vptree_distances = vptree.searchKNN(queries, K)

```

## Searching Binary Features

```python

import pyvptree
import numpy as np

 np.random.seed(seed=42)

dimension = 32 # 32 bytes = 256 bit examples
num_points = 2021
data = np.random.normal(scale=255, loc=0, size=(num_points, dimension)).astype(dtype=np.uint8)

num_queries = 8
queries = np.random.normal(scale=255, loc=0, size=(num_queries, dimension)).astype(dtype=np.uint8)

K = 2  # search 2 nearest neighbors

vptree = pyvptree.VPTreeBinaryIndex()
vptree.set(data)
vptree_indices, vptree_distances = vptree.searchKNN(queries, K)

```

# Searching for the nearest neighbor (K=1)

There is ano optimized version of KNN for K=1:

```python

vptree = pyvptree.VPTreeBinaryIndex()
vptree.set(data)
vptree_indices, vptree_distances = vptree.search1NN(queries)

```

Which is considerably faster than calling the generic searchKNN with K=1. This option is available for both L2 and Binary indices.


# Benchmarks

(Work in progress, soon)

# Using the C++ library
You can install vptree C++ library header using cmake. The library is a single header only.

To install, run:

```console
mkdir build
cd build

cmake ..
make
make install
```

# Development

The C++ code is a one header file within include/VPTree.hpp.

To use the project with some compiling tools on can use CMake to export the compile commands (optional):

```console
mkdir build
cd build

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ../

make install
```

To run C++ tests, run the below command, after running cmake:

```console
make test
```

