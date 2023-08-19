# Basic Usage

## Available Indices

Pyvptree has several available indexes that will use different distance functions or algorithms to perform the search.
Available indices are:

| Index Name                     | Description                                                                                                                                                                                                                                       |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| pyvptree.VPTreeL2Index         | Uses AVX2 optimized L2 (euclidean norm) distance function and VPTree algorithm to perform exact searches.                                                                                                                                         |
| pyvptree.VPTreeL1Index         | Uses L1 (manhattan) distance function and VPTree algorithm to perform exact searches.                                                                                                                                                             |
| pyvptree.VPTreeBinaryIndex     | Uses AVX2 optimized Hamming distances function and VPTree algorithm to perform exact searches. Supports 16, 32, 64, 128 and 256 bit dimensional vectors.                                                                                                                                                     |
| pyvptree.VPTreeChebyshevIndex  | Uses [Chebyshev](https://en.wikipedia.org/wiki/Chebyshev_distance) distance function and VPTree algorithm to perform exact searches. |

## Usage example

### Creating the index

All indices need to be initialized with `set()` method before being used. This will copy the data and build the index.


Examples.


#### pyvptree.VPTreeBinaryIndex

```
np.random.seed(seed=42)

dimension = 32 # 32 bytes are 256 bit dimensional vectos
num_points = 2021
data = np.random.normal(scale=255, loc=0, size=(num_points, dimension)).astype(dtype=np.uint8)

num_queries = 8
queries = np.random.normal(scale=255, loc=0, size=(num_queries, dimension)).astype(dtype=np.uint8)

k = 2

vptree = pyvptree.VPTreeBinaryIndex()
vptree.set(data)
vptree_indices, vptree_distances = vptree.searchKNN(queries, k)
```

For convenience, apart from `searchKNN` function, vptree also provides `search1NN` for searching the closest nearest neighbor.

#### pyvptree.VPTreeBinaryIndex

```
np.random.seed(seed=42)

dimension = 8
num_points = 2021
data = np.random.rand(num_points, dimension).astype(dtype=np.float32)

num_queries = 8
queries = np.random.rand(num_queries, dimension).astype(dtype=np.float32)

k = num_points

vptree = pyvptree.VPTreeL2Index()
vptree.set(data)
vptree_indices, vptree_distances = vptree.searchKNN(queries, k)
```

Usage is analog to other index types.

