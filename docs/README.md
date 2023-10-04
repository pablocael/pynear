# Basic Usage

## Available Indices

PyNear has several available indexes that will use different distance functions or algorithms to perform the search.
Available indices are:

### Threshold based KNN Indices:

| Index Name                     | Description                                                                                                                                                                                                                                       |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| pynear.BKTreeBinaryIndex     | Uses AVX2 optimized Hamming distance function and [BKTree](https://en.wikipedia.org/wiki/BK-tree) algorithm to perform exact searches within a threshold distance.                                                                                                                                         |

### Non Threshold based KNN Indices:

| Index Name                     | Description                                                                                                                                                                                                                                       |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| pynear.VPTreeL2Index         | Uses AVX2 optimized L2 (euclidean norm) distance function and VPTree algorithm to perform exact searches.                                                                                                                                         |
| pynear.VPTreeL1Index         | Uses L1 (manhattan) distance function and VPTree algorithm to perform exact searches.                                                                                                                                                             |
| pynear.VPTreeBinaryIndex     | Uses AVX2 optimized Hamming distance function and VPTree algorithm to perform exact searches. Supports 16, 32, 64, 128 and 256 bit dimensional vectors only.                                                                                                                                                     |
| pynear.VPTreeChebyshevIndex  | Uses [Chebyshev](https://en.wikipedia.org/wiki/Chebyshev_distance) distance function and VPTree algorithm to perform exact searches. |


## Usage example

### Creating the index

All indices need to be initialized with `set()` method before being used. This will copy the data and build the index.


Examples.


#### pynear.VPTreeBinaryIndex

```python
np.random.seed(seed=42)

dimension = 32 # 32 bytes are 256 bit dimensional vectos
num_points = 2021
data = np.random.normal(scale=255, loc=0, size=(num_points, dimension)).astype(dtype=np.uint8)

num_queries = 8
queries = np.random.normal(scale=255, loc=0, size=(num_queries, dimension)).astype(dtype=np.uint8)

k = 2

vptree = pynear.VPTreeBinaryIndex()
vptree.set(data)
vptree_indices, vptree_distances = vptree.searchKNN(queries, k)
```

#### pynear.BKTreeBinaryIndex

```python
np.random.seed(seed=42)

dimension = 32 # 32 bytes are 256 bit dimensional vectos
num_points = 2021
data = np.random.normal(scale=255, loc=0, size=(num_points, dimension)).astype(dtype=np.uint8)

num_queries = 8
queries = np.random.normal(scale=255, loc=0, size=(num_queries, dimension)).astype(dtype=np.uint8)

vptree = pynear.BKTreeBinaryIndex()
vptree.set(data)

# To search using maximum threshold use dimension * 8 (the maximum distance) or set any other threshold
indices, distances, keys = tree.find_threshold(data, dimensions * 8)
```

For convenience, apart from `searchKNN` function, vptree also provides `search1NN` for searching the closest nearest neighbor.

#### pynear.VPTreeL2Index

```python
np.random.seed(seed=42)

dimension = 8
num_points = 2021
data = np.random.rand(num_points, dimension).astype(dtype=np.float32)

num_queries = 8
queries = np.random.rand(num_queries, dimension).astype(dtype=np.float32)

k = num_points

vptree = pynear.VPTreeL2Index()
vptree.set(data)
vptree_indices, vptree_distances = vptree.searchKNN(queries, k)
```

Usage is analog for all other index types.

