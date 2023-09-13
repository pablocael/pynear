#
# MIT Licence
# Copyright 2021 Pablo Carneiro Elias
#

from collections import Counter
from functools import partial
import os
import sys
from typing import Callable
from typing import Tuple

import numpy as np
import pytest

import pynear
from pynear import dist_chebyshev
from pynear import dist_hamming
from pynear import dist_l1
from pynear import dist_l2

seed = os.environ.get("PYNEAR_TEST_SEED", None)
if seed is not None:
    np.random.seed(int(seed))
    print(f"using test seed: {seed}")

np.set_printoptions(threshold=sys.maxsize)

def hamming_distance_pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = []
    for ai in a:
        dists = []
        for bi in b:
            dists.append(dist_hamming(ai, bi))

        result.append(dists)

    return np.array(result).astype(np.uint8)

def euclidean_distance_pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = []
    for ai in a:
        dists = []
        for bi in b:
            dists.append(dist_l2(ai, bi))

        result.append(dists)

    return np.array(result)


def manhattan_distance_pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = []
    for ai in a:
        dists = []
        for bi in b:
            dists.append(dist_l1(ai, bi))

        result.append(dists)

    return np.array(result)

def chebyshev_distance_pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = []
    for ai in a:
        dists = []
        for bi in b:
            dists.append(dist_chebyshev(ai, bi))

        result.append(dists)

    return np.array(result)


def test_empty_index():
    try:
        vptree = pynear.VPTreeBinaryIndex()
        i, d = vptree.search1NN(np.random.rand(1, 8).astype(dtype=np.uint8))
        # expect exception and should now reach this line
        # since index is empty, we should not be able to search
        assert False
    except Exception:
        pass

    try:
        vptree = pynear.VPTreeL2Index()
        i, d = vptree.search1NN(np.random.rand(1, 8).astype(dtype=np.uint8))
        # expect exception and should now reach this line
        # since index is empty, we should not be able to search
        assert False
    except Exception:
        pass

    try:
        vptree = pynear.VPTreeL2Index()
        empty = np.array([]).reshape(-1, 2)
        vptree.set(empty)
        i, d = vptree.search1NN(np.random.rand(1, 8).astype(dtype=np.uint8))
        assert False

        # expect no exception and should now reach this line
        # since index is empty, we should not be able to search
    except Exception:
        pass


def test_hamming():
    def hamming_distance(a, b) -> np.ndarray:
        r = (1 << np.arange(8))[:, None]
        return np.count_nonzero((np.bitwise_xor(a, b) & r) != 0)

    arr1 = np.random.randint(0, 10, (5, 8), dtype=np.uint8)
    arr2 = np.random.randint(0, 10, (3, 8), dtype=np.uint8)

    truth = np.empty((arr1.shape[0], arr2.shape[0]), dtype=np.uint64)
    for i in range(arr1.shape[0]):
        for j in range(arr2.shape[0]):
            truth[i, j] = hamming_distance(arr1[i], arr2[j])

    result = hamming_distance_pairwise(arr1, arr2)

    assert np.array_equal(truth, result)


def exhaustive_search(
    metric_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    data: np.ndarray,
    queries: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    distances = metric_func(queries, data)
    indices = np.argpartition(distances, range(k), axis=-1)[:, :k]
    distances = np.take_along_axis(distances, indices, axis=-1)

    return indices, distances


exhaustive_search_hamming = partial(exhaustive_search, hamming_distance_pairwise)
exhaustive_search_euclidean = partial(exhaustive_search, euclidean_distance_pairwise)
exhaustive_search_manhattan = partial(exhaustive_search, manhattan_distance_pairwise)
exhaustive_search_chebyshev = partial(exhaustive_search, chebyshev_distance_pairwise)


def _num_dups(distances):
    dups = 0
    for i in range(len(distances)):
        c = Counter(distances[i].tolist())
        dups += sum(1 for k, c in c.most_common() if c > 1)
    return dups


CLASSES = [
    (pynear.VPTreeL2Index, exhaustive_search_euclidean),
    (pynear.VPTreeL1Index, exhaustive_search_manhattan),
    (pynear.VPTreeChebyshevIndex, exhaustive_search_chebyshev),
]


@pytest.mark.parametrize("num_points, k", [(2021, 2), (40021, 3)])
def test_binary(num_points, k):
    dimension = 32
    data = np.random.normal(scale=255, loc=0, size=(num_points, dimension)).astype(dtype=np.uint8)

    num_queries = 8
    queries = np.random.normal(scale=255, loc=0, size=(num_queries, dimension)).astype(dtype=np.uint8)

    exaustive_indices, exaustive_distances = exhaustive_search_hamming(data, queries, k)

    vptree = pynear.VPTreeBinaryIndex()
    vptree.set(data)
    vptree_indices, vptree_distances = vptree.searchKNN(queries, k)

    vptree_indices = np.array(vptree_indices, dtype=np.uint64)[:, ::-1]
    vptree_distances = np.array(vptree_distances, dtype=np.int64)[:, ::-1]

    assert np.array_equal(exaustive_distances, vptree_distances)
    # assert np.array_equal(exaustive_indices, vptree_indices)  # indices order can vary for same distances


def test_binary_duplicates():
    dimension = 32
    num_points = 2
    data = np.zeros((num_points, dimension), dtype=np.uint8)

    k = 2

    vptree = pynear.VPTreeBinaryIndex()
    vptree.set(data)
    indices, distances = vptree.searchKNN(data, k)

    assert [sorted(i) for i in indices] == [list(range(num_points))] * num_points
    assert distances == [[0] * k] * num_points


@pytest.mark.parametrize("vptree_cls, exaustive_metric", CLASSES)
def test_k_equals_dataset(vptree_cls, exaustive_metric):
    dimension = 8
    num_points = 2021
    data = np.random.rand(num_points, dimension).astype(dtype=np.float32)

    num_queries = 8
    queries = np.random.rand(num_queries, dimension).astype(dtype=np.float32)

    k = num_points

    exaustive_indices, exaustive_distances = exaustive_metric(data, queries, k)

    vptree = vptree_cls()
    vptree.set(data)
    vptree_indices, vptree_distances = vptree.searchKNN(queries, k)

    vptree_indices = np.array(vptree_indices, dtype=np.uint64)[:, ::-1]
    vptree_distances = np.array(vptree_distances, dtype=np.float64)[:, ::-1]
    dist_diff = vptree_distances - exaustive_distances 
    ind_diff = vptree_indices - exaustive_indices
    print(">>>>>>>>>>>>", dist_diff[dist_diff > 1e-7] )
    print(">>>>>>>>>>>>", np.argwhere(ind_diff != 0))

    np.testing.assert_allclose(exaustive_distances, vptree_distances, rtol=1e-06)
    if _num_dups(exaustive_distances) == 0:
        assert np.array_equal(exaustive_indices, vptree_indices)


@pytest.mark.parametrize("vptree_cls, exaustive_metric", CLASSES)
def test_large_dataset(vptree_cls, exaustive_metric):
    dimension = 8
    num_points = 401001
    data = np.random.rand(num_points, dimension).astype(dtype=np.float32)

    num_queries = 8
    queries = np.random.rand(num_queries, dimension).astype(dtype=np.float32)

    k = 3

    exaustive_indices, exaustive_distances = exaustive_metric(data, queries, k)

    vptree = vptree_cls()
    vptree.set(data)
    vptree_indices, vptree_distances = vptree.searchKNN(queries, k)

    vptree_indices = np.array(vptree_indices, dtype=np.uint64)[:, ::-1]
    vptree_distances = np.array(vptree_distances, dtype=np.float32)[:, ::-1]

    vptree_distances2 = np.sort(vptree_distances, axis=-1)
    assert np.array_equal(vptree_distances, vptree_distances2)  # distances are sorted

    assert np.array_equal(exaustive_indices, vptree_indices)
    np.testing.assert_allclose(exaustive_distances, vptree_distances, rtol=1e-06)


@pytest.mark.parametrize("vptree_cls, exaustive_metric", CLASSES)
def test_large_dataset_highdim(vptree_cls, exaustive_metric):
    dimension = 16
    num_points = 401001
    data = np.random.rand(num_points, dimension).astype(dtype=np.float32)

    num_queries = 8
    queries = np.random.rand(num_queries, dimension).astype(dtype=np.float32)

    k = 3

    exaustive_indices, exaustive_distances = exaustive_metric(data, queries, k)

    vptree = vptree_cls()
    vptree.set(data)
    vptree_indices, vptree_distances = vptree.searchKNN(queries, k)

    vptree_indices = np.array(vptree_indices, dtype=np.uint64)[:, ::-1]
    vptree_distances = np.array(vptree_distances, dtype=np.float32)[:, ::-1]

    vptree_distances2 = np.sort(vptree_distances, axis=-1)
    assert np.array_equal(vptree_distances, vptree_distances2)  # distances are sorted

    assert np.array_equal(exaustive_indices, vptree_indices)
    np.testing.assert_allclose(exaustive_distances, vptree_distances, rtol=1e-06)


@pytest.mark.parametrize("vptree_cls, exaustive_metric", CLASSES)
def test_dataset_split_less_than_k(vptree_cls, exaustive_metric):
    """doc
    Test the case where on of the splits of the dataset eliminates half of points but contains less than k points
    """

    data = np.array([[-2.5, 0], [-2.58, 0], [0, 0], [2.5, 0], [2.6, 0]])
    queries = np.array([[-2.55, 0]])
    k = 4

    exaustive_indices, exaustive_distances = exaustive_metric(data, queries, k)

    vptree = vptree_cls()
    vptree.set(data)
    vptree_indices, vptree_distances = vptree.searchKNN(queries, k)

    vptree_indices = np.array(vptree_indices, dtype=np.uint64)[:, ::-1]
    vptree_distances = np.array(vptree_distances, dtype=np.float32)[:, ::-1]

    assert np.array_equal(exaustive_indices, vptree_indices)
    np.testing.assert_allclose(exaustive_distances, vptree_distances, rtol=1e-06)


@pytest.mark.parametrize("vptree_cls, exaustive_metric", CLASSES)
def test_query_larger_than_dataset(vptree_cls, exaustive_metric):
    num_points = 5
    dimension = 8
    data = np.random.rand(num_points, dimension).astype(dtype=np.float32)
    print(f"data: {data}")

    num_queries = 8
    queries = np.random.rand(num_queries, dimension).astype(dtype=np.float32)
    print(f"queries: {queries}")

    k = 3

    exaustive_indices, exaustive_distances = exaustive_metric(data, np.array([queries[0]]), k)

    vptree = vptree_cls()
    vptree.set(data)
    vptree_indices, vptree_distances = vptree.searchKNN(np.array([queries[0]]), k)

    vptree_indices = np.array(vptree_indices, dtype=np.uint64)[:, ::-1]
    vptree_distances = np.array(vptree_distances, dtype=np.float32)[:, ::-1]

    assert np.array_equal(exaustive_indices, vptree_indices)
    np.testing.assert_allclose(exaustive_distances, vptree_distances, rtol=1e-06)


@pytest.mark.parametrize("vptree_cls, exaustive_metric", CLASSES)
def test_compare_with_exaustive_knn(vptree_cls, exaustive_metric):
    num_points = 21231
    dimension = 8
    data = np.random.rand(num_points, dimension).astype(dtype=np.float32)

    num_queries = 23
    queries = np.random.rand(num_queries, dimension).astype(dtype=np.float32)

    k = 3

    exaustive_indices, exaustive_distances = exaustive_metric(data, queries, k)

    vptree = vptree_cls()
    vptree.set(data)
    vptree_indices, vptree_distances = vptree.searchKNN(queries, k)

    vptree_indices = np.array(vptree_indices, dtype=np.uint64)[:, ::-1]
    vptree_distances = np.array(vptree_distances, dtype=np.float32)[:, ::-1]

    assert np.array_equal(exaustive_indices, vptree_indices)
    np.testing.assert_allclose(exaustive_distances, vptree_distances, rtol=1e-06)


@pytest.mark.parametrize("vptree_cls, exaustive_metric", CLASSES)
def test_compare_with_exaustive_1nn(vptree_cls, exaustive_metric):
    num_points = 21231
    dimension = 8
    data = np.random.rand(num_points, dimension).astype(dtype=np.float32)

    num_queries = 23
    queries = np.random.rand(num_queries, dimension).astype(dtype=np.float32)

    exaustive_indices, exaustive_distances = exaustive_metric(data, queries, 1)

    exaustive_indices = exaustive_indices.reshape(exaustive_indices.shape[:-1])
    exaustive_distances = exaustive_distances.reshape(exaustive_distances.shape[:-1])

    vptree = vptree_cls()
    vptree.set(data)
    vptree_indices, vptree_distances = vptree.search1NN(queries)

    vptree_indices = np.array(vptree_indices, dtype=np.uint64)
    vptree_distances = np.array(vptree_distances, dtype=np.float32)

    assert np.array_equal(exaustive_indices, vptree_indices)
    np.testing.assert_allclose(exaustive_distances, vptree_distances, rtol=1e-06)
