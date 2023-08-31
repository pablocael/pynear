import numpy as np
import pytest

import pynear

CLASSES = [
    (pynear.BKTreeBinaryIndex64, 8),
    (pynear.BKTreeBinaryIndex128, 16),
    (pynear.BKTreeBinaryIndex256, 32),
    (pynear.BKTreeBinaryIndex512, 64),
    (pynear.BKTreeBinaryIndexN, 1),
]


def hamming_distance_pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = (1 << np.arange(8))[:, None, None, None]
    return np.count_nonzero((np.bitwise_xor(a[:, None, :], b[None, :, :]) & r) != 0, axis=(0, -1))


@pytest.mark.parametrize("bktree_cls, dimensions", CLASSES)
def test_bktree_empty_index(bktree_cls, dimensions):
    num_points = 2
    data = np.random.randint(0, 255, size=(num_points, dimensions), dtype=np.uint8)
    empty = np.array([], dtype=np.uint8)

    tree = bktree_cls()
    indices, distances, keys = tree.find_threshold(data, 1)
    truth = [[]] * num_points
    assert truth == indices
    assert truth == distances
    assert tree.empty()
    assert tree.values() == []

    tree.set(empty)
    indices, distances, keys = tree.find_threshold(data, 1)
    truth = [[]] * num_points
    assert truth == indices
    assert truth == distances
    assert tree.empty()
    assert tree.values() == []


@pytest.mark.parametrize("bktree_cls, dimensions", CLASSES)
def test_bktree_find_self(bktree_cls, dimensions):
    num_points = 2
    data = np.random.randint(0, 255, size=(num_points, dimensions), dtype=np.uint8)

    tree = bktree_cls()
    tree.set(data)
    indices, distances, keys = tree.find_threshold(data, 0)
    assert indices == [[i] for i in range(num_points)]
    assert distances == [[0]] * num_points
    assert keys == data[:, None, :].tolist()
    assert tree.size() == num_points
    assert sorted(tree.values()) == sorted(data.tolist())


@pytest.mark.parametrize("bktree_cls, dimensions", CLASSES)
def test_bktree_find_all(bktree_cls, dimensions):
    num_points = 2
    data = np.random.randint(0, 255, size=(num_points, dimensions), dtype=np.uint8)

    tree = bktree_cls()
    tree.set(data)
    indices, distances, keys = tree.find_threshold(data, 255)

    assert indices == [list(range(num_points))] * num_points
    assert distances == hamming_distance_pairwise(data, data).tolist()
    assert keys == np.broadcast_to(data, (num_points, num_points, dimensions)).tolist()
    assert tree.size() == num_points
    assert sorted(tree.values()) == sorted(data.tolist())


@pytest.mark.parametrize("bktree_cls, dimensions", CLASSES)
def test_bktree_find_duplicates(bktree_cls, dimensions):
    num_points = 2
    data = np.zeros((num_points, dimensions), dtype=np.uint8)

    tree = bktree_cls()
    tree.set(data)
    indices, distances, keys = tree.find_threshold(data, 255)

    assert indices == [list(range(num_points))] * num_points
    assert distances == [[0] * num_points] * num_points
    assert keys == np.broadcast_to(data, (num_points, num_points, dimensions)).tolist()
    assert tree.size() == num_points
    assert sorted(tree.values()) == sorted(data.tolist())
