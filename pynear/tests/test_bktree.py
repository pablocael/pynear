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
    distances, keys = tree.find_threshold(data, 1)
    truth = [[]] * num_points
    assert truth == distances

    tree.set(empty)
    distances, keys = tree.find_threshold(data, 1)
    truth = [[]] * num_points
    assert truth == distances


@pytest.mark.parametrize("bktree_cls, dimensions", CLASSES)
def test_bktree_find_self(bktree_cls, dimensions):
    num_points = 2
    data = np.random.randint(0, 255, size=(num_points, dimensions), dtype=np.uint8)

    tree = bktree_cls()
    tree.set(data)
    distances, keys = tree.find_threshold(data, 0)
    assert distances == [[0]] * num_points
    assert keys == data[:, None, :].tolist()


@pytest.mark.parametrize("bktree_cls, dimensions", CLASSES)
def test_bktree_find_all(bktree_cls, dimensions):
    num_points = 2
    data = np.random.randint(0, 255, size=(num_points, dimensions), dtype=np.uint8)

    tree = bktree_cls()
    tree.set(data)
    distances, keys = tree.find_threshold(data, 255)

    assert distances == hamming_distance_pairwise(data, data).tolist()
    assert keys == np.broadcast_to(data, (num_points, num_points, dimensions)).tolist()
