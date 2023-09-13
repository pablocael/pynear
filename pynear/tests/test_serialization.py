import pickle

import numpy as np

import pynear


def test_empty_index_serialization():
    vptree = pynear.VPTreeL2Index()
    data = pickle.dumps(vptree)
    recovered = pickle.loads(data)
    data_rec = pickle.dumps(recovered)
    assert data_rec == data

    # not initializing data with .set() should
    # be equivalent of initializing with empty array
    vptree = pynear.VPTreeL2Index()
    empty = np.array([]).reshape(-1, 8)
    vptree.set(empty)
    data = pickle.dumps(vptree)
    recovered = pickle.loads(data)
    data_rec = pickle.dumps(recovered)
    assert data_rec == data

    vptree = pynear.VPTreeBinaryIndex()
    data = pickle.dumps(vptree)
    recovered = pickle.loads(data)
    data_rec = pickle.dumps(recovered)
    assert data_rec == data


def test_basic_serialization():
    num_points = 10
    dimension = 8
    num_queries = 2
    data = np.random.rand(num_points, dimension).astype(dtype=np.float32)

    queries = np.random.rand(num_queries, dimension).astype(dtype=np.float32)

    vptree = pynear.VPTreeL2Index()
    vptree.set(data)

    vptree_indices, vptree_distances = vptree.search1NN(queries)

    data = pickle.dumps(vptree)
    recovered = pickle.loads(data)
    recovered_data = pickle.dumps(recovered)

    assert data == recovered_data

    vptree_indices_rec, vptree_distances_rec = recovered.search1NN(queries)
    assert vptree_indices_rec == vptree_indices and vptree_distances_rec == vptree_distances


def test_string_serialization():
    num_points = 27
    dimension = 8
    data = np.random.rand(num_points, dimension).astype(dtype=np.float32)

    vptree = pynear.VPTreeL2Index()
    vptree.set(data)

    string_state = vptree.to_string()

    binary_state = pickle.dumps(vptree)
    restored = pickle.loads(binary_state)

    restored_string_state = restored.to_string()

    assert restored_string_state == string_state


def test_binary_serialization():
    num_points = 20000
    dimension = 32
    num_queries = 2
    data = np.random.rand(num_points, dimension).astype(dtype=np.uint8)

    queries = np.random.rand(num_queries, dimension).astype(dtype=np.uint8)

    vptree = pynear.VPTreeBinaryIndex()
    vptree.set(data)

    vptree_indices, vptree_distances = vptree.search1NN(queries)

    data = pickle.dumps(vptree)
    recovered = pickle.loads(data)
    recovered_data = pickle.dumps(recovered)

    assert data == recovered_data

    vptree_indices_rec, vptree_distances_rec = recovered.search1NN(queries)
    assert vptree_distances_rec == vptree_distances
