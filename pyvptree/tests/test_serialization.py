import pickle

import numpy as np

import pyvptree


def test_empty_index_serialization():
    vptree = pyvptree.VPTreeL2Index()
    data = pickle.dumps(vptree)
    recovered = pickle.loads(data)
    data_rec = pickle.dumps(recovered)
    assert data_rec == data

    vptree = pyvptree.VPTreeBinaryIndex()
    data = pickle.dumps(vptree)
    recovered = pickle.loads(data)
    data_rec = pickle.dumps(recovered)
    assert data_rec == data


def test_basic_serialization():
    np.random.seed(seed=42)

    num_points = 20000
    dimension = 8
    num_queries = 2
    data = np.random.rand(num_points, dimension).astype(dtype=np.float32)

    queries = np.random.rand(num_queries, dimension).astype(dtype=np.float32)

    vptree = pyvptree.VPTreeL2Index()
    vptree.set(data)

    vptree_indices, vptree_distances = vptree.search1NN(queries)

    data = pickle.dumps(vptree)
    recovered = pickle.loads(data)
    recovered_data = pickle.dumps(recovered)

    assert data == recovered_data

    vptree_indices_rec, vptree_distances_rec = recovered.search1NN(queries)
    assert vptree_indices_rec == vptree_indices and vptree_distances_rec == vptree_distances


def test_binary_serialization():
    np.random.seed(seed=42)

    num_points = 20000
    dimension = 32
    num_queries = 2
    data = np.random.rand(num_points, dimension).astype(dtype=np.uint8)

    queries = np.random.rand(num_queries, dimension).astype(dtype=np.uint8)

    vptree = pyvptree.VPTreeBinaryIndex()
    vptree.set(data)

    vptree_indices, vptree_distances = vptree.search1NN(queries)

    data = pickle.dumps(vptree)
    recovered = pickle.loads(data)
    recovered_data = pickle.dumps(recovered)

    assert data == recovered_data

    vptree_indices_rec, vptree_distances_rec = recovered.search1NN(queries)
    assert vptree_distances_rec == vptree_distances
