#
# MIT Licence
# Copyright 2021 Pablo Carneiro Elias
#

import pyvptree
import numpy as np
import heapq
import math

def hamming_distance(a, b):
    r = (1 << np.arange(8))[:,None]
    return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)

def compare_results(distances1, distances2):
    assert len(distances1) == len(distances2)
    if type(distances1[0]) is not list:
        distances1.sort()
        distances2.sort()

    for i in range(len(distances1)):
        x = distances1[i]
        y = distances2[i]
        if type(x) == list:
            xsorted = sorted(x)
            ysorted = sorted(y)
            for j in range(len(xsorted)):
                assert math.isclose(xsorted[j], ysorted[j], abs_tol=1e-2)
        else:
            assert math.isclose(x, y, abs_tol=1e-1)

def test_binary():

    np.random.seed(seed=42)

    dimension = 32
    num_points = 2021
    data = np.random.normal(scale=255, loc=0, size=(num_points, dimension)).astype(dtype=np.uint8)

    num_queries = 8
    queries = np.random.normal(scale=255, loc=0, size=(num_queries, dimension)).astype(dtype=np.uint8)

    K = 2

    # check exaustivelly and find the correct answer
    exaustive_indices = []
    exaustive_distances = []
    for qi, q in enumerate(queries):

        max_heap = []
        tau = float('inf')
        for i, p in enumerate(data):
            d = hamming_distance(p, q)
            l = len(max_heap)
            if d < tau or l < K:

                if len(max_heap) == K:
                    heapq.heappop(max_heap)

                heapq.heappush(max_heap, (-d,i))
                tau = -max_heap[0][0]

        exaustive_distances.append([-v[0] for v in max_heap])
        exaustive_indices.append([v[1] for v in max_heap])


    # now try same search with vp tree

    vptree = pyvptree.VPTreeBinaryIndex()
    vptree.set(data)
    vptree_indices, vptree_distances = vptree.searchKNN(queries, K)

    compare_results(vptree_distances, exaustive_distances)

def test_large_binary():

    np.random.seed(seed=42)

    dimension = 32
    num_points = 40021
    data = np.random.normal(scale=255, loc=0, size=(num_points, dimension)).astype(dtype=np.uint8)

    num_queries = 8
    queries = np.random.normal(scale=255, loc=0, size=(num_queries, dimension)).astype(dtype=np.uint8)

    K = 3

    # check exaustivelly and find the correct answer
    exaustive_indices = []
    exaustive_distances = []
    for qi, q in enumerate(queries):

        max_heap = []
        tau = float('inf')
        for i, p in enumerate(data):
            d = hamming_distance(p, q)
            l = len(max_heap)
            if d < tau or l < K:

                if len(max_heap) == K:
                    heapq.heappop(max_heap)

                heapq.heappush(max_heap, (-d,i))
                tau = -max_heap[0][0]

        exaustive_distances.append([-v[0] for v in max_heap])
        exaustive_indices.append([v[1] for v in max_heap])


    # now try same search with vp tree

    vptree = pyvptree.VPTreeBinaryIndex()
    vptree.set(data)
    vptree_indices, vptree_distances = vptree.searchKNN(queries, K)

    compare_results(vptree_distances, exaustive_distances)

def test_k_equals_dataset():

    np.random.seed(seed=42)

    dimension = 8
    num_points = 2021
    data = np.random.rand(int(num_points), dimension).astype(dtype=np.float32)

    num_queries = 8
    queries = np.random.rand(num_queries, dimension).astype(dtype=np.float32)

    K = num_points

    # check exaustivelly and find the correct answer
    exaustive_indices = []
    exaustive_distances = []
    for qi, q in enumerate(queries):

        max_heap = []
        tau = float('inf')
        for i, p in enumerate(data):
            d = np.linalg.norm(q-p)
            l = len(max_heap)
            if d < tau or l < K:

                if len(max_heap) == K:
                    heapq.heappop(max_heap)

                heapq.heappush(max_heap, (-d,i))
                tau = -max_heap[0][0]

        exaustive_distances.append([-v[0] for v in max_heap])
        exaustive_indices.append([v[1] for v in max_heap])


    # now try same search with vp tree

    vptree = pyvptree.VPTreeL2Index()
    vptree.set(data)
    vptree_indices, vptree_distances = vptree.searchKNN(queries, K)

    compare_results(vptree_distances, exaustive_distances)

def test_large_dataset():

    np.random.seed(seed=42)

    dimension = 8
    num_points = 401001
    data = np.random.rand(int(num_points), dimension).astype(dtype=np.float32)

    num_queries = 8
    queries = np.random.rand(num_queries, dimension).astype(dtype=np.float32)

    K = 3

    # check exaustivelly and find the correct answer
    exaustive_indices = []
    exaustive_distances = []
    for qi, q in enumerate(queries):

        max_heap = []
        tau = float('inf')
        for i, p in enumerate(data):
            d = np.linalg.norm(q-p)
            l = len(max_heap)
            if d < tau or l < K:

                if len(max_heap) == K:
                    heapq.heappop(max_heap)

                heapq.heappush(max_heap, (-d,i))
                tau = -max_heap[0][0]

        exaustive_distances.append([-v[0] for v in max_heap])
        exaustive_indices.append([v[1] for v in max_heap])


    # now try same search with vp tree

    vptree = pyvptree.VPTreeL2Index()
    vptree.set(data)
    vptree_indices, vptree_distances = vptree.searchKNN(queries, K)

    compare_results(vptree_distances, exaustive_distances)

def test_dataset_split_less_than_k():
    """doc
        Test the case where on of the splits of the dataset eliminates half of points but contains less than K points
    """
    # set seed for predictability
    np.random.seed(seed=42)

    num_points = 4
    dimension = 2
    data = np.array([[-2.5, 0], [-2.58, 0], [0,0], [2.5, 0], [2.6, 0]])

    num_queries = 8
    queries = np.array([[-2.55, 0]])

    K = 4

    # check exaustivelly and find the correct answer
    exaustive_indices = []
    exaustive_distances = []
    for qi, q in enumerate(queries):

        max_heap = []
        tau = float('inf')
        for i, p in enumerate(data):
            d = np.linalg.norm(q-p)
            l = len(max_heap)
            if d < tau or l < K:

                if len(max_heap) == K:
                    heapq.heappop(max_heap)

                heapq.heappush(max_heap, (-d,i))
                tau = -max_heap[0][0]

        exaustive_distances.append([-v[0] for v in max_heap])
        exaustive_indices.append([v[1] for v in max_heap])


    # now try same search with vp tree

    vptree = pyvptree.VPTreeL2Index()
    vptree.set(data)
    vptree_indices, vptree_distances = vptree.searchKNN(queries, K)

    compare_results(vptree_distances, exaustive_distances)

def test_query_larger_than_dataset():

    # set seed for predictability
    np.random.seed(seed=42)

    num_points = 5
    dimension = 8
    data = np.random.rand(int(num_points), dimension).astype(dtype=np.float32)

    num_queries = 8
    queries = np.random.rand(num_queries, dimension).astype(dtype=np.float32)

    K = 3

    # check exaustivelly and find the correct answer
    exaustive_indices = []
    exaustive_distances = []
    for qi, q in enumerate(queries):

        max_heap = []
        tau = float('inf')
        for i, p in enumerate(data):
            d = np.linalg.norm(q-p)
            l = len(max_heap)
            if d < tau or l < K:

                if len(max_heap) == K:
                    heapq.heappop(max_heap)

                heapq.heappush(max_heap, (-d,i))
                tau = -max_heap[0][0]

        exaustive_distances.append([-v[0] for v in max_heap])
        exaustive_indices.append([v[1] for v in max_heap])


    # now try same search with vp tree

    vptree = pyvptree.VPTreeL2Index()
    vptree.set(data)
    vptree_indices, vptree_distances = vptree.searchKNN(queries, K)

    compare_results(vptree_distances, exaustive_distances)

def test_compare_with_exaustive_KNN():

    # set seed for predictability
    np.random.seed(seed=42)

    num_points = 21231
    dimension = 8
    data = np.random.rand(int(num_points), dimension).astype(dtype=np.float32)

    num_queries = 23
    queries = np.random.rand(num_queries, dimension).astype(dtype=np.float32)

    K = 3

    # check exaustivelly and find the correct answer
    exaustive_indices = []
    exaustive_distances = []
    for qi, q in enumerate(queries):

        max_heap = []
        tau = float('inf')
        for i, p in enumerate(data):
            d = np.linalg.norm(q-p)
            l = len(max_heap)
            if d < tau or l < K:

                if len(max_heap) == K:
                    heapq.heappop(max_heap)

                heapq.heappush(max_heap, (-d,i))
                tau = -max_heap[0][0]

        exaustive_distances.append([-v[0] for v in max_heap])
        exaustive_indices.append([v[1] for v in max_heap])


    # now try same search with vp tree

    vptree = pyvptree.VPTreeL2Index()
    vptree.set(data)
    vptree_indices, vptree_distances = vptree.searchKNN(queries, K)

    compare_results(vptree_distances, exaustive_distances)

def test_compare_with_exaustive_1NN():

    # set seed for predictability
    np.random.seed(seed=42)

    num_points = 21231
    dimension = 8
    data = np.random.rand(int(num_points), dimension).astype(dtype=np.float32)

    num_queries = 23
    queries = np.random.rand(num_queries, dimension).astype(dtype=np.float32)

    # check exaustivelly and find the correct answer
    exaustive_indices = []
    exaustive_distances = []
    for q in queries:

        tau = float('inf')
        sel_i = -1
        for i, p in enumerate(data):
            d = np.linalg.norm(q-p)
            if d < tau:
                tau = d
                sel_i = i

        exaustive_distances.append(tau)
        exaustive_indices.append(sel_i)

    # now try same search with vp tree

    vptree = pyvptree.VPTreeL2Index()
    vptree.set(data)
    vptree_indices, vptree_distances = vptree.search1NN(queries)

    compare_results(vptree_distances, exaustive_distances)
