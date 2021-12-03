#
# MIT Licence
# Copyright 2021 Pablo Carneiro Elias
#

import pyvptree
import numpy as np
import heapq

def hamming_distance(a, b):
    r = (1 << np.arange(8))[:,None]
    return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)

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

    assert(len(vptree_indices) == len(exaustive_indices))
    assert(len(vptree_distances) == len(exaustive_distances))

    # check if distances are same
    for i in range(len(exaustive_distances)):
        vp = vptree_distances[i]
        ex = exaustive_distances[i]
        assert(sorted(vp) == sorted(ex))

    # do not check indices since in discrete binary data there can be multiple valid solutions to KNN with a higher probability than in the continuous case

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

    exaustive_result = list(zip(exaustive_indices, exaustive_distances))
    vptree_result = list(zip(vptree_indices, vptree_distances))

    assert(len(exaustive_result) == len(vptree_result))

    # check if distances are same
    for i in range(len(exaustive_distances)):
        vp = vptree_distances[i]
        ex = exaustive_distances[i]
        assert(sorted(vp) == sorted(ex))

    # do not check indices since in discrete binary data there can be multiple valid solutions to KNN with a higher probability than in the continuous case

def test_k_equals_dataset():

    np.random.seed(seed=42)

    dimension = 5
    num_points = 2021
    data = np.random.rand(int(num_points), dimension)

    num_queries = 8
    queries = np.random.rand(num_queries, dimension)

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

    exaustive_result = list(zip(exaustive_indices, exaustive_distances))
    vptree_result = list(zip(vptree_indices, vptree_distances))

    assert(len(exaustive_result) == len(vptree_result))

    # check if indices are same
    for i in range(len(exaustive_result)):
        vp = vptree_result[i]
        ex = exaustive_result[i]
        assert(sorted(vp[0]) == sorted(ex[0]))
        assert(sorted(vp[1]) == sorted(ex[1]))

def test_large_dataset():

    np.random.seed(seed=42)

    dimension = 8
    num_points = 401001
    data = np.random.rand(int(num_points), dimension)

    num_queries = 8
    queries = np.random.rand(num_queries, dimension)

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

    exaustive_result = list(zip(exaustive_indices, exaustive_distances))
    vptree_result = list(zip(vptree_indices, vptree_distances))

    assert(len(exaustive_result) == len(vptree_result))

    # check if indices are same
    for i in range(len(exaustive_result)):
        vp = vptree_result[i]
        ex = exaustive_result[i]
        assert(sorted(vp[0]) == sorted(ex[0]))
        assert(sorted(vp[1]) == sorted(ex[1]))

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

    exaustive_result = list(zip(exaustive_indices, exaustive_distances))
    vptree_result = list(zip(vptree_indices, vptree_distances))

    assert(len(exaustive_result) == len(vptree_result))

    # check if indices are same
    for i in range(len(exaustive_result)):
        vp = vptree_result[i]
        ex = exaustive_result[i]
        assert(sorted(vp[0]) == sorted(ex[0]))
        assert(sorted(vp[1]) == sorted(ex[1]))


def test_query_larger_than_dataset():

    # set seed for predictability
    np.random.seed(seed=42)

    num_points = 5
    dimension = 8
    data = np.random.rand(int(num_points), dimension)

    num_queries = 8
    queries = np.random.rand(num_queries, dimension)

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

    exaustive_result = list(zip(exaustive_indices, exaustive_distances))
    vptree_result = list(zip(vptree_indices, vptree_distances))

    assert(len(exaustive_result) == len(vptree_result))

    # check if indices are same
    for i in range(len(exaustive_result)):
        vp = vptree_result[i]
        ex = exaustive_result[i]
        assert(sorted(vp[0]) == sorted(ex[0]))
        assert(sorted(vp[1]) == sorted(ex[1]))

def test_compare_with_exaustive_KNN():

    # set seed for predictability
    np.random.seed(seed=42)

    num_points = 21231
    dimension = 8
    data = np.random.rand(int(num_points), dimension)

    num_queries = 23
    queries = np.random.rand(num_queries, dimension)

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

    exaustive_result = list(zip(exaustive_indices, exaustive_distances))
    vptree_result = list(zip(vptree_indices, vptree_distances))

    assert(len(exaustive_result) == len(vptree_result))

    # check if indices are same
    for i in range(len(exaustive_result)):
        vp = vptree_result[i]
        ex = exaustive_result[i]
        assert(sorted(vp[0]) == sorted(ex[0]))
        assert(sorted(vp[1]) == sorted(ex[1]))


def test_compare_with_exaustive_1NN():

    # set seed for predictability
    np.random.seed(seed=42)

    num_points = 21231
    dimension = 8
    data = np.random.rand(int(num_points), dimension)

    num_queries = 23
    queries = np.random.rand(num_queries, dimension)

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

    exaustive_result = list(zip(exaustive_indices, exaustive_distances))
    vptree_result = list(zip(vptree_indices, vptree_distances))

    assert(len(exaustive_result) == len(vptree_result))

    # check if indices are same
    for i in range(len(exaustive_result)):
        vp = vptree_result[i]
        ex = exaustive_result[i]
        assert(vp[0] == ex[0])
        assert(vp[1] == ex[1])

