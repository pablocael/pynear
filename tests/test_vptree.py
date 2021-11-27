import pyvptree
import numpy as np
import heapq

# def build_knn_index(features, features_ids):

#     # d is the bit dimension of the binary vector (in bits), tipically 256
#     # so, if your feature has 32 uint8 dimensions, it will have 256 bits (recommended size is 256 bit due to optimized settings)
#     d = features.shape[1] * 8
#     quantizer = faiss.IndexBinaryFlat(d)

#     # Number of clusters.
#     nlist = int(np.sqrt(features.shape[0]))

#     logger.debug(f'building index with dimension size = {d}, input features shape is {features.shape}, number of clusters is {nlist}')
#     index = faiss.IndexBinaryIVF(quantizer, d, nlist)
#     index.nprobe = d  # Number of nearest clusters to be searched per query.
#     data = np.uint8(features)
#     index.train(data)
#     index.add_with_ids(data, np.array(features_ids, dtype=np.int64))
#     return index

# def find_feature_neighbors(index, query_features, k=1):

#     if query_features is None or len(query_features) == 0:
#         return None, None

#     logger.debug(f'performing search in index with query feature shape = {query_features.shape}')
#     return index.search(np.uint8(query_features), k=k)

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

    vptree = pyvptree.VPTree()
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

    vptree = pyvptree.VPTree()
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

    vptree = pyvptree.VPTree()
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

    vptree = pyvptree.VPTree()
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

