import time
import pyvptree
import numpy as np

def build_vptree_binary_knn_index(features):
    vptree = pyvptree.VPTreeL2Index()
    vptree.set(features)
    return vptree

def find_faiss_feature_neighbors(index, query_features, k=1):
    return index.search(np.uint8(query_features), k=k)

def find_vptree_feature_neighbors(index, query_features, k=1):
    return index.searchKNN(query_features, k)

if __name__ == '__main__':
    np.random.seed(seed=42)

    dimension = 4
    num_points = 821030
    data = np.random.normal(scale=255, loc=0, size=(num_points, dimension)).astype(dtype=np.uint8)

    num_queries = 5000
    queries = np.random.normal(scale=255, loc=0, size=(num_queries, dimension)).astype(dtype=np.uint8)

    K = 32

    start = time.time()
    print('start creation of vptree index ...')
    vp_index = build_vptree_binary_knn_index(data)
    print(f'end pyvptree index creation, took {time.time() - start} ...')

    start = time.time()
    print('start pyvptree search ...')
    vp_indices, vp_distances = find_vptree_feature_neighbors(vp_index, queries, K)
    print(f'end pyvptree search, took {time.time() - start} ...')
