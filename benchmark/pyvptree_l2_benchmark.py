import json
import time
import pyvptree
import numpy as np

import dicutil

def build_vptree_l2_knn_index(features):
    vptree = pyvptree.VPTreeL2Index()
    vptree.set(features)
    return vptree

def find_vptree_feature_neighbors(index, query_features, k=1):
    return index.searchKNN(query_features, k)

if __name__ == '__main__':
    np.random.seed(seed=42)

    test_results = {}
    print('start starting test of vptree L2 index ...')
    num_queries = 5000
    for dimension in [3, 4, 8, 16, 32, 64, 128, 256]:
        for K in [1, 3, 8, 32]:
            for num_points in [200000, 500000, 1000000]:
                print(f'running case K = {K}, num_points = {num_points}')
                data = np.random.random((num_points, dimension)).astype(np.float32)

                start = time.time()
                vptree_index = build_vptree_l2_knn_index(data)
                creation_time = time.time() - start

                # average 3 measurements
                avg_search_time = 0
                for i in range(3):
                    queries = np.random.random((num_queries, dimension)).astype(np.float32)
                    start = time.time()
                    d, i = find_vptree_feature_neighbors(vptree_index, queries, K)
                    avg_search_time += time.time() - start

                avg_search_time = avg_search_time / 3
                test_results = dicutil.set_dict(test_results, [dimension, K, num_points],
                    {
                        'K': K,
                        'index_type': 'pyvptree.VPTreeL2Index',
                        'dimension': dimension,
                        'num_points': num_points,
                        'creation_time': creation_time,
                        'avg_search_time': avg_search_time
                    }
                )
                print(f'test time: {avg_search_time}')


    with open('benchmark/results/pyvptree_l2_test_result.json', 'w') as outfile:
        json.dump(test_results, outfile)

    print('pyvptree test index finished ...')
