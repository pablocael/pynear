import json
import time
import faiss
import numpy as np

def build_faiss_l2_knn_index(features, features_ids):

    # d is the bit dimension of the binary vector (in bits), tipically 256
    # so, if your feature has 32 uint8 dimensions, it will have 256 bits (recommended size is 256 bit due to optimized settings)
    d = features.shape[1]

    index = faiss.IndexFlatL2(d)
    index.add(features)
    return index

def find_faiss_feature_neighbors(index, query_features, k=1):
    return index.search(query_features, k=k)

if __name__ == '__main__':
    np.random.seed(seed=42)

    test_results = {}
    print('start starting test of faiss L2 index ...')
    num_queries = 5000
    for dimension in [3, 4, 8, 10, 12]:
        for K in [1, 2, 3, 8, 16, 32]:
            for num_points in [100000, 200000, 500000, 1000000]:
                print(f'running case K = {K}, num_points = {num_points}')
                data = np.random.random((num_points, dimension)).astype(np.float32)  # inputs of faiss must be float32


                start = time.time()
                faiss_index = build_faiss_l2_knn_index(data, np.arange(num_points))
                creation_time = time.time() - start

                # average 3 measurements
                avg_search_time = 0
                for i in range(3):
                    queries = np.random.random((num_queries, dimension)).astype(np.float32)  # inputs of faiss must be float32
                    start = time.time()
                    faiss_indices, faiss_distances = find_faiss_feature_neighbors(faiss_index, queries, K)
                    avg_search_time += time.time() - start

                avg_search_time = avg_search_time / 3
                test_results[dimension] = {
                    K: {
                        num_points: {
                            'K': K,
                            'index_type': 'faiss.IndexFlatL2',
                            'dimension': dimension,
                            'num_points': num_points,
                            'creation_time': creation_time,
                            'avg_search_time': avg_search_time
                        }
                    },
                }

                print(f'test time: {avg_search_time}')

    with open('benchmark/results/faiss_l2_test_result.json', 'w') as outfile:
        json.dump(test_results, outfile)

    print('faiss test index finished ...')
