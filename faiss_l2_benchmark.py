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

    dimension = 32
    num_points = 821030
    data = np.random.random((num_points, dimension)).astype(np.float32)  # inputs of faiss must be float32

    num_queries = 5000
    queries = np.random.random((num_queries, dimension)).astype(np.float32)  # inputs of faiss must be float32

    K = 2

    start = time.time()
    print('start creation of faiss index ...')
    faiss_index = build_faiss_l2_knn_index(data, np.arange(num_points))
    print(f'end faiss index creation, took {time.time() - start} ...')

    start = time.time()
    print('start faiss search ...')
    faiss_indices, faiss_distances = find_faiss_feature_neighbors(faiss_index, queries, K)
    print(f'end faiss search, took {time.time() - start} ...')
