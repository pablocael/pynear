import time
import faiss
import numpy as np

def build_faiss_binary_knn_index(features, features_ids):

    # d is the bit dimension of the binary vector (in bits), tipically 256
    # so, if your feature has 32 uint8 dimensions, it will have 256 bits (recommended size is 256 bit due to optimized settings)
    d = features.shape[1] * 8
    quantizer = faiss.IndexBinaryFlat(d)

    # Number of clusters.
    nlist = int(np.sqrt(features.shape[0]))

    index = faiss.IndexBinaryIVF(quantizer, d, nlist)
    index.nprobe = d  # Number of nearest clusters to be searched per query.
    data = np.uint8(features)
    index.train(data)
    index.add_with_ids(data, np.array(features_ids, dtype=np.int64))
    return index

def find_faiss_feature_neighbors(index, query_features, k=1):
    return index.search(np.uint8(query_features), k=k)

if __name__ == '__main__':
    np.random.seed(seed=42)

    dimension = 32
    num_points = 821030
    data = np.random.normal(scale=255, loc=0, size=(num_points, dimension)).astype(dtype=np.uint8)

    num_queries = 5000
    queries = np.random.normal(scale=255, loc=0, size=(num_queries, dimension)).astype(dtype=np.uint8)

    K = 1

    start = time.time()
    print('start creation of faiss index ...')
    faiss_index = build_faiss_binary_knn_index(data, np.arange(num_points))
    print(f'end faiss index creation, took {time.time() - start} ...')

    start = time.time()
    print('start faiss search ...')
    faiss_indices, faiss_distances = find_faiss_feature_neighbors(faiss_index, queries, K)
    print(f'end faiss search, took {time.time() - start} ...')
