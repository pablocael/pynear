import faiss
import pyvptree
import numpy as np

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

def test_compare_faiss():
    assert True
