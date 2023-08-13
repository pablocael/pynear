import time
import annoy
import faiss
import pyvptree
import numpy as np
from sklearn.neighbors import NearestNeighbors
from abc import ABC, abstractmethod


def create_index_adapter(index_name: str):
    # Supported 3rd party indices are: FaissIndexFlatL2, FaissIndexBinaryFlat, AnnoyL2, AnnoyManhattan, AnnoyHamming, SKLearnL2
    mapper = {
        'FaissIndexFlatL2': FaissIndexFlatL2Adapter,
        'FaissIndexBinaryFlat': FaissIndexBinaryFlatAdapter,
        'AnnoyL2': AnnoyL2Adapter,
        'AnnoyManhattan': AnnoyManhattanAdapter,
        'AnnoyHamming': AnnoyHammingAdapter,
        'SKLearnL2': SKLearnL2Adapter,
        'VPTreeL2Index': pyvptree.VPTreeL2Index,
        'VPTreeBinaryIndex': pyvptree.VPTreeBinaryIndex,
        'VPTreeChebyshevIndex': pyvptree.VPTreeChebyshevIndex,
        'VPTreeL1Index': pyvptree.VPTreeL1Index
    }
    if index_name not in mapper:
        raise ValueError(f'Index name {index_name} not supported')

    if index_name.startswith('VPTree'):
        return PyVPtreeAdapter(index_name)

    return mapper[index_name]()

# Supported 3rd party indices are: FaissIndexFlatL2, FaissIndexBinaryFlat, AnnoyL2, AnnoyManhattan, AnnoyHamming, SKLearnL2
class IndexAdapter(ABC):
    @abstractmethod
    def build_index(self, data: np.ndarray):
        pass

    def clock_search(self, query: np.ndarray, k: int) -> float:
        """
        Searchs in index and retrieves the time it took to search
        """
        s = time.time()
        self._search_implementation(query, k)
        return time.time() - s

    @abstractmethod
    def _search_implementation(self, query, k: int):
        pass

class PyVPtreeAdapter(IndexAdapter):

    def __init__(self, pyvp_index_name: str):
        self._index = None
        self._pyvp_index_name = pyvp_index_name
        self._pyvp_index_map = {
            'VPTreeL2Index': pyvptree.VPTreeL2Index,
            'VPTreeBinaryIndex': pyvptree.VPTreeBinaryIndex,
            'VPTreeChebyshevIndex': pyvptree.VPTreeChebyshevIndex,
            'VPTreeL1Index': pyvptree.VPTreeL1Index
        }

    def build_index(self, data: np.ndarray):
        self._index = self._pyvp_index_map[self._pyvp_index_name]()
        self._index.set(data)

    def _search_implementation(self, query, k: int):
        self._index.searchKNN(query, k=k)

class FaissIndexFlatL2Adapter(IndexAdapter):
    def __init__(self):
        self._index = None

    def build_index(self, data: np.ndarray):
        d = data.shape[1]
        self._index = faiss.IndexFlatL2(d)
        self._index.add(features)

    def _search_implementation(self, query, k: int):
        self._index.search(query, k=k)


class FaissIndexBinaryFlatAdapter(IndexAdapter):
    def __init__(self):
        self._index = None

    def build_index(self, data: np.ndarray):
        d = data.shape[1] * 8
        self._index = faiss.IndexBinaryFlat(d)
        self._index.add(data)

    def _search_implementation(self, query, k: int):
        self._index.search(query, k=k)

class AnnoyL2Adapter(IndexAdapter):
    def __init__(self):
        self._index = None
        pass

    def build_index(self, data: np.ndarray):
        self._index = annoy.AnnoyIndex(data.shape[1], 'euclidean')
        for i, v in enumerate(data):
            self._index.add_item(i, v)

        self._index.build(10)

    def _search_implementation(self, query, k: int):
        self._index.get_nns_by_vector(query, k)

class AnnoyManhattanAdapter(IndexAdapter):
    def __init__(self):
        self._index = None

    def build_index(self, data: np.ndarray):
        self._index = annoy.AnnoyIndex(data.shape[1], 'manhattan')
        for i, v in enumerate(data):
            self._index.add_item(i, v)

    def clock_search(self, query: np.ndarray, k: int):
        return self._index.get_nns_by_vector(query, k)

class AnnoyHammingAdapter(IndexAdapter):
    def __init__(self):
        self._index = None

    def build_index(self, data: np.ndarray):
        self._index = annoy.AnnoyIndex(data.shape[1], 'hamming')
        for i, v in enumerate(data):
            self._index.add_item(i, v)

        self._index.build(10)

    def _search_implementation(self, query, k: int):
        self.index.get_nns_by_vector(query, k)

class SKLearnL2Adapter(IndexAdapter):
    def __init__(self):
        self._index = None

    def build_index(self, data: np.ndarray):
        pass

    def _search_implementation(self, query, k: int):
        # sklearn uses index based on k, so need to build on the fly
        pass

    def clock_search(self, query: np.ndarray, k: int) -> float:
        """
        Searchs in index and retrieves the time it took to search
        """
        # sklearn uses index based on k, so need to build on the fly
        self._index = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', metric='euclidean')

        s = time.time()
        self._index.kneighbors(query)
        return time.time() - s
