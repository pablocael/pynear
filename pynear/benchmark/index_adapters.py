from abc import ABC
from abc import abstractmethod
import time

import annoy
import faiss
import numpy as np
from sklearn.neighbors import NearestNeighbors

import pynear


def create_index_adapter(index_name: str):
    # Supported 3rd party indices are: FaissIndexFlatL2, FaissIndexBinaryFlat, AnnoyL2, AnnoyManhattan, AnnoyHamming, SKLearnL2
    mapper = {
        "FaissIndexFlatL2": FaissIndexFlatL2Adapter,
        "FaissIndexBinaryFlat": FaissIndexBinaryFlatAdapter,
        "AnnoyL2": AnnoyL2Adapter,
        "AnnoyManhattan": AnnoyManhattanAdapter,
        "AnnoyHamming": AnnoyHammingAdapter,
        "SKLearnL2": SKLearnL2Adapter,
        "BKTreeBinaryIndex": PyNearBKTreeAdapter,
        "VPTreeL2Index": pynear.VPTreeL2Index,
        "VPTreeL1Index": pynear.VPTreeL1Index,
        "VPTreeBinaryIndex": pynear.VPTreeBinaryIndex,
        "VPTreeChebyshevIndex": pynear.VPTreeChebyshevIndex,
    }
    if index_name not in mapper:
        raise ValueError(f"Index name {index_name} not supported")

    if index_name.startswith("VPTree"):
        return PyNearVPAdapter(index_name)

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


class PyNearVPAdapter(IndexAdapter):
    def __init__(self, pyvp_index_name: str):
        self._index = None
        self._pyvp_index_name = pyvp_index_name
        self._pyvp_index_map = {
            "VPTreeL2Index": pynear.VPTreeL2Index,
            "VPTreeBinaryIndex": pynear.VPTreeBinaryIndex,
            "VPTreeChebyshevIndex": pynear.VPTreeChebyshevIndex,
            "VPTreeL1Index": pynear.VPTreeL1Index,
        }

    def build_index(self, data: np.ndarray):
        self._index = self._pyvp_index_map[self._pyvp_index_name]()
        self._index.set(data)

    def _search_implementation(self, query, k: int):
        self._index.searchKNN(query, k)

class PyNearBKTreeAdapter(IndexAdapter):
    def __init__(self):
        self._index = pynear.BKTreeBinaryIndex()
        self._dimensions = 0

    def build_index(self, data: np.ndarray):
        self._index.set(data)
        self._dimensions = data.shape[1]

    def _search_implementation(self, query, k: int):
        self._index.find_threshold(query, self._dimensions)

class FaissIndexFlatL2Adapter(IndexAdapter):
    def __init__(self):
        self._index = None

    def build_index(self, data: np.ndarray):
        d = data.shape[1]
        self._index = faiss.IndexFlatL2(d)
        self._index.add(data)

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
        self._index = annoy.AnnoyIndex(data.shape[1], "euclidean")
        for i, v in enumerate(data):
            self._index.add_item(i, v)

        self._index.build(10)

    def _search_implementation(self, query, k: int):
        for v in query:
            self._index.get_nns_by_vector(v, k)


class AnnoyManhattanAdapter(IndexAdapter):
    def __init__(self):
        self._index = None

    def build_index(self, data: np.ndarray):
        self._index = annoy.AnnoyIndex(data.shape[1], "manhattan")
        for i, v in enumerate(data):
            self._index.add_item(i, v)

    def _search_implementation(self, query, k: int):
        for v in query:
            self._index.get_nns_by_vector(v, k)


class AnnoyHammingAdapter(IndexAdapter):
    def __init__(self):
        self._index = None

    def build_index(self, data: np.ndarray):
        self._index = annoy.AnnoyIndex(data.shape[1], "hamming")
        for i, v in enumerate(data):
            self._index.add_item(i, v)

        self._index.build(10)

    def _search_implementation(self, query, k: int):
        for v in query:
            self._index.get_nns_by_vector(v, k)


class SKLearnL2Adapter(IndexAdapter):
    def __init__(self):
        self._index = None

    def build_index(self, data: np.ndarray):
        self._data = data
        pass

    def _search_implementation(self, query, k: int):
        # sklearn uses index based on k, so need to build on the fly
        pass

    def clock_search(self, query: np.ndarray, k: int) -> float:
        """
        Searchs in index and retrieves the time it took to search
        """
        # sklearn uses index based on k, so need to build on the fly
        # we will not clock index training stage
        self._index = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", metric="euclidean")
        self._index.fit(self._data)

        s = time.time()
        self._index.kneighbors(query)
        return time.time() - s
