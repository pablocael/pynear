from typing import List
from typing import Tuple

from _pynear import BKTreeBinaryIndex64
from _pynear import BKTreeBinaryIndex128
from _pynear import BKTreeBinaryIndex256
from _pynear import BKTreeBinaryIndex512
from _pynear import BKTreeBinaryIndex as BKTreeBinaryIndexN
from _pynear import VPTreeBinaryIndex64
from _pynear import VPTreeBinaryIndex128
from _pynear import VPTreeBinaryIndex256
from _pynear import VPTreeBinaryIndex512
from _pynear import VPTreeBinaryIndex as VPTreeBinaryIndexN
from _pynear import VPTreeChebyshevIndex
from _pynear import VPTreeL1Index
from _pynear import VPTreeL2Index
from _pynear import dist_chebyshev
from _pynear import dist_hamming_64
from _pynear import dist_hamming_128
from _pynear import dist_hamming_256
from _pynear import dist_hamming_512
from _pynear import dist_l1
from _pynear import dist_l2
import numpy as np

from ._version import __version__


def dist_hamming(a: List, b: List):
    if len(a) != len(b):
        raise ValueError(
            f"invalid data dimension: a and b dimensions must agree."
        )
    dim = len(a)
    if dim == 64:
        return dist_hamming_512(a, b)
    elif dim == 32:
        return dist_hamming_256(a, b)
    elif dim == 16:
        return dist_hamming_128(a, b)
    elif dim == 8:
        return dist_hamming_64(a, b)
    else:
        raise ValueError(
            f"invalid data dimension: hamming distance only supports 64, 32, 16 or 8 bytes of data"
        )



class VPTreeBinaryIndex:
    def __init__(self) -> None:
        self._index = None
        self._dimension = None

    def set(self, data: np.ndarray) -> None:
        self._validate(data)

        dim = data.shape[1]
        if dim == 64:
            self._index = VPTreeBinaryIndex512()
        elif dim == 32:
            self._index = VPTreeBinaryIndex256()
        elif dim == 16:
            self._index = VPTreeBinaryIndex128()
        elif dim == 8:
            self._index = VPTreeBinaryIndex64()
        else:
            self._index = VPTreeBinaryIndexN()

        self._dimension = dim
        self._index.set(data)

    def searchKNN(self, queries: np.ndarray, k: int) -> Tuple[list, list]:
        dim = queries.shape[1]
        if dim != self._dimension:
            raise ValueError(
                f"invalid data dimension: index built data and query data dimensions must agree, index built data dimension is {dim}"
            )

        if self._index is None:
            return [], []

        self._validate(queries)
        return self._index.searchKNN(queries, k)

    def search1NN(self, queries: np.ndarray) -> Tuple[list, list]:
        if self._index is None:
            return [], []

        self._validate(queries)
        return self._index.search1NN(queries)

    def _validate(self, data: np.ndarray) -> None:
        if len(data.shape) != 2:
            raise ValueError("invalid data shape: binary indexes must be 2D")

        if data.dtype != "uint8":
            raise TypeError("invalid data type: binary indexes must be uint8")


class BKTreeBinaryIndex:
    def __init__(self) -> None:
        self._index = None
        self._dimension = None

    def set(self, data: np.ndarray) -> None:
        self._validate(data)

        dim = data.shape[1]
        if dim == 64:
            self._index = BKTreeBinaryIndex512()
        elif dim == 32:
            self._index = BKTreeBinaryIndex256()
        elif dim == 16:
            self._index = BKTreeBinaryIndex128()
        elif dim == 8:
            self._index = BKTreeBinaryIndex64()
        else:
            self._index = BKTreeBinaryIndexN()

        self._dimension = dim
        self._index.set(data)

    def find_threshold(self, queries: np.ndarray, threshold: int) -> Tuple[list, list]:
        dim = queries.shape[1]
        if dim != self._dimension:
            raise ValueError(
                f"invalid data dimension: index built data and query data dimensions must agree, index built data dimension is {dim}"
            )

        if self._index is None:
            return [], []

        self._validate(queries)
        return self._index.find_threshold(queries, threshold)

    def empty(self) -> bool:
        if self._index is None:
            return True

        return self._index.empty()

    def values(self) -> list:
        if self._index is None:
            return []

        return self._index.values()

    def _validate(self, data: np.ndarray) -> None:
        if len(data.shape) != 2:
            raise ValueError("invalid data shape: binary indexes must be 2D")

        if data.dtype != "uint8":
            raise TypeError("invalid data type: binary indexes must be uint8")
