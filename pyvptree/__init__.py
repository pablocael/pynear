from typing import Type
from _pyvptree import VPTreeBinaryIndex512
from _pyvptree import VPTreeBinaryIndex256
from _pyvptree import VPTreeBinaryIndex128
from _pyvptree import VPTreeBinaryIndex64
from _pyvptree import VPTreeChebyshevIndex
from _pyvptree import VPTreeL1Index
from _pyvptree import VPTreeL2Index

from ._version import __version__


class VPTreeBinaryIndex:

    def __init__(self):
        self._index = None
        self._dimension = None

    def set(self, data):
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

        self._dimension = dim
        self._index.set(data)

    def searchKNN(self, queries, k):
        if self._index is None:
            return [], []

        dim = queries.shape[1]
        if dim != self._dimension:
            raise ValueError(f"invalid data dimension: index built data and query data dimensions must agree, index built data dimension is {dim}")

        self._validate(queries)
        return self._index.searchKNN(queries, k)

    def search1NN(self, queries):
        if self._index is None:
            return [], []

        self._validate(queries)
        return self._index.search1NN(queries)

    def _validate(self, input):
        if len(input.shape) != 2:
            raise ValueError("invalid data shape: binary indexes must be 2D")

        if input.dtype != 'uint8':
            raise TypeError("invalid data type: binary indexes must be uint8")

        dim = input.shape[1]
        if dim not in [8, 16, 32, 64]:
            raise ValueError("invalid dimension: binary indexes must be 8, 16, 32 or 64 bytes")
