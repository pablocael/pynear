import h5py
import numpy as np
import pyvptree
import pandas as pd
from typing import Union

class BenchmarkDataset:
    def __init__(
        self,
        data: np.ndarray,
        name: str,
        pyvpindex_type: Union[pyvptree.VPTreeL2Index, pyvptree.VPTreeBinaryIndex],
        description: str,
    ):
        """
        Creates a new dataset.

        Args:
           data (np.ndarray): a (N, DIM) array where N is number of examples and DIM is the dimension of each example.
           name (str): the name of this dataset.
           description (str): a short description of what kind of data is within this dataset.
           pyvpindex_type (pyvptree.VPTreeL2Index or pyvptree.VPTreeBinaryIndex):
           the pyvptree index that will be used with to index this dataset.
        """
        self._name: str = name
        self._pyvpindex_type: Union[pyvptree.VPTreeL2Index, pyvptree.VPTreeBinaryIndex] = pyvpindex_type
        self._description: str = description
        self._data: np.ndarray = data

    def save(self, filepath: str):
        with h5py.File(filepath, "w") as outfile:
            dataset = outfile.create_dataset("dataset", data=self._data)

            dataset.attrs["name"] = self._name
            dataset.attrs["description"] = self._description

    def load(self, filepath: str):
        with h5py.File(filepath, "r") as infile:
            dataset = infile["dataset"]
            self._data = dataset[...]  # Load it into memory. Could also slice a subset.

            self._name = dataset.attrs["name"]
            self._description = dataset.attrs["description"]

    def dimension(self):
        if self._data is None:
            return 0

        return self._data.shape[1]

