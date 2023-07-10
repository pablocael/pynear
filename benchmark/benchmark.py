import h5py
import numpy as np
import pyvptree
import pandas as pd
from typing import Callable, Any, List, Tuple, Union
from dataclasses import dataclass

class BenchmarkDataset:

    def __init__(self, data: np.ndarray, name: str, description: str):
        self._name = name
        self._description = description
        self._data = data

    def save(self, filepath: str):
        with h5py.File(filepath, "w") as outfile:
            dataset = outfile.create_dataset('dataset', data=self._data)

            dataset.attrs["name"] = self._name
            dataset.attrs["description"] = self._description

    def load(self, filepath: str):

        with h5py.File(filepath, "r") as infile:
            dataset = infile["dataset"]
            self._data = dataset[...] # Load it into memory. Could also slice a subset.

            self._name = dataset.attrs["name"]
            self._description = dataset.attrs["description"]


@dataclass
class FaissComparatorBenchmarkCase:
    # K (how many neighbors to search)
    k: int

    # benchmark test case name
    name: str

    # the data to use in the test case
    dataset: BenchmarkDataset

    # the index to use with the  dataset for this case
    pyvpindex: Union[pyvptree.VPTreeL2Index, pyvptree.VPTreeBinaryIndex]



class FaissComparatorBenchmark:

    """
    A generic benchmark build helper class to automate benchmark generation.
    """
    def __init__(self, benchmark_cases: List[FaissComparatorBenchmarkCase]):
        """
        Build a benchmark generator.

        Args:
            benchmark_cases (List[BenchmarkCase]): the list of benchmark cases to perform.
        """
        self._benchmark_cases = benchmark_cases


    def run(self):
        pass



    def result(self) -> pd.DataFrame:
        pass

