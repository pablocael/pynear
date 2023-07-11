import numpy as np
import pandas as pd
from typing import Callable, Any, List, Tuple, Union
from dataclasses import dataclass
from benchmark.dataset import BenchmarkDataset


@dataclass
class FaissComparatorBenchmarkCase:
    # K (how many neighbors to search)
    k: int

    # benchmark test case name
    name: str

    # the data to use in the test case
    dataset: BenchmarkDataset

    # the random seed for reproducibility
    seed: int = 12246


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
