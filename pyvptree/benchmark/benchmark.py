import time
import numpy as np
import pandas as pd
import faiss
import pyvptree
from sklearn.neighbors import NearestNeighbors
from typing import Callable, Any, List, Tuple, Union
from dataclasses import dataclass
from pyvptree.benchmark.dataset import BenchmarkDataset
from pyvptree.logging import create_and_configure_log

logger = create_and_configure_log(__name__)


@dataclass
class BenchmarkCase:
    """
    - name: "Pyvptree Indexes Comparison"
      k: [2, 4, 8]
      num_queries: [16, 32, 128]
      dataset:
      - total_size: 2500000
      - num_clusters: 50
      index_types:
      - VPTreeL2Index
      - VPTreeBinaryIndex
      - VPTreeChebyshevIndex
      - VPTreeL1Index # (manhattan distance)
    """

    name: str

    # list of test case K values (how many neighbors to search)
    # each case will result in multiple tests, one for each k value
    ks: List[int]
    num_queries: List[int]
    index_types: List[str]

    dataset_total_size: int
    num_clusters: int

    # the random seed for reproducibility
    seed: int = 12246

    def id(self):
        # return name in a dns compatible format
        return self.name.replace(" ", "-").lower()

    def __str__(self):
        return f"{self.id()}: ks={self.ks}, num_queries={self.num_queries}, index_types={self.index_types}, dataset_total_size={self.dataset_total_size}, num_clusters={self.num_clusters}, seed={self.seed}"



class Benchmark:
    """
    A generic benchmark build helper class to automate benchmark generation.
    """

    def __init__(self, benchmark_yaml_file: str):
        """
        Build a benchmark comparator between pyvptree and faiss.

        Args:
            benchmark_cases (List[BenchmarkCase]): the list of benchmark cases to perform.
        """
        self._benchmark_cases = benchmark_cases
        self._results = pd.DataFrame()


    def read_cases_from_yaml(self, yaml_file: str):
        """
        Read benchmark cases from a yaml file.

        Args:
            yaml_file (str): the yaml file to read from.
        """
        with open(yaml_file, "r") as stream:
            try:
                benchmark_configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)
    def run(self):
        num_cases = len(self._benchmark_cases)
        logger.info(f"start running benchmark for {num_cases} cases ...")

        results = []
        for case in self._benchmark_cases:
            logger.info("***** begin case *****\n")
            start_case = time.time()
            logger.info(f"starting case {str(case)} ...")
            logger.info("splittting dataset into train / test... ")
            train, test = self._split_test_train_case(case.dataset)
            logger.info(f"split sizes test = {test.shape[0]}, train = {train.shape[0]}... ")

            start = time.time()
            logger.info("generating faiss index ... ")
            faiss_index = self._generate_faiss_index(train, case.dataset._pyvpindex_type)
            logger.info(f"done, faiss index took {time.time()-start:0.3f} seconds... ")

            start = time.time()
            logger.info("generating pyvp index ... ")
            pyvptree_index = self._generate_pyvptree_index(train, case.dataset._pyvpindex_type)
            logger.info(f"done, pyvptree index took {time.time()-start:0.3f} seconds... ")
            for k in case.ks:
                logger.info("searching into faiss index ... ")
                start = time.time()
                self._search_knn_faiss(faiss_index, test, k)
                end = time.time()
                faiss_time = end - start
                logger.info(f"faiss search for k = {k} took {faiss_time:0.3f} seconds... ")

                logger.info("searching into pyvptree index ... ")
                start = time.time()
                self._search_knn_pyvptree(pyvptree_index, test, k)
                end = time.time()
                pyvptree_time = end - start
                logger.info(f"pyvptree search for k = {k} took {pyvptree_time:0.3f} seconds... ")

                results.append(
                    {
                        "k": k,
                        "dimension": case.dataset.dimension(),
                        "size": case.dataset.size(),
                        "index_type": case.dataset.index_type().__name__,
                        "query_size": test.shape[0],
                        "faiss_time": faiss_time,
                        "pyvptree_time": pyvptree_time,
                    }
                )

                if case.dataset.index_type() == pyvptree.VPTreeL2Index:
                    start = time.time()
                    logger.info("generating sklearn index ... ")
                    sklearn_index = self._generate_sklearn_index(train, k)
                    logger.info(f"done, sklearn index took {time.time()-start:0.3f} seconds... ")

                    logger.info("searching into sklearn index ... ")
                    start = time.time()
                    self._search_knn_sklearn(sklearn_index, test)
                    end = time.time()
                    sklearn_time = end - start
                    logger.info(f"sklearn search for k = {k} took {sklearn_time:0.3f} seconds... ")

                    results[-1].update({"sklearn_time": sklearn_time})

            # make sure dataset is unloaded to prevent memory overflow
            logger.info(f"case took {time.time()-start_case:0.3f}")
            case.dataset.unload_data()
            logger.info("***** end case *****\n")

        self._results = pd.DataFrame(results)

    def result(self) -> pd.DataFrame:
        return self._results

    def _split_test_train_case(self, dataset: BenchmarkDataset, num_queries: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        n_test = num_queries  # perform 8 queries for test and rest for train
        data: np.ndarray = dataset.data()
        np.random.shuffle(data)
        n = dataset.size()
        n_train = n - n_test
        return data[0:n_train, :], data[n_train:, :]
