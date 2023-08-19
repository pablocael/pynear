import time
import yaml
import numpy as np
import pandas as pd
import faiss
import pynear
from sklearn.neighbors import NearestNeighbors
from typing import Callable, Any, Dict, Generator, List, Optional, Tuple, Union
from dataclasses import dataclass
from pynear.benchmark.dataset import generate_gaussian_dataset
from pynear.logging import create_and_configure_log
from pynear.benchmark.index_adapters import create_index_adapter

logger = create_and_configure_log(__name__)

# number of searchs to perform for averaging the result
# in order to reduce effect of high outliers
NUM_AVG_SEARCHS = 8


@dataclass
class BenchmarkCase:
    """
    A benchmark case to run.
    """

    name: str

    # list of test case K values (how many neighbors to search)
    # each case will result in multiple tests, one for each k value
    k: List[int]
    num_queries: List[int]
    index_types: List[str]
    dimensions: List[int]
    dataset_total_size: int
    dataset_num_clusters: int
    dataset_type: str = "float32"

    def run(self):
        results = []
        for dimension in self.dimensions:
            logger.info("generating dataset ...")
            data = generate_gaussian_dataset(
                self.dataset_total_size,
                self.dataset_num_clusters,
                dimension,
                data_type=np.dtype(self.dataset_type),
            )
            logger.info(f"generating dataset done")
            for k_value in self.k:
                for index_type in self.index_types:
                    logger.info(f"processing index_type: {index_type} for k = {k_value} and dimension = {dimension}")
                    index = create_index_adapter(index_type)
                    logger.info(f"buiding index ...")
                    index.build_index(data)
                    logger.info(f"buiding index done")
                    logger.info("start performing queries")
                    for num_queries in self.num_queries:
                        query = generate_gaussian_dataset(
                            num_queries,
                            1,
                            dimension,
                            data_type=np.dtype(self.dataset_type),
                        )
                        logger.info(f"start performing queries (num_queries = {num_queries})")
                        runs = np.array([index.clock_search(query, k_value) for _ in range(NUM_AVG_SEARCHS)])
                        logger.info(f"5 runs set: {runs}")
                        runs = BenchmarkCase.reject_outliers(runs)
                        logger.info(f"rejected {NUM_AVG_SEARCHS-len(runs)} outliers.")

                        # reject outliers
                        avg = sum(runs) / len(runs)
                        logger.info(f"avg of {NUM_AVG_SEARCHS} searches runtime is {avg:0.4f}")

                        results.append({
                            "time": avg,
                            "k": k_value,
                            "num_seraches_avg": NUM_AVG_SEARCHS,
                            "num_queries": num_queries,
                            "index_type": index_type,
                            "dimension": dimension,
                            "dataset_total_size": self.dataset_total_size,
                            "dataset_num_clusters": self.dataset_num_clusters,
                        })
                        logger.info(f"done performing queries for (num_queries = {num_queries})")
                    logger.info(f"queries done")
        return {"benchmark_case_id": self.id(), "benchmark_case_name": self.name, "results": results}

    # the random seed for reproducibility
    seed: int = 12246

    def id(self):
        # return name in a dns compatible format
        return self.name.replace(" ", "-").lower()

    def __str__(self):
        return f"{self.id()}: ks={self.k}, num_queries={self.num_queries}, index_types={self.index_types}, dataset_total_size={self.dataset_total_size}, num_clusters={self.dataset_num_clusters}, seed={self.seed}"

    @staticmethod
    def reject_outliers(data, m=1.2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]

class BenchmarkRunner:
    """
    A generic benchmark build helper class to automate benchmark generation.
    """

    def __init__(self, benchmark_yaml_file: str):
        """
        Build a benchmark comparator between pynear and faiss.

        Args:
            benchmark_cases (List[BenchmarkCase]): the list of benchmark cases to perform.
        """
        self._benchmark_cases = BenchmarkRunner.read_cases_from_yaml(benchmark_yaml_file)
        self._results = pd.DataFrame()

    @staticmethod
    def read_cases_from_yaml(yaml_file: str) -> List[BenchmarkCase]:
        """
        Read benchmark cases from a yaml file.

        Args:
            yaml_file (str): the yaml file to read from.
        """
        with open(yaml_file, "r") as stream:
            benchmark_configs = yaml.safe_load(stream)

        cases = benchmark_configs.get("benchmark", {}).get("cases", [])

        results = []
        for case in cases:
            results.append(BenchmarkCase(**case))

        return results

    def run(self) -> Generator[Dict[str, Any], None, None]:
        num_cases = len(self._benchmark_cases)
        logger.info(f"start running benchmark for {num_cases} cases ...")

        for case in self._benchmark_cases:
            logger.info(f"***** begin case {case.name} *****\n")
            case_result = case.run()
            logger.info(f"******* end case {case.name} *****\n")
            yield case_result
