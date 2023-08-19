import os
import sys
import argparse
from pyvptree.logging import create_and_configure_log

from pyvptree.benchmark import BenchmarkCase
from pyvptree.benchmark import BenchmarkRunner

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
import seaborn as sns
import yaml

logger = create_and_configure_log(__name__)


def create_performance_plot(result: pd.DataFrame, case_name: str, output_folder: str):

    groups = result.groupby("k")

    for k, group in groups:
        index_types = group.groupby("index_type")
        for index_type, df in index_types:

            # create one plot per k
            # resetting index before melting to save the current index in 'index' column...
            plt.plot([str(v) for v in df["dimension"]], df["time"], label=index_type, marker='o')
            data_size = df.iloc[0]["dataset_total_size"]
            query_size = df.iloc[0]["num_queries"]
            num_avg_searchs = df.iloc[0]["num_seraches_avg"]
            plt.title(
                f"{case_name}\nData Dimensions x Query Time (avg of {num_avg_searchs})\ndata_size={data_size}), k={k}\nquery_size={query_size})")
            plt.legend(loc='upper center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"result_k={k}.png"))
        plt.clf()


def main():

    parser = argparse.ArgumentParser(description="Creates a stress test report for a segemaker endpoint")
    parser.add_argument(
        "--config-file",
        default="benchmark_config.yaml",
        type=str,
        help="The config file describing the benchmark cases to run",
        required=False,
    )

    args = parser.parse_args()

    runner = BenchmarkRunner(args.config_file)

    print("start running benchmarks... ")

    for result in runner.run():
        case_id = result["benchmark_case_id"]
        case_name = result["benchmark_case_name"]
        case_results = result["results"]
        dir_name = f"./results/{case_id}"
        os.makedirs(dir_name, exist_ok=True)
        create_performance_plot(pd.DataFrame(case_results), case_name, dir_name)

    print("benchmarks done.")
    print("result images written to ./results folder")


if __name__ == "__main__":
    main()
