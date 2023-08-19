import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

from pyvptree.benchmark import BenchmarkDataset, ComparatorBenchmark, ComparatorBenchmarkCase
from pyvptree.logging import create_and_configure_log

logger = create_and_configure_log(__name__)


def create_performance_plot(result: pd.DataFrame, output_folder: str):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    index_types = result.groupby("index_type")

    for index_type, df in index_types:
        groups = result.groupby("k")
        for k, group in groups:
            # create one plot per k
            # resetting index before melting to save the current index in 'index' column...

            data_size = group.iloc[0]["size"]
            query_size = group.iloc[0]["query_size"]
            filtered = group.filter(regex="dimension|time*", axis=1)
            df = filtered.melt("dimension", var_name="cols", value_name="query time")
            g = sns.catplot(
                x="dimension", y="query time", hue="cols", data=df, kind="point", errorbar=None, legend=False
            )
            g.set(
                title=f"Data Dimensions x Query Time\nindex_type={index_type}\ndata_size={data_size}), k={k}\nquery_size={query_size})"
            )
            plt.legend(loc="upper center")

            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"{index_type}_k_{k}.png"))
            plt.clf()


def main():
    parser = argparse.ArgumentParser(description="Creates a stress test report for a segemaker endpoint")
    parser.add_argument(
        "--min-dimension",
        default=2,
        type=int,
        help="The minimum dimensionality of the data to generate the benchmarks",
        required=False,
    )
    parser.add_argument(
        "--max-dimension",
        default=32,
        type=int,
        help="The maximum dimensionality of the data to generate the benchmarks",
        required=False,
    )

    args = parser.parse_args()

    min_dim = args.min_dimension
    max_dim = args.max_dimension

    print(f"creating cases for dims from {min_dim} to {max_dim}... ")
    datasets = BenchmarkDataset.generate_gaussian_euclidean_cluster_datasets(min_dim=min_dim, max_dim=max_dim)

    cases = []
    for ds in datasets:
        case = ComparatorBenchmarkCase(ks=[2**i for i in range(0, 5)], dataset=ds)
        print("case created", str(case))
        cases.append(case)

    print("start running benchmarks... ")

    runner = ComparatorBenchmark(benchmark_cases=cases)
    runner.run()
    results = runner.result()
    create_performance_plot(results, f"./results/from_{min_dim}_to_{max_dim}/")

    print("benchmarks done.")
    print("result images written to ./results folder")


if __name__ == "__main__":
    main()
