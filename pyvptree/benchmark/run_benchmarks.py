from pyvptree.logging import create_and_configure_log

from pyvptree.benchmark import FaissComparatorBenchmarkCase
from pyvptree.benchmark import FaissComparatorBenchmark
from pyvptree.benchmark import BenchmarkDataset


logger = create_and_configure_log(__name__)

def main():
    datasets = BenchmarkDataset.available_datasets()
    print("datasets", len(datasets))

    cases = []
    print("creating cases ... ")
    for dataset in datasets:
        case = FaissComparatorBenchmarkCase(ks=[2**i for i in range(0,5)], dataset=dataset)
        print("case created", str(case))
        cases.append(case)
    

    print("running ... ")
    runner = FaissComparatorBenchmark(benchmark_cases=cases)
    runner.run()

    print(">>> RESULT", runner.result())


if __name__ == "__main__":
    main()
