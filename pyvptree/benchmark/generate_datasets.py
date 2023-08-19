import glob
import os

import numpy as np
from PIL import Image

import pyvptree
from pyvptree.benchmark.dataset import BenchmarkDataset


def main():
    datasets = BenchmarkDataset.available_datasets()
    for dataset in datasets:
        print(">>>>>>>>>", dataset.name(), dataset.dimension(), dataset.size())
        # force memory release
        dataset.unload_data()


if __name__ == "__main__":
    main()
