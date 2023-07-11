import wget
from benchmark import BenchmarkDataset

datasets =  []

def generate_high_dim_gaussian_dataset():
    pass


def generate_coco_dataset():
    wget.download("http://images.cocodataset.org/zips/train2017.zip", out="/tmp/datasets/coco_train2017.zip")
    pass

