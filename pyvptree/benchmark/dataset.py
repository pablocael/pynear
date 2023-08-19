import glob
import os
from functools import partial
from tempfile import TemporaryDirectory
from typing import Any, Callable, List, Optional, Union
from zipfile import BadZipFile, ZipFile

import h5py
import numpy as np
import pandas as pd
import wget
from img2vec_pytorch import Img2Vec
from PIL import Image

import pyvptree
from pyvptree.logging import create_and_configure_log

logger = create_and_configure_log(__name__)


class BenchmarkDataset:
    def __init__(
        self,
        data: Union[np.ndarray, Callable[..., Any]],
        name: str,
        dim: int,
        pyvpindex_type: Union[pyvptree.VPTreeL2Index, pyvptree.VPTreeBinaryIndex],
        description: str,
    ):
        """
        Creates a new dataset.

        Args:
           data (np.ndarray or callable that will build ndarray):
            a (N, DIM) array where N is number of examples and DIM is the dimension of each example.
           name (str): the name of this dataset.
           dim (int): the dimensionality of this dataset.
           description (str): a short description of what kind of data is within this dataset.
           pyvpindex_type (pyvptree.VPTreeL2Index or pyvptree.VPTreeBinaryIndex):
           the pyvptree index that will be used with to index this dataset.
        """
        self._name: str = name
        self._pyvpindex_type: Union[pyvptree.VPTreeL2Index, pyvptree.VPTreeBinaryIndex] = pyvpindex_type
        self._description: str = description
        self._original_data: Optional[Union[np.ndarray, Callable[..., np.ndarray]]] = data
        self._loaded_data: Optional[np.ndarray] = None
        self._dim = dim

    def save(self, filepath: str):
        with h5py.File(filepath, "w") as outfile:
            dataset = outfile.create_dataset("dataset", data=self.data())

            dataset.attrs["name"] = self._name
            dataset.attrs["description"] = self._description

    def data(self) -> np.ndarray:
        if self._loaded_data is not None:
            return self._loaded_data

        # if self._data is callable, lazy load it
        if callable(self._original_data):
            logger.info(f"lazy loading data for {self._name} ...")
            self._loaded_data = self._original_data()
        else:
            self._loaded_data = self._original_data

        return self._loaded_data

    def load_from_file(self, filepath: str):
        with h5py.File(filepath, "r") as infile:
            dataset = infile["dataset"]
            self._loaded_data = dataset[...]  # Load it into memory. Could also slice a subset.

            self._name = dataset.attrs["name"]
            self._description = dataset.attrs["description"]

        self._original_data = self._loaded_data

    def dimension(self):
        return self._dim

        return d.shape[1]

    def size(self):
        d = self.data()
        if d is None:
            return 0

        return d.shape[0]

    def name(self):
        return self._name

    def index_type(self):
        return self._pyvpindex_type

    def unload_data(self):
        logger.info(f"unloading data for {self._name} ...")
        self._loaded_data = None

    @staticmethod
    def generate_gaussian_euclidean_cluster_datasets(
        min_dim: int, max_dim: int, total_size=2500000, num_clusters=50
    ) -> List["BenchmarkDataset"]:
        datasets = []
        for dim in range(min_dim, max_dim + 1):
            datasets.append(
                BenchmarkDataset(
                    name=f"gaussian_euclidean_clusters_dim={dim}",
                    description=f"An euclidean gaussian clusters dataset of {dim} dimensions, type is float64",
                    data=partial(
                        generate_euclidean_gaussian_dataset,
                        num_clusters=num_clusters,
                        cluster_size=(total_size // num_clusters),
                        dim=dim,
                    ),
                    dim=dim,
                    pyvpindex_type=pyvptree.VPTreeL2Index,
                )
            )

        return datasets


def extract_zip_file(extract_path):
    try:
        # remove zipfile
        zfileTOremove = f"{extract_path}"
        if os.path.isfile(zfileTOremove):
            os.remove(zfileTOremove)
        else:
            print("Error: %s file not found" % zfileTOremove)
    except BadZipFile as e:
        print("Error:", e)


def generate_coco_img2vec_dataset() -> np.ndarray:
    temp = TemporaryDirectory()

    input_dataset = os.path.join(temp.name, "coco_val2017.zip")
    wget.download("http://images.cocodataset.org/zips/val2017.zip", out=input_dataset)

    images_dir = os.path.join(temp.name, "images")
    os.makedirs(images_dir)

    with ZipFile(input_dataset) as zfile:
        zfile.extractall(images_dir)

    images = glob.glob(f"{images_dir}/val2017/*.jpg")

    vecs = []
    for image_path in images:
        # Initialize Img2Vec with GPU
        img2vec = Img2Vec(cuda=False)

        # Read in an image (rgb format)
        img = Image.open(image_path)

        if len(img.getbands()) < 3:
            continue

        # Get a vector from img2vec, returned as a torch FloatTensor
        vec = img2vec.get_vec(img, tensor=True)
        vecs.append(vec.reshape((1, 512)))

    temp.cleanup()
    return np.concatenate(vecs)


def generate_euclidean_gaussian_dataset(
    num_clusters: int, cluster_size: int, dim: int, data_type: Any = np.float64
) -> np.ndarray:
    """
    Generate a set of gaussian clusters of specific size and dimention
    """
    # sample gaussan center points
    centers = np.random.uniform(-10000, 10000, size=(cluster_size, dim))
    scalers = np.random.uniform(5000, 10000, size=(cluster_size, dim))
    datas: list = []
    for cluster in range(num_clusters):
        center = centers[cluster, :]
        scale = scalers[cluster, :]
        data = np.random.normal(loc=center, scale=scale, size=(cluster_size, dim))
        datas.append(data)

    return np.concatenate(datas, axis=0).astype(data_type)
