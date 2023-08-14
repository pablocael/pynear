from typing import Any
import numpy as np


def generate_gaussian_dataset(total_size: int, num_clusters: int, dim: int, data_type: Any = np.float64) -> np.ndarray:
    """
    Generate a set of gaussian clusters of specific size and dimention
    """
    # sample gaussan center points
    cluster_size = total_size // num_clusters

    centers = None
    scalers = None
    if data_type == np.uint8:
        centers = np.random.uniform(0, 255, size=(cluster_size, dim))
        scalers = np.random.uniform(1, 1, size=(cluster_size, dim))
    else:
        centers = np.random.uniform(-10000, 10000, size=(cluster_size, dim))
        scalers = np.random.uniform(1, 10, size=(cluster_size, dim))
    datas: list = []
    for cluster in range(num_clusters):
        center = centers[cluster, :]
        scale = scalers[cluster, :]
        data = np.random.normal(loc=center, scale=scale, size=(cluster_size, dim))
        datas.append(data)

    return np.concatenate(datas, axis=0).astype(data_type)
