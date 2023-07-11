import os
import wget
import numpy as np
import glob
import pyvptree
from typing import Any
from zipfile import ZipFile, BadZipFile
from tempfile import TemporaryDirectory
from pyvptree.benchmark.dataset import BenchmarkDataset
from img2vec_pytorch import Img2Vec
from PIL import Image


def extract_zip_file(extract_path):
    try:
        # remove zipfile
        zfileTOremove=f"{extract_path}"
        if os.path.isfile(zfileTOremove):
            os.remove(zfileTOremove)
        else:
            print("Error: %s file not found" % zfileTOremove)    
    except BadZipFile as e:
        print("Error:", e)

def generate_coco_img2vec_dataset():
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
        vecs.append(vec.reshape((1,512)))

    temp.cleanup()

    data = np.concatenate(vecs)
    return BenchmarkDataset(
        name=f"coco_img2vec_512",
        description=f"An Image2Vec representation of Coco dataset using resnet, with vectors of 512 dimensions",
        data=data,
        pyvpindex_type=pyvptree.VPTreeL2Index,
    )

def generate_euclidean_gaussian_dataset(num_clusters: int, cluster_size: int, dim: int, type: Any = np.float64):
    """
    Generate a set of gaussian clusters of specific size and dimention
    """
    # sample gaussan center points
    centers = np.random.uniform(-10000,10000,size=(cluster_size, dim))
    scalers = np.random.uniform(5000,10000,size=(cluster_size, dim))
    datas: list = []
    for cluster in range(num_clusters):
        center = centers[cluster, :]
        scale = scalers[cluster, :]
        data = np.random.normal(loc=center, scale=scale, size=(cluster_size, dim))
        datas.append(data)
    
    return np.concatenate(datas, axis=0)

def main():

    # generate small dim gaussian clusters
    for dim in [16, 32, 256, 512]:
        print(f"generating euclidean datasets dim = {dim} ...")
        ds = BenchmarkDataset(
            name=f"gaussian_euclidean_clusters_dim={dim}",
            description=f"An euclidean gaussian clusters dataset of {dim} dimensions, type is float64",
            data=generate_euclidean_gaussian_dataset(num_clusters=50, cluster_size=10000, dim=dim),
            pyvpindex_type=pyvptree.VPTreeL2Index
        )
        ds.save(os.path.join("./datasets", ds.name()))
        print(f"done")

        print(f"generating binary datasets dim = {dim} ...")
        ds = BenchmarkDataset(
            name=f"gaussian_binary_clusters_dim={dim}",
            description=f"A binary gaussian clusters dataset of {dim} dimensions",
            data=generate_euclidean_gaussian_dataset(num_clusters=50, cluster_size=10000, dim=dim, type=np.uint8),
            pyvpindex_type=pyvptree.VPTreeL2Index
        )
        ds.save(os.path.join("./datasets", ds.name()))
        print(f"done")

    print("generating coco img2vec dataset dim = 512")
    coco_img2vec_dataset = generate_coco_img2vec_dataset()
    print(f"done")

if __name__ == "__main__":

    main()
