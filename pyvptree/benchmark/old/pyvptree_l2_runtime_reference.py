import time
import numpy as np
import pyvptree

np.random.seed(seed=42)

dimension = 64
num_points = 500000
data = np.random.rand(int(num_points), dimension)

num_queries = 5000
queries = np.random.rand(num_queries, dimension)

K = 16

# now try same search with vp tree
vptree = pyvptree.VPTreeL2Index()
vptree.set(data)

start = time.time()
vptree_indices, vptree_distances = vptree.searchKNN(queries, K)
print(f'dimension = {dimension}, num_points = {num_points}, num_queries = {num_queries}, K = {K}')
print(f'took {(time.time() - start):0.2f}')
