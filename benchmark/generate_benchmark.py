import json
import matplotlib.pyplot as plt

faiss_l2_data = None
with open("benchmark/results/faiss_l2_test_result.json", "r") as read_file:
    faiss_l2_data = json.load(read_file)
    print(faiss_l2_data)

pyvptree_l2_data = None
with open("benchmark/results/pyvptree_l2_test_result.json", "r") as read_file:
    pyvptree_l2_data = json.load(read_file)
    print(pyvptree_l2_data)

# L2 result:

# Compare fixed K and fixed dimension, time vs numpoints (for various Ks)
dimension = 8
for K in [1, 2, 3, 8, 16, 32]:
    x = [100000, 200000, 500000, 1000000]
    y = [faiss_l2_data['8'][str(K)][str(v)] for v in x]
    plt.figure().set_dpi(100);
    plt.title(f'Search L2 Index, faiss vs pyvptree K={K}, dimension={dimension}')
    plt.plot(x, y, "-b", label="faiss")
    y = [pyvptree_l2_data['8'][str(K)][str(v)]['avg_search_time'] for v in x]
    plt.plot(x, y, "-b", label="pyvptree")
    plt.legend(loc="upper left")
    plt.savefig(f'benchmark/results/l2_K{K}_vs_time.png')

# Compare fixed K and fixed dimension, time vs numpoints (for various dimensions)
