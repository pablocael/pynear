import json
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

faiss_l2_data = None
with open("benchmark/results/faiss_l2_test_result.json", "r") as read_file:
    faiss_l2_data = json.load(read_file)

pyvptree_l2_data = None
with open("benchmark/results/pyvptree_l2_test_result.json", "r") as read_file:
    pyvptree_l2_data = json.load(read_file)

print(pyvptree_l2_data)
print('generating L2 index bench mark ... ')
# L2 result:

# Compare fixed K and fixed dimension, time vs numpoints (for various Ks)
dimension = 16 
for K in [1, 3, 8, 32]:
    x = [200000, 500000, 1000000]
    y = [faiss_l2_data[str(dimension)][str(K)][str(v)]['avg_search_time'] for v in x]
    plt.figure().set_dpi(120);
    plt.gcf().set_size_inches(8, 5)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.title(f'Search L2 Index, faiss vs pyvptree K={K}, dimension={dimension}')
    plt.plot(x, y, "-b", label="faiss", color='red')
    y = [pyvptree_l2_data[str(dimension)][str(K)][str(v)]['avg_search_time'] for v in x]
    plt.plot(x, y, "-b", label="pyvptree", color='blue')
    plt.xlabel('number of points')
    plt.ylabel('search time (seconds)')
    plt.legend(loc="upper left")
    plt.savefig(f'benchmark/results/l2_K{K}_{dimension}D_vs_time.png')

# Compare fixed number of points and fixed K for varing dimensions

K = 3
num_points = 1000000
x = [3, 4, 8, 16, 32]
y = [faiss_l2_data[str(v)][str(K)][str(num_points)]['avg_search_time'] for v in x]
plt.figure().set_dpi(120);
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.title(f'Search L2 Index, faiss vs pyvptree K={K}, number of points={num_points}')
plt.plot(x, y, "-b", label="faiss", color='red')
y = [pyvptree_l2_data[str(v)][str(K)][str(num_points)]['avg_search_time'] for v in x]
plt.plot(x, y, "-b", label="pyvptree", color='blue')
plt.xlabel('data dimension')
plt.ylabel('search time (seconds)')
plt.legend(loc="upper left")
plt.savefig(f'benchmark/results/l2_K{K}_{num_points}P_vs_time.png')

print('done generating benchmarks ... ')
