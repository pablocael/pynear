# Choosing PyNear or Faiss

The [README's Why PyNear? section](../README.md#why-pynear) introduces the
main use cases. Choose by workload, required recall, and integration needs.

## Where PyNear is worth evaluating

- **Binary near-duplicate search:** `MIHBinaryIndex` targets small Hamming
  neighbourhoods in perceptual hashes and binary descriptors such as
  ORB/BRIEF. Wide codes and small radii are promising workloads. Increasing
  the radius increases probing cost, so general binary k-NN can favour a
  different index. See the [binary benchmarks](../results/faiss_comparison.md).
- **Exact metric search:** VP-trees prune by metric bounds while preserving
  exact nearest-neighbour results. Consider them for geometric data, feature
  matching, or ground-truth generation. Pruning efficiency depends on the
  data distribution as well as dimensionality; benchmark against a flat scan.
- **Binary threshold queries:** `BKTreeBinaryIndex.find_threshold(q, t)`
  returns all codes at Hamming distance at most `t`. This is useful when the
  number of matches is unknown and a top-k limit would discard valid matches.
- **scikit-learn integration:** PyNear provides neighbour lookup, classifier,
  and regressor adapters for supported metrics and options. They require
  scikit-learn and use float32 inputs for float metrics. Check compatibility
  with your pipeline rather than assuming every sklearn option is supported.
- **A shared CPU API:** exact trees, binary indexes, and HNSW are available
  through one package with NumPy as its core Python dependency. PyNear itself
  contains compiled C++ code; wheel availability and CPU requirements still
  matter. See [installation](../README.md#installation).

### Understand the guarantees and memory tradeoffs

MIH enumerates every candidate within its configured Hamming radius, but
`searchKNN` returns at most `k` results. It can return candidates beyond that
radius too. If the true kth neighbour is within the radius, the returned
top-k distances are exact; equal-distance ties may select different IDs.
Use the BK-tree threshold API for all matches within a radius. Neither index
guarantees that a descriptor captures every semantically similar item.

`HNSWL2IndexSQ8` stores vector components as int8 rather than float32, reducing
vector storage by 4×. Total index memory also includes the graph, metadata,
and other overhead. Quantisation can affect recall. Faiss also supports
scalar-quantised HNSW, so SQ8 is a useful PyNear option, not an exclusive
advantage; compare equivalent formats at matched recall.

## Where Faiss is a stronger starting point

- **GPU search:** PyNear's search indexes are CPU-only; Faiss provides GPU
  implementations. See the [Faiss overview](https://github.com/facebookresearch/faiss).
- **Compressed large-scale retrieval:** Faiss offers PQ/OPQ and a broader
  range of index combinations. PyNear's SQ8 does not replace these options.
  See the [Faiss index guide](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).
- **Dense embedding throughput:** Faiss wins the float HNSW and IVF cases in
  our [published comparison](../README.md#pynear-vs-faiss-in-numbers).
- **Exact binary scans:** Faiss's optimised flat Hamming scan is a strong
  baseline, particularly for narrow codes or high-recall queries where MIH
  would need extensive probing.

Faiss also supports exact search, binary hashing, binary range search, and
additional metrics including L1 and L∞. These capabilities alone are not
reasons to switch. See its [binary index documentation](https://github.com/facebookresearch/faiss/wiki/Binary-indexes)
and [metric documentation](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances).

## How to decide

Measure build time, query latency or throughput, total memory, and recall
using representative data, batch sizes, and thread counts. For approximate
indexes, compare at matched recall; for compressed indexes, compare equivalent
storage formats. For exact search, account for ties and numerical precision.
The [benchmark suite](../pynear/benchmark/) provides starting points, and the
[README methodology note](../README.md#pynear-vs-faiss-in-numbers) describes
an OpenMP runtime interaction observed when benchmarking both libraries in
one process.
