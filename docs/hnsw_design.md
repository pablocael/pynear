# HNSW design document for pynear

This is the design we are building against, written before the implementation
so the code has a clear target.

## Scope

Three indices, three release tiers in this branch:

| Tier | Class | Distance | Source of speed |
|---|---|---|---|
| Core (paper-faithful) | `HNSWL2Index` | Euclidean (`float32`) | Multi-layer NSW graph, α-heuristic, AVX2 SIMD, prefetching, cache-friendly layout |
| Cosine wrapper | `HNSWCosineIndex` | Cosine (`float32`) | L2-normalised input → reuses the L2 core (same identity we used for `VPTreeCosineIndex`) |
| Novel | `HNSWBinaryIndex` | Hamming (`uint8`) | HNSW graph navigation seeded by `MIHBinaryIndex` candidates — see the *Novel variant* section below |

## References

- **Original HNSW paper.** Y. Malkov, D. Yashunin, *"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"*, [arXiv:1603.09320](https://arxiv.org/abs/1603.09320) (2016, journal version 2020).
- **hnswlib** (reference implementation by the paper's author): [github.com/nmslib/hnswlib](https://github.com/nmslib/hnswlib). ~3,000 LOC, very readable. The de-facto baseline we are matching.
- **Faiss HNSW source** for comparison patterns: [Faiss/IndexHNSW.cpp](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexHNSW.cpp).
- **ParlayHNSW** (lock-free parallel HNSW build), Princeton CS, [arXiv:2403.01797](https://arxiv.org/abs/2403.01797) (2024). Not implemented here — single-threaded build in v1.
- **MIH paper** for the novel variant: Norouzi, Punjani, Fleet, *"Fast Search in Hamming Space with Multi-Index Hashing"*, CVPR 2012, [paper PDF](https://www.cs.toronto.edu/~norouzi/research/papers/multi_index_hashing.pdf). pynear already implements MIH (see `pynear/include/MIH.hpp`).

## Algorithm summary (HNSW, no novelty)

A multi-layer navigable small-world graph. Each layer is a graph over a subset of points; higher layers contain fewer points and serve as "long-range" entry points. Search descends layer by layer:

1. At the top layer, start at the entry point. Greedy walk to the nearest point.
2. Use that point as the entry into the next layer down; repeat the greedy walk.
3. At the bottom layer (layer 0, containing all points), do a beam search with width `ef_search` and return the top `k`.

Build proceeds one point at a time: pick a random level using a geometric distribution with parameter `mL = 1 / ln(M)`, then for each layer from the chosen level down to 0, find the nearest `ef_construction` candidates and link to up to `M` (or `2M` at layer 0) of them via the **α-heuristic** (see below).

### α-heuristic for neighbor selection

After collecting `ef_construction` nearest candidates for a new point at a given layer, instead of just keeping the `M` closest, we apply the heuristic from §4 of the HNSW paper. A candidate `c` is kept only if every already-selected neighbour is *farther* from `c` than `c` is from the inserted point. This biases the graph toward long-range edges that make the small-world property emerge — search becomes O(log N) rather than O(N) on uniform data.

Without this heuristic, recall collapses at high dimensions because the graph becomes too "clustered" locally and search gets stuck in basins.

### Parameters and defaults

| Parameter | Default | Effect |
|---|---|---|
| `M` | 16 | Max edges per node above layer 0. Layer 0 gets `2*M`. Higher = more memory, better recall, slower build. |
| `ef_construction` | 200 | Candidate set size during build. Higher = better-quality graph, slower build. |
| `ef_search` | 50 | Candidate set size during query. Tunable at runtime via `set_ef`. Higher = better recall, slower search. |
| `n_threads` | 1 | OpenMP threads for the build loop. `1` = deterministic; higher = much faster build, slightly non-deterministic graph. |
| `mL` | `1 / ln(M)` | Layer assignment exponential parameter. Fixed by paper. |

Memory budget per point at layer 0: `dim * 4 bytes (vector) + 2*M * 4 bytes (edges) + ~8 bytes overhead`. For `d=128, M=16`: ~656 bytes/point → ~625 MB for 1M points.

#### Query-latency notes

Profiling against Faiss with identical params (N=20k, d=128, M=16,
ef_construction=200, ef_search=256) shows:

| | pynear | Faiss |
|---|---|---|
| Distance computations per query (`ndis`) | ~4 810 | ~4 930 |
| Query time | ~115 µs | ~9 µs |
| Per distance | ~24 ns | ~2 ns* |

*Faiss's `ndis` may count distance-computer invocations rather than
individual SIMD distances, so the 2 ns figure is misleading. The
real gap is closer to ~3–4 × on per-distance throughput plus a
larger gap in heap-management overhead — Faiss uses a custom
`MinimaxHeap` whereas pynear uses `std::vector` + `std::push_heap`.

Closing the heap gap would require ~2 days of work to write a custom
data structure; closing the per-distance gap would require AVX-512
kernels (~1 week). Both are post-release follow-ups.

For profiling, the index exposes:

```python
idx.reset_dist_calls()
idx.searchKNN(queries, k=k)
print(idx.dist_calls())  # total distance computations since reset
```

The HNSW internal pipeline uses **squared L2** distance to skip ~4 800
`sqrt` operations per query (each ~10 cycle latency). The public
`searchKNN` / `search1NN` still return sqrt'd L2 distances — sqrt is
applied only to the final top-k.

#### Recall guidance

Empirical recall on random Gaussian data (N=20 000, d=128, k=10):

| M | ef_construction | ef_search | pynear recall@10 | Faiss recall@10 |
|---|---|---|---|---|
| 16 | 200 | 128 | 0.85 | 0.88 |
| 16 | 200 | 256 | 0.94 | 0.96 |
| 32 | 200 | 128 | 0.96 | 0.96 |
| 32 | 200 | 256 | **0.99** | **0.99** |

At `M=32` pynear matches Faiss exactly. Use `M=32` when high recall matters more than memory; `M=16` is fine for most applications.

## API surface

Mirrors the existing pynear pattern (`VPTreeL2Index`, `IVFFlatL2Index`):

```python
index = pynear.HNSWL2Index(M=16, ef_construction=200, ef_search=50)
index.set(vectors)                                    # build the index
indices, distances = index.searchKNN(queries, k=10)   # query
index.set_ef(100)                                     # retune at runtime
nn_idx, nn_dist = index.search1NN(queries)
```

Same `set()` / `searchKNN()` / `search1NN()` / pickle conventions as the rest of pynear. Returned distances follow the existing pynear convention (sorted within top-k farthest-first; the test helper in `test_vptree.py` reverses to nearest-first).

## Memory layout — what makes it fast

The performance gap between "a correct HNSW" and "a fast HNSW" lives in three places:

1. **Vectors stored in a flat backing array** (`std::vector<float>`), 32-byte aligned, row-major. No `std::vector<Vector>` indirection.
2. **Adjacency lists stored in fixed-size contiguous blocks**. At layer 0 each node owns a slot of `2*M` int32 IDs; at higher layers, `M` IDs. All of layer 0's edges live in one contiguous array indexed by `node_id * 2*M`. No `std::vector<int>` per node.
3. **Prefetching** during the inner candidate loop: `_mm_prefetch` the next vector before computing the current distance. Cuts L3 misses dramatically.

This matches hnswlib's layout. The cost is that resizing requires a rebuild (no incremental `add()` in v1; that's a follow-up feature).

## Pickle format

Distinct from VP-Tree's. Stored as a tuple of 6 byte blobs:

| Field | Type | Notes |
|---|---|---|
| `flat_vectors` | `float32[N * D]` | The raw vector data, contiguous |
| `dim` | `uint64` | D |
| `level_offsets` | `int32[N + 1]` | Where each node's adjacency block starts (per layer) |
| `levels` | `int32[N]` | The randomly-assigned level for each node |
| `adjacency` | `int32[total_edges]` | All edges concatenated, indexed by `level_offsets` |
| `entry_point` | `int32` | Top-layer entry point ID |
| `params` | tuple | `(M, ef_construction, ef_search, max_level)` |

## Threading

- **Search** is single-threaded (one query at a time, no internal parallelism).
- **Build** is opt-in parallel via the `n_threads` constructor parameter.
  Default is `n_threads=1` (sequential, fully deterministic given the seed).
  Pass `n_threads=os.cpu_count()` for max-speed builds.

Parallel build is implemented with OpenMP `parallel for` over the insertion
loop. Per-node `std::shared_mutex` instances protect adjacency modifications;
reads inside `search_layer` take shared locks (build-time only) so concurrent
inserts and graph traversals are safe. Per-thread visited-version buffers and
per-thread `std::mt19937` RNGs avoid cross-thread contention.

Observed speedup (N=20 000, d=128, M=16, ef_construction=200, 24-core box):

| n_threads | Build time | vs nt=1 |
|---|---|---|
| 1 | 2.41 s | 1.00× |
| 24 | 0.16 s | 15.1× |

Parallel build is **non-deterministic** — the resulting graph topology depends
on the OpenMP thread schedule. Recall at the same `ef_search` is empirically
within ~0.5 % of the sequential build. If you need bit-identical reproducibility
across runs, use `n_threads=1`.

The cosine wrapper inherits the same threading model.

## Novel variant: MIH-seeded HNSW for Hamming

This is the differentiator. Pure HNSW with Hamming distance already works and hnswlib supports it, but the literature on combining HNSW with MIH-style sub-table hashing for binary descriptors is thin. The intuition:

- **MIH** is *exact* and *very fast* for queries with small Hamming radius — it exploits the pigeonhole principle on bit substrings.
- **HNSW** is *robust* — it gives reasonable recall even when the answer is far from the query, but can suffer on binary distributions where many points sit at identical distances and the greedy walk has no gradient to follow.

Combining them: at query time, MIH first generates the candidate set for a small Hamming radius `r` (say, `r = log2(d)` — large enough to find true near-duplicates, small enough to remain MIH-fast). The HNSW graph search then starts the beam search seeded with those candidates instead of just the global entry point. The MIH candidates are *already* close to the query, so the graph walk has a head start.

Pseudocode:

```text
HNSWBinaryIndex.searchKNN(query, k):
    mih_seeds = MIH.searchKNN(query, k=ef_search, radius=r)  # exact within radius
    # Descend HNSW layers as usual to layer 0
    candidates = greedy_descent(query, top_layer_entry)
    # Layer 0 beam search seeded with both the descended candidates AND MIH seeds
    beam = candidates | mih_seeds
    return beam_search(query, beam, ef_search)
```

**Expected wins:**
1. **Near-duplicate queries** (small Hamming radius): MIH returns the answer directly with 100% recall; HNSW just ratifies and ranks.
2. **Larger-radius queries**: HNSW behaves as usual but starts from a better seed set, so recall is higher at the same `ef_search`.
3. **Robustness on adversarial binary distributions**: where HNSW alone gets stuck in distance plateaus, MIH guarantees that any candidate within radius `r` is found.

**Memory cost:** roughly 2× — both an HNSW graph *and* an MIH index. Reasonable for the niche where pynear already shines (image dedup, perceptual hashes).

This variant is the part most worth a blog post once we have numbers.

## Testing strategy

Each tier gets its own tests in `pynear/tests/test_hnsw.py`:

- **Correctness:** ≥0.95 recall@10 vs brute force at default params on random Gaussian data
- **1-NN exactness:** at `ef_search = N` the index should be effectively exact
- **`set_ef` runtime tunability:** recall increases monotonically with `ef_search`
- **Pickle round-trip:** rebuilt index returns identical results
- **Zero-vector safety** for L2 and cosine
- **Distance convention parity** with VPTree (farthest-first within top-k)
- **Binary correctness** for `HNSWBinaryIndex`: ≥0.95 recall@10
- **MIH-seeded variant:** at small Hamming radius, recall@10 = 1.000 (exact, by construction)

## sklearn adapter integration — explicitly deferred

The existing `pynear.sklearn_adapter` adapters mirror scikit-learn's exact-search API. HNSW is approximate (recall < 1 in general), so routing it through the `metric` parameter would conflate *what distance* with *how to compute it*. Users opt into HNSW directly:

```python
idx = pynear.HNSWL2Index(M=16, ef_construction=200, ef_search=50)
idx.set(X_train); idx.searchKNN(X_query, k=5)
```

A future `PyNearApproximateNearestNeighbors` adapter could wrap HNSW + IVF behind one API — out of scope for this branch.

## What's NOT in v1

These are deliberate scope cuts so the branch ships in reasonable time. Each can become a follow-up release:

- ✅ ~~Parallel build~~ — shipped (OpenMP `parallel for` with per-node `std::shared_mutex`; opt-in via `n_threads` parameter). Lock-free ParlayHNSW remains a v2 candidate.
- ❌ Incremental `add()` / `remove()`
- ❌ Scalar / product quantization
- ❌ Disk-resident index (DiskANN-style)
- ❌ Filtered search
- ❌ GPU backend

## Out-of-scope SOTA techniques (explicitly not pursued)

The honest picture of what we are *not* matching:

- **ScaNN's anisotropic vector quantization** ([arXiv:2008.02464](https://arxiv.org/abs/2008.02464)) — Google's product-quantization-loss formulation. Beats HNSW on text embeddings via better PQ. Months of work.
- **SOAR routing** ([arXiv:2404.00774](https://arxiv.org/abs/2404.00774)) — spilling-based routing on top of ScaNN. Combined with ScaNN, beats HNSW more. Research project.
- **DiskANN / Vamana** (NeurIPS 2019) — single-layer graph designed for SSD-resident billion-scale indexes. Different problem from ours.

These are documented here for posterity; we are not implementing them in pynear.
