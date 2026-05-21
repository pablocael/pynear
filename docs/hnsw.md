# HNSW Indices — user guide

pynear ships five HNSW (Hierarchical Navigable Small World) index variants
for approximate nearest-neighbour search. This document is the practical
user guide; for the algorithm internals see
[`docs/hnsw_design.md`](./hnsw_design.md), and for a from-scratch tutorial
on what HNSW *is*, the README's "How HNSW works" section.

> **Why a separate guide for HNSW?** It's the only index family in pynear
> with a **mutation API** (`add()` / `remove()` / `rebuild()`). The VP-Tree,
> IVF-Flat, MIH and BK-Tree families are build-once: their internal
> structures (balanced trees, fixed centroids, bit-prefix hash tables)
> can't accept incremental updates without rebalancing. HNSW's graph
> structure is naturally amenable to in-place change.

---

## The five variants

| Class | Distance | Data type | When to use |
|---|---|---|---|
| `HNSWL2Index` | L2 (Euclidean) | `float32` | The default for generic float ANN |
| `HNSWCosineIndex` | Cosine | `float32` | Text / image embeddings (RAG, semantic search) |
| `HNSWL2IndexSQ8` | L2 (int8-quantised) | `float32` in, int8 stored | Memory-constrained — 4× less RAM, 2-3× faster queries, ~1-3% recall hit |
| `HNSWBinaryIndex` | Hamming | `uint8` (packed bits) | Binary descriptors — perceptual hashes, ORB, BRIEF, SimHash |
| `MIHSeededHNSWBinaryIndex` | Hamming | `uint8` (packed bits) | Binary + need exact recovery within a Hamming radius. Novel — combines MIH for seed selection with HNSW for traversal. |

All five share the same public API: `set()`, `add()`, `remove()`,
`rebuild()`, `searchKNN()`, `search1NN()`, `set_ef()`, pickle, plus
read-only properties `size`, `num_deleted`, `ef_search` (and `scale`
on the SQ8 variant).

---

## Quickstart

```python
import numpy as np
import pynear

# Build
db = np.random.randn(100_000, 384).astype(np.float32)
idx = pynear.HNSWCosineIndex(M=16, ef_construction=200, ef_search=64, n_threads=8)
idx.set(db)

# Query
queries = np.random.randn(10, 384).astype(np.float32)
indices, distances = idx.searchKNN(queries, k=10)
# indices[i] = top-k DB ids for query i (farthest-first within top-k by convention)
# distances[i] = cosine distances, matching ordering

# Convert to nearest-first if you prefer:
indices = np.array(indices)[:, ::-1]
distances = np.array(distances)[:, ::-1]
```

---

## Parameters that matter

### Build-time (set once)

| Param | Default | What it controls |
|---|---|---|
| `M` | 16 | Neighbours per node. Higher = better recall + more memory. Bump to 32 for parity with Faiss's defaults. |
| `ef_construction` | 200 | Beam width during graph build. Higher = better graph + slower build. 200 is the safe sweet spot. |
| `seed` | 42 | RNG seed for deterministic builds (with `n_threads=1`). |
| `n_threads` | 1 | Parallel build via OpenMP. Set to `os.cpu_count()` for fastest build. Build becomes non-deterministic; results are still correct just topologically different across runs. |

### Query-time (tunable per query)

| Param | How |
|---|---|
| `ef_search` | `idx.set_ef(128)` — beam width during search. Higher = better recall, slower queries. Default 50; try 64-256 for production. |
| `k` | Passed per query: `idx.searchKNN(q, k=10)` |
| `filter` | Optional bool/uint8 mask; see [Filtered search](#filtered-search) |

### Parameter intuition

| If you want… | Change |
|---|---|
| Higher recall | ↑ `ef_search` first, then ↑ `ef_construction`, then ↑ `M` |
| Faster queries | ↓ `ef_search` |
| Faster build | ↑ `n_threads` |
| Less memory | switch to `HNSWL2IndexSQ8` (4× reduction) |
| Reproducible builds | `n_threads=1`, fixed `seed` |

---

## The mutation API (`add`, `remove`, `rebuild`)

HNSW is the **only** pynear index family with incremental mutation. The
graph structure tolerates inserts and tombstones without rebuilding from
scratch.

### `add(vectors) → new_ids`

Append vectors to an existing index. Returns the new node IDs.

```python
extras = np.random.randn(1000, 384).astype(np.float32)
new_ids = idx.add(extras)   # [100_000, 100_001, ..., 100_999]
assert idx.size == 101_000
```

- **Batched**: pass a 2-D array. Per-vector cost is ~5× cheaper than
  many `add(single)` calls because of allocation overhead.
- **Parallel** when `n_threads > 1` (~6.6× speedup at 8 threads).
- **For SQ8**: the new vectors are quantised using the **existing**
  global scale (refitting the scale would invalidate prior data).
  If your new vectors have a very different magnitude range, recall on
  the new partition will degrade — consider rebuilding from scratch
  instead.
- **For Cosine**: new vectors are L2-normalised internally, same as `set()`.
- **For Binary**: pass a 2-D `uint8` array, same shape as `set()`.

### `remove(node_id)`

Mark a node as deleted (tombstone). Cost is O(1) — just flips a byte.

```python
idx.remove(42)
idx.remove(100)
idx.num_deleted   # 2
```

Search still traverses through deleted nodes (preserves graph
reachability) but excludes them from the result top-k. The inflation
factor (~`k * 8` raw candidates fetched) keeps result quality intact
up to very high deletion ratios (see [Performance](#performance)).

### `rebuild() → mapping`

Compact away tombstones. Returns a list mapping `old_id → new_id`
(or `-1` if the slot was deleted).

```python
mapping = idx.rebuild()
# mapping[42] == -1   (deleted in the previous step)
# mapping[43] is the new id of the surviving node that was 43
assert idx.num_deleted == 0
```

**When to call `rebuild()`:**

| % deleted | Should you rebuild? |
|---|---|
| < 25% | No. Speed unchanged, memory waste tolerable. |
| 25-50% | Optional. Run before pickling to shrink the persisted file. |
| 50-75% | Yes if memory is constrained. |
| > 75% | Yes. Significant memory waste, ID space getting messy. |

In a streaming production scenario, `rebuild()` is typically a nightly
cron or runs after a big purge.

### What about VPTree / IVF / MIH?

These don't have `add()` / `remove()` because their internal structures
require rebalancing or re-fitting on change:

| Family | Why no incremental update |
|---|---|
| VP-Tree | Balanced tree from recursive median-split — adding a point breaks balance |
| IVF-Flat | K-Means centroids are fixed at build time — adding a vector skews the partition |
| MIH | Hash table buckets are a fixed bit-prefix partitioning |
| BK-Tree | Like VP-Tree, recursive parent/child structure |

If you need streaming inserts on a non-HNSW workload, the pattern is:
buffer new vectors externally, batch-rebuild periodically.

---

## Filtered search

Top-k nearest neighbours subject to a per-vector predicate. Pass a
1-D bool or uint8 array of length `idx.size`:

```python
# Metadata stored externally — pandas, dict, numpy, whatever
categories = np.array([...])      # shape (N,)
prices     = np.array([...])      # shape (N,)

# Compute mask in Python (one byte per vector)
mask = (categories == "shoes") & (prices < 100)
print(mask.dtype, mask.shape)     # bool, (N,)

hits, dists = idx.searchKNN(queries, k=10, filter=mask)
# Only ids where mask[id] == True appear in the result
```

Works with **all five HNSW variants** and `search1NN` too.

### How it works

Same pattern as the tombstone filter: fetch `k * 8` raw candidates from
the beam, then a single pass excludes (a) tombstoned ids and (b)
filter-zero ids, returning up to k eligible results.

### When it works well / poorly

| Filter selectivity (fraction of `True` in mask) | Behaviour |
|---|---|
| 100% (no filter) | No-op fast path — zero overhead |
| 50% | Almost no measurable cost |
| 10% | Slightly fewer hits than k possible; bump `ef_search` if needed |
| 1% | May return fewer than k results — HNSW's beam can't easily find restrictive matches via graph traversal |
| 0.1% | Use a per-partition pre-filter or a separate index per category instead |

For very selective filters (say tenant_id = 42 where each tenant has
1% of data), the right answer is usually **per-shard indices** rather
than filter — see [`ShardedHNSWIndex`](#shardedhnswindex) (coming
soon) or just keep one HNSW per tenant.

A future v2.5 enhancement plumbs the filter into the beam itself
(Qdrant-style), eliminating the inflation factor. Out of scope for
v2.4.

### Combining with tombstones

`filter` and `remove()` compose naturally. A node is returned only if:
- it's not tombstoned, **and**
- `mask[id] == True` (when mask is provided)

```python
idx.remove(7)
mask = np.ones(idx.size, dtype=bool)   # mask allows everyone
hits, _ = idx.searchKNN(q, k=10, filter=mask)
# id 7 will not appear despite mask[7] == True
```

---

## Pickle persistence

Every HNSW variant is pickle-serialisable. Tombstones survive the
round-trip; SQ8 also preserves the quantisation scale.

```python
import pickle

# Save
with open("index.pkl", "wb") as f:
    pickle.dump(idx, f)

# Load
with open("index.pkl", "rb") as f:
    idx2 = pickle.load(f)

assert idx2.size == idx.size
assert idx2.num_deleted == idx.num_deleted
```

**Tip**: call `rebuild()` before pickling if you've accumulated many
tombstones — the persisted file shrinks by the deleted fraction.

---

## Performance

Numbers from one machine (Intel Core Ultra 9 285K, Arrow Lake, AVX2 only).
See [`pynear/benchmark/hnsw_benchmark.py`](../pynear/benchmark/hnsw_benchmark.py)
to reproduce.

### Build

| N | d | n_threads | Time |
|---|---|---|---|
| 20k | 128 | 1 | ~3.0 s |
| 20k | 128 | 8 | ~0.5 s |
| 20k | 128 | 24 | **0.18 s** (competitive with Faiss 0.20 s) |
| 50k | 128 | 4 | ~2.0 s |
| 100k | 128 | 8 | ~7 s |

### Query latency (N=20 000, ef_search=256, k=10, 8-thread build)

| dim | `HNSWL2Index` | `HNSWL2IndexSQ8` | Faiss `IndexHNSWFlat` (ref) |
|---|---|---|---|
| 128 | 88 µs | **70 µs** | 9 µs |
| 384 | 181 µs | 113 µs | 24 µs |
| 768 | 349 µs | **173 µs** | 94 µs |

SQ8 is faster than full-float because of 4× memory-bandwidth reduction
and 4× wider SIMD lanes.

### Mutation API

| Op | N=50k, d=128, n_threads=8 |
|---|---|
| `add(10k)` | 294 ms (~34 000 vec/s) — near-linear scaling with threads |
| `add(1)` single item | 17 ms — batch instead! |
| `remove(one)` | 89 ns — flips a byte |
| `rebuild()` after 90% deleted | 0.11 s (recovers 9× memory) |

### Search vs % deleted

Latency is **remarkably flat** as tombstones accumulate — `_deleted[id]`
is one byte read per visited node, negligible:

| % deleted | Query latency |
|---|---|
| 0% | 83 µs |
| 50% | 84 µs |
| 90% | 80 µs |

So `rebuild()` is mostly for memory reclamation, not speed.

---

## Tips and pitfalls

### Use the right variant

| Symptom | Likely cause | Fix |
|---|---|---|
| "Slow queries, lots of memory" | Using `HNSWL2Index` at N > 1M | Switch to `HNSWL2IndexSQ8` (4× less RAM, 2-3× faster) |
| "Recall too low for cosine workload" | Using `HNSWL2Index` on un-normalised vectors | Use `HNSWCosineIndex` (normalises internally) |
| "Need exact near-duplicate recovery on binary data" | Plain `HNSWBinaryIndex` misses some at low ef | Use `MIHSeededHNSWBinaryIndex` |
| "Need exact answers" | HNSW is approximate — recall < 100% at any ef | Use `VPTreeL2Index` etc. for exact search up to d ≈ 256 |

### Build-time gotchas

- **Single-threaded by default** for reproducibility. Set `n_threads`
  explicitly for production builds.
- **`M=16` is conservative** — bump to `M=32` if you want recall
  parity with Faiss's defaults at any ef_search.
- **`ef_construction=200` is the safe choice.** Going higher rarely
  helps; going lower (< 100) hurts recall noticeably.

### Query-time gotchas

- **Return order is farthest-first within the top-k.** Pynear's
  convention. Reverse with `[::-1]` if you want nearest-first.
- **`ef_search` defaults to 50** — fine for quick smoke tests, often
  too low for production. Try 64-256 and measure.
- **`searchKNN(queries, k)` is batched and multi-threaded** inside C++.
  Pass a full batch instead of looping in Python.

### Mutation gotchas

- **Always batch `add()`** — single-item adds pay an allocation cost
  per call. Group 100-10 000 vectors per call instead.
- **SQ8 add() reuses the build-time scale.** If your new vectors have
  much larger magnitudes than the training set, they'll be clamped to
  the int8 range and quantisation error rises. For drastic
  distribution shifts, prefer rebuild from scratch.
- **`rebuild()` invalidates external id mappings.** It returns a
  `old_id → new_id` array so callers can update their bookkeeping.

### Filter gotchas

- **Mask length must equal `idx.size`** — including tombstoned slots.
  Easiest to keep your metadata array aligned with the index and use
  the same indices throughout.
- **Mask dtype must be `bool` or `uint8`.** numpy stores both as one
  byte per element. Other dtypes raise.
- **Very selective filters return fewer than `k` results.** This is
  correct behaviour — pynear returns what's reachable. Crank
  `ef_search` if you need more, or pre-partition the index.

---

## ShardedHNSWIndex — many indices, one handle

For multi-tenant SaaS, per-category catalogues, or any "many small
partitions" workload, `pynear.ShardedHNSWIndex` wraps N independent
HNSW indices behind one API. Each shard is a normal pynear HNSW
underneath — pickled to its own file in a directory.

```python
import pynear, numpy as np

shards = pynear.ShardedHNSWIndex(
    index_cls=pynear.HNSWCosineIndex,
    M=16, ef_construction=200, ef_search=64,
)
# Build N shards from one vector batch + per-row labels
shards.set(vectors, shard_keys=tenant_ids)

# Single-tenant query — only one shard scanned (fastest)
hits, dists = shards.searchKNN(query, k=10, shard="tenant_42")

# Cross-tenant query — all shards in parallel, top-k merged
hits, dists = shards.searchKNN(query, k=10)

# Per-shard mutation
shards.add(more_vectors, shard="tenant_42")
shards.remove(node_id=7, shard="tenant_42")
shards.rebuild(shard="tenant_42")

# Persist to a directory of .pkl files + manifest.json
shards.save("./tenants/")
shards2 = pynear.ShardedHNSWIndex.load("./tenants/")
```

Use it for:
- **Tenant isolation** — search within one tenant at native speed
- **Faster incremental rebuilds** — only the affected shard rebuilds
- **Parallel build** — one OS thread per shard for free
- **Manageable persistence** — one .pkl per shard, copy/version/ship individually

Don't use it for:
- **Memory reduction** — shards still live in RAM (use SQ8 quantisation
  for that, or wait for v2.5+ mmap support)
- **Higher recall** — a single large HNSW gives slightly better recall
  than N×smaller ones (typically < 2 % difference)
- **Very selective filters** — for tenant-shaped filters, sharding is
  the right answer; for content filters (`category="shoes"`), use the
  `filter` kwarg on `searchKNN` directly

## Related

- [`docs/hnsw_design.md`](./hnsw_design.md) — algorithm internals, the
  α-heuristic, SIMD paths, recall tuning, profiling notes
- [`docs/comparison.md`](./comparison.md) — pynear vs Faiss / hnswlib /
  Annoy / scikit-learn
- [`pynear/benchmark/hnsw_benchmark.py`](../pynear/benchmark/hnsw_benchmark.py)
  — reproducible perf comparison vs Faiss
- [`pynear/benchmark/arm64_neon_benchmark.py`](../pynear/benchmark/arm64_neon_benchmark.py)
  — ARM64 / NEON micro-benchmark (Apple Silicon, Graviton)
