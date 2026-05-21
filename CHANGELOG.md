# Changelog

All notable changes to PyNear are documented in this file. Versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html): MAJOR.MINOR.PATCH.

## 2.4.0 — 2026-05-21

### Added

- **`HNSWL2Index`** — paper-faithful Hierarchical Navigable Small World graph
  (Malkov & Yashunin, 2016) with the α-heuristic + `keepPrunedConnections`
  neighbour selection. Native C++ + pybind11, same template-on-distance
  pattern as the VPTree family. Optional parallel build via `n_threads=N`
  (per-node `std::shared_mutex` locking; default `n_threads=1` for
  deterministic builds).
- **`HNSWCosineIndex`** — HNSW with cosine distance via L2 normalisation
  (same identity used by `VPTreeCosineIndex`).
- **`HNSWBinaryIndex`** — HNSW over Hamming distance for packed `uint8`
  descriptors (perceptual hashes, ORB, etc.).
- **`HNSWL2IndexSQ8`** — HNSW with **int8 scalar quantisation** of vectors.
  4× memory reduction and 2-3× faster queries vs the float HNSW at the
  same parameters, with a small recall hit (~1-3 % on typical embeddings).
  Distances returned are L2 (scaled back).
- **`MIHSeededHNSWBinaryIndex`** — **novel variant** combining HNSW graph
  navigation with Multi-Index Hashing for the seed set. Guarantees exact
  recovery of near-duplicates within a configurable Hamming radius (where
  MIH is exact) while keeping HNSW's logarithmic graph traversal for
  larger queries. Not previously published, as far as we know.
- **AVX-512 distance kernels** (gated by `__AVX512F__`) for `HNSWL2Index`
  and `HNSWL2IndexSQ8` — ~2× distance throughput on supporting hardware
  (Zen 4, Sapphire Rapids, Xeon Gold). Falls back to AVX2 on older CPUs.
  PyPI wheels ship AVX2 only; users on AVX-512 hardware can recompile via
  `pip install --no-binary :all: pynear` to get the wider path.

### Performance

HNSW query latency on N=20 000, d=128, ef_search=256, 8 build threads:

| Stage | Query µs | Notes |
|---|---:|---|
| Original implementation | ~410 | First commit on the HNSW branch |
| After visited-version + heap optimisations | ~135 | |
| After 4-acc ILP + squared-L2 internal | ~115 | sqrt only applied to top-k |
| After 8-way batched dot + precomputed norms | ~110 | dot-product trick (Faiss-style) |
| After visited-table prefetching | ~110 | |
| `HNSWL2IndexSQ8` at the same params | ~70 | 4× memory reduction |
| Faiss `IndexHNSWFlat` (reference) | ~9 | |

At higher dimensions the gap to Faiss tightens — at D=768 pynear SQ8
runs at 173 µs vs Faiss 94 µs (within 2×).

Build time is now competitive with Faiss when using `n_threads = cpu_count()`:
~0.18 s vs Faiss 0.20 s on the same N=20 000, d=128 dataset.

### Compatibility

- **Windows MSVC**: bumped from `/arch:AVX` to `/arch:AVX2` so the FMA + AVX2
  int8 intrinsics used by the new SQ8 kernels are available. Minimum CPU
  is now AVX2 (Haswell, 2013+) on all platforms — matches existing Linux
  and macOS x86_64 wheels.
- **macOS x86_64 cross-compile**: added scalar fallbacks for every new
  `*_avx2` helper so the no-AVX cibuildwheel path compiles cleanly.

### Tests / CI

- 22 new HNSW tests + 3 new SQ8 tests — total suite 134 passing.
- New CI job (`build-check-avx512`) that forces `-march=skylake-avx512`
  and confirms the AVX-512 kernels compile without errors on every push.

### Notes

A separate `pynear` algorithm wrapper is in flight for the
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks) suite —
the PR will be submitted once 2.4.0 lands on PyPI.

## 2.3.0 — 2026-05-20

### Added

- **`VPTreeCosineIndex`** — exact cosine-distance KNN, native C++ with the existing AVX2 SIMD path. Input vectors are L2-normalised at `set()` and queries at `searchKNN()` / `search1NN()`; returned distances are cosine distances in `[0, 2]` (0 = identical direction, 1 = orthogonal, 2 = antiparallel). Pickle-serialisable, zero-vector safe.
- **`IVFFlatCosineIndex`** — approximate cosine IVF index using spherical K-Means (centroids projected back to the unit sphere after K-Means) so cluster assignment is monotonic with cosine ranking. Inner-cluster scoring uses a single BLAS SGEMV per probed cluster.
- **scikit-learn cosine adapter** — `metric='cosine'` now resolves to `VPTreeCosineIndex` across `PyNearNearestNeighbors`, `PyNearKNeighborsClassifier`, and `PyNearKNeighborsRegressor`.

### Changed

- README rewritten for first-impression value density: opinionated tagline + the 257× claim above the fold, runnable example in the first viewport, use-case gallery, lifted benchmark chart. Moved the curse-of-dimensionality derivation to `docs/approximate.md`, the layman intro to `docs/intro.md`, and the full feature-comparison matrix to `docs/comparison.md`.

### Notes

The cosine implementation uses the L2-normalisation identity:
`‖u−v‖² = 2 − 2(u·v)` for unit vectors, so `argmin L2 = argmin cosine`,
and the underlying L2 distance is a true metric — VP-Tree pruning stays correct
under the cosine wrapper.

## 2.2.0 — Previous release

See [release notes on GitHub](https://github.com/pablocael/pynear/releases) for prior versions.
