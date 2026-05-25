# Changelog

All notable changes to PyNear are documented in this file. Versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html): MAJOR.MINOR.PATCH.

## 2.3.1 — 2026-05-25

### Added

- **OpenMP parallelism in `MIHBinaryIndex.searchKNN`** — batched queries now run in parallel across cores.
- **`demo_faiss_comparison.py` + `results/faiss_comparison.md`** — reproducible, thread-matched comparison of pynear's binary indices against Faiss `IndexBinaryFlat` and `IndexBinaryMultiHash`, on both SIFT1M (128-bit) and a 512-bit near-duplicate workload.
- **Multi-Index Hashing section in `docs/approximate.md`** — the pigeonhole guarantee, `radius` / `m` tuning, and honest "when MIH wins vs when brute force wins" guidance.

### Changed

- **Corrected the binary-descriptor performance claims.** The previous "257× faster than Faiss" headline compared `MIHBinaryIndex` against a *single-threaded* brute-force scan on a synthetic d=512 workload. It is replaced throughout the README and docs with reproducible, thread-matched results: **~40× faster than Faiss's brute-force `IndexBinaryFlat`** on 512-bit near-duplicates (100% Recall@10), and **1.3–1.6× faster than Faiss's own `IndexBinaryMultiHash`** at matched recall on SIFT1M (128-bit). Added the honest caveat that an optimised brute-force POPCNT scan outperforms MIH on narrow descriptors at high recall.
- **`demo_binary.py`** now reports the standard `recall@k` (`|returned ∩ true| / k`) instead of the lenient "≥1 hit in top-k" metric, and labels its baseline explicitly as a naive numpy scan.

## 2.3.0 — 2026-05-20

### Added

- **`VPTreeCosineIndex`** — exact cosine-distance KNN, native C++ with the existing AVX2 SIMD path. Input vectors are L2-normalised at `set()` and queries at `searchKNN()` / `search1NN()`; returned distances are cosine distances in `[0, 2]` (0 = identical direction, 1 = orthogonal, 2 = antiparallel). Pickle-serialisable, zero-vector safe.
- **`IVFFlatCosineIndex`** — approximate cosine IVF index using spherical K-Means (centroids projected back to the unit sphere after K-Means) so cluster assignment is monotonic with cosine ranking. Inner-cluster scoring uses a single BLAS SGEMV per probed cluster.
- **scikit-learn cosine adapter** — `metric='cosine'` now resolves to `VPTreeCosineIndex` across `PyNearNearestNeighbors`, `PyNearKNeighborsClassifier`, and `PyNearKNeighborsRegressor`.

### Changed

- README rewritten for first-impression value density: opinionated tagline + a headline binary-speed claim above the fold, runnable example in the first viewport, use-case gallery, lifted benchmark chart. Moved the curse-of-dimensionality derivation to `docs/approximate.md`, the layman intro to `docs/intro.md`, and the full feature-comparison matrix to `docs/comparison.md`. (The binary-speed claim was corrected in a later release — see Unreleased.)

### Notes

The cosine implementation uses the L2-normalisation identity:
`‖u−v‖² = 2 − 2(u·v)` for unit vectors, so `argmin L2 = argmin cosine`,
and the underlying L2 distance is a true metric — VP-Tree pruning stays correct
under the cosine wrapper.

## 2.2.0 — Previous release

See [release notes on GitHub](https://github.com/pablocael/pynear/releases) for prior versions.
