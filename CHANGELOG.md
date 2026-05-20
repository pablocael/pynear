# Changelog

All notable changes to PyNear are documented in this file. Versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html): MAJOR.MINOR.PATCH.

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
