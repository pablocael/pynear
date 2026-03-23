"""
VPForest indices — IVF-style partitioned search using VPTrees.

Data is split into ``n_clusters`` Voronoi cells via K-Means.  Each cell gets
its own VPTree.  A query probes the ``n_probe`` nearest centroids and merges
their results into a global top-k ranking.

This trades a small, configurable recall loss for a large speed gain on
high-dimensional data (e.g. 512-D / 1024-D image embeddings) where a single
VPTree would otherwise spend too much time exploring the whole space.

Rule of thumb
-------------
* n_clusters ≈ sqrt(N)  for a dataset of N points
* n_probe    ≈ 10–20   for ~95 % recall; increase toward n_clusters for exact results
"""

import heapq

import numpy as np

from _pynear import VPTreeChebyshevIndex
from _pynear import VPTreeL1Index
from _pynear import VPTreeL2Index


class _VPForestIndex:
    def __init__(self, n_clusters: int, n_probe: int, index_class):
        self._n_clusters = n_clusters
        self._n_probe = min(n_probe, n_clusters)
        self._index_class = index_class
        self._centroids = None   # (C, D) float32
        self._trees = []         # one VPTree per non-empty cluster
        self._orig_indices = []  # original row indices per cluster

    # ── Build ──────────────────────────────────────────────────────────────

    def set(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("data must be a 2-D array of shape (N, D)")

        n = len(data)
        n_clusters = min(self._n_clusters, n)

        labels, centroids = self._kmeans(data, n_clusters)

        self._centroids = centroids
        self._trees = []
        self._orig_indices = []

        for c in range(len(centroids)):
            mask = labels == c
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            tree = self._index_class()
            tree.set(data[idx])
            self._trees.append(tree)
            self._orig_indices.append(idx)

    # ── Search ─────────────────────────────────────────────────────────────

    def searchKNN(self, queries: np.ndarray, k: int):
        """Return (indices, distances) for the k nearest neighbours of each query."""
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[np.newaxis]
        if self._centroids is None:
            raise RuntimeError("Index is empty — call set() first")

        n_queries = len(queries)
        n_probe = min(self._n_probe, len(self._trees))

        # Find the n_probe nearest centroids for every query at once
        centroid_dists = _l2sq_pairwise(queries, self._centroids)  # (Q, C)
        if n_probe == len(self._trees):
            probe_clusters = np.tile(np.arange(len(self._trees)), (n_queries, 1))
        else:
            probe_clusters = np.argpartition(centroid_dists, n_probe - 1, axis=1)[:, :n_probe]

        all_indices, all_distances = [], []

        for qi in range(n_queries):
            # Max-heap of size k: entries are (-dist, orig_idx)
            heap: list = []

            for ci in probe_clusters[qi]:
                tree = self._trees[ci]
                orig_idx = self._orig_indices[ci]
                local_k = min(k, len(orig_idx))

                local_indices, local_dists = tree.searchKNN(queries[qi : qi + 1], local_k)

                for li, ld in zip(local_indices[0], local_dists[0]):
                    if len(heap) < k:
                        heapq.heappush(heap, (-ld, int(orig_idx[li])))
                    elif ld < -heap[0][0]:
                        heapq.heapreplace(heap, (-ld, int(orig_idx[li])))

            # Sort nearest-first
            results = sorted(heap, key=lambda x: -x[0])
            all_indices.append([item[1] for item in results])
            all_distances.append([-item[0] for item in results])

        return all_indices, all_distances

    def search1NN(self, queries: np.ndarray):
        """Shortcut for k=1 nearest neighbour search."""
        return self.searchKNN(queries, 1)

    # ── Info ───────────────────────────────────────────────────────────────

    @property
    def n_clusters(self) -> int:
        """Number of clusters actually built (may be less than requested)."""
        return len(self._trees)

    @property
    def n_probe(self) -> int:
        return self._n_probe

    # ── K-Means ────────────────────────────────────────────────────────────

    def _kmeans(self, data: np.ndarray, n_clusters: int):
        try:
            from sklearn.cluster import MiniBatchKMeans

            km = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=3,  # type: ignore[arg-type]
                batch_size=min(4096, len(data)),
            )
            labels = km.fit_predict(data)
            centroids = km.cluster_centers_.astype(np.float32)  # type: ignore[union-attr]
            return labels, centroids
        except ImportError:
            pass
        return _numpy_kmeans(data, n_clusters)


# ── Public index classes ───────────────────────────────────────────────────────


class VPForestL2Index(_VPForestIndex):
    """
    IVF-style index over **L2 (Euclidean)** distance.

    Clusters data with K-Means and builds one :class:`VPTreeL2Index` per
    cluster.  Ideal for float32 image / text embeddings (CLIP, ResNet, etc.)
    in 128-D to 1024-D.

    Parameters
    ----------
    n_clusters : int
        Number of Voronoi cells.  Suggested starting point: ``int(sqrt(N))``.
    n_probe : int
        Cells probed per query.  ``n_probe=1`` is fast but approximate;
        ``n_probe=n_clusters`` is exact.  Values in 10–30 usually give
        ≥ 95 % recall.
    """

    def __init__(self, n_clusters: int = 100, n_probe: int = 10):
        super().__init__(n_clusters, n_probe, VPTreeL2Index)


class VPForestL1Index(_VPForestIndex):
    """
    IVF-style index over **L1 (Manhattan)** distance.

    Parameters
    ----------
    n_clusters : int
        Number of Voronoi cells (K-Means partitioning uses L2 for speed;
        the fine search inside each cell uses L1).
    n_probe : int
        Cells probed per query.
    """

    def __init__(self, n_clusters: int = 100, n_probe: int = 10):
        super().__init__(n_clusters, n_probe, VPTreeL1Index)


class VPForestChebyshevIndex(_VPForestIndex):
    """
    IVF-style index over **Chebyshev (L∞)** distance.

    Parameters
    ----------
    n_clusters : int
        Number of Voronoi cells.
    n_probe : int
        Cells probed per query.
    """

    def __init__(self, n_clusters: int = 100, n_probe: int = 10):
        super().__init__(n_clusters, n_probe, VPTreeChebyshevIndex)


# ── Utilities ──────────────────────────────────────────────────────────────────


def _l2sq_pairwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared L2 distance matrix between rows of A (N,D) and B (M,D)."""
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b  →  O(NMD) via BLAS gemm
    a_norms = np.einsum("ij,ij->i", A, A)[:, np.newaxis]
    b_norms = np.einsum("ij,ij->i", B, B)[np.newaxis, :]
    return np.maximum(0.0, a_norms + b_norms - 2.0 * (A @ B.T))


def _numpy_kmeans(data: np.ndarray, n_clusters: int, max_iter: int = 100):
    """Fallback Lloyd's K-Means using only numpy (no sklearn required)."""
    rng = np.random.default_rng(42)
    centroids = data[rng.choice(len(data), n_clusters, replace=False)].copy()
    labels = np.zeros(len(data), dtype=np.intp)

    for _ in range(max_iter):
        dists = _l2sq_pairwise(data, centroids)
        labels = np.argmin(dists, axis=1)

        new_centroids = np.zeros_like(centroids)
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                new_centroids[c] = data[mask].mean(axis=0)
            else:
                new_centroids[c] = data[rng.integers(len(data))]

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return labels, centroids
