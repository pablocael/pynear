"""
IVFFlat index — IVF-style partitioned search using flat numpy scans.

Data is split into ``n_clusters`` Voronoi cells via K-Means.  Each cell stores
its raw vectors.  A query probes the ``n_probe`` nearest centroids and merges
their results into a global top-k ranking.

Inner-cluster distance computation for L2 uses a BLAS-backed identity:

* ``‖q−x‖² = ‖q‖² + ‖x‖² − 2 qᵀx``  — the ``qᵀX`` term is a
  BLAS SGEMV call (single query) or SGEMM call (batched queries), with
  per-cluster squared norms precomputed at build time.

Rule of thumb
-------------
* n_clusters ≈ sqrt(N)  for a dataset of N points
* n_probe    ≈ 10–20   for ~95 % recall; increase toward n_clusters for exact
"""

import heapq

import numpy as np


class _IVFFlatIndex:
    def __init__(self, n_clusters: int, n_probe: int):
        self._n_clusters = n_clusters
        self._n_probe = min(n_probe, n_clusters)
        self._centroids = None      # (C, D) float32
        self._cluster_data = []     # list of (n_c, D) float32 arrays
        self._cluster_norms = []    # list of (n_c,) float32 — precomputed for L2
        self._orig_indices = []     # list of (n_c,) int arrays — global row indices

    # ── Build ──────────────────────────────────────────────────────────────

    def set(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("data must be a 2-D array of shape (N, D)")

        n = len(data)
        n_clusters = min(self._n_clusters, n)

        labels, centroids = self._kmeans(data, n_clusters)

        self._centroids = centroids
        self._cluster_data = []
        self._cluster_norms = []
        self._orig_indices = []

        for c in range(len(centroids)):
            mask = labels == c
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            pts = data[idx]
            self._cluster_data.append(pts)
            self._cluster_norms.append((pts * pts).sum(axis=1))  # ||x||² per point
            self._orig_indices.append(idx)

    # ── Search ─────────────────────────────────────────────────────────────

    def searchKNN(self, queries: np.ndarray, k: int):
        """Return ``(indices, distances)`` for the k nearest neighbours of each query."""
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[np.newaxis]
        if self._centroids is None:
            raise RuntimeError("Index is empty — call set() first")

        n_queries = len(queries)
        n_clusters = len(self._cluster_data)
        n_probe = min(self._n_probe, n_clusters)

        # Nearest centroids for every query (Q, C) → argpartition → (Q, n_probe)
        centroid_dists = _l2sq_pairwise(queries, self._centroids)
        if n_probe == n_clusters:
            probe_clusters = np.tile(np.arange(n_clusters), (n_queries, 1))
        else:
            probe_clusters = np.argpartition(centroid_dists, n_probe - 1, axis=1)[:, :n_probe]

        all_indices, all_distances = [], []

        for qi in range(n_queries):
            heap: list = []
            q = queries[qi]

            for ci in probe_clusters[qi]:
                dists = self._flat_distances(q, int(ci))   # (n_c,)
                orig_idx = self._orig_indices[ci]
                local_k = min(k, len(orig_idx))

                # Partial sort: find the local_k smallest distances
                if local_k >= len(dists):
                    top = np.arange(len(dists))
                else:
                    top = np.argpartition(dists, local_k - 1)[:local_k]

                for li in top:
                    ld = float(dists[li])
                    gidx = int(orig_idx[li])
                    if len(heap) < k:
                        heapq.heappush(heap, (-ld, gidx))
                    elif ld < -heap[0][0]:
                        heapq.heapreplace(heap, (-ld, gidx))

            results = sorted(heap, key=lambda x: -x[0])
            all_indices.append([item[1] for item in results])
            all_distances.append([-item[0] for item in results])

        return all_indices, all_distances

    def search1NN(self, queries: np.ndarray):
        """Shortcut for k=1 nearest neighbour search."""
        return self.searchKNN(queries, 1)

    # ── Distance kernel (overridden per subclass) ───────────────────────────

    def _flat_distances(self, query: np.ndarray, ci: int) -> np.ndarray:
        """Return distance from *query* (D,) to every point in cluster *ci*."""
        raise NotImplementedError

    # ── Info ───────────────────────────────────────────────────────────────

    @property
    def n_clusters(self) -> int:
        """Number of clusters actually built (may be less than requested)."""
        return len(self._cluster_data)

    @property
    def n_probe(self) -> int:
        return self._n_probe

    # ── K-Means ────────────────────────────────────────────────────────────

    def _kmeans(self, data: np.ndarray, n_clusters: int):
        # Fast path: C++ Lloyd's K-Means with K-Means++ init and OpenMP parallelism
        try:
            from _pynear import kmeans_l2  # type: ignore[import]
            labels, centroids = kmeans_l2(data, n_clusters, 100, 42)
            return labels.astype(np.intp), centroids
        except (ImportError, AttributeError):
            pass

        # Fallback: sklearn MiniBatchKMeans
        try:
            from sklearn.cluster import MiniBatchKMeans

            km = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=1,  # type: ignore[arg-type]
                batch_size=min(len(data), max(4096, len(data) // 8)),
            )
            labels = km.fit_predict(data)
            centroids = km.cluster_centers_.astype(np.float32)  # type: ignore[union-attr]
            return labels, centroids
        except ImportError:
            pass

        return _numpy_kmeans(data, n_clusters)


# ── Public index classes ───────────────────────────────────────────────────────


class IVFFlatL2Index(_IVFFlatIndex):
    """
    IVF-style approximate index over **L2 (Euclidean)** distance.

    Partitions data into Voronoi cells via K-Means.  Each query probes the
    ``n_probe`` nearest centroids; inner-cluster distances are computed with a
    BLAS SGEMV using the identity ``‖q−x‖² = ‖q‖² + ‖x‖² − 2 qᵀX``, with
    per-cluster squared norms precomputed at build time.

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
        super().__init__(n_clusters, n_probe)

    def _flat_distances(self, query: np.ndarray, ci: int) -> np.ndarray:
        # ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q·x
        # cluster_data[ci] @ query  →  BLAS SGEMV: (n_c, D) × (D,) = (n_c,)
        q_norm = float(np.dot(query, query))
        cross = self._cluster_data[ci] @ query          # SGEMV
        return np.maximum(0.0, q_norm + self._cluster_norms[ci] - 2.0 * cross)


# ── Utilities ──────────────────────────────────────────────────────────────────


def _l2sq_pairwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared L2 distance matrix between rows of A (N,D) and B (M,D)."""
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
