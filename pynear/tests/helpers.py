"""Shared test helpers for the pynear test suite.

Plain importable functions (not fixtures) so test modules can use them
directly: ``from pynear.tests.helpers import _nearest_first, ...``.
"""

import numpy as np


def _reference_topk_cosine(db: np.ndarray, queries: np.ndarray, k: int):
    """Brute-force cosine top-k. Returns (indices, distances) nearest-first."""
    db_n = db / np.linalg.norm(db, axis=1, keepdims=True)
    q_n = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    cos_dists = 1.0 - q_n @ db_n.T  # (Q, N), values in [0, 2]
    order = np.argsort(cos_dists, axis=1)[:, :k]
    sorted_dists = np.take_along_axis(cos_dists, order, axis=1)
    return order, sorted_dists


def _nearest_first(indices, distances):
    """Reverse pynear's farthest-first-within-top-k convention."""
    indices = np.array(indices, dtype=np.int64)[:, ::-1]
    distances = np.array(distances, dtype=np.float64)[:, ::-1]
    return indices, distances
