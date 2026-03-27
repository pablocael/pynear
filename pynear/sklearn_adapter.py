"""
scikit-learn compatible adapters for PyNear indices.

These classes mirror the interface of ``sklearn.neighbors.NearestNeighbors``,
``KNeighborsClassifier``, and ``KNeighborsRegressor``, allowing drop-in
migration from scikit-learn to PyNear with minimal code changes.

Example migration
-----------------
Before::

    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

After::

    from pynear.sklearn_adapter import PyNearKNeighborsClassifier
    clf = PyNearKNeighborsClassifier(n_neighbors=5, metric='euclidean')

The ``fit`` / ``predict`` / ``score`` / ``kneighbors`` API is identical.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted

# ---------------------------------------------------------------------------
# Metric → index class mapping
# (built lazily to avoid circular import with pynear/__init__.py)
# ---------------------------------------------------------------------------

_FLOAT_METRIC_NAMES = {"euclidean", "l2", "manhattan", "l1", "chebyshev", "linf"}
_BINARY_METRIC_NAMES = {"hamming"}
_ALL_METRIC_NAMES = _FLOAT_METRIC_NAMES | _BINARY_METRIC_NAMES


def _resolve_index(metric: str):
    """Instantiate the PyNear index for *metric*, or raise ValueError."""
    import pynear  # local import breaks the circular dependency with __init__.py

    metric_map = {
        "euclidean": pynear.VPTreeL2Index,
        "l2":        pynear.VPTreeL2Index,
        "manhattan": pynear.VPTreeL1Index,
        "l1":        pynear.VPTreeL1Index,
        "chebyshev": pynear.VPTreeChebyshevIndex,
        "linf":      pynear.VPTreeChebyshevIndex,
        "hamming":   pynear.VPTreeBinaryIndex,
    }
    cls = metric_map.get(metric.lower())
    if cls is None:
        raise ValueError(
            f"Unsupported metric '{metric}'. "
            f"Supported values: {sorted(metric_map.keys())}"
        )
    return cls()


def _input_dtype(metric: str) -> str:
    return "uint8" if metric.lower() in _BINARY_METRIC_NAMES else "float32"


def _to_arrays(indices_ll, distances_ll):
    """
    Convert PyNear list-of-lists to (n_queries, k) numpy arrays sorted
    nearest-first, matching sklearn's convention.

    PyNear returns results farthest-first; we sort by ascending distance here.
    Note: sklearn also returns ``(distances, indices)``, opposite of PyNear's
    ``(indices, distances)``.
    """
    ind = np.array(indices_ll, dtype=np.intp)
    dst = np.array(distances_ll, dtype=np.float64)
    order = np.argsort(dst, axis=1)
    return np.take_along_axis(dst, order, axis=1), np.take_along_axis(ind, order, axis=1)


def _compute_weights(distances: np.ndarray, weights: str) -> np.ndarray:
    """
    Return a weight array of shape ``(n_queries, k)``.

    For ``weights='distance'``, each weight is the inverse of the
    corresponding distance.  When a neighbour is at distance zero (exact
    match), that neighbour receives weight 1 and all others in the same
    query receive weight 0, consistent with scikit-learn's behaviour.
    """
    if weights == "uniform":
        return np.ones_like(distances)

    if weights == "distance":
        with np.errstate(divide="ignore"):
            w = 1.0 / distances
        zero_rows = (distances == 0.0).any(axis=1)
        if zero_rows.any():
            w[zero_rows] = np.where(distances[zero_rows] == 0.0, 1.0, 0.0)
        return w

    raise ValueError(f"weights must be 'uniform' or 'distance', got '{weights}'")


# ---------------------------------------------------------------------------
# PyNearNearestNeighbors
# ---------------------------------------------------------------------------

class PyNearNearestNeighbors(BaseEstimator):
    """
    Unsupervised nearest-neighbour lookup backed by PyNear VP-Trees.

    Drop-in replacement for ``sklearn.neighbors.NearestNeighbors``.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbours to return by default.
    metric : str, default='euclidean'
        Distance metric.  One of ``'euclidean'`` (alias ``'l2'``),
        ``'manhattan'`` (alias ``'l1'``), ``'chebyshev'`` (alias
        ``'linf'``), or ``'hamming'``.
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y=None):
        """
        Build the VP-Tree index from training data *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        X = check_array(X, dtype=_input_dtype(self.metric))
        self._index = _resolve_index(self.metric)
        self._index.set(X)
        self._fit_X = X
        self.n_samples_fit_ = X.shape[0]
        return self  # y is intentionally unused (unsupervised)

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """
        Find the *n_neighbors* nearest neighbours of each query point.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), optional
            Query points.  When *None*, returns neighbours of each training
            point (excluding the point itself).
        n_neighbors : int, optional
            Overrides ``self.n_neighbors`` for this call.
        return_distance : bool, default=True
            When *True* returns ``(distances, indices)``; otherwise only
            ``indices``.

        Returns
        -------
        distances : ndarray of shape (n_queries, n_neighbors)
            Only present when *return_distance* is *True*.
        indices : ndarray of shape (n_queries, n_neighbors)
        """
        check_is_fitted(self)
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        exclude_self = X is None
        if exclude_self:
            X = self._fit_X
        else:
            X = check_array(X, dtype=_input_dtype(self.metric))

        # Fetch one extra neighbour when excluding self so we can strip it
        k_fetch = n_neighbors + 1 if exclude_self else n_neighbors
        indices_ll, distances_ll = self._index.searchKNN(X, k_fetch)

        # Sort nearest-first before stripping so we always remove the true self (distance=0)
        dst, ind = _to_arrays(indices_ll, distances_ll)

        if exclude_self:
            dst = dst[:, 1:]
            ind = ind[:, 1:]

        return (dst, ind) if return_distance else ind

    def kneighbors_graph(self, X=None, n_neighbors=None, mode="connectivity"):
        """
        Return a sparse graph of k-nearest-neighbour connections.

        Parameters
        ----------
        X : array-like, optional
            Query points.  Defaults to the training set.
        n_neighbors : int, optional
        mode : {'connectivity', 'distance'}, default='connectivity'
            Edge values: 1s for ``'connectivity'``, actual distances for
            ``'distance'``.

        Returns
        -------
        A : scipy.sparse.csr_matrix of shape (n_queries, n_samples_fit_)
        """
        from scipy.sparse import csr_matrix

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        dst, ind = self.kneighbors(X, n_neighbors=n_neighbors, return_distance=True)
        n_queries = ind.shape[0]
        row = np.repeat(np.arange(n_queries), n_neighbors)
        col = ind.ravel()
        data = np.ones(len(col)) if mode == "connectivity" else dst.ravel()
        return csr_matrix((data, (row, col)), shape=(n_queries, self.n_samples_fit_))


# ---------------------------------------------------------------------------
# PyNearKNeighborsClassifier
# ---------------------------------------------------------------------------

class PyNearKNeighborsClassifier(ClassifierMixin, BaseEstimator):
    """
    k-nearest-neighbours classifier backed by PyNear VP-Trees.

    Drop-in replacement for ``sklearn.neighbors.KNeighborsClassifier``.

    Parameters
    ----------
    n_neighbors : int, default=5
    metric : str, default='euclidean'
        Supported: ``'euclidean'``, ``'l2'``, ``'manhattan'``, ``'l1'``,
        ``'chebyshev'``, ``'linf'``, ``'hamming'``.
    weights : {'uniform', 'distance'}, default='uniform'
        ``'uniform'``: all neighbours vote equally.
        ``'distance'``: neighbours are weighted by the inverse of their
        distance to the query point.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

    def fit(self, X, y):
        """
        Build the VP-Tree index and store training labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X = check_array(X, dtype=_input_dtype(self.metric))
        self._y = np.asarray(y)
        self.classes_ = unique_labels(y)
        self._class_index = {c: i for i, c in enumerate(self.classes_)}
        self._index = _resolve_index(self.metric)
        self._index.set(X)
        return self

    def predict(self, X):
        """
        Predict class labels for *X*.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_queries,)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        """
        Return class probability estimates for *X*.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        proba : ndarray of shape (n_queries, n_classes)
            Rows sum to 1.  Column order follows ``self.classes_``.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=_input_dtype(self.metric))

        indices_ll, distances_ll = self._index.searchKNN(X, self.n_neighbors)
        dst, ind = _to_arrays(indices_ll, distances_ll)
        w = _compute_weights(dst, self.weights)

        n_queries = X.shape[0]
        proba = np.zeros((n_queries, len(self.classes_)))
        for q in range(n_queries):
            for j, nb_idx in enumerate(ind[q]):
                ci = self._class_index[self._y[nb_idx]]
                proba[q, ci] += w[q, j]

        row_sums = proba.sum(axis=1, keepdims=True)
        proba /= np.where(row_sums == 0, 1.0, row_sums)
        return proba


# ---------------------------------------------------------------------------
# PyNearKNeighborsRegressor
# ---------------------------------------------------------------------------

class PyNearKNeighborsRegressor(RegressorMixin, BaseEstimator):
    """
    k-nearest-neighbours regressor backed by PyNear VP-Trees.

    Drop-in replacement for ``sklearn.neighbors.KNeighborsRegressor``.

    Parameters
    ----------
    n_neighbors : int, default=5
    metric : str, default='euclidean'
        Supported: ``'euclidean'``, ``'l2'``, ``'manhattan'``, ``'l1'``,
        ``'chebyshev'``, ``'linf'``, ``'hamming'``.
    weights : {'uniform', 'distance'}, default='uniform'
        ``'uniform'``: predict the unweighted mean of neighbour targets.
        ``'distance'``: predict the inverse-distance-weighted mean.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

    def fit(self, X, y):
        """
        Build the VP-Tree index and store training targets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self
        """
        X = check_array(X, dtype=_input_dtype(self.metric))
        self._y = np.asarray(y, dtype=np.float64)
        self._index = _resolve_index(self.metric)
        self._index.set(X)
        self.n_samples_fit_ = X.shape[0]
        return self

    def predict(self, X):
        """
        Predict target values for *X*.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_queries,) or (n_queries, n_outputs)
        """
        check_is_fitted(self)
        X = check_array(X, dtype=_input_dtype(self.metric))

        indices_ll, distances_ll = self._index.searchKNN(X, self.n_neighbors)
        dst, ind = _to_arrays(indices_ll, distances_ll)
        w = _compute_weights(dst, self.weights)           # (n_queries, k)

        neighbour_y = self._y[ind]                        # (n_queries, k, ...)
        w_sum = w.sum(axis=1, keepdims=True)

        if neighbour_y.ndim == 2:
            # Single output: (n_queries, k)
            return (w * neighbour_y).sum(axis=1) / w_sum.squeeze(axis=1)
        else:
            # Multi-output: (n_queries, k, n_outputs)
            return (w[:, :, np.newaxis] * neighbour_y).sum(axis=1) / w_sum

    # score() is inherited from RegressorMixin (returns R²)
