"""
Tests for pynear.sklearn_adapter.

Each PyNear adapter is verified against scikit-learn's brute-force
implementation to ensure identical (or equivalent) results.
"""

import numpy as np
import pytest
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)

from pynear.sklearn_adapter import (
    PyNearKNeighborsClassifier,
    PyNearKNeighborsRegressor,
    PyNearNearestNeighbors,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def make_X(n=300, d=8):
    return RNG.standard_normal((n, d)).astype(np.float32)


def make_clf_data(n=300, d=8, n_classes=3):
    X = make_X(n, d)
    y = RNG.integers(0, n_classes, n)
    return X, y


def make_reg_data(n=300, d=8):
    X = make_X(n, d)
    y = RNG.standard_normal(n)
    return X, y


# ---------------------------------------------------------------------------
# PyNearNearestNeighbors
# ---------------------------------------------------------------------------

class TestPyNearNearestNeighbors:

    def test_indices_match_sklearn_brute(self):
        X = make_X()
        q = make_X(n=20)
        k = 5

        sk = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean").fit(X)
        pn = PyNearNearestNeighbors(n_neighbors=k, metric="euclidean").fit(X)

        _, sk_ind = sk.kneighbors(q)
        _, pn_ind = pn.kneighbors(q)

        np.testing.assert_array_equal(pn_ind, sk_ind)

    def test_distances_match_sklearn_brute(self):
        X = make_X()
        q = make_X(n=20)
        k = 5

        sk = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean").fit(X)
        pn = PyNearNearestNeighbors(n_neighbors=k, metric="euclidean").fit(X)

        sk_dst, _ = sk.kneighbors(q)
        pn_dst, _ = pn.kneighbors(q)

        np.testing.assert_allclose(pn_dst, sk_dst, rtol=1e-4)

    def test_return_distance_false_returns_only_indices(self):
        X = make_X()
        pn = PyNearNearestNeighbors(n_neighbors=4).fit(X)
        result = pn.kneighbors(make_X(n=10), return_distance=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 4)

    def test_kneighbors_no_query_excludes_self(self):
        X = make_X(n=50)
        pn = PyNearNearestNeighbors(n_neighbors=4).fit(X)
        _, ind = pn.kneighbors()
        for i, row in enumerate(ind):
            assert i not in row, f"point {i} found itself as a neighbour"

    def test_n_neighbors_override(self):
        X = make_X()
        pn = PyNearNearestNeighbors(n_neighbors=5).fit(X)
        _, ind = pn.kneighbors(make_X(n=5), n_neighbors=3)
        assert ind.shape == (5, 3)

    def test_kneighbors_graph_connectivity_shape_and_nnz(self):
        X = make_X(n=60)
        k = 3
        pn = PyNearNearestNeighbors(n_neighbors=k).fit(X)
        G = pn.kneighbors_graph(make_X(n=10), n_neighbors=k, mode="connectivity")
        assert G.shape == (10, 60)
        assert G.nnz == 10 * k

    def test_kneighbors_graph_distance_values_positive(self):
        X = make_X(n=60)
        pn = PyNearNearestNeighbors(n_neighbors=3).fit(X)
        G = pn.kneighbors_graph(make_X(n=5), mode="distance")
        assert (G.data > 0).all()

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
    def test_supported_float_metrics_run(self, metric):
        X = make_X()
        pn = PyNearNearestNeighbors(n_neighbors=3, metric=metric).fit(X)
        dst, ind = pn.kneighbors(make_X(n=5))
        assert dst.shape == (5, 3)
        assert ind.shape == (5, 3)

    def test_unsupported_metric_raises(self):
        with pytest.raises(ValueError, match="Unsupported metric"):
            PyNearNearestNeighbors(metric="cosine").fit(make_X())

    def test_not_fitted_raises(self):
        from sklearn.exceptions import NotFittedError
        with pytest.raises(NotFittedError):
            PyNearNearestNeighbors().kneighbors(make_X(n=5))


# ---------------------------------------------------------------------------
# PyNearKNeighborsClassifier
# ---------------------------------------------------------------------------

class TestPyNearKNeighborsClassifier:

    def test_predict_matches_sklearn_brute(self):
        X, y = make_clf_data()
        q = make_X(n=30)

        sk = KNeighborsClassifier(n_neighbors=5, algorithm="brute", metric="euclidean").fit(X, y)
        pn = PyNearKNeighborsClassifier(n_neighbors=5, metric="euclidean").fit(X, y)

        np.testing.assert_array_equal(pn.predict(q), sk.predict(q))

    def test_predict_proba_sums_to_one(self):
        X, y = make_clf_data()
        pn = PyNearKNeighborsClassifier(n_neighbors=5).fit(X, y)
        proba = pn.predict_proba(make_X(n=20))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(20), rtol=1e-6)

    def test_predict_proba_shape(self):
        X, y = make_clf_data(n_classes=4)
        pn = PyNearKNeighborsClassifier(n_neighbors=5).fit(X, y)
        proba = pn.predict_proba(make_X(n=10))
        assert proba.shape == (10, 4)

    def test_classes_stored_correctly(self):
        X, y = make_clf_data(n_classes=4)
        pn = PyNearKNeighborsClassifier().fit(X, y)
        assert set(pn.classes_) == set(np.unique(y))

    def test_score_on_training_data(self):
        X, y = make_clf_data(n=500, n_classes=2)
        pn = PyNearKNeighborsClassifier(n_neighbors=5).fit(X, y)
        assert pn.score(X, y) > 0.5

    def test_distance_weights_proba_sums_to_one(self):
        X, y = make_clf_data()
        pn = PyNearKNeighborsClassifier(n_neighbors=5, weights="distance").fit(X, y)
        proba = pn.predict_proba(make_X(n=10))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(10), rtol=1e-6)

    def test_distance_weights_predict_matches_sklearn(self):
        X, y = make_clf_data()
        q = make_X(n=20)
        sk = KNeighborsClassifier(n_neighbors=5, weights="distance", algorithm="brute").fit(X, y)
        pn = PyNearKNeighborsClassifier(n_neighbors=5, weights="distance").fit(X, y)
        np.testing.assert_array_equal(pn.predict(q), sk.predict(q))

    def test_invalid_weights_raises(self):
        X, y = make_clf_data()
        pn = PyNearKNeighborsClassifier(weights="invalid").fit(X, y)
        with pytest.raises(ValueError, match="weights must be"):
            pn.predict(make_X(n=5))


# ---------------------------------------------------------------------------
# PyNearKNeighborsRegressor
# ---------------------------------------------------------------------------

class TestPyNearKNeighborsRegressor:

    def test_predict_matches_sklearn_brute_uniform(self):
        X, y = make_reg_data()
        q = make_X(n=20)

        sk = KNeighborsRegressor(n_neighbors=5, algorithm="brute", metric="euclidean").fit(X, y)
        pn = PyNearKNeighborsRegressor(n_neighbors=5, metric="euclidean").fit(X, y)

        np.testing.assert_allclose(pn.predict(q), sk.predict(q), rtol=1e-4)

    def test_predict_matches_sklearn_brute_distance_weights(self):
        X, y = make_reg_data()
        q = make_X(n=20)

        sk = KNeighborsRegressor(n_neighbors=5, weights="distance", algorithm="brute").fit(X, y)
        pn = PyNearKNeighborsRegressor(n_neighbors=5, weights="distance").fit(X, y)

        np.testing.assert_allclose(pn.predict(q), sk.predict(q), rtol=1e-4)

    def test_predict_shape(self):
        X, y = make_reg_data()
        pn = PyNearKNeighborsRegressor(n_neighbors=5).fit(X, y)
        preds = pn.predict(make_X(n=15))
        assert preds.shape == (15,)

    def test_score_r2_positive_on_training_data(self):
        X, y = make_reg_data(n=500)
        pn = PyNearKNeighborsRegressor(n_neighbors=5).fit(X, y)
        assert pn.score(X, y) > 0.0

    def test_multi_output_predict(self):
        X = make_X(n=200)
        y = RNG.standard_normal((200, 3))
        pn = PyNearKNeighborsRegressor(n_neighbors=5).fit(X, y)
        preds = pn.predict(make_X(n=10))
        assert preds.shape == (10, 3)

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
    def test_supported_metrics(self, metric):
        X, y = make_reg_data()
        pn = PyNearKNeighborsRegressor(n_neighbors=3, metric=metric).fit(X, y)
        preds = pn.predict(make_X(n=5))
        assert preds.shape == (5,)
