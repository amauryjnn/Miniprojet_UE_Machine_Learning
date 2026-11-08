"""Tests unitaires des modèles du cours codés à la main (LDA/QDA, régression logistique) — responsable : Anthony.

Lancer tous les tests avec `pytest -v` depuis la racine du dépôt.
"""

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml_workflow import GaussianBayesClassifier, LogisticRegressionGD


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 5)) + y[:, None] * np.array([1.0, 0.5, 0.0, -0.5, 2.0])
    X[:, 2] *= 50                       # échelles très différentes
    return X, y

def test_lda_manual_matches_sklearn(toy_data):
    X, y = toy_data
    ours = GaussianBayesClassifier(covariance="shared", reg=0.0).fit(X, y)
    ref = LinearDiscriminantAnalysis(solver="lsqr").fit(X, y)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
    # écart minime : sklearn estime la covariance poolée en divisant par n, nous par n - K (sans biais)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=2e-3)

def test_qda_probabilities_are_valid(toy_data):
    X, y = toy_data
    P = GaussianBayesClassifier(covariance="full", reg=1e-3).fit(X, y).predict_proba(X)
    assert np.all((P >= 0) & (P <= 1))
    np.testing.assert_allclose(P.sum(1), 1)

def test_logistic_regression_gd_matches_sklearn(toy_data):
    X, y = toy_data
    X = StandardScaler().fit_transform(X)
    ours = LogisticRegressionGD(C=0.5, tol=1e-8, max_iter=20000).fit(X, y)
    ref = LogisticRegression(C=0.5, tol=1e-10, max_iter=10000).fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, atol=1e-4)
    np.testing.assert_allclose(ours.intercept_, ref.intercept_, atol=1e-4)
    # la fonction de coût doit décroître (descente de gradient)
    assert ours.loss_history_[-1] < ours.loss_history_[0]
