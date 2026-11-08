"""Tests unitaires du socle (import, prétraitement, séparation, métriques) — responsable : Amaury.

Lancer tous les tests avec `pytest -v` depuis la racine du dépôt.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from ml_workflow import (
    GaussianBayesClassifier,
    PCAManual,
    StandardScalerManual,
    build_pipeline,
    compute_metrics,
    confusion_counts,
    read_spambase_names,
    split_data,
)


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 5)) + y[:, None] * np.array([1.0, 0.5, 0.0, -0.5, 2.0])
    X[:, 2] *= 50                       # échelles très différentes
    return X, y

def test_read_spambase_names():
    names = read_spambase_names("data/spambase.names")
    assert len(names) == 57
    assert names[0] == "word_freq_make" and names[-1] == "capital_run_length_total"

def test_imputation_uses_train_median_only():
    """L'imputation doit utiliser la médiane du TRAIN (pas celle du test, ni la moyenne)."""
    X_train = pd.DataFrame({"a": [1.0, np.nan, 3.0, 100.0], "b": [0.0, 1.0, 1.0, np.nan]})
    X_test = pd.DataFrame({"a": [np.nan, 1000.0], "b": [np.nan, 0.0]})
    pipe = build_pipeline(GaussianBayesClassifier(), impute="median", scale=False)
    imputer = pipe.named_steps["impute"].fit(X_train)
    out = imputer.transform(X_test)
    assert out[0, 0] == 3.0             # médiane de [1, 3, 100] ; la moyenne vaudrait 34,7
    assert out[0, 1] == 1.0
    assert not np.isnan(imputer.transform(X_train)).any()

def test_standard_scaler_manual_matches_sklearn(toy_data):
    X, _ = toy_data
    Xtr, Xte = X[:400], X[400:]
    ours = StandardScalerManual().fit(Xtr)
    ref = StandardScaler().fit(Xtr)
    np.testing.assert_allclose(ours.transform(Xte), ref.transform(Xte))
    Z = ours.transform(Xtr)
    np.testing.assert_allclose(Z.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(Z.std(0), 1, atol=1e-12)

def test_standard_scaler_constant_column():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    Z = StandardScalerManual().fit_transform(X)
    assert np.all(np.isfinite(Z)) and np.all(Z[:, 1] == 0)

def test_pca_manual_matches_sklearn_up_to_sign(toy_data):
    X, _ = toy_data
    X = StandardScaler().fit_transform(X)
    ours = PCAManual(n_components=3).fit(X)
    ref = PCA(n_components=3).fit(X)
    np.testing.assert_allclose(ours.explained_variance_, ref.explained_variance_, rtol=1e-8)
    # les axes ne sont définis qu'au signe près
    np.testing.assert_allclose(np.abs(ours.components_), np.abs(ref.components_), atol=1e-8)
    np.testing.assert_allclose(np.abs(ours.transform(X)), np.abs(ref.transform(X)), atol=1e-8)

def test_pca_manual_variance_threshold(toy_data):
    X, _ = toy_data
    X = StandardScaler().fit_transform(X)
    pca = PCAManual(n_components=0.9).fit(X)
    cum = np.cumsum(pca.all_explained_variance_ratio_)
    assert cum[pca.n_components_ - 1] >= 0.9
    assert pca.n_components_ == 1 or cum[pca.n_components_ - 2] < 0.9

def test_split_is_stratified():
    y = np.r_[np.zeros(860), np.ones(140)]
    X = np.arange(1000).reshape(-1, 1)
    Xtr, Xte, ytr, yte = split_data(X, y, test_size=0.2, seed=0)
    assert len(Xte) == 200
    assert abs(ytr.mean() - 0.14) < 0.005 and abs(yte.mean() - 0.14) < 0.005
    assert len(np.intersect1d(Xtr.ravel(), Xte.ravel())) == 0

def test_confusion_counts_and_metrics_match_sklearn():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    assert (tn, fp, fn, tp) == tuple(confusion_matrix(y_true, y_pred).ravel())
    m = compute_metrics(y_true, y_pred)
    assert m["precision"] == pytest.approx(precision_score(y_true, y_pred))
    assert m["rappel"] == pytest.approx(recall_score(y_true, y_pred))
    assert m["f1"] == pytest.approx(f1_score(y_true, y_pred))
    assert m["specificite"] == pytest.approx(4 / 6)
