"""Tests unitaires du choix du seuil de décision (Neyman-Pearson, risque bayésien) — responsable : Lucas.

Lancer tous les tests avec `pytest -v` depuis la racine du dépôt.
"""

import numpy as np
import pytest

from ml_workflow import bayes_risk_threshold, compute_metrics, neyman_pearson_threshold


def test_neyman_pearson_threshold_controls_false_positive_rate():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(2000), np.ones(2000)].astype(int)
    s = np.r_[rng.normal(0, 1, 2000), rng.normal(2, 1, 2000)]
    thr = neyman_pearson_threshold(y, s, alpha=0.05)
    m = compute_metrics(y, (s >= thr).astype(int))
    assert m["taux_faux_positifs"] <= 0.05 + 1e-12
    assert m["taux_faux_positifs"] > 0.045      # on utilise bien tout le budget d'erreur

def test_bayes_risk_threshold():
    assert bayes_risk_threshold(1, 1) == pytest.approx(0.5)      # coûts égaux : règle MPE
    assert bayes_risk_threshold(cost_fn=5, cost_fp=1) == pytest.approx(1 / 6)
