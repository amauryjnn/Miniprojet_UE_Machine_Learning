"""Tests du module ml_workflow.py (à la racine du dépôt).

Lancer depuis la racine du dépôt : `pytest test -v`.
"""

import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))


@pytest.fixture(autouse=True)
def run_from_root(monkeypatch):
    # les chemins des données (data/...) sont relatifs à la racine du dépôt
    monkeypatch.chdir(RACINE)
