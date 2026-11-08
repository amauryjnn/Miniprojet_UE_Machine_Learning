"""Les tests restent en local (hors rendu) et testent le code de ../rendu.

Lancer depuis la racine du mini-projet : `pytest tests -v`.
"""

import sys
from pathlib import Path

import pytest

RENDU = Path(__file__).resolve().parent.parent / "rendu"
sys.path.insert(0, str(RENDU))


@pytest.fixture(autouse=True)
def run_from_rendu(monkeypatch):
    # les chemins des données (data/...) sont relatifs à la racine du dépôt rendu
    monkeypatch.chdir(RENDU)
