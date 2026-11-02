import time
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_spambase_names(names_path):
    """Lit les noms des 57 variables dans spambase.names (le fichier .data n'a pas d'en-tête)."""
    names = []
    with open(names_path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            # les lignes utiles ressemblent à « word_freq_make:   continuous. »
            if line.endswith("continuous.") and not line.startswith("|"):
                names.append(line.split(":")[0])
    return names



def load_spambase(data_dir="data"):
    """Charge Spambase avec ses noms de colonnes. La cible est is_spam (1 = spam)."""
    data_dir = Path(data_dir)
    names = read_spambase_names(data_dir / "spambase.names")
    df = pd.read_csv(data_dir / "spambase.data", header=None, names=names + ["is_spam"])
    return df



def load_diabetes(csv_path="data/diabetes_binary_health_indicators_BRFSS2015.csv"):
    """Charge Diabetes Health Indicators. La cible est Diabetes_binary.

    Le CSV stocke tout en float alors que les valeurs sont entières : on convertit en int.
    """
    df = pd.read_csv(csv_path)
    return df.astype(int)



def split_features_target(df, target_col):
    """Sépare les variables explicatives X et la cible y."""
    X = df.drop(columns=target_col)
    y = df[target_col]
    return X, y



def describe_dataset(df, target_col):
    """Premier contrôle du jeu : taille, valeurs manquantes, doublons, proportion de positifs."""
    X, y = split_features_target(df, target_col)
    # mêmes variables mais cible différente : impossible à classer correctement
    contradictions = df.drop_duplicates().duplicated(subset=list(X.columns)).sum()
    summary = {
        "n_lignes": len(df),
        "n_features": X.shape[1],
        "valeurs_manquantes": df.isna().sum().sum(),
        "lignes_dupliquees": df.duplicated().sum(),
        "doublons_X_avec_labels_contradictoires": contradictions,
        "proportion_positifs": y.mean(),
        "accuracy_classe_majoritaire": max(y.mean(), 1 - y.mean()),
    }
    return pd.Series(summary)



def variable_types(X, max_ordinal_levels=13):
    """Indique pour chaque variable si elle est binaire, ordinale/discrète ou continue."""
    types = {}
    for col in X.columns:
        n_values = X[col].nunique()
        if n_values <= 2:
            types[col] = "binaire"
        elif n_values <= max_ordinal_levels:
            types[col] = "ordinale/discrète"
        else:
            types[col] = "continue"
    return pd.Series(types)

