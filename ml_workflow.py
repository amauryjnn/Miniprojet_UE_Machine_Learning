import time
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.base import BaseEstimator, ClassifierMixin,TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, fbeta_score, make_scorer,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import (FixedThresholdClassifier, TunedThresholdClassifierCV, cross_val_predict, cross_validate)
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (FixedThresholdClassifier,
                                     StratifiedKFold, TunedThresholdClassifierCV, cross_val_predict,
                                     cross_validate, train_test_split)




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

def remove_duplicates(df):
    """Supprime les lignes identiques (rien n'est appris, on peut le faire avant le split)."""
    return df.drop_duplicates().reset_index(drop=True)

def simulate_missing(X, rate, seed=0):
    """Efface au hasard une proportion `rate` des valeurs, pour tester l'imputation.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(X.shape) < rate
    return X.mask(mask)

class StandardScalerManual(TransformerMixin, BaseEstimator):
    """Centrée-réduite : z = (x - moyenne) / écart-type, appris sur le train.

    BaseEstimator et TransformerMixin permettent de l'utiliser dans un Pipeline scikit-learn.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0 # variable constante : on la centre seulement
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

class PCAManual(TransformerMixin, BaseEstimator):
    """ACP : diagonalisation de la matrice de covariance.

    n_components : nombre d'axes gardés (entier), ou part de variance à garder (réel entre 0 et 1).

    (Ça fonctionne.)
    """

    def __init__(self, n_components=0.95):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = Xc.T @ Xc / (len(X) - 1)

        # eigh : pour une matrice symétrique, valeurs propres rangées par ordre croissant
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0, None)
        eigvecs = eigvecs[:, order]

        # un vecteur propre défini au signe près : on rend positive sa plus grande composante
        for j in range(eigvecs.shape[1]):
            biggest = np.argmax(np.abs(eigvecs[:, j]))
            if eigvecs[biggest, j] < 0:
                eigvecs[:, j] = -eigvecs[:, j]

        ratio = eigvals / eigvals.sum()
        if self.n_components < 1:
            # premier nombre d'axes dont la variance cumulée atteint le seuil
            k = np.argmax(np.cumsum(ratio) >= self.n_components) + 1
        else:
            k = self.n_components

        self.n_components_ = int(k)
        self.components_ = eigvecs[:, :k].T
        self.explained_variance_ = eigvals[:k]
        self.explained_variance_ratio_ = ratio[:k]
        self.all_explained_variance_ratio_ = ratio
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

def log1p_transformer():
    """Transformation x -> log(1 + x), qui écrase les grandes valeurs des variables positives."""
    return FunctionTransformer(np.log1p, feature_names_out="one-to-one")


def build_pipeline(model, impute="median", log_transform=False, scale=True,
                   selector=None, pca=None, sampler=None):
    """Assemble le prétraitement et le modèle dans un Pipeline.

    Ordre : imputation, log(1 + x), centrage-réduction, sélection de variables, ACP,
    rééchantillonnage, modèle. Avec model=None, on obtient le prétraitement seul.
    """
    steps = []
    if impute is not None:
        steps.append(("impute", SimpleImputer(strategy=impute)))
    if log_transform:
        steps.append(("log1p", log1p_transformer()))
    if scale:
        steps.append(("scale", StandardScalerManual()))
    if selector is not None:
        steps.append(("select", selector))
    if pca is not None:
        steps.append(("pca", PCAManual(n_components=pca)))
    if sampler is not None:
        steps.append(("sample", sampler))
    if model is not None:
        steps.append(("model", model))

    # le Pipeline d'imbalanced-learn rééchantillonne au fit seulement, jamais au predict
    if sampler is not None:
        return ImbPipeline(steps)
    return Pipeline(steps)

def split_data(X, y, test_size=0.2, seed=42):
    """Séparation train/test stratifiée : même proportion de positifs dans les deux parties."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def make_cv(n_splits=5, seed=42):
    """Validation croisée en k plis, stratifiée et mélangée."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)




class GaussianBayesClassifier(ClassifierMixin, BaseEstimator):
    """Classifieur bayésien gaussien : on suppose X | Y=k ~ N(mu_k, Sigma_k).

    On prédit la classe k qui maximise
        g_k(x) = -1/2 (x - mu_k)^T Sigma_k^-1 (x - mu_k) - 1/2 log|Sigma_k| + log(pi_k).

    covariance = "shared" : une covariance commune aux deux classes (LDA, frontière linéaire)
                 "full"   : une covariance par classe (QDA, frontière quadratique)
                 "diag"   : covariances diagonales (Bayes naïf gaussien)
    reg : petit nombre ajouté à la diagonale des covariances pour pouvoir les inverser.
    """

    def __init__(self, covariance="shared", reg=1e-6):
        self.covariance = covariance
        self.reg = reg

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, p = X.shape
        self.classes_ = np.unique(y)

        self.priors_ = []
        self.means_ = []
        covs = []
        counts = []
        for k in self.classes_:
            Xk = X[y == k]
            mu = Xk.mean(axis=0)
            self.priors_.append(len(Xk) / n)
            self.means_.append(mu)
            covs.append((Xk - mu).T @ (Xk - mu) / (len(Xk) - 1))
            counts.append(len(Xk))

        if self.covariance == "shared":
            # covariance intra-classe commune (moyenne des covariances pondérée par les effectifs)
            pooled = sum((counts[i] - 1) * covs[i] for i in range(len(covs))) / (n - len(covs))
            covs = [pooled for _ in covs]
        elif self.covariance == "diag":
            covs = [np.diag(np.diag(C)) for C in covs]

        self.inv_covs_ = []
        self.log_dets_ = []
        for C in covs:
            C = C + self.reg * np.eye(p)
            self.inv_covs_.append(np.linalg.inv(C))
            # log|C| = somme des log des valeurs propres de C
            self.log_dets_.append(np.sum(np.log(np.linalg.eigvalsh(C))))
        return self

    def _discriminant(self, X):
        """Tableau des g_k(x) : une ligne par individu, une colonne par classe."""
        X = np.asarray(X, dtype=float)
        G = np.zeros((len(X), len(self.classes_)))
        for k in range(len(self.classes_)):
            D = X - self.means_[k]
            mahalanobis = np.sum((D @ self.inv_covs_[k]) * D, axis=1)
            G[:, k] = -0.5 * mahalanobis - 0.5 * self.log_dets_[k] + np.log(self.priors_[k])
        return G

    def decision_function(self, X):
        """g_1(x) - g_0(x) : log du rapport de vraisemblance + log(pi_1 / pi_0)."""
        G = self._discriminant(X)
        return G[:, 1] - G[:, 0]

    def predict_proba(self, X):
        G = self._discriminant(X)
        G = G - G.max(axis=1, keepdims=True) # évite que exp() déborde
        P = np.exp(G)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        G = self._discriminant(X)
        return self.classes_[np.argmax(G, axis=1)]

def sigmoid(t):
    """Fonction sigmoïde 1 / (1 + exp(-t)) ; on borne t pour que exp() ne déborde pas."""
    t = np.clip(t, -500, 500)
    return 1 / (1 + np.exp(-t))

class LogisticRegressionGD(ClassifierMixin, BaseEstimator):
    """Régression logistique binaire, entraînée par descente de gradient accélérée (Nesterov).

    Modèle : P(Y=1 | x) = sigmoid(w.x + b).
    Coût minimisé (le même que scikit-learn, divisé par n) :
        J(w, b) = moyenne des s_i * entropie_croisée_i + ||w||^2 / (2 C n)
    où s_i est le poids de l'exemple i (1, ou n / (2 n_k) si class_weight="balanced").
    Pas de descente = 1 / L, avec L la constante de Lipschitz du gradient : la convergence est garantie.
    """

    def __init__(self, C=1.0, class_weight=None, max_iter=5000, tol=1e-6):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        t = (y == self.classes_[1]).astype(float) # cible codée en 0 / 1
        n, p = X.shape

        # poids des exemples
        s = np.ones(n)
        if self.class_weight == "balanced":
            n_pos = t.sum()
            s[t == 1] = n / (2 * n_pos)
            s[t == 0] = n / (2 * (n - n_pos))

        lam = 1 / (self.C * n) # coefficient de la pénalité L2
        Xb = np.hstack([X, np.ones((n, 1))]) # colonne de 1 pour le biais b
        L = 0.25 * np.linalg.eigvalsh(Xb.T @ (Xb * s[:, None]) / n).max() + lam
        step = 1 / L

        theta = np.zeros(p + 1) # theta = (w, b)
        z = theta.copy() # point où l'on calcule le gradient
        self.loss_history_ = []
        for it in range(self.max_iter):
            grad = self._gradient(z, Xb, t, s, lam)
            new_theta = z - step * grad
            z = new_theta + it / (it + 3) * (new_theta - theta) # élan de Nesterov
            theta = new_theta
            if it % 10 == 0:
                self.loss_history_.append(self._loss(theta, Xb, t, s, lam))
            if np.linalg.norm(grad) < self.tol:
                break

        self.n_iter_ = it + 1
        self.coef_ = theta[:p].reshape(1, -1)
        self.intercept_ = theta[p:]
        return self

    def _gradient(self, theta, Xb, t, s, lam):
        prob = sigmoid(Xb @ theta)
        grad = Xb.T @ (s * (prob - t)) / len(t)
        penalty = lam * theta
        penalty[-1] = 0 # le biais n'est pas pénalisé
        return grad + penalty

    def _loss(self, theta, Xb, t, s, lam):
        prob = np.clip(sigmoid(Xb @ theta), 1e-12, 1 - 1e-12)
        cross_entropy = -np.mean(s * (t * np.log(prob) + (1 - t) * np.log(1 - prob)))
        return cross_entropy + 0.5 * lam * np.sum(theta[:-1] ** 2)

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        p1 = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

# ------------------------------------------------------------------------
# Seuils de décision

def neyman_pearson_threshold(y_true, y_score, alpha=0.01):
    """Seuil de Neyman-Pearson : le plus petit seuil (donc le meilleur rappel) qui garde
    un taux de faux positifs <= alpha."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    allowed = np.where(fpr <= alpha)[0]
    return float(thresholds[allowed[-1]])



def bayes_risk_threshold(cost_fn, cost_fp):
    """Seuil qui minimise le risque bayésien, si les probabilités p = P(Y=1 | x) sont calibrées.

    On prédit 1 quand cost_fn * p > cost_fp * (1 - p), c'est-à-dire p > cost_fp / (cost_fp + cost_fn).
    Avec des coûts égaux on retrouve le seuil 0,5.
    """
    return cost_fp / (cost_fp + cost_fn)



def fbeta_threshold(y_true, y_score, beta=1.0):
    """Seuil qui maximise le F-beta (beta > 1 favorise le rappel, beta < 1 la précision)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # la courbe a un point de plus que les seuils : on retire le dernier
    precision = precision[:-1]
    recall = recall[:-1]
    f = (1 + beta**2) * precision * recall / np.maximum(beta**2 * precision + recall, 1e-12)
    return float(thresholds[np.argmax(f)])



def fbeta_scorer(beta=2.0):
    """F-beta sous forme de scorer (beta > 1 : le rappel compte plus que la précision)."""
    return make_scorer(fbeta_score, beta=beta)



def out_of_fold_scores(model, X, y, cv):
    """Scores « hors pli » sur le train : chaque individu est noté par un modèle qui ne l'a pas vu.

    Sert à choisir un seuil de décision sans utiliser le jeu de test.
    """
    if hasattr(model, "predict_proba"):
        return cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return cross_val_predict(model, X, y, cv=cv, method="decision_function", n_jobs=-1)


# ---------------------------------------------------------------------------------------
# Déséquilibre de classes


def imbalance_strategies(model, beta=2.0, cost_fn=5.0, cost_fp=1.0, seed=42):
    """Six variantes d'un même modèle pour traiter le déséquilibre des classes.

    Le rééchantillonnage (sous-échantillonnage, SMOTE) n'est appliqué qu'au train, dans le pipeline.
    """
    weighted_model = clone(model).set_params(class_weight="balanced")
    threshold = bayes_risk_threshold(cost_fn, cost_fp)
    f_beta = make_scorer(fbeta_score, beta=beta)

    strategies = {}
    strategies["aucun traitement (seuil 0,5)"] = build_pipeline(clone(model))
    strategies["class_weight='balanced'"] = build_pipeline(weighted_model)
    strategies["sous-échantillonnage"] = build_pipeline(clone(model), sampler=RandomUnderSampler(random_state=seed))
    strategies["SMOTE"] = build_pipeline(clone(model), sampler=SMOTE(random_state=seed))
    # seuil fixé à cost_fp / (cost_fp + cost_fn)
    strategies[f"seuil bayésien (c_FN/c_FP={cost_fn / cost_fp:g} → {threshold:.3f})"] = FixedThresholdClassifier(
        build_pipeline(clone(model)), threshold=threshold, response_method="predict_proba")
    # seuil qui maximise le F-beta, choisi par une validation croisée interne en 3 plis
    strategies[f"seuil max F{beta:g} (CV interne)"] = TunedThresholdClassifierCV(
        build_pipeline(clone(model)), scoring=f_beta, cv=3, response_method="predict_proba", random_state=seed)
    return strategies



def synthetic_vs_real_auc(X_train, y_train, sampler, n_max=20000, seed=42):
    """Un classifieur sait-il distinguer les vrais positifs des positifs synthétiques (SMOTE) ?

    AUC proche de 0,5 : les points synthétiques ressemblent aux vrais.
    AUC proche de 1 : ils sont faciles à repérer (par exemple 0,37 pour une variable binaire).
    """
    Z = StandardScalerManual().fit_transform(X_train)
    y = np.asarray(y_train)
    Z_resampled, _ = sampler.fit_resample(Z, y)
    synthetic = Z_resampled[len(Z):] # imbalanced-learn ajoute les points créés à la fin
    real = Z[y == 1]

    n = min(n_max, len(real), len(synthetic))
    rng = np.random.default_rng(seed)
    real = real[rng.choice(len(real), n, replace=False)]
    synthetic = synthetic[rng.choice(len(synthetic), n, replace=False)]

    A = np.vstack([real, synthetic])
    is_synthetic = np.r_[np.zeros(n), np.ones(n)]
    clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
    res = cross_validate(clf, A, is_synthetic, cv=make_cv(3, seed), scoring="roc_auc")
    return res["test_score"].mean()


# --------------------------------------------------------------------------------------
# Metrics et Évals

def confusion_counts(y_true, y_pred):
    """Matrice de confusion binaire  : (VN, FP, FN, VP)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return int(tn), int(fp), int(fn), int(tp)

def divide(a, b):
    """a / b, ou 0 si b = 0 (par exemple une précision quand le modèle ne prédit aucun positif)."""
    if b == 0:
        return 0.0
    return a / b

def compute_metrics(y_true, y_pred, y_score=None, beta=1.0):
    """Métriques de classification binaire calculées à partir de la matrice de confusion.

    Si les scores y_score sont donnés, on ajoute l'AUC-ROC et l'AUC-PR (qui ne dépendent pas du seuil).
    """
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    metrics = {
        "accuracy": divide(tp + tn, tn + fp + fn + tp),
        "precision": precision,
        "rappel": recall,
        "specificite": specificity,
        "taux_faux_positifs": 1 - specificity,
        "f1": divide(2 * precision * recall, precision + recall),
        f"f{beta:g}": divide((1 + beta**2) * precision * recall, beta**2 * precision + recall),
        "accuracy_equilibree": (recall + specificity) / 2,
        "VN": tn, "FP": fp, "FN": fn, "VP": tp,
    }
    if y_score is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_score)
        metrics["auc_pr"] = average_precision_score(y_true, y_score)
    return metrics

def get_scores(model, X):
    """Score continu du modèle : probabilité de la classe 1, ou decision_function à défaut (SVM)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate_on_test(model, X_test, y_test, threshold=None, beta=1.0):
    """Évaluation finale d'un modèle déjà entraîné, au seuil donné (par défaut : model.predict)."""
    scores = get_scores(model, X_test)
    if threshold is None:
        y_pred = model.predict(X_test)
    else:
        y_pred = (scores >= threshold).astype(int)
    return compute_metrics(y_test, y_pred, scores, beta=beta)


def results_table(results, cols):
    """Transforme un dictionnaire {nom: métriques} en tableau, en gardant les colonnes cols."""
    return pd.DataFrame(results).T[list(cols)]


def missing_values_experiment(make_model, X_train, y_train, X_test, y_test, rates=(0.0, 0.1, 0.3), seed=0):
    """Compare l'imputation par la moyenne et par la médiane quand on efface des valeurs.

    make_model(strategy) renvoie un pipeline dont l'imputation utilise strategy ("mean" ou "median").
    On mesure l'AUC-ROC sur le test.
    """
    rows = []
    for rate in rates:
        X_tr = simulate_missing(X_train, rate, seed)
        X_te = simulate_missing(X_test, rate, seed + 1)
        for strategy in ["mean", "median"]:
            model = make_model(strategy).fit(X_tr, y_train)
            auc = roc_auc_score(y_test, get_scores(model, X_te))
            rows.append({"taux_manquant": rate, "imputation": strategy, "auc_roc_test": auc})
    table = pd.DataFrame(rows)
    return table.pivot(index="taux_manquant", columns="imputation", values="auc_roc_test")

# --------------------------------------------------------------------------------------
# Les plots

def plot_class_balance(y, labels=("0", "1"), ax=None):
    """Diagramme en barres du nombre d'individus de chaque classe."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    counts = pd.Series(y).value_counts().sort_index()
    bars = ax.bar(labels, counts.values, color=["#4C72B0", "#DD8452"])
    for bar, count in zip(bars, counts.values):
        x = bar.get_x() + bar.get_width() / 2
        ax.text(x, count, f"{count}\n({count / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_ylabel("effectif")
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_title("Répartition des classes")
    return ax

def plot_pca_variance(pca, threshold=0.95, ax=None):
    """Variance expliquée par axe (barres) et cumulée (courbe), pour choisir le nombre d'axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
    ratio = pca.all_explained_variance_ratio_
    cumulated = np.cumsum(ratio)
    axes_numbers = np.arange(1, len(ratio) + 1)
    k = np.argmax(cumulated >= threshold) + 1
    ax.bar(axes_numbers, ratio, alpha=0.6, label="par axe")
    ax.plot(axes_numbers, cumulated, "o-", ms=3, color="C1", label="cumulée")
    ax.axhline(threshold, ls="--", color="grey")
    ax.axvline(k, ls=":", color="grey")
    ax.set_title(f"ACP : {k} axes pour {threshold:.0%} de la variance")
    ax.set_xlabel("composante")
    ax.set_ylabel("part de variance")
    ax.legend()
    return ax

def plot_pca_2d(Z, y, labels=("0", "1"), max_points=4000, seed=0, ax=None):
    """Individus projetés sur les deux premiers axes de l'ACP, colorés par classe."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    y = np.asarray(y)
    idx = np.random.default_rng(seed).choice(len(Z), min(max_points, len(Z)), replace=False)
    Z, y = Z[idx], y[idx]
    for c, label in zip([0, 1], labels):
        ax.scatter(Z[y == c, 0], Z[y == c, 1], s=4, alpha=0.4, label=label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=3)
    ax.set_title("Projection sur les 2 premiers axes")
    return ax

def plot_confusion(metrics, labels=("0", "1"), normalize=False, ax=None, title=""):
    """Matrice de confusion dessinée à partir des comptes VN, FP, FN, VP de compute_metrics."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
    cm = np.array([[metrics["VN"], metrics["FP"]], [metrics["FN"], metrics["VP"]]], dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    ax.imshow(shown, cmap="Blues")
    for i in range(2):
        for j in range(2):
            text = f"{shown[i, j]:.1%}\n({int(cm[i, j])})" if normalize else str(int(cm[i, j]))
            color = "white" if shown[i, j] > shown.max() / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color)
    ax.set_xticks([0, 1], ["prédit " + labels[0], "prédit " + labels[1]])
    ax.set_yticks([0, 1], ["réel " + labels[0], "réel " + labels[1]])
    ax.set_title(title)
    return ax