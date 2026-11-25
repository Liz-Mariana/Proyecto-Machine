"""Definición de los tres modelos y de sus espacios de búsqueda de hiperparámetros."""
from scipy.stats import loguniform, randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .preprocessing import build_preprocessor

MODEL_NAMES = ["Regresión logística", "Árbol de decisión", "Random Forest"]
MODEL_FILES = {
    "Regresión logística": "regresion_logistica",
    "Árbol de decisión": "arbol_decision",
    "Random Forest": "random_forest",
}


def base_estimator(name: str):
    """Estimador con hiperparámetros por defecto (línea base antes de optimizar)."""
    if name == "Regresión logística":
        # Con solver lbfgs y y multiclase, scikit-learn ajusta la versión multinomial
        # (softmax). Por defecto: regularización L2 con C = 1.
        return LogisticRegression(max_iter=5000, random_state=C.RANDOM_STATE)
    if name == "Árbol de decisión":
        return DecisionTreeClassifier(random_state=C.RANDOM_STATE)
    if name == "Random Forest":
        return RandomForestClassifier(random_state=C.RANDOM_STATE, n_jobs=C.N_JOBS)
    raise ValueError(name)


def build_pipeline(name: str, include_anthropometric: bool = False) -> Pipeline:
    """Pipeline completo: preprocesamiento + clasificador."""
    scale = name == "Regresión logística"
    return Pipeline(
        [
            ("prep", build_preprocessor(include_anthropometric, scale=scale)),
            ("clf", base_estimator(name)),
        ]
    )


# ---------------------------------------------------------------------------
# Espacios de búsqueda (prefijo "clf__" porque el modelo está dentro del Pipeline)
# ---------------------------------------------------------------------------
# Regresión logística: el tipo de regularización se controla con l1_ratio
# (0 = L2/Ridge, 1 = L1/Lasso, 0.5 = Elastic Net). lbfgs solo admite L2; saga
# admite las tres.
_C_GRID = [0.01, 0.1, 1, 10, 100]

GRID_SPACES = {
    "Regresión logística": [
        {"clf__solver": ["lbfgs"], "clf__l1_ratio": [0.0], "clf__C": _C_GRID},
        {"clf__solver": ["saga"], "clf__l1_ratio": [0.0, 0.5, 1.0], "clf__C": _C_GRID},
    ],
    "Árbol de decisión": {
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [3, 5, 8, 12, None],
        "clf__min_samples_split": [2, 10, 20],
        "clf__min_samples_leaf": [1, 5, 10],
    },
    "Random Forest": {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [10, 20, None],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
        "clf__max_features": ["sqrt", "log2"],
    },
}

RANDOM_SPACES = {
    "Regresión logística": {
        "clf__solver": ["saga"],
        "clf__l1_ratio": [0.0, 0.5, 1.0],
        "clf__C": loguniform(1e-3, 1e3),
    },
    "Árbol de decisión": {
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [None] + list(range(2, 26)),
        "clf__min_samples_split": randint(2, 41),
        "clf__min_samples_leaf": randint(1, 21),
    },
    "Random Forest": {
        "clf__n_estimators": randint(50, 501),
        "clf__max_depth": [None] + list(range(5, 31)),
        "clf__min_samples_split": randint(2, 21),
        "clf__min_samples_leaf": randint(1, 11),
        "clf__max_features": ["sqrt", "log2", 0.5],
    },
}

N_ITER = {"Regresión logística": 20, "Árbol de decisión": 40, "Random Forest": 30}
