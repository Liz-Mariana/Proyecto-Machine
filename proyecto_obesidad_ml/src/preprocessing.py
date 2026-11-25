"""Preprocesamiento con ColumnTransformer.

Todo el preprocesamiento vive dentro de un Pipeline de scikit-learn. Así, en cada
fold de la validación cruzada el escalador y los codificadores se ajustan solo con
los datos de entrenamiento de ese fold, evitando fugas de información.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config as C


def build_preprocessor(include_anthropometric: bool = False, scale: bool = True) -> ColumnTransformer:
    """Construye el transformador de columnas.

    - Numéricas: estandarización z = (x - media) / desviación (si ``scale``).
    - Ordinales (CAEC, CALC): codificación 0 < 1 < 2 < 3 respetando su orden.
    - Nominales: One-Hot (las binarias quedan en una sola columna 0/1).

    ``scale=False`` se usa para los modelos de árboles, que son invariantes a
    transformaciones monótonas de las variables y no necesitan estandarización.
    """
    numeric = C.NUMERIC_FEATURES + (C.ANTHROPOMETRIC if include_anthropometric else [])
    ordinal_cols = list(C.ORDINAL_FEATURES)
    ordinal_cats = [C.ORDINAL_FEATURES[c] for c in ordinal_cols]

    ordinal_steps = [("ordinal", OrdinalEncoder(categories=ordinal_cats))]
    if scale:
        ordinal_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler() if scale else "passthrough", numeric),
            ("ord", Pipeline(ordinal_steps), ordinal_cols),
            (
                "nom",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
                C.NOMINAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
