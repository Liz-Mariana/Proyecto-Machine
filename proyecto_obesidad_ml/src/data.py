"""Carga, limpieza y división de los datos."""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def load_raw() -> pd.DataFrame:
    """Lee el CSV original sin modificaciones."""
    return pd.read_csv(C.DATA_RAW)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas duplicadas exactas.

    Justificación: un registro idéntico podría quedar a la vez en entrenamiento y en
    prueba, lo que inflaría artificialmente la métrica de prueba (fuga de información).
    """
    return df.drop_duplicates().reset_index(drop=True)


def add_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el Índice de Masa Corporal (IMC = peso / estatura²).

    Solo se usa para el análisis de fuga de información; no es una variable de los
    modelos del escenario principal.
    """
    out = df.copy()
    out["BMI"] = out["Weight"] / out["Height"] ** 2
    return out


def feature_list(include_anthropometric: bool = False) -> list[str]:
    """Lista de predictores según el escenario."""
    numeric = C.NUMERIC_FEATURES + (C.ANTHROPOMETRIC if include_anthropometric else [])
    return numeric + list(C.ORDINAL_FEATURES) + C.NOMINAL_FEATURES


def split(df: pd.DataFrame, save: bool = True):
    """División estratificada 80/20. Devuelve los DataFrames completos de entrenamiento
    y prueba (todas las columnas) para que ambos escenarios usen exactamente las mismas
    filas."""
    train_df, test_df = train_test_split(
        df,
        test_size=C.TEST_SIZE,
        stratify=df[C.TARGET],
        random_state=C.RANDOM_STATE,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    if save:
        train_df.to_csv(C.DATA_PROCESSED / "train.csv", index=False)
        test_df.to_csv(C.DATA_PROCESSED / "test.csv", index=False)
    return train_df, test_df


def get_xy(df: pd.DataFrame, include_anthropometric: bool = False):
    """Separa predictores (X) y variable objetivo (y)."""
    return df[feature_list(include_anthropometric)], df[C.TARGET]


def load_split():
    """Atajo: carga, limpia y divide. Usado por todos los módulos."""
    df = clean(load_raw())
    return split(df)
