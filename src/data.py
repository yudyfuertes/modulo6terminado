"""Carga y partición de datos. Si cambian los datos, se cambia solo este archivo."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RUTA_DATOS, SEMILLA

COLUMNAS_NUMERICAS = ["Age", "SibSp", "Parch", "Fare"]
COLUMNAS_CATEGORICAS = ["Pclass", "Sex", "Embarked"]
OBJETIVO = "Survived"

# La ruta se calcula desde este archivo, así funciona sin importar desde dónde se ejecute
RAIZ = Path(__file__).resolve().parent.parent


def cargar_dataframe():
    return pd.read_csv(RAIZ / RUTA_DATOS)


def cargar_datos(test_size=0.3):
    """Devuelve X_train, X_test, y_train, y_test con la misma partición siempre."""
    df = cargar_dataframe()
    X = df[COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS]
    y = df[OBJETIVO]
    return train_test_split(X, y, test_size=test_size, random_state=SEMILLA, stratify=y)
