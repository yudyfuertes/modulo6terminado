"""Carga un modelo del registro por alias y predice sobre un archivo CSV."""
import argparse

import mlflow
import pandas as pd

from config import MODELO_REGISTRADO
from data import COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS

parser = argparse.ArgumentParser(description="Predicción con un modelo del registro")
parser.add_argument("csv", help="archivo con las columnas del Titanic (ej. data/Titanic-Dataset.csv)")
parser.add_argument("--alias", default="champion", help="alias del modelo (champion, challenger...)")
args = parser.parse_args()

modelo = mlflow.pyfunc.load_model(f"models:/{MODELO_REGISTRADO}@{args.alias}")
datos = pd.read_csv(args.csv)
entrada = datos[COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS]
datos["prediccion"] = modelo.predict(entrada)
print(datos["prediccion"].value_counts().sort_index().to_string())
datos.to_csv("predicciones.csv", index=False)
print("Guardado en predicciones.csv")
