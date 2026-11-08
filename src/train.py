"""Entrena un modelo y registra parámetros, métricas, artefactos y modelo en MLflow."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # genera gráficos sin abrir ventanas (necesario en servidores y scripts)
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import EXPERIMENTO, SEMILLA
from data import COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS, cargar_datos

# Tipos de scikit-learn en los que confiamos al guardar el modelo (es nuestro)
TIPOS_CONFIABLES = ["sklearn.tree._tree.Tree", "numpy.dtype"]


def construir_modelo(n_estimators, max_depth):
    """Pipeline: limpia los datos y entrena. Así el modelo guardado acepta datos crudos."""
    preprocesado = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), COLUMNAS_NUMERICAS),
        ("cat", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore")),
        ]), COLUMNAS_CATEGORICAS),
    ])
    return Pipeline([
        ("preprocesado", preprocesado),
        ("modelo", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=SEMILLA)),
    ])


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento con seguimiento en MLflow")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--run-name", default=None, help="nombre legible del run")
    args = parser.parse_args()

    # `mlflow run` ya asigna el experimento; con `python src/train.py` lo asignamos aquí
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment(EXPERIMENTO)

    X_train, X_test, y_train, y_test = cargar_datos()

    with mlflow.start_run(run_name=args.run_name):
        # 1) Qué se probó
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "seed": SEMILLA,
        })
        mlflow.set_tags({"modelo": "RandomForest", "dataset": "titanic"})

        # 2) Entrenar
        modelo = construir_modelo(args.n_estimators, args.max_depth).fit(X_train, y_train)

        # 3) Evaluar con datos que el modelo no vio y guardar las métricas
        pred = modelo.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        mlflow.log_metrics({"accuracy": accuracy, "f1_macro": f1_score(y_test, pred, average="macro")})

        # 4) Un gráfico como artefacto
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
        mlflow.log_figure(fig, "matriz_confusion.png")
        plt.close(fig)

        # 5) El modelo, con su firma y un ejemplo de entrada
        info = mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            signature=infer_signature(X_train, modelo.predict(X_train)),
            input_example=X_train.head(3),
            skops_trusted_types=TIPOS_CONFIABLES,
        )
        mlflow.set_tag("model_uri", info.model_uri)  # para encontrarlo después
        print(f"accuracy={accuracy:.3f}  model_uri={info.model_uri}")


if __name__ == "__main__":
    main()
