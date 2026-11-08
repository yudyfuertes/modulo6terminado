# mlflow-titanic

Clasificación de supervivencia en el Titanic con seguimiento de experimentos en MLflow.

## Estructura
- `data/Titanic-Dataset.csv`: datos de ejemplo
- `notebooks/01_exploracion.ipynb`: exploración de los datos
- `src/`: configuración, carga de datos, entrenamiento, registro y predicción
- `MLproject`: punto de entrada para `mlflow run .`

## Instalar (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Entrenar y registrar
```powershell
$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
python src/train.py --n-estimators 50 --max-depth 3 --run-name probando
mlflow run . --env-manager local --experiment-name clasificacion-titanic -P n_estimators=200 -P max_depth=5
python src/register.py
python src/predict.py data/Titanic-Dataset.csv --alias challenger
```

## Ver resultados
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Abrir http://127.0.0.1:5000
