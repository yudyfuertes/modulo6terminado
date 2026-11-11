FROM python:3.10-slim

WORKDIR /opt/mlflow

# Instalar dependencias
RUN pip install --no-cache-dir mlflow gunicorn scikit-learn pandas numpy

# Copiar el modelo descargado y los archivos del proyecto
COPY . /opt/mlflow

EXPOSE 8080

# Servir directamente la carpeta local exportada
ENTRYPOINT ["mlflow", "models", "serve", "-m", "/opt/mlflow/modelo_exportado", "-h", "0.0.0.0", "-p", "8080", "--no-conda"]