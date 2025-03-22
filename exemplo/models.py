import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")

# Verifica o modelo registrado
models = client.search_registered_models()
for model in models:
    print(f"Modelo: {model.name}, Local: {model.latest_versions}")