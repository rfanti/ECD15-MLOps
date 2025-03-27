import mlflow

# Configura o backend store e o diretório de artefatos
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Inicia um experimento
with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.89)
    # Certifique-se de que o arquivo existe no caminho especificado
    mlflow.log_artifact("dataset/brasil_estado_cidade.csv")