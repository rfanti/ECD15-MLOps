import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
model_name = "RandomForestGridSearch"

# Definir os limites de F1-score para Staging e Production
staging_threshold = 0.56  # Apenas modelos acima deste F1-score vão para Staging

# Buscar todas as versões do modelo
versions = client.search_model_versions(f"name='{model_name}'")

best_model = None  # Para armazenar o modelo Champion
best_f1_score = 0  # Para rastrear o melhor F1

for version in versions:
    run_id = version.run_id
    metrics = client.get_run(run_id).data.metrics

    if "f1_score" in metrics:
        f1 = metrics["f1_score"]

        # Adicionar modelos qualificados para Staging
        if f1 > staging_threshold:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Staging"
            )
            print(f"Modelo versão {version.version} com F1-score {f1} movido para Staging.")

        # Encontrar o melhor modelo para Produção
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model = version.version

# Atualizar o Champion (Produção)
if best_model:
    client.transition_model_version_stage(
        name=model_name,
        version=best_model,
        stage="Production"
    )
    print(f"Modelo versão {best_model} agora é o Champion com F1-score {best_f1_score}.")
else:
    print("Nenhum modelo atende ao critério para ser Champion.")