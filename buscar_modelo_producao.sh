echo "Aguardando MLflow server iniciar..."
until curl -s http://localhost:8080/health | grep -q 'OK'; do
  sleep 2
done
echo "MLflow server está pronto."

#Carregar API com modelo de produção
MODEL_SERVED=0

# Lista de nomes de modelos
MODEL_NAMES=(
  "gradient_boosting_model"
  "linear_regression_model"
  "xgboost_model"
  "random_forest_model"
  "decision_tree_model"
)

#Carrega dinamicamente na API do mlflow o modelo que estiver em produção, se encontrar algum.

for MODEL_NAME in "${MODEL_NAMES[@]}"
do
  echo "[INFO] Verificando modelo: ${MODEL_NAME}"

  RESPONSE=$(curl -s -X GET "http://localhost:8080/api/2.0/mlflow/model-versions/search?filter=name='${MODEL_NAME}'")

  echo "[DEBUG] Requisição feita para modelo '${MODEL_NAME}'"
  echo "[DEBUG] Resposta recebida:"
  echo "$RESPONSE"

  # Filtra o modelo em produção no script, analisando a resposta
  MODEL_VERSION=$(echo "$RESPONSE" | jq -r '.model_versions[] | select(.current_stage=="Production") | .version' | head -n 1)

  if [ "$MODEL_VERSION" != "null" ] && [ -n "$MODEL_VERSION" ]; then
    echo "[INFO] Servindo modelo: ${MODEL_NAME} versão ${MODEL_VERSION} (Production)"
    mlflow models serve -m "models:/${MODEL_NAME}/${MODEL_VERSION}" --no-conda --host 0.0.0.0 --port 5000 
    MODEL_SERVED=1
    break
  else
    echo "[INFO] Nenhuma versão em produção encontrada para ${MODEL_NAME}"
  fi
done

#Em uma primeira execucao, quando não tem nenhum modelo treinado em producao, não sobe a API do MLFLOW na porta 5000

if [ $MODEL_SERVED -eq 0 ]; then
  echo "Nenhum modelo em produção encontrado."
  #exit 1
fi