#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
exec &> >(tee -a /app/init.log)

chmod +x /app/*.sh

# Inicializa serviços do Container
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 8080 &
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password='' --allow-root &

#mlflow models serve -m "models:/xgboost_model/7" --no-conda --host 0.0.0.0 --port 5000

echo "Aguardando MLflow server iniciar..."
until curl -s http://localhost:8080/health | grep -q 'OK'; do
  sleep 2
done
echo "MLflow server está pronto."

#Carregar API com modelo de produção
bash reiniciar _mlflow_serve.sh

wait