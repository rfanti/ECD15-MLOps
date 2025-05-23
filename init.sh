#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
exec &> >(tee -a /app/init.log)

chmod +x /app/*.sh

# Inicializa serviços do Container
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 8080 &
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password='' --allow-root &
./buscar_modelo_producao.sh

#mlflow models serve -m "models:/xgboost_model/7" --no-conda --host 0.0.0.0 --port 5000

wait