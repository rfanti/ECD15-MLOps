#!/bin/bash

# Incializa serviços do Container
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password='' --allow-root

mlflow ui --backend-store-uri sqlite:///mlflow.db