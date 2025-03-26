#!/bin/bash

# Incializa servi�os do Container
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password='' 

#mlflow ui --backend-store-uri sqlite:///mlflow.db