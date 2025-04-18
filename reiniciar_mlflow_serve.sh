#!/bin/bash
 
 # Define the model URI and other parameters
 if [ -z "$1" ]; then
     MODEL_URI="models:/xgboost_model/7"
 else
     MODEL_URI="$1"
 fi
 
 HOST="0.0.0.0"
 PORT=5000
 
 # Run the mlflow models serve command
 setsid mlflow models serve -m "$MODEL_URI" --no-conda --host "$HOST" --port "$PORT" > mlflow.log 2>&1 < /dev/null &