import pandas as pd
import numpy as np
import requests
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, ClassificationPreset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from scipy import stats
import json

import os
#import json

def check_for_drift(drift_score, drift_by_columns):
    num_columns_drift = sum(1 for col, values in drift_by_columns.items() if values.get("drift_detected", False))
    print(drift_score)
    if drift_score > 0.5:
        print("Drift detectado no Dataset")
        os.system("jupyter nbconvert --to notebook --execute treinamento.ipynb --output treinamento_exec.ipynb")
        os.system("jupyter nbconvert --to notebook --execute predicao.ipynb --output predicao_exec.ipynb")  # <-- aqui
    else:
        if num_columns_drift > 2:
            print(f"Drift detectado em {num_columns_drift} colunas! Treinando novo modelo...")
            os.system("jupyter nbconvert --to notebook --execute treinamento.ipynb --output treinamento_exec.ipynb")
            os.system("jupyter nbconvert --to notebook --execute predicao.ipynb --output predicao_exec.ipynb")  # <-- e aqui também
        else:
            print("Modelo ainda está bom, sem necessidade de re-treinamento.")
            print("Nenhum drift detectado nas colunas e no dataset")

def remover_outliers_por_cidade(df):
    # Remove outliers da coluna 'price_brl' agrupando por cidade.

    def remover_outliers_grupo(grupo):
        # Remove outliers de um grupo usando o método Z-score.
        z_scores = np.abs(stats.zscore(grupo["price_brl"]))
        return grupo[(z_scores < 3)]

    df_filtrado = df.groupby("city").apply(remover_outliers_grupo).reset_index(drop=True)
    return df_filtrado

def simulate_drift(dados):
    new_data = dados.copy()
    # Mudando coluna de preço de imóveis para simular mudanças nos padrões dos dados
    # new_data["tenure"] = np.random.randint(0, 10, new_data.shape[0])  # Mudamos a duração do cliente aleatoriamente
    #new_data["price_brl"] *= 1.038  # Aumentamos o custo mensal em 3.80%, considerando IPCA dos últimos 6 meses https://www.ibge.gov.br/explica/inflacao.php
    
    new_data["price_brl"] *= 5 # Simulando drift com uma inflação de 15% 
    new_data["area_m2"] *= 2

    print("Criado dataset artificialmente alterado para simular drift.")
    return new_data

def get_predictions(data):
    print(data.head())

    
    # Crie uma lista de dicionários, onde cada dicionário representa uma instância
    instances = []
    for _, row in data.iterrows():
        instance = {col: row[col] for col in data.columns if col != "preco_brl"}
        instances.append(instance)


    url = "http://localhost:5000/invocations"
    headers = {"Content-Type": "application/json"}
    payload = {"instances": instances}
    
    response = requests.post(url, headers=headers, json=payload)
    predictions = response.json()
    predictions = predictions.get("predictions")
    print(predictions)
    return predictions

# Avaliar degradação do modelo
def evaluate_model(df, y, new_data):
   
    data_to_evaluate = new_data if new_data is not None else df
    evaluation_type = "artificiais" if new_data is not None else "originais"
    print(f"Avaliando modelo com dados {evaluation_type}")
 
    data_to_evaluate["prediction"] = get_predictions(data_to_evaluate)
    data_to_evaluate["prediction"] = data_to_evaluate["prediction"].astype(int)
    print(data_to_evaluate["prediction"].unique())
 
    data_to_evaluate["target"] = y
    print(data_to_evaluate["target"].unique())
 
    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    report.run(reference_data=df, current_data=data_to_evaluate)
    
    report_filename = "monitoring_report_df_new_data.html" if new_data is not None else "monitoring_report_df.html"
    report.save_html(report_filename)
 
    report_dict = report.as_dict()
    drift_score = report_dict["metrics"][0]["result"]["dataset_drift"]
    print(f"Score de drift: {drift_score}")
 
    drift_by_columns = report_dict["metrics"][1]["result"].get("drift_by_columns", {})
    print(f"Colunas com drift: {drift_by_columns}")
 
    return drift_score, drift_by_columns

def preprocess_data(df):

    df.dropna(inplace=True)

    # float64
    df = df.astype({col: 'float64' for col in df.select_dtypes(include='int').columns})

    X = df.drop(columns=["price_brl"])
    y = df["price_brl"]

    print(df.head())

    return X, y.astype(float).round(2)



if __name__ == "__main__":

    dados = pd.read_csv("dataset/brasil_estado_cidade.csv", encoding="utf-8")
    sample = dados.sample(1000)  # Pegamos exemplos aleatórios para testar

    df_examples, y = preprocess_data(sample)

    drift_score, drift_by_columns = evaluate_model(df_examples, y, None)

    #new_data = simulate_drift(dados.sample(1000))
    #df_examples, y = preprocess_data(new_data)
    #drift_score, drift_by_columns = evaluate_model(df_examples, y, new_data)

    new_data = simulate_drift(sample)
    drift_score, drift_by_columns = evaluate_model(df_examples, y, new_data)

    check_for_drift(drift_score, drift_by_columns)



