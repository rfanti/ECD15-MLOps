import pandas as pd
import numpy as np
import requests
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, ClassificationPreset
from evidently import ColumnMapping
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
        os.system("jupyter nbconvert --to notebook --execute --inplace treinamento.ipynb --output treinamento_exec.ipynb")
        os.system("jupyter nbconvert --to notebook --execute --inplace predicao.ipynb --output predicao_exec.ipynb")  # <-- aqui
    else:
        if num_columns_drift > 2:
            print(f"Drift detectado em {num_columns_drift} colunas! Treinando novo modelo...")
            os.system("jupyter nbconvert --to notebook --execute --inplace treinamento.ipynb --output treinamento_exec.ipynb")
            os.system("jupyter nbconvert --to notebook --execute --inplace predicao.ipynb --output predicao_exec.ipynb")  # <-- e aqui também
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

def ajuste_valor_inflacao(dados):
    data = dados.copy()
    # Mudando coluna de preço de imóveis para simular mudanças nos padrões dos dados
    # new_data["tenure"] = np.random.randint(0, 10, new_data.shape[0])  # Mudamos a duração do cliente aleatoriamente
    data["price_brl"] *= 1.038  # Aumentamos o custo mensal em 3.80%, considerando IPCA dos últimos 6 meses https://www.ibge.gov.br/explica/inflacao.php

    return data

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

def evaluate_model(df, y, new_data=None, y_new=None):

    column_mapping = ColumnMapping(
        target='target',
        prediction='prediction',
        numerical_features=['area_m2', 'lat', 'lon'],
        categorical_features=['city', 'state', 'property_type']
    )

    data_to_evaluate = new_data if new_data is not None else df
    target = y_new if y_new is not None else y

    data_to_evaluate = data_to_evaluate.copy()
    data_to_evaluate["prediction"] = get_predictions(data_to_evaluate)
    data_to_evaluate["prediction"] = data_to_evaluate["prediction"].astype(float)
    data_to_evaluate["target"] = target

    reference_data = df.copy()
    reference_data["target"] = y
    reference_data["prediction"] = y  # simula previsão perfeita no ref

    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])

    #report.run(reference_data=reference_data, current_data=data_to_evaluate)
    report.run(reference_data=reference_data, current_data=data_to_evaluate, column_mapping=column_mapping)


    report_filename = "monitoring_report_df_new_data.html" if new_data is not None else "monitoring_report_df.html"
    report.save_html(report_filename)

    report_dict = report.as_dict()
    drift_score = report_dict["metrics"][0]["result"]["dataset_drift"]
    drift_by_columns = report_dict["metrics"][1]["result"].get("drift_by_columns", {})

    print(f"Score de drift: {drift_score}")
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

    dados_antigos = pd.read_csv("dataset/brasil_estado_cidade.csv", encoding="utf-8")
    dados_novos = pd.read_csv("dataset/sao_paulo.csv", encoding="utf-8")

    sample = dados_antigos.sample(1000)  # Pegamos exemplos aleatórios para testar

    df, y = preprocess_data(sample)

    drift_score, drift_by_columns = evaluate_model(df, y, None)

    # Dados antigos, Testando a detecção de drift, enviando  nova amostra dos dados antigos para comparação, 
    #  - não deve detectar drift relevante nos dados 
    #sample_novos = dados_antigos.sample(1000)    
    
    # Dados novos reais (novo data set)
    sample_novos = dados_novos.sample(1000)       

    # Rotina de ajuste do valor do imóvel de acordo com inflação, não é mais utilizado
    #new_data = ajuste_valor_inflacao(sample_novos) 

    new_data = sample_novos

    df_driftado, y_driftado = preprocess_data(new_data) 

    # Avalia o modelo com os dados que passaram por drift e seus targets atualizados
    drift_score, drift_by_columns = evaluate_model(
        df,                     # <-- base de referência (antiga, sem drift)
        y,                      # <-- target da base de referência
        new_data=df_driftado,
        y_new=y_driftado        # <-- target dos dados com drift
    )



