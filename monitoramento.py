import pandas as pd
import numpy as np
import requests
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, ClassificationPreset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from scipy import stats

import os
#import json

def check_for_drift(drift_score, drift_by_columns):
    num_columns_drift = sum(1 for col, values in drift_by_columns.items() if values.get("drift_detected", False))
    if drift_score > 0.5:
        print("Drift detectado no Dataset")
        os.system("jupyter nbconvert --to notebook --execute treinamento.ipynb --output treinamento_exec.ipynb")
    else:
        if num_columns_drift > 2:
            print(f"Drift detectado em {num_columns_drift} colunas! Treinando novo modelo...")
            os.system("jupyter nbconvert --to notebook --execute treinamento.ipynb --output treinamento_exec.ipynb")
        else:
            print("Modelo ainda está bom, sem necessidade de re-treinamento.")
            print("Nenhum drift detectado nas colunas e no dataset")

# Carregar o conjunto de dados
dados = pd.read_csv("dataset/brasil_estado_cidade.csv", encoding="utf-8")

# Eliminando registros com valores null
dados.dropna(inplace=True)

# float64
dados = dados.astype({col: 'float64' for col in dados.select_dtypes(include='int').columns})

def remover_outliers_por_cidade(df):
    # Remove outliers da coluna 'price_brl' agrupando por cidade.

    def remover_outliers_grupo(grupo):
        # Remove outliers de um grupo usando o método Z-score.
        z_scores = np.abs(stats.zscore(grupo["price_brl"]))
        return grupo[(z_scores < 3)]

    df_filtrado = df.groupby("city").apply(remover_outliers_grupo).reset_index(drop=True)
    return df_filtrado

dados = remover_outliers_por_cidade(dados)

dados.sample(10)

#def load_new_data():
 #   df = pd.read_csv("dataset/brasil_estado_cidade.csv", encoding="utf-8")
  #  df = df.sample(1000)  # Pegamos exemplos aleatórios para testar
    #X, y = preprocess_data(df)
    #return X, y

def simulate_drift(dados):
    new_data = dados.copy()
    # Mudando coluna de preço de imóveis para simular mudanças nos padrões dos dados
    # new_data["tenure"] = np.random.randint(0, 10, new_data.shape[0])  # Mudamos a duração do cliente aleatoriamente
    new_data["price_brl"] *= 1.038  # Aumentamos o custo mensal em 3.80%, considerando IPCA dos últimos 6 meses https://www.ibge.gov.br/explica/inflacao.php
    
    print("Criado dataset artificialmente alterado para simular drift.")
    return new_data

# Exemplo de pré-processamento
x_features = dados.drop(["price_brl"], axis=1)  # Features
y_target = dados["price_brl"]  # Variável alvo

# Identificando colunas numéricas e categóricas
numeric_features = x_features.select_dtypes(include=['number']).columns
categorical_features = x_features.select_dtypes(include=['object']).columns

# Criando transformadores
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Criando o ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


#def preprocess_data(df):
    #df.drop(columns=["customerID"], inplace=True, errors="ignore")
 #   df.replace({"Yes": 1, "No": 0}, inplace=True)
  #  df = df.infer_objects(copy=False)
    # for col in df.select_dtypes(include=["int64"]).columns:
     #   df[col] = df[col].astype("float64")
   # for col in df.select_dtypes(include=["object"]).columns:
     #   df[col] = df[col].astype(str)
      #  df[col] = LabelEncoder().fit_transform(df[col])

    #df.fillna(0, inplace=True)

    #X = df.drop(columns=["Churn"])
    #y = df["Churn"]

    #print(df.head())

   # return X, y.astype(int)

# Fazer previsões com o modelo
def get_predictions(data):
    print(data.head())

    
    # Crie uma lista de dicionários, onde cada dicionário representa uma instância
    instances = []
    for _, row in data.iterrows():
        instance = {col: row[col] for col in data.columns}
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
    if new_data is None:
        print("Avaliando modelo com dados originais")
        df["prediction"] = get_predictions(df)
        df["prediction"] = df["prediction"].astype(int)
        print(df["prediction"].unique())
        df["target"] = y
        print(df["target"].unique())
        report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
        report.run(reference_data=df, current_data=df)
        report.save_html("monitoring_report_df.html")
        report_dict = report.as_dict()
        drift_score = report_dict["metrics"][0]["result"]["dataset_drift"]
        print(f"Score de drift: {drift_score}")
        drift_by_columns = report_dict["metrics"][1]["result"].get("drift_by_columns", {})
        print(f"Coluns drift: {drift_by_columns}")
        return drift_score, drift_by_columns
    else:
        print("Avaliando modelo com dados artificiais")
        new_data["prediction"] = get_predictions(new_data)
        new_data["prediction"] = new_data["prediction"].astype(int)
        print(new_data["prediction"].unique())
        new_data["target"] = y
        print(new_data["target"].unique())
        report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
        report.run(reference_data=df, current_data=new_data)
        report.save_html("monitoring_report_df_new_data.html")
        report_dict = report.as_dict()
        drift_score = report_dict["metrics"][0]["result"]["dataset_drift"]
        print(f"Score de drift: {drift_score}")
        drift_by_columns = report_dict["metrics"][1]["result"].get("drift_by_columns", {})
        print(f"Coluns drift: {drift_by_columns}")
        return drift_score, drift_by_columns

def main():
    df_examples = dados.copy()
    y = y_target
    drift_score, drift_by_columns = evaluate_model(df_examples, y, None)
    new_data = simulate_drift(df_examples)
    drift_score, drift_by_columns = evaluate_model(df_examples, y, new_data)
    check_for_drift(drift_score, drift_by_columns)

if __name__ == "__main__":
    main()



