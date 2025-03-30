import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

mlflow.set_tracking_uri("sqlite:///models/mlflow.db")

mlflow.set_experiment("ecd15")

with mlflow.start_run() as run:
    mlflow.log_param("model_type", "Decision Tree")
    # Carregar o conjunto de dados
    dados = pd.read_csv("dataset/brasil_estado_cidade.csv", encoding="latin1")

    # Eliminando registros com valores null
    dados.dropna(inplace=True)

    def remover_outliers_por_cidade(df):
        """Remove outliers da coluna 'price_brl' agrupando por cidade."""

        def remover_outliers_grupo(grupo):
            """Remove outliers de um grupo usando o método Z-score."""
            z_scores = np.abs(stats.zscore(grupo["price_brl"]))
            return grupo[(z_scores < 3)]

        df_filtrado = df.groupby("city").apply(remover_outliers_grupo).reset_index(drop=True)
        return df_filtrado

    dados = remover_outliers_por_cidade(dados)


    # Exemplo de pré-processamento
    X = dados.drop(["price_brl"], axis=1)  # Features
    y = dados["price_brl"]  # Variável alvo


    # Identificando colunas numéricas e categóricas
    numeric_features = X.select_dtypes(include=['number']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Criando transformadores
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Criando o ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Arvore de Decisões

    model_dt = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', DecisionTreeRegressor())])
    model_dt.fit(X_train, y_train)

    # Avaliação do modelo Arvore de Decisão
    y_pred_dt = model_dt.predict(X_test)
    mse_dt = mean_squared_error(y_test, y_pred_dt)
    r2_dt = r2_score(y_test, y_pred_dt)
    mae_dt = mean_absolute_error(y_test, y_pred_dt)

    signature = infer_signature(X_test, model_dt)

    print(f"Arvore de Decisão: MSE={mse_dt:.2f}, R2={r2_dt:.2f}, MAE={mae_dt:.2f}")

    mlflow.log_metrics({"mse": mse_dt, "r2": r2_dt, "mae": mae_dt})

    mlflow.sklearn.log_model(
        sk_model=model_dt,
        artifact_path="'sklearn'-model",
        signature=signature,
        registered_model_name="decision-tree-model",
    )
