import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("ecd15-gb")

with mlflow.start_run() as run:

    # Carregar o conjunto de dados
    dados = pd.read_csv("../dataset/brasil_estado_cidade.csv", encoding="latin1")

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

    #Gradient Boost

    model_gb = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor())])
    model_gb.fit(X_train, y_train)

    # Avaliação do modelo Gradient Boosting
    y_pred_gb = model_gb.predict(X_test)
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)
    mae_gb = mean_absolute_error(y_test, y_pred_gb)

    print(f"Gradient Boosting: MSE={mse_gb:.2f}, R2={r2_gb:.2f}, MAE={mae_gb:.2f}")
