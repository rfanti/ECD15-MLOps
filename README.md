# ECD15 - MLOps

## Introdu��o

O objetivo deste trabalho � aplicar conceitos e pr�ticas de MLOps para desenvolver um pipeline de Machine Learning funcional e automatizado. Os grupos (m�ximo de 3 pessoas) dever�o explorar um conjunto de dados real, implementar modelos preditivos e integrar o processo com ferramentas de monitoramento, versionamento e deploy.

O foco do projeto est� na constru��o de um fluxo completo, contemplando desde a prepara��o dos dados at� a entrega do modelo em produ��o, garantindo rastreabilidade e reprodutibilidade.

### Integrantes
- Douglas L�zaro (@dlazarosps)
- Ilenice Trojahn (@ilenicetr)
- Rafael Fanti (@rfanti)
---

## Dataset e Problema
O problema de predi��o do valor de im�veis � um problema cl�ssico de regress�o em Machine Learning, onde o objetivo � prever o pre�o de uma propriedade com base em um conjunto de caracter�sticas (features) que descrevem o im�vel e sua localiza��o.

## Dataset
- **Treinamento e Valida��o**: O dataset [Brasil Real Estate Data](https://www.kaggle.com/datasets/ashishkumarjayswal/brasil-real-estate/data) ser� utilizado para o treinamento e valida��o do modelo no MLflow.
- **Predi��es e Monitoramento de Drift**: O dataset [House Price Data of S�o Paulo](https://www.kaggle.com/datasets/kaggleshashankk/house-price-data-of-sao-paulo/data), previamente pr�-processado e mapeado para manter consist�ncia com o dataset de treinamento, ser� utilizado para realizar predi��es e avaliar o drift do modelo treinado.


### Descri��o do Problema

O objetivo � construir um modelo preditivo que estime o valor de um im�vel com base em vari�veis como:

- **Caracter�sticas do im�vel**: n�mero de quartos, banheiros, �rea constru�da, n�mero de vagas na garagem, entre outros.
- **Localiza��o**: bairro, proximidade de servi�os essenciais (escolas, hospitais, transporte p�blico), �ndice de criminalidade, etc.
- **Condi��es do mercado**: ano de constru��o, tend�ncias de mercado, infla��o, entre outros fatores econ�micos.

### Desafios do Problema

1. **Dados Ausentes ou Inconsistentes**: � comum que datasets imobili�rios apresentem valores ausentes ou inconsist�ncias, como dados duplicados ou fora do intervalo esperado.
2. **Multicolinearidade**: Algumas vari�veis podem estar altamente correlacionadas, o que pode impactar a performance de modelos lineares.
3. **Distribui��o N�o-Uniforme**: Os pre�os dos im�veis podem variar amplamente, resultando em uma distribui��o enviesada que pode dificultar o treinamento do modelo.
4. **Overfitting**: Modelos complexos podem se ajustar demais aos dados de treinamento, prejudicando a generaliza��o para novos dados.
5. **Drift de Dados**: Mudan�as no mercado imobili�rio ao longo do tempo podem tornar o modelo obsoleto, exigindo monitoramento cont�nuo e re-treinamento.

### Abordagem com Modelos Cl�ssicos de ML

Para resolver o problema, ser�o utilizados modelos cl�ssicos de regress�o, como:

- **Regress�o Linear**: Simples e interpret�vel, ideal para estabelecer uma linha de base.
- **�rvores de Decis�o**: Capturam rela��es n�o-lineares entre as vari�veis.
- **Random Forest**: Combina��o de m�ltiplas �rvores para melhorar a robustez e reduzir overfitting.
- **Gradient Boosting (ex.: XGBoost, LightGBM)**: Modelos baseados em boosting que geralmente apresentam alta performance em problemas de regress�o.

### M�tricas de Avalia��o

As m�tricas utilizadas para avaliar o desempenho dos modelos incluir�o:

- **Mean Absolute Error (MAE)**: M�dia dos erros absolutos entre os valores reais e preditos.
- **Mean Squared Error (MSE)**: M�dia dos quadrados dos erros, penalizando erros maiores.
- **R� Score**: Mede a propor��o da vari�ncia explicada pelo modelo.

A escolha do modelo final ser� baseada em um equil�brio entre desempenho (m�tricas) e interpretabilidade, considerando tamb�m a facilidade de integra��o no pipeline de MLOps.

---

## Requisitos T�cnicos
- **Linguagem**: Python 3.10
- **Ferramentas Utilizadas**:
    - Jupyter Notebooks (para explora��o do dataset e modelos)
    - MLflow (para rastreamento e versionamento de modelos)
    - Evidently AI (para monitoramento de drift)
    - FastAPI/Flask (para disponibiliza��o do modelo via API, ex.: usando MLflow)
    - GitHub/GitLab (para controle de vers�o)
    - Docker/Docker Compose (para conteineriza��o e execu��o do pipeline)

### Execu��o com Docker Compose

O projeto inclui um arquivo `docker-compose.yml` que facilita a execu��o do pipeline completo em cont�ineres. Siga os passos abaixo para executar o projeto:

1. **Certifique-se de ter o Docker e o Docker Compose instalados**:
   - [Instalar Docker](https://docs.docker.com/get-docker/)
   - [Instalar Docker Compose](https://docs.docker.com/compose/install/)

2. **Clone o reposit�rio do projeto**:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DO_DIRETORIO>
   ```

3. **Construa e inicie os cont�ineres**:
   ```bash
   docker-compose up --build
   ```

4. **Acesse os servi�os**:
   - O Jupyter Notebook estar� dispon�vel em `http://localhost:8888`.
   - A API estar� dispon�vel em `http://localhost:5000`.
   - O MLflow estar� dispon�vel em `http://localhost:8080`.

5. **Parar os cont�ineres**:
   Para encerrar os servi�os, utilize:
   ```bash
   docker-compose down
   ```
---

## Etapas do Projeto (MLOps Pipeline)

1. **Explora��o e Pr�-processamento dos Dados**
     - An�lise explorat�ria e tratamento de valores ausentes.
     - Normaliza��o/Padroniza��o dos dados, quando necess�rio.

2. **Treinamento e Avalia��o do Modelo**
     - Implementa��o de pelo menos dois modelos e compara��o de m�tricas.
     - Utiliza��o do MLflow para rastrear experimentos.

3. **Versionamento e Armazenamento do Modelo**
     - Registro do modelo no MLflow Model Registry.

4. **Implanta��o do Modelo**
     - Constru��o de uma API com FastAPI ou Flask para servir previs�es (MLflow).
     - Deploy local ou em nuvem (ex.: AWS, GCP, Azure).

5. **Monitoramento e Re-treinamento**
     - Implementa��o de monitoramento de drift de dados com Evidently AI.
     - Defini��o de uma estrat�gia para re-treinamento autom�tico do modelo.

6. **Conteineriza��o e Documenta��o**
     - Instru��es de execu��o/documenta��o do pipeline no reposit�rio.

---

## Entreg�veis

- **C�digo-fonte** em um reposit�rio Git (GitHub/GitLab) contendo:
    - Pipeline de dados e treinamento.
    - C�digo da API para infer�ncia.
    - Scripts de monitoramento e re-treinamento.
    - Arquivos de configura��o.

- **Relat�rio** (.DOCX ou .PDF) explicando:
    - Escolha do dataset e problema abordado.
    - Metodologia e ferramentas utilizadas.
    - Resultados e m�tricas dos modelos.
    - Fluxo completo do pipeline e considera��es finais.

---
