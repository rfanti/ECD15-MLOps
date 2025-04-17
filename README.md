# ECD15 - MLOps

## Introdução

O objetivo deste trabalho é aplicar conceitos e práticas de MLOps para desenvolver um pipeline de Machine Learning funcional e automatizado. Os grupos (máximo de 3 pessoas) deverão explorar um conjunto de dados real, implementar modelos preditivos e integrar o processo com ferramentas de monitoramento, versionamento e deploy.

O foco do projeto está na construção de um fluxo completo, contemplando desde a preparação dos dados até a entrega do modelo em produção, garantindo rastreabilidade e reprodutibilidade.

### Integrantes
- Douglas Lázaro (@dlazarosps)
- Ilenice Trojahn (@ilenicetr)
- Rafael Fanti (@rfanti)
---

## Dataset e Problema
O problema de predição do valor de imóveis é um problema clássico de regressão em Machine Learning, onde o objetivo é prever o preço de uma propriedade com base em um conjunto de características (features) que descrevem o imóvel e sua localização.

## Dataset
- **Treinamento e Validação**: O dataset [Brasil Real Estate Data](https://www.kaggle.com/datasets/ashishkumarjayswal/brasil-real-estate/data) será utilizado para o treinamento e validação do modelo no MLflow.
- **Predições e Monitoramento de Drift**: O dataset [House Price Data of São Paulo](https://www.kaggle.com/datasets/kaggleshashankk/house-price-data-of-sao-paulo/data), previamente pré-processado e mapeado para manter consistência com o dataset de treinamento, será utilizado para realizar predições e avaliar o drift do modelo treinado.


### Descrição do Problema

O objetivo é construir um modelo preditivo que estime o valor de um imóvel com base em variáveis como:

- **Características do imóvel**: número de quartos, banheiros, área construída, número de vagas na garagem, entre outros.
- **Localização**: bairro, proximidade de serviços essenciais (escolas, hospitais, transporte público), índice de criminalidade, etc.
- **Condições do mercado**: ano de construção, tendências de mercado, inflação, entre outros fatores econômicos.

### Desafios do Problema

1. **Dados Ausentes ou Inconsistentes**: É comum que datasets imobiliários apresentem valores ausentes ou inconsistências, como dados duplicados ou fora do intervalo esperado.
2. **Multicolinearidade**: Algumas variáveis podem estar altamente correlacionadas, o que pode impactar a performance de modelos lineares.
3. **Distribuição Não-Uniforme**: Os preços dos imóveis podem variar amplamente, resultando em uma distribuição enviesada que pode dificultar o treinamento do modelo.
4. **Overfitting**: Modelos complexos podem se ajustar demais aos dados de treinamento, prejudicando a generalização para novos dados.
5. **Drift de Dados**: Mudanças no mercado imobiliário ao longo do tempo podem tornar o modelo obsoleto, exigindo monitoramento contínuo e re-treinamento.

### Abordagem com Modelos Clássicos de ML

Para resolver o problema, serão utilizados modelos clássicos de regressão, como:

- **Regressão Linear**: Simples e interpretável, ideal para estabelecer uma linha de base.
- **Árvores de Decisão**: Capturam relações não-lineares entre as variáveis.
- **Random Forest**: Combinação de múltiplas árvores para melhorar a robustez e reduzir overfitting.
- **Gradient Boosting (ex.: XGBoost, LightGBM)**: Modelos baseados em boosting que geralmente apresentam alta performance em problemas de regressão.

### Métricas de Avaliação

As métricas utilizadas para avaliar o desempenho dos modelos incluirão:

- **Mean Absolute Error (MAE)**: Média dos erros absolutos entre os valores reais e preditos.
- **Mean Squared Error (MSE)**: Média dos quadrados dos erros, penalizando erros maiores.
- **R² Score**: Mede a proporção da variância explicada pelo modelo.

A escolha do modelo final será baseada em um equilíbrio entre desempenho (métricas) e interpretabilidade, considerando também a facilidade de integração no pipeline de MLOps.

---

## Requisitos Técnicos
- **Linguagem**: Python 3.10
- **Ferramentas Utilizadas**:
    - Jupyter Notebooks (para exploração do dataset e modelos)
    - MLflow (para rastreamento e versionamento de modelos)
    - Evidently AI (para monitoramento de drift)
    - FastAPI/Flask (para disponibilização do modelo via API, ex.: usando MLflow)
    - GitHub/GitLab (para controle de versão)
    - Docker/Docker Compose (para conteinerização e execução do pipeline)

### Execução com Docker Compose

O projeto inclui um arquivo `docker-compose.yml` que facilita a execução do pipeline completo em contêineres. Siga os passos abaixo para executar o projeto:

1. **Certifique-se de ter o Docker e o Docker Compose instalados**:
   - [Instalar Docker](https://docs.docker.com/get-docker/)
   - [Instalar Docker Compose](https://docs.docker.com/compose/install/)

2. **Clone o repositório do projeto**:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DO_DIRETORIO>
   ```

3. **Construa e inicie os contêineres**:
   ```bash
   docker-compose up -d --build
   ```

4. **Acesse os serviços**:
   - O Jupyter Notebook estará disponível em `http://localhost:8888`.
   - A API estará disponível em `http://localhost:5000`.
   - O MLflow estará disponível em `http://localhost:8080`.

5. **Parar os contêineres**:
   Para encerrar os serviços, utilize:
   ```bash
   docker-compose down
   ```

6. **Executar container docker**
     docker exec -it ecd15 bash
---

## Etapas do Projeto (MLOps Pipeline)

1. **Exploração e Pré-processamento dos Dados**
     - Análise exploratória e tratamento de valores ausentes.
     - Normalização/Padronização dos dados, quando necessário.

2. **Treinamento e Avaliação do Modelo**
     - Implementação de pelo menos dois modelos e comparação de métricas.
     - Utilização do MLflow para rastrear experimentos.

3. **Versionamento e Armazenamento do Modelo**
     - Registro do modelo no MLflow Model Registry.

4. **Implantação do Modelo**
     - Construção de uma API com FastAPI ou Flask para servir previsões (MLflow).
     - Deploy local ou em nuvem (ex.: AWS, GCP, Azure).

5. **Monitoramento e Re-treinamento**
     - Implementação de monitoramento de drift de dados com Evidently AI.
     - Definição de uma estratégia para re-treinamento automático do modelo.

6. **Conteinerização e Documentação**
     - Instruções de execução/documentação do pipeline no repositório.

---

## Entregáveis

- **Código-fonte** em um repositório Git (GitHub/GitLab) contendo:
    - Pipeline de dados e treinamento.
    - Código da API para inferência.
    - Scripts de monitoramento e re-treinamento.
    - Arquivos de configuração.

- **Relatório** (.DOCX ou .PDF) explicando:
    - Escolha do dataset e problema abordado.
    - Metodologia e ferramentas utilizadas.
    - Resultados e métricas dos modelos.
    - Fluxo completo do pipeline e considerações finais.

---
