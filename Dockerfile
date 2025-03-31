# Utiliza a imagem oficial do python 3
FROM python:3.10-slim

# Atualiza pip
RUN pip install --upgrade pip

# Instala o Jupyter, MLFlow e as bibliotecas necessárias
# RUN pip install --no-cache-dir \
RUN pip install \
    evidently \
    jupyter \
    matplotlib \
    mlflow \
    nbformat \
    numpy \
    pandas \
    plotly \
    psycopg2-binary \
    requests \
    scikit-learn \
    scipy \
    seaborn \
    sqlparse  \
    geopandas \
    xgboost

# Instala o Git
RUN apt-get update && apt-get install -y git && apt-get clean

# Define a variável de ambiente para o Git Python
ENV GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git

# Cria um diretório de trabalho
RUN mkdir -p /app
WORKDIR /app

# Exponha as portas
EXPOSE 8080 
EXPOSE 8888 
EXPOSE 5000

# Executa o Jupyter Notebook
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]

# Copiar o script de inicialização e torná-lo executável
COPY init.sh /init.sh
RUN chmod +x /init.sh

# Definir o script de inicialização como o entrypoint do contêiner
CMD ["/bin/bash", "/init.sh"]
