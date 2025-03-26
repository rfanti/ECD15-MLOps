# Utiliza a imagem oficial do python 3
FROM python:3.10-slim

# Atualiza pip
RUN pip install --upgrade pip

# Instala o Jupyter, MLFlow e as bibliotecas necess�rias
# RUN pip install --no-cache-dir \
RUN pip install \
    # evidently \
    jupyter \
    matplotlib \
    mlflow \
    # nbformat \
    numpy \
    pandas \
    plotly \
    psycopg2-binary \
    requests \
    scikit-learn \
    scipy \
    seaborn \
    sqlparse  \
    geopandas 

# Cria um diret�rio de trabalho
RUN mkdir -p /app
WORKDIR /app

# exp�e a portas
EXPOSE 8080 8888 5000

# Executa o Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]

# Copiar o script de inicializa��o e torn�-lo execut�vel
#COPY init.sh /init.sh
#RUN chmod +x /init.sh

# Definir o script de inicializa��o como o entrypoint do cont�iner
#CMD ["/bin/sh", "/init.sh"]
