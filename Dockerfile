# Utiliza a imagem oficial do python 3
FROM python:3.10-slim

# Cria um diretório de trabalho
RUN mkdir -p /app
WORKDIR /app

# Atualiza pip
RUN pip install --upgrade pip

# Copia o arquivo requirements.txt para o contêiner
COPY requirements.txt /app/requirements.txt

# Instala o Jupyter, MLFlow e as bibliotecas necessárias
RUN pip install -r requirements.txt

# Instala o Git
RUN apt-get update && apt-get install -y git && apt-get clean


RUN apt-get update && apt-get install -y curl jq

# Instala fuser
RUN apt-get install -y psmisc

# Define a variável de ambiente para o Git Python
ENV GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git

# Configura a variável de ambiente MLFLOW_TRACKING_URI no arquivo .bashrc
RUN echo 'export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"' >> ~/.bashrc

# Configura a variável de ambiente MLFLOW_TRACKING_URI para o contêiner
ENV MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

# Copiar o script de inicialização e torná-lo executável
COPY init.sh /init.sh
RUN chmod +x /init.sh

# Exponha as portas
EXPOSE 8080 
EXPOSE 8888 
EXPOSE 5000

# Definir o script de inicialização como o entrypoint do contêiner
RUN sed -i 's/\r$//' /init.sh 
CMD ["/bin/bash", "/init.sh"]
