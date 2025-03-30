@echo off
echo Iniciando a execução dos scripts Python na pasta "models"...

rem Loop para percorrer todos os arquivos .py na pasta "models"
for %%f in (models\*.py) do (
    echo Executando %%f...
    start python "%%f"
)

echo Todos os códigos phyton em execução!
pause
