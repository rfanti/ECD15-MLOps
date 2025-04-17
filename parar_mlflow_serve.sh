# Use o comando abaixo para matar o processo que está usando a porta 5000
if fuser -k 5000/tcp; then
    echo "O processo na porta 5000 foi encerrado com sucesso."
else
    echo "Falha ao tentar encerrar o processo na porta 5000."
fi