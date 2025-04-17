#!/bin/bash

# Tenta matar o processo na porta 5000
if fuser -k 5000/tcp > /dev/null 2>&1; then
    echo "O processo na porta 5000 foi encerrado com sucesso."
else
    echo "Falha ao tentar encerrar o processo na porta 5000 (ou nenhum processo estava ativo)."
fi