#!/bin/bash

cabal build

if [ $? -ne 0 ]; then
    echo "Erro na compilação do código. Corrija os erros e tente novamente."
    exit 1
fi

echo "Compilação concluída com sucesso! Iniciando a bateria de treinos..."

# Loop pelas épocas pedidas
for EPOCAS in 1 2 4 8 16 32 64 128 256 512 1024 2048; do
    echo "====================================================="
    echo "Iniciando treino de $EPOCAS épocas..."
    
    cabal run lambda-ai $EPOCAS
    
    # Verifica se a execução teve sucesso ou se houve overflow/crash
    if [ $? -eq 0 ]; then
        echo "Sucesso: Treino e inferência de $EPOCAS épocas finalizados."
    else
        echo "ERRO ou Out Of Memory (OOM) detectado no modelo de $EPOCAS épocas!"
        echo "Parando a bateria de experimentos. Os modelos menores que rodaram antes estão salvos."
        break
    fi
done

echo "====================================================="
echo "Bateria de experimentos finalizada!"