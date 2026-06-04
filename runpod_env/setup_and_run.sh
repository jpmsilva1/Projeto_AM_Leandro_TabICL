#!/bin/bash
# Otimizador Extremo para Runpod

echo "Preparando o terreno na RTX 4090..."

# Instala todas as dependências exigidas
pip install numpy pandas scikit-learn optuna openml lightgbm xgboost catboost "autogluon.tabular[all]" tabicl "pytabkit[models]"

# Inicia o pipeline em segundo plano com nohup
# (Isso garante que se o terminal SSH fechar, o processo continua rodando)
nohup python run_runpod.py > runpod_log.txt 2>&1 &

echo ""
echo "🚀 PIPELINE INICIADO NA GPU COM SUCESSO! 🚀"
echo ""
echo "O código está rodando no background. Pode fechar o terminal se quiser."
echo "Para espiar o progresso ao vivo, digite:"
echo "tail -f runpod_log.txt"
