#!/bin/bash
#SBATCH --job-name=Optuna_Tuning
#SBATCH --partition=long-simple
#SBATCH --output=resultado_optuna.txt
#SBATCH --error=erros_optuna.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

echo "Iniciando a bateria de Optuna Hyperparameter Tuning!"
echo "Node alocado: $SLURMD_NODENAME"

# Ativa o seu ambiente virtual
source $HOME/tabarena_env/bin/activate

# Garante que o Optuna esteja instalado no ambiente
pip install optuna

# Inicia a execução usando o Python diretamente de dentro do ambiente virtual
$HOME/tabarena_env/bin/python run_optuna.py
