#!/bin/bash
#SBATCH --job-name=TabICL_Eval         # Nome do job no cluster
#SBATCH --output=resultado_tabarena.txt  # Onde o print() do Python vai ser salvo
#SBATCH --error=erros_tabarena.txt       # Onde os erros graves (tracebacks) vão ser salvos
#SBATCH --time=48:00:00                # Limite máximo de tempo (48 horas)
#SBATCH --gpus=1                       # Pedimos 1 Placa de Vídeo (Essencial para TabICL e AutoGluon)
#SBATCH --cpus-per-task=8              # Pedimos 8 processadores (Essencial para pré-processamento rápido)

# === INSTRUÇÕES ===
# 1. Este arquivo diz ao SLURM como configurar a máquina antes de rodar seu código.
# 2. Quando você tiver sua conta, abra o terminal do cluster e rode: sbatch job.slurm
# 3. O cluster vai colocar na fila e rodar no background!

# Python 3.10.12 já está instalado no sistema, não precisa de 'module load'

# Ativa o seu ambiente virtual
source $HOME/tabarena_env/bin/activate

# Inicia a execução do script
echo "Iniciando a bateria de testes no Cluster Apuana..."
python3 run_cluster.py
echo "Finalizado!"
