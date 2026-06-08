#!/bin/bash
#SBATCH --job-name=TabICL_Final_Run        # Nome do job no cluster
#SBATCH --output=resultado_final_run.txt   # Onde o print() do Python vai ser salvo
#SBATCH --error=erros_final_run.txt        # Onde os erros graves (tracebacks) vão ser salvos
#SBATCH --partition=long-simple            # Fila principal (permite até 7 dias)
#SBATCH --time=72:00:00                    # Limite máximo de tempo (72 horas)
#SBATCH --gpus=1                           # Pedimos 1 Placa de Vídeo 
#SBATCH --cpus-per-task=16                 # SETUP SEGURO: 16 processadores
#SBATCH --mem=128G                         # SETUP SEGURO: 128GB de RAM para evitar SegFault no XGBoost

# === INSTRUÇÕES PARA A RODADA FINAL ===
# 1. Apague os arquivos .csv antigos (se quiser recomeçar do zero).
# 2. Rode este arquivo com: sbatch job_apuana_final.slurm
# 3. Este setup vai alocar uma máquina muito forte (pode demorar um pouco na fila PENDING).

source $HOME/tabarena_env/bin/activate

python3 run_cluster_final.py
