#!/bin/bash
#SBATCH --job-name=tabicl_v2
#SBATCH --output=logs/run_v2_%j.out
#SBATCH --error=logs/run_v2_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=40:00:00

# Inicializa o ambiente
source /home/seu_usuario/miniconda3/etc/profile.d/conda.sh
conda activate tabicl_env

echo "=========================================="
echo "Iniciando Job v2 (Correção de Leakage)"
echo "Data/Hora: $(date)"
echo "Nó alocado: $SLURMD_NODENAME"
echo "CPUs alocadas: $SLURM_CPUS_PER_TASK"
echo "GPU alocada: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Cria diretório de logs se não existir
mkdir -p logs

# Executa o pipeline corrigido
python run_cluster_final_v2.py

echo "=========================================="
echo "Job Finalizado."
echo "Data/Hora: $(date)"
echo "=========================================="
