#!/bin/bash
#SBATCH --job-name=ag_joao
#SBATCH --output=logs/ag_extreme_joao_%j.out
#SBATCH --error=logs/ag_extreme_joao_%j.err
#SBATCH --partition=long-simple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# ==============================================
# TATICA ENXAME - JOAO
# AutoGluon Extreme 4h - Datasets LARGE
# SEM GPU / 4 CPUs (para caber ao lado do Job V2)
# ==============================================

source $HOME/tabarena_env/bin/activate

echo "=========================================="
echo "TATICA ENXAME - JOAO (Batch LARGE)"
echo "Data/Hora: $(date)"
echo "No alocado: $SLURMD_NODENAME"
echo "CPUs alocadas: $SLURM_CPUS_PER_TASK"
echo "=========================================="

mkdir -p logs

python run_ag_extreme_joao.py

echo "=========================================="
echo "Job Finalizado."
echo "Data/Hora: $(date)"
echo "=========================================="
