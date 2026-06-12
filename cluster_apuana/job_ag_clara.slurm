#!/bin/bash
#SBATCH --job-name=ag_clara
#SBATCH --output=logs/ag_extreme_clara_%j.out
#SBATCH --error=logs/ag_extreme_clara_%j.err
#SBATCH --partition=long-simple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# ==============================================
# TATICA ENXAME - CLARA
# AutoGluon Extreme 4h - Datasets MEDIUM
# SEM GPU / 8 CPUs
# ==============================================

source $HOME/tabarena_env/bin/activate

echo "=========================================="
echo "TATICA ENXAME - CLARA (Batch MEDIUM)"
echo "Data/Hora: $(date)"
echo "No alocado: $SLURMD_NODENAME"
echo "CPUs alocadas: $SLURM_CPUS_PER_TASK"
echo "=========================================="

mkdir -p logs

python run_ag_extreme_clara.py

echo "=========================================="
echo "Job Finalizado."
echo "Data/Hora: $(date)"
echo "=========================================="
