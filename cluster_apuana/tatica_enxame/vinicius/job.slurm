#!/bin/bash
#SBATCH --job-name=ag_viniciu
#SBATCH --output=logs/ag_extreme_vinicius_%j.out
#SBATCH --error=logs/ag_extreme_vinicius_%j.err
#SBATCH --partition=long-simple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# ==============================================
# TATICA ENXAME - VINICIUS
# AutoGluon Extreme 4h - Datasets SMALL/MEDIUM
# SEM GPU / 8 CPUs
# ==============================================

source $HOME/tabarena_env/bin/activate

echo "=========================================="
echo "TATICA ENXAME - VINICIUS (Batch SMALL/MEDIUM)"
echo "Data/Hora: $(date)"
echo "No alocado: $SLURMD_NODENAME"
echo "CPUs alocadas: $SLURM_CPUS_PER_TASK"
echo "=========================================="

mkdir -p logs

cd $HOME/Projeto_AM_Leandro_TabICL/cluster_apuana/tatica_enxame/vinicius
python run_ag_extreme.py

echo "=========================================="
echo "Job Finalizado."
echo "Data/Hora: $(date)"
echo "=========================================="
