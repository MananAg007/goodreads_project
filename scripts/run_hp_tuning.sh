#!/bin/bash

#SBATCH --job-name=goodreads_hp_tuning
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --output=logs/hp_tuning_%j.out
#SBATCH --error=logs/hp_tuning_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/sheels/Spring2026/10718_mlip/env

# Print diagnostics
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo "Working directory: $(pwd)"
echo "========================================"

# Run hyperparameter tuning
python scripts/hyperparameter_tuning.py

echo "========================================"
echo "End time: $(date)"
echo "========================================"