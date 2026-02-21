#!/bin/bash

#SBATCH --job-name=llm_eval
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --output=logs/llm_eval_%j.out
#SBATCH --error=logs/llm_eval_%j.err

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

# Available models (swap via --model, no other changes needed):
#   Qwen/Qwen2.5-7B-Instruct       (7B, default, recommended)
#   Qwen/Qwen2.5-3B-Instruct       (3B)
#   meta-llama/Llama-3.2-3B-Instruct (3B)
#   meta-llama/Llama-3.1-8B-Instruct (8B)

python util/run_llm_eval.py \
    --input util/book_preference_dataset.jsonl \
    --output_dir /data/user_data/saksham3/courses/10-718-project/llm_eval_results \
    --template util/templates/default_prompt.txt \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num_book_reviews 5 \
    --num_user_reviews 1 \
    --max_new_tokens 256 \
    --batch_size 4 \
    --random_seed 86 \
    --device auto

echo "========================================"
echo "End time: $(date)"
echo "========================================"
