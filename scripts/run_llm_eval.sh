#!/bin/bash
mkdir -p logs

# Activate conda environment
# source ~/miniconda/etc/profile.d/conda.sh
# conda activate /data/user_data/sheels/Spring2026/10718_mlip/env

# Available models (swap via --model, no other changes needed):
#   Qwen/Qwen2.5-7B-Instruct       (7B, default, recommended)
#   Qwen/Qwen2.5-3B-Instruct       (3B)
#   meta-llama/Llama-3.2-3B-Instruct (3B)
#   meta-llama/Llama-3.1-8B-Instruct (8B)

# Set HuggingFace cache directories to avoid filling up login node
export HF_HOME=/data/user_data/mananaga/huggingface
export HUGGINGFACE_HUB_CACHE=/data/user_data/mananaga/huggingface/hub
export TRANSFORMERS_CACHE=/data/user_data/mananaga/huggingface/transformers

python util/run_llm_eval.py \
    --input data/book_preference_dataset.jsonl \
    --output_dir /home/mananaga/goodreads/results \
    --template util/templates/default_prompt.txt \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num_book_reviews 5 \
    --num_user_reviews 1 \
    --max_new_tokens 256 \
    --batch_size 8 \
    --random_seed 86 \
    --device auto \
    --num_entries 8

echo "========================================"
echo "End time: $(date)"
echo "========================================"
