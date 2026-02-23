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

# Configuration
MODEL="Qwen/Qwen2.5-7B-Instruct"
INPUT_FILE="data/book_preference_dataset.jsonl"
TEMPLATE="util/templates/default_prompt.txt"
BASE_OUTPUT_DIR="/home/mananaga/goodreads/results"
MAX_NEW_TOKENS=256
BATCH_SIZE=16
RANDOM_SEED=86
DEVICE="auto"
NUM_ENTRIES=100

# Define the values to iterate over
NUM_BOOK_REVIEWS_LIST=(1 2 4 8)
NUM_USER_REVIEWS_LIST=(1)

echo "========================================"
echo "Starting evaluation runs: $(date)"
echo "========================================"

# Iterate over num_book_reviews
for num_book_reviews in "${NUM_BOOK_REVIEWS_LIST[@]}"; do
    # Iterate over num_user_reviews
    for num_user_reviews in "${NUM_USER_REVIEWS_LIST[@]}"; do
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/num_book_reviews_${num_book_reviews}_num_user_reviews_${num_user_reviews}"

        echo ""
        echo "========================================"
        echo "Running: num_book_reviews=${num_book_reviews}, num_user_reviews=${num_user_reviews}"
        echo "Output directory: ${OUTPUT_DIR}"
        echo "Start time: $(date)"
        echo "========================================"

        python util/run_llm_eval.py \
            --input "${INPUT_FILE}" \
            --output_dir "${OUTPUT_DIR}" \
            --template "${TEMPLATE}" \
            --model "${MODEL}" \
            --num_book_reviews ${num_book_reviews} \
            --num_user_reviews ${num_user_reviews} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --batch_size ${BATCH_SIZE} \
            --random_seed ${RANDOM_SEED} \
            --device ${DEVICE} \
            --num_entries ${NUM_ENTRIES}

        echo "Completed: num_book_reviews=${num_book_reviews}, num_user_reviews=${num_user_reviews}"
        echo "End time: $(date)"
    done
done

echo ""
echo "========================================"
echo "All evaluation runs completed: $(date)"
echo "========================================"
