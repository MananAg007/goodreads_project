#!/bin/bash
mkdir -p logs

# Available Claude models:
#   claude-3-haiku-20240307        (fastest, cheapest)
#   claude-3-5-sonnet-20240620     (balanced, recommended)
#   claude-3-opus-20240229         (most capable, expensive)

# Configuration
INPUT_FILE="data/book_preference_dataset.jsonl"
TEMPLATE="util/templates/strict_format_prompt.txt"
BASE_OUTPUT_DIR="/home/mananaga/goodreads/results/claude"
MAX_TOKENS=1024
CONCURRENT_REQUESTS=5  # Number of parallel API requests (adjust based on rate limits)
RANDOM_SEED=86
NUM_ENTRIES=100
RATE_LIMIT_DELAY=0.1

# Define the values to iterate over
MODELS=("claude-3-haiku-20240307")
NUM_BOOK_REVIEWS_LIST=(8)
NUM_USER_REVIEWS_LIST=(1)

echo "========================================"
echo "Starting Claude API evaluation runs: $(date)"
echo "========================================"

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY environment variable not set"
    echo "Set it with: export ANTHROPIC_API_KEY='your-api-key-here'"
    exit 1
fi

# Iterate over models
for MODEL in "${MODELS[@]}"; do
    # Extract model name for output directory (e.g., "haiku", "sonnet", "opus")
    if [[ $MODEL == *"haiku"* ]]; then
        MODEL_NAME="haiku"
    elif [[ $MODEL == *"sonnet"* ]]; then
        MODEL_NAME="sonnet"
    elif [[ $MODEL == *"opus"* ]]; then
        MODEL_NAME="opus"
    else
        MODEL_NAME="claude"
    fi

    # Iterate over num_book_reviews
    for num_book_reviews in "${NUM_BOOK_REVIEWS_LIST[@]}"; do
        # Iterate over num_user_reviews
        for num_user_reviews in "${NUM_USER_REVIEWS_LIST[@]}"; do
            OUTPUT_DIR="${BASE_OUTPUT_DIR}/model_${MODEL_NAME}_num_book_reviews_${num_book_reviews}_num_user_reviews_${num_user_reviews}"

            echo ""
            echo "========================================"
            echo "Running: model=${MODEL}, num_book_reviews=${num_book_reviews}, num_user_reviews=${num_user_reviews}"
            echo "Output directory: ${OUTPUT_DIR}"
            echo "Start time: $(date)"
            echo "========================================"

            python util/run_claude_eval.py \
                --input "${INPUT_FILE}" \
                --output_dir "${OUTPUT_DIR}" \
                --template "${TEMPLATE}" \
                --model "${MODEL}" \
                --num_book_reviews ${num_book_reviews} \
                --num_user_reviews ${num_user_reviews} \
                --max_tokens ${MAX_TOKENS} \
                --concurrent_requests ${CONCURRENT_REQUESTS} \
                --random_seed ${RANDOM_SEED} \
                --num_entries ${NUM_ENTRIES} \
                --rate_limit_delay ${RATE_LIMIT_DELAY}

            echo "Completed: model=${MODEL}, num_book_reviews=${num_book_reviews}, num_user_reviews=${num_user_reviews}"
            echo "End time: $(date)"
        done
    done
done

echo ""
echo "========================================"
echo "All Claude evaluation runs completed: $(date)"
echo "========================================"
