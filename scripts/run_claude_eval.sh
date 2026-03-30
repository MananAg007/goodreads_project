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
CONCURRENT_REQUESTS=10  # Number of parallel API requests (adjust based on rate limits)
TEMPERATURE=0.0  # 0 = deterministic, 1 = default randomness
NUM_ENTRIES=100
RATE_LIMIT_DELAY=0.01  # Very small delay for Build tier (1000 req/min). Use 0 to disable

# Define the values to iterate over
MODELS=("claude-3-haiku-20240307")
NUM_BOOK_REVIEWS_LIST=(8)
NUM_USER_REVIEWS_LIST=(1)
AVERAGE_RATINGS_LIST=("true")
REVIEWS_FILTER_MODE_LIST=("none" "most_popular" "least_popular")

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
            # Iterate over average_ratings_mode
            for avg_ratings_mode in "${AVERAGE_RATINGS_LIST[@]}"; do
                # Iterate over reviews_filter_mode
                for reviews_filter_mode in "${REVIEWS_FILTER_MODE_LIST[@]}"; do
                    OUTPUT_DIR="${BASE_OUTPUT_DIR}/model_${MODEL_NAME}_num_book_reviews_${num_book_reviews}_num_user_reviews_${num_user_reviews}_avg_ratings_${avg_ratings_mode}_filter_${reviews_filter_mode}"

                    echo ""
                    echo "========================================"
                    echo "Running: model=${MODEL}, num_book_reviews=${num_book_reviews}, num_user_reviews=${num_user_reviews}, avg_ratings=${avg_ratings_mode}, filter=${reviews_filter_mode}"
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
                        --temperature ${TEMPERATURE} \
                        --num_entries ${NUM_ENTRIES} \
                        --rate_limit_delay ${RATE_LIMIT_DELAY} \
                        --average_ratings_mode ${avg_ratings_mode} \
                        --reviews_filter_mode ${reviews_filter_mode}

                    echo "Completed: model=${MODEL}, num_book_reviews=${num_book_reviews}, num_user_reviews=${num_user_reviews}, avg_ratings=${avg_ratings_mode}, filter=${reviews_filter_mode}"
                    echo "End time: $(date)"
                done
            done
        done
    done
done

echo ""
echo "========================================"
echo "All Claude evaluation runs completed: $(date)"
echo "========================================"
