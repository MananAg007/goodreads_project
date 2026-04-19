#!/bin/bash
#SBATCH --job-name=gr_llm_sweep
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --output=logs/llm_sweep_%A_%a.out
#SBATCH --error=logs/llm_sweep_%A_%a.err
#SBATCH --array=0-15

# Array sweep over the LLM experiment axes and both prompt variants for
# Qwen2.5-14B-Instruct. Each array index launches one independent run; SLURM
# schedules them in parallel. Re-launch a subset with --array=0,3,5 on sbatch.
#
# Each config is "subset|user_context_mode|reference_selection|community_selection|tag".
# The prompt variant is derived from the array index:
#   0-7   -> answer_only
#   8-15  -> answer_reason
# The swapped_user task re-uses the top-line personalized config as a
# falsification test: accuracy should drop if the model is actually using the
# user context.

set -euo pipefail
mkdir -p logs runs

cd "$SLURM_SUBMIT_DIR"
# Keep each array task on node-local storage to avoid shared-cache contention.
export HF_HOME="/tmp/hf_cache_${SLURM_ARRAY_TASK_ID}/"
# shellcheck disable=SC1091
source scripts/slurm/setup_env.sh

MODEL="Qwen/Qwen2.5-14B-Instruct"
PROMPT_FILES=(
    "util/templates/answer_only_prompt.txt"
    "util/templates/strict_format_prompt.txt"
)
PROMPT_TAGS=(
    "answer_only"
    "answer_reason"
)

declare -a CONFIGS=(
    "full|none|random|random|baseline_no_user"
    "full|ratings_only|random|random|ratings_only"
    "full|full_reviews|random|random|full_reviews_random_ref"
    "full|full_reviews|tfidf|random|full_reviews_tfidf_ref"
    "full|full_reviews|random|random|full_reviews_random_comm"
    "full|full_reviews|random|top_upvoted|full_reviews_top_comm"
    "full|full_reviews|tfidf|top_upvoted|full_reviews_tfidf_top"
    "swapped_user|full_reviews|tfidf|top_upvoted|swapped_user_tfidf_top"
)

N_CONFIGS=${#CONFIGS[@]}
prompt_idx=$((SLURM_ARRAY_TASK_ID / N_CONFIGS))
cfg_idx=$((SLURM_ARRAY_TASK_ID % N_CONFIGS))
template="${PROMPT_FILES[$prompt_idx]}"
prompt_tag="${PROMPT_TAGS[$prompt_idx]}"

cfg="${CONFIGS[$cfg_idx]}"
IFS='|' read -r subset uctx refsel commsel tag <<< "$cfg"
out="runs/${tag}_${prompt_tag}"

echo "========================================"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL"
echo "Template: $template"
echo "Config: subset=$subset user_context=$uctx ref_sel=$refsel comm_sel=$commsel"
echo "Output: $out"
echo "========================================"

"$PY" util/run_llm_eval.py \
    --model "$MODEL" \
    --template "$template" \
    --subset "$subset" \
    --splits data/splits/splits.json \
    --user_context_mode "$uctx" \
    --reference_selection "$refsel" \
    --community_selection "$commsel" \
    --num_book_reviews 8 \
    --num_user_reviews 1 \
    --output_dir "$out"

echo "[$(date)] done: ${tag}_${prompt_tag}"
