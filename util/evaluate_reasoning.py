#!/usr/bin/env python3
"""Evaluate the reasoning quality of LLM predictions for both correct and incorrect traces."""

import anthropic
import argparse
import json
import os
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate reasoning quality of LLM predictions (correct and incorrect separately).")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/mananaga/goodreads/results/model_14B_num_book_reviews_8_num_user_reviews_1/raw_outputs.jsonl",
        help="Path to raw_outputs.jsonl file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for reasoning evaluation results (default: same directory as input)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size for processing (default: 5)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-haiku-20240307",
        help="Claude model to use for evaluation (default: claude-3-haiku-20240307)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Maximum number of samples to evaluate for each category (correct/incorrect) (default: 10)"
    )
    return parser.parse_args()


def load_predictions(input_path, max_samples=10):
    """Load and separate entries into correct and incorrect predictions."""
    correct_entries = []
    incorrect_entries = []
    with open(input_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["parse_success"]:
                if entry["ground_truth"] == entry["predicted"]:
                    if len(correct_entries) < max_samples:
                        correct_entries.append(entry)
                else:
                    if len(incorrect_entries) < max_samples:
                        incorrect_entries.append(entry)

            # Stop reading if we have enough samples of both types
            if len(correct_entries) >= max_samples and len(incorrect_entries) >= max_samples:
                break

    return correct_entries, incorrect_entries


def build_reasoning_evaluation_prompt(raw_input, raw_response):
    """Build a prompt asking Claude to evaluate the reasoning quality without revealing correctness."""
    # Remove the ground truth answer from the input prompt to hide it from the evaluator
    # Find and remove the line that shows which book the user actually preferred
    input_lines = raw_input.split('\n')
    filtered_lines = []
    for line in input_lines:
        # Skip lines that reveal the ground truth
        if 'actually preferred' in line.lower() or 'ground truth' in line.lower():
            continue
        filtered_lines.append(line)
    cleaned_input = '\n'.join(filtered_lines)

    return f"""You are a strict evaluator of reasoning quality for an LLM's book recommendation task.

The LLM was asked to predict which of two books a user would prefer based on their past reviews and community reviews.

Your task is to evaluate the quality of the LLM's reasoning, WITHOUT knowing whether the prediction was correct or not.

Below is the original input prompt given to the LLM, followed by the LLM's response.

=== ORIGINAL PROMPT TO LLM ===
{cleaned_input}

=== LLM'S RESPONSE ===
{raw_response}

=== YOUR TASK ===
Critically analyze the LLM's reasoning in the REASON section. Be STRICT in your evaluation.

Classify the reasoning into ONE of three categories:

**SOUND**: The reasoning must meet ALL these criteria:
- Makes specific references to the user's actual review content or preferences
- Draws clear, logical connections between user preferences and the chosen book
- Supports claims with concrete evidence from the provided reviews
- Demonstrates genuine understanding of what the user values in books
- No logical fallacies or contradictions

**VAGUE**: The reasoning has significant weaknesses:
- Makes generic or superficial observations that could apply to many books
- Uses vague language without specific evidence ("seems like", "probably", "might enjoy")
- Makes assumptions not clearly supported by the user's reviews
- Mentions themes or preferences without showing they actually appear in the user's reviews
- Provides only surface-level analysis
- Makes claims that are partially supported but not thoroughly justified

**FLAWED**: The reasoning is seriously deficient:
- Contains logical contradictions or fallacies
- Makes claims directly contradicted by the provided reviews
- Ignores important evidence from the user's reviews
- Draws conclusions that don't follow from the evidence
- Fabricates preferences or patterns not present in the reviews
- Reasoning is incoherent or doesn't actually support the conclusion

Respond in this EXACT format:
REASONING_QUALITY: [SOUND or VAGUE or FLAWED]
EXPLANATION: [2-4 sentences explaining your assessment with specific examples from the reasoning]

BE STRICT. Most reasoning should fall into VAGUE or FLAWED categories. Only exceptionally well-justified reasoning with specific evidence should be rated SOUND."""


def evaluate_reasoning_batch(client, entries, model="claude-3-haiku-20240307", batch_size=5):
    """Evaluate reasoning for a batch of entries."""
    results = []

    for i in tqdm(range(0, len(entries), batch_size), desc="Evaluating reasoning"):
        batch = entries[i:i+batch_size]

        for entry in batch:
            prompt = build_reasoning_evaluation_prompt(
                entry["raw_input"],
                entry["raw_response"]
            )

            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                evaluation_text = response.content[0].text

                # Parse the response
                reasoning_quality = None
                explanation = None

                for line in evaluation_text.split('\n'):
                    if line.startswith('REASONING_QUALITY:'):
                        reasoning_quality = line.split(':', 1)[1].strip()
                    elif line.startswith('EXPLANATION:'):
                        explanation = line.split(':', 1)[1].strip()

                results.append({
                    "entry_idx": entry["entry_idx"],
                    "user_id": entry["user_id"],
                    "book_a_title": entry["book_a_title"],
                    "book_b_title": entry["book_b_title"],
                    "ground_truth": entry["ground_truth"],
                    "predicted": entry["predicted"],
                    "original_reason": entry["raw_response"],
                    "reasoning_quality": reasoning_quality,
                    "explanation": explanation,
                    "full_evaluation": evaluation_text
                })

            except Exception as e:
                print(f"\nError evaluating entry {entry['entry_idx']}: {e}")
                results.append({
                    "entry_idx": entry["entry_idx"],
                    "user_id": entry["user_id"],
                    "book_a_title": entry["book_a_title"],
                    "book_b_title": entry["book_b_title"],
                    "ground_truth": entry["ground_truth"],
                    "predicted": entry["predicted"],
                    "original_reason": entry["raw_response"],
                    "reasoning_quality": "ERROR",
                    "explanation": str(e),
                    "full_evaluation": None
                })

    return results


def save_results(results, output_path):
    """Save evaluation results to a JSONL file."""
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


def compute_summary_stats(results, prediction_type=""):
    """Compute summary statistics of reasoning quality."""
    total = len(results)
    sound = sum(1 for r in results if r["reasoning_quality"] == "SOUND")
    vague = sum(1 for r in results if r["reasoning_quality"] == "VAGUE")
    flawed = sum(1 for r in results if r["reasoning_quality"] == "FLAWED")
    error = sum(1 for r in results if r["reasoning_quality"] == "ERROR")

    return {
        "prediction_type": prediction_type,
        "total_predictions": total,
        "sound_reasoning": sound,
        "vague_reasoning": vague,
        "flawed_reasoning": flawed,
        "evaluation_errors": error,
        "sound_reasoning_rate": sound / total if total > 0 else 0,
        "vague_reasoning_rate": vague / total if total > 0 else 0,
        "flawed_reasoning_rate": flawed / total if total > 0 else 0
    }


def main():
    args = parse_args()

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-api-key-here'")
        return

    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(args.input)
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    print(f"Input: {args.input}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max samples per category: {args.max_samples}")
    print()

    # Initialize Claude client
    client = anthropic.Anthropic(api_key=api_key)

    # Load predictions
    print(f"Loading predictions from {args.input}...")
    correct_entries, incorrect_entries = load_predictions(args.input, max_samples=args.max_samples)
    print(f"Loaded {len(correct_entries)} correct predictions and {len(incorrect_entries)} incorrect predictions")

    if len(correct_entries) == 0 and len(incorrect_entries) == 0:
        print("No predictions found. Exiting.")
        return

    # Process correct predictions
    if len(correct_entries) > 0:
        print("\n" + "=" * 60)
        print("EVALUATING CORRECT PREDICTIONS")
        print("=" * 60)
        correct_results = evaluate_reasoning_batch(client, correct_entries, model=args.model, batch_size=args.batch_size)

        # Save correct results
        correct_output_path = os.path.join(output_dir, "reasoning_evaluation_correct.jsonl")
        print(f"\nSaving correct prediction results to {correct_output_path}...")
        save_results(correct_results, correct_output_path)

        # Compute and display summary statistics for correct predictions
        correct_stats = compute_summary_stats(correct_results, "correct")
        print("\n" + "=" * 60)
        print("CORRECT PREDICTIONS - SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total correct predictions evaluated: {correct_stats['total_predictions']}")
        print(f"Sound reasoning: {correct_stats['sound_reasoning']} ({correct_stats['sound_reasoning_rate']:.1%})")
        print(f"Vague reasoning: {correct_stats['vague_reasoning']} ({correct_stats['vague_reasoning_rate']:.1%})")
        print(f"Flawed reasoning: {correct_stats['flawed_reasoning']} ({correct_stats['flawed_reasoning_rate']:.1%})")
        print(f"Evaluation errors: {correct_stats['evaluation_errors']}")
        print("=" * 60)

        # Save summary stats
        correct_summary_path = os.path.join(output_dir, "reasoning_evaluation_correct_summary.json")
        with open(correct_summary_path, 'w') as f:
            json.dump(correct_stats, f, indent=2)
        print(f"Summary statistics saved to {correct_summary_path}")

    # Process incorrect predictions
    if len(incorrect_entries) > 0:
        print("\n" + "=" * 60)
        print("EVALUATING INCORRECT PREDICTIONS")
        print("=" * 60)
        incorrect_results = evaluate_reasoning_batch(client, incorrect_entries, model=args.model, batch_size=args.batch_size)

        # Save incorrect results
        incorrect_output_path = os.path.join(output_dir, "reasoning_evaluation_incorrect.jsonl")
        print(f"\nSaving incorrect prediction results to {incorrect_output_path}...")
        save_results(incorrect_results, incorrect_output_path)

        # Compute and display summary statistics for incorrect predictions
        incorrect_stats = compute_summary_stats(incorrect_results, "incorrect")
        print("\n" + "=" * 60)
        print("INCORRECT PREDICTIONS - SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total incorrect predictions evaluated: {incorrect_stats['total_predictions']}")
        print(f"Sound reasoning: {incorrect_stats['sound_reasoning']} ({incorrect_stats['sound_reasoning_rate']:.1%})")
        print(f"Vague reasoning: {incorrect_stats['vague_reasoning']} ({incorrect_stats['vague_reasoning_rate']:.1%})")
        print(f"Flawed reasoning: {incorrect_stats['flawed_reasoning']} ({incorrect_stats['flawed_reasoning_rate']:.1%})")
        print(f"Evaluation errors: {incorrect_stats['evaluation_errors']}")
        print("=" * 60)

        # Save summary stats
        incorrect_summary_path = os.path.join(output_dir, "reasoning_evaluation_incorrect_summary.json")
        with open(incorrect_summary_path, 'w') as f:
            json.dump(incorrect_stats, f, indent=2)
        print(f"Summary statistics saved to {incorrect_summary_path}")

    # Print overall summary
    if len(correct_entries) > 0 and len(incorrect_entries) > 0:
        print("\n" + "=" * 60)
        print("OVERALL COMPARISON")
        print("=" * 60)
        print(f"\nCorrect Predictions:")
        print(f"  Sound:  {correct_stats['sound_reasoning']} ({correct_stats['sound_reasoning_rate']:.1%})")
        print(f"  Vague:  {correct_stats['vague_reasoning']} ({correct_stats['vague_reasoning_rate']:.1%})")
        print(f"  Flawed: {correct_stats['flawed_reasoning']} ({correct_stats['flawed_reasoning_rate']:.1%})")
        print(f"\nIncorrect Predictions:")
        print(f"  Sound:  {incorrect_stats['sound_reasoning']} ({incorrect_stats['sound_reasoning_rate']:.1%})")
        print(f"  Vague:  {incorrect_stats['vague_reasoning']} ({incorrect_stats['vague_reasoning_rate']:.1%})")
        print(f"  Flawed: {incorrect_stats['flawed_reasoning']} ({incorrect_stats['flawed_reasoning_rate']:.1%})")
        print("=" * 60)


if __name__ == "__main__":
    main()
