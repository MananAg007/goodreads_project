#!/usr/bin/env python3
"""Evaluate the reasoning quality of LLM predictions that were correct."""

import anthropic
import argparse
import json
import os
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate reasoning quality of correct LLM predictions.")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/mananaga/goodreads/results/model_14B_num_book_reviews_8_num_user_reviews_1/raw_outputs.jsonl",
        help="Path to raw_outputs.jsonl file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for reasoning evaluation results (default: auto-generated in same directory as input)"
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
    return parser.parse_args()


def load_correct_predictions(input_path):
    """Load entries where ground_truth == predicted."""
    correct_entries = []
    with open(input_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["ground_truth"] == entry["predicted"] and entry["parse_success"]:
                correct_entries.append(entry)
    return correct_entries


def build_reasoning_evaluation_prompt(raw_input, raw_response):
    """Build a prompt asking Claude to evaluate the reasoning quality."""
    return f"""You are evaluating the reasoning quality of an LLM's book recommendation.

The LLM was asked to predict which of two books a user would prefer based on their past reviews and community reviews. The LLM's prediction was CORRECT (it matched the ground truth).

However, we want to know: Did the LLM arrive at the correct answer through sound reasoning, or was the reasoning flawed despite getting the right answer?

Below is the original input prompt given to the LLM, followed by the LLM's response.

=== ORIGINAL PROMPT TO LLM ===
{raw_input}

=== LLM'S RESPONSE ===
{raw_response}

=== YOUR TASK ===
Analyze the LLM's reasoning in the REASON section of its response. Answer these questions:

1. Is the reasoning logically sound and well-supported by the information provided?
2. Does the reasoning actually connect the user's preferences to the predicted book choice?
3. Are there any logical fallacies, unsupported claims, or flawed inferences?

Respond in this format:
REASONING_QUALITY: [SOUND or FLAWED]
EXPLANATION: [2-3 sentences explaining your assessment]

Be critical but fair. A response is SOUND if the reasoning logically connects the user's preferences to the book choice, even if it's brief. It's FLAWED if the reasoning is illogical, contradictory, or doesn't actually support the conclusion."""


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


def compute_summary_stats(results):
    """Compute summary statistics of reasoning quality."""
    total = len(results)
    sound = sum(1 for r in results if r["reasoning_quality"] == "SOUND")
    flawed = sum(1 for r in results if r["reasoning_quality"] == "FLAWED")
    error = sum(1 for r in results if r["reasoning_quality"] == "ERROR")

    return {
        "total_correct_predictions": total,
        "sound_reasoning": sound,
        "flawed_reasoning": flawed,
        "evaluation_errors": error,
        "sound_reasoning_rate": sound / total if total > 0 else 0,
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

    # Determine output path
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        output_path = os.path.join(input_dir, "reasoning_evaluation.jsonl")
    else:
        output_path = args.output

    print(f"Input: {args.input}")
    print(f"Output: {output_path}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Initialize Claude client
    client = anthropic.Anthropic(api_key=api_key)

    # Load correct predictions
    print(f"Loading correct predictions from {args.input}...")
    entries = load_correct_predictions(args.input)
    print(f"Found {len(entries)} correct predictions to evaluate")

    if len(entries) == 0:
        print("No correct predictions found. Exiting.")
        return

    # Evaluate reasoning
    print("Evaluating reasoning quality...")
    results = evaluate_reasoning_batch(client, entries, model=args.model, batch_size=args.batch_size)

    # Save results
    print(f"\nSaving results to {output_path}...")
    save_results(results, output_path)

    # Compute and display summary statistics
    stats = compute_summary_stats(results)
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total correct predictions evaluated: {stats['total_correct_predictions']}")
    print(f"Sound reasoning: {stats['sound_reasoning']} ({stats['sound_reasoning_rate']:.1%})")
    print(f"Flawed reasoning: {stats['flawed_reasoning']} ({stats['flawed_reasoning_rate']:.1%})")
    print(f"Evaluation errors: {stats['evaluation_errors']}")
    print("=" * 60)

    # Save summary stats
    summary_path = output_path.replace('.jsonl', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Summary statistics saved to {summary_path}")


if __name__ == "__main__":
    main()
