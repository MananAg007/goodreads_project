#!/usr/bin/env python3
"""Claude API evaluation pipeline for book preference prediction."""

import anthropic
import argparse
import asyncio
import json
import os
import re
import time
from typing import List, Dict, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Claude API on book preference prediction.")
    parser.add_argument("--input", type=str, default="data/book_preference_dataset.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--template", type=str, default="util/templates/strict_format_prompt.txt")
    parser.add_argument("--model", type=str, default="claude-3-haiku-20240307",
                        help="Claude model (e.g., claude-3-haiku-20240307, claude-3-5-sonnet-20240620)")
    parser.add_argument("--num_book_reviews", type=int, default=8)
    parser.add_argument("--num_user_reviews", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Max tokens in response (not max_new_tokens)")
    parser.add_argument("--concurrent_requests", type=int, default=5,
                        help="Number of concurrent API requests (respect rate limits)")
    parser.add_argument("--random_seed", type=int, default=86)
    parser.add_argument("--num_entries", type=int, default=None,
                        help="Number of entries to process (default: all)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rate_limit_delay", type=float, default=0.1,
                        help="Delay between batches of requests in seconds")
    return parser.parse_args()


def load_entries(input_path: str) -> List[Dict]:
    """Load dataset entries from JSONL file."""
    entries = []
    with open(input_path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def build_user_review_map(entries: List[Dict]) -> Dict[str, List[Dict]]:
    """Build a map from user_id to list of unique reference book reviews."""
    user_reviews = {}
    seen = set()
    for entry in entries:
        user_id = entry["user_id"]
        # Handle both old format (reference_book) and new format (reference_books)
        ref_books = entry.get("reference_books", [entry.get("reference_book")] if "reference_book" in entry else [])

        for ref in ref_books:
            if ref is None:
                continue
            key = (user_id, ref["book_id"])
            if key in seen:
                continue
            seen.add(key)
            review = {
                "title": ref["title"],
                "rating": ref["rating"],
                "review_text": ref["review_text"],
            }
            if user_id not in user_reviews:
                user_reviews[user_id] = []
            user_reviews[user_id].append(review)
    return user_reviews


def format_user_reviews_block(reviews: List[Dict], y: int) -> str:
    """Format user's past reviews."""
    selected = reviews[:y]
    parts = []
    for i, r in enumerate(selected, start=1):
        parts.append(f'Review {i} of "{r["title"]}" (rated {r["rating"]}/5): "{r["review_text"]}"')
    return "\n\n".join(parts)


def format_sample_reviews_block(sample_reviews: List[Dict], x: int) -> str:
    """Format community reviews for a book."""
    selected = sample_reviews[:x]
    parts = []
    for i, r in enumerate(selected, start=1):
        parts.append(f'Review {i} (rated {r["rating"]}/5): "{r["review_text"]}"')
    return "\n\n".join(parts)


def build_prompt(template: str, entry: Dict, user_reviews: List[Dict], x: int, y: int) -> str:
    """Build the prompt for a single entry."""
    user_reviews_block = format_user_reviews_block(user_reviews, y)
    book_a_reviews_block = format_sample_reviews_block(entry["book_a"]["sample_reviews"], x)
    book_b_reviews_block = format_sample_reviews_block(entry["book_b"]["sample_reviews"], x)
    return template.format_map({
        "user_reviews_block": user_reviews_block,
        "book_a_title": entry["book_a"]["title"],
        "book_b_title": entry["book_b"]["title"],
        "book_a_reviews_block": book_a_reviews_block,
        "book_b_reviews_block": book_b_reviews_block,
    })


async def call_claude_api(client: anthropic.AsyncAnthropic, prompt: str, model: str, max_tokens: int) -> Tuple[str, bool]:
    """Make an async call to Claude API."""
    try:
        message = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response_text = message.content[0].text
        return response_text, True
    except Exception as e:
        print(f"\nAPI Error: {e}")
        return f"ERROR: {str(e)}", False


async def process_batch(client: anthropic.AsyncAnthropic, prompts: List[str], model: str,
                       max_tokens: int, start_idx: int) -> List[Tuple[int, str, bool]]:
    """Process a batch of prompts concurrently."""
    tasks = []
    for i, prompt in enumerate(prompts):
        task = call_claude_api(client, prompt, model, max_tokens)
        tasks.append((start_idx + i, task))

    results = []
    for idx, task in tasks:
        response, success = await task
        results.append((idx, response, success))

    return results


def parse_response(raw_response: str) -> Tuple[str, bool]:
    """Parse the model response to extract the answer."""
    match = re.search(r"ANSWER:\s*\[?(A|B)\]?", raw_response, re.IGNORECASE)
    if match:
        return match.group(1).upper(), True
    return None, False


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute accuracy metrics."""
    parsed = [r for r in results if r["parse_success"]]
    n_total = len(results)
    n_parsed = len(parsed)
    n_parse_failed = n_total - n_parsed

    if n_parsed == 0:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_total": n_total,
            "n_parsed": n_parsed,
            "n_parse_failed": n_parse_failed,
        }

    n_correct = sum(1 for r in parsed if r["predicted"] == r["ground_truth"])
    accuracy = n_correct / n_parsed

    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
        "n_parsed": n_parsed,
        "n_parse_failed": n_parse_failed,
    }


def save_raw_outputs(results: List[Dict], output_dir: str):
    """Save all results to JSONL file."""
    path = os.path.join(output_dir, "raw_outputs.jsonl")

    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "entry_idx": r["entry_idx"],
                "user_id": r["user_id"],
                "book_a_title": r["book_a_title"],
                "book_b_title": r["book_b_title"],
                "ground_truth": r["ground_truth"],
                "predicted": r["predicted"],
                "rating_difference": r["rating_difference"],
                "raw_input": r["raw_input"],
                "raw_response": r["raw_response"],
                "parse_success": r["parse_success"],
                "api_success": r["api_success"],
            }) + "\n")

    parsed_count = sum(1 for r in results if r["parse_success"])
    api_failed = sum(1 for r in results if not r["api_success"])
    print(f"  Saved {len(results)} results ({parsed_count} parsed, {api_failed} API errors)")


def save_metrics(metrics: Dict, output_dir: str):
    """Save metrics to JSON file."""
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


async def main():
    args = parse_args()

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-api-key-here'")
        return

    # Load template
    with open(args.template) as f:
        template = f.read()

    # Load and prepare data
    entries = load_entries(args.input)
    if args.num_entries is not None:
        entries = entries[:args.num_entries]
        print(f"Processing {len(entries)} entries (limited by --num_entries)")
    else:
        print(f"Processing all {len(entries)} entries")

    user_review_map = build_user_review_map(entries)

    # Build all prompts
    print("Building prompts...")
    prompts = []
    for entry in entries:
        user_reviews = user_review_map[entry["user_id"]]
        prompt = build_prompt(template, entry, user_reviews, args.num_book_reviews, args.num_user_reviews)
        prompts.append(prompt)

    if args.debug and len(prompts) > 0:
        print(f"\n{'=' * 60}\n[DEBUG] Sample prompt (entry 0):\n{'=' * 60}\n{prompts[0]}\n{'=' * 60}\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Claude client
    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Process all prompts with concurrent requests
    print(f"Processing with {args.concurrent_requests} concurrent requests...")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")

    results = []
    batch_size = args.concurrent_requests

    start_time = time.time()

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]

        # Process batch concurrently
        batch_results = await process_batch(
            client, batch_prompts, args.model, args.max_tokens, batch_start
        )

        # Parse and store results
        for idx, raw_response, api_success in batch_results:
            entry = entries[idx]
            predicted, parse_success = parse_response(raw_response) if api_success else (None, False)

            results.append({
                "entry_idx": idx,
                "user_id": entry["user_id"],
                "book_a_title": entry["book_a"]["title"],
                "book_b_title": entry["book_b"]["title"],
                "ground_truth": entry["preferred"],
                "predicted": predicted,
                "rating_difference": entry["rating_difference"],
                "raw_input": prompts[idx],
                "raw_response": raw_response,
                "parse_success": parse_success,
                "api_success": api_success,
            })

        processed = min(batch_start + batch_size, len(entries))
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"Processed {processed}/{len(entries)} entries ({rate:.1f} entries/sec)")

        # Rate limiting delay between batches
        if batch_start + batch_size < len(prompts):
            await asyncio.sleep(args.rate_limit_delay)

    # Save results
    print("\nSaving results...")
    save_raw_outputs(results, args.output_dir)

    metrics = compute_metrics(results)
    save_metrics(metrics, args.output_dir)

    # Print summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total time:       {total_time:.1f}s ({len(entries)/total_time:.2f} entries/sec)")
    print(f"Accuracy:         {metrics['n_correct']}/{metrics['n_parsed']} = {metrics['accuracy']:.2%}")
    print(f"Parsed:           {metrics['n_parsed']}/{metrics['n_total']}")
    print(f"Parse failed:     {metrics['n_parse_failed']}")
    api_failed = sum(1 for r in results if not r["api_success"])
    print(f"API errors:       {api_failed}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
