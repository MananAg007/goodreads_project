"""LLM evaluation pipeline for book preference prediction."""

import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an LLM on book preference prediction.")
    parser.add_argument("--input", type = str, default = "util/book_preference_dataset.jsonl")
    parser.add_argument("--output_dir", type = str, required = True)
    parser.add_argument("--template", type = str, default = "util/templates/default_prompt.txt")
    parser.add_argument("--model", type = str, default = "Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_book_reviews", type = int, default = 5)
    parser.add_argument("--num_user_reviews", type = int, default = 1)
    parser.add_argument("--max_new_tokens", type = int, default = 256)
    parser.add_argument("--batch_size", type = int, default = 4)
    parser.add_argument("--random_seed", type = int, default = 86)
    parser.add_argument("--device", type = str, default = "auto")
    parser.add_argument("--num_entries", type = int, default = None, help = "Number of entries to process (default: all)")
    parser.add_argument("--debug", action = "store_true")
    return parser.parse_args()


def load_entries(input_path):
    entries = []
    with open(input_path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def build_user_review_map(entries):
    """Build a map from user_id to list of unique reference book reviews."""
    user_reviews = {}
    seen = set()
    for entry in entries:
        user_id = entry["user_id"]
        ref = entry["reference_book"]
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


def format_user_reviews_block(reviews, y):
    selected = reviews[:y]
    parts = []
    for i, r in enumerate(selected, start = 1):
        parts.append(f'Review {i} of "{r["title"]}" (rated {r["rating"]}/5): "{r["review_text"]}"')
    return "\n\n".join(parts)


def format_sample_reviews_block(sample_reviews, x):
    selected = sample_reviews[:x]
    parts = []
    for i, r in enumerate(selected, start = 1):
        parts.append(f'Review {i} (rated {r["rating"]}/5): "{r["review_text"]}"')
    return "\n\n".join(parts)


def build_prompt(template, entry, user_reviews, x, y):
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


def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
    # Set left padding for decoder-only models (required for correct batch inference)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.float16,
        device_map = device,
    )
    model.eval()
    return model, tokenizer


def apply_chat_wrapper(prompt, tokenizer):
    if tokenizer.chat_template is not None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    return prompt


def run_batch_inference(prompts, model, tokenizer, max_new_tokens, device):
    inputs = tokenizer(
        prompts,
        return_tensors = "pt",
        padding = True,
        truncation = True,
        max_length = 3072,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens,
            do_sample = False,
            pad_token_id = tokenizer.pad_token_id,
        )

    decoded = []
    for i, output in enumerate(outputs):
        new_tokens = output[input_ids.shape[1]:]
        decoded.append(tokenizer.decode(new_tokens, skip_special_tokens = True))
    return decoded


def parse_response(raw_response):
    match = re.search(r"ANSWER:\s*(A|B)", raw_response, re.IGNORECASE)
    if match:
        return match.group(1).upper(), True
    return None, False


def compute_metrics(results):
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


def save_raw_outputs(results, output_dir):
    """Save only successfully parsed results to the output file."""
    path = os.path.join(output_dir, "raw_outputs.jsonl")
    # Filter to only parsed results
    parsed_results = [r for r in results if r["parse_success"]]

    with open(path, "w") as f:
        for r in parsed_results:
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
            }) + "\n")

    print(f"  Saved {len(parsed_results)} successfully parsed results (out of {len(results)} total)")


def save_metrics(metrics, output_dir):
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent = 2)


def main():
    args = parse_args()
    torch.manual_seed(args.random_seed)

    with open(args.template) as f:
        template = f.read()

    entries = load_entries(args.input)
    if args.num_entries is not None:
        entries = entries[:args.num_entries]
        print(f"Processing {len(entries)} entries (limited by --num_entries)")
    else:
        print(f"Processing all {len(entries)} entries")
    user_review_map = build_user_review_map(entries)

    prompts = []
    for entry in entries:
        user_reviews = user_review_map[entry["user_id"]]
        prompts.append(build_prompt(template, entry, user_reviews, args.num_book_reviews, args.num_user_reviews))

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    wrapped_prompts = [apply_chat_wrapper(p, tokenizer) for p in prompts]

    os.makedirs(args.output_dir, exist_ok = True)

    results = []
    for batch_start in range(0, len(wrapped_prompts), args.batch_size):
        batch = wrapped_prompts[batch_start : batch_start + args.batch_size]
        if args.debug:
            for i, prompt in enumerate(batch):
                print(f"\n{'=' * 60}\n[DEBUG] Entry {batch_start + i} prompt:\n{'=' * 60}\n{prompt}\n{'=' * 60}\n")
        raw_outputs = run_batch_inference(batch, model, tokenizer, args.max_new_tokens, args.device)
        for i, raw_response in enumerate(raw_outputs):
            idx = batch_start + i
            entry = entries[idx]
            predicted, parse_success = parse_response(raw_response)
            results.append({
                "entry_idx": idx,
                "user_id": entry["user_id"],
                "book_a_title": entry["book_a"]["title"],
                "book_b_title": entry["book_b"]["title"],
                "ground_truth": entry["preferred"],
                "predicted": predicted,
                "rating_difference": entry["rating_difference"],
                "raw_input": wrapped_prompts[idx],
                "raw_response": raw_response,
                "parse_success": parse_success,
            })
        print(f"Processed {min(batch_start + args.batch_size, len(entries))}/{len(entries)} entries")

    save_raw_outputs(results, args.output_dir)
    metrics = compute_metrics(results)
    save_metrics(metrics, args.output_dir)

    print(f"Accuracy:         {metrics['n_correct']}/{metrics['n_parsed']}")
    print(f"Parsed:           {metrics['n_parsed']}/{metrics['n_total']}")


if __name__ == "__main__":
    main()