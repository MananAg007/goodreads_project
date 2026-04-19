"""LLM evaluation pipeline for pairwise book preference prediction.

Supports experiment flags for the personalization / prompt / subset ablations
described in the project plan:

  --user_context_mode     {none, ratings_only, full_reviews}
  --reference_selection   {first, random, tfidf}
  --community_selection   {first, random, top_upvoted}
  --prompt_mode           {answer_only, answer_plus_reason}
  --subset                {full, hard_rating_matched, low_popularity,
                           high_popularity, same_genre, swapped_user}
  --splits                path to JSON produced by scripts/build_splits.py
                          (required for --subset other than 'full')
"""

import argparse
import json
import math
import os
import random
import re
from collections import Counter

# torch / transformers are imported lazily inside the model-loading functions
# so that the prompt-construction utilities can be imported on login nodes.


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description = "Evaluate an LLM on book preference prediction.")
    parser.add_argument("--input", type = str, default = "data/book_preference_dataset.jsonl")
    parser.add_argument("--output_dir", type = str, required = True)
    parser.add_argument("--template", type = str, default = None,
                        help = "Explicit template path. If omitted, chosen from --prompt_mode.")
    parser.add_argument("--model", type = str, default = "Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_book_reviews", type = int, default = 5,
                        help = "Number of community reviews per candidate book (x).")
    parser.add_argument("--num_user_reviews", type = int, default = 5,
                        help = "Number of user reference reviews (y).")
    parser.add_argument("--max_new_tokens", type = int, default = 256)
    parser.add_argument("--batch_size", type = int, default = 4)
    parser.add_argument("--random_seed", type = int, default = 86)
    parser.add_argument("--device", type = str, default = "auto")
    parser.add_argument("--num_entries", type = int, default = None,
                        help = "Limit on number of entries to process (after subset filtering).")
    parser.add_argument("--debug", action = "store_true")

    # Experiment flags ------------------------------------------------------
    parser.add_argument("--user_context_mode", type = str, default = "full_reviews",
                        choices = ["none", "ratings_only", "full_reviews"])
    parser.add_argument("--reference_selection", type = str, default = "first",
                        choices = ["first", "random", "tfidf"])
    parser.add_argument("--community_selection", type = str, default = "first",
                        choices = ["first", "random", "top_upvoted"])
    parser.add_argument("--prompt_mode", type = str, default = "answer_plus_reason",
                        choices = ["answer_only", "answer_plus_reason"])
    parser.add_argument("--subset", type = str, default = "full",
                        choices = ["full", "hard_rating_matched", "low_popularity",
                                   "high_popularity", "same_genre", "swapped_user"])
    parser.add_argument("--splits", type = str, default = "data/splits/splits.json")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loading entries + splits
# ---------------------------------------------------------------------------

def load_entries(input_path):
    entries = []
    with open(input_path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def apply_subset(entries, subset, splits_path):
    """Return (kept_entries, kept_indices, swap_map).

    swap_map is a dict entry_idx_in_kept -> source_entry_idx_in_original for the
    swapped-user variant, or None otherwise.
    """
    if subset == "full":
        return entries, list(range(len(entries))), None

    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f"--splits file not found at {splits_path}. "
            f"Run `python scripts/build_splits.py` first."
        )
    with open(splits_path) as f:
        splits = json.load(f)

    if subset == "swapped_user":
        idxs = list(range(len(entries)))
    else:
        idxs = splits["subsets"][subset]

    kept = [entries[i] for i in idxs]

    swap_map = None
    if subset == "swapped_user":
        swap_pairs = {int(k): v for k, v in splits["swap_pairs"].items()}
        swap_map = {pos: swap_pairs[orig_idx] for pos, orig_idx in enumerate(idxs)}

    return kept, idxs, swap_map


# ---------------------------------------------------------------------------
# User-context construction
# ---------------------------------------------------------------------------

def build_user_review_map(entries):
    """Build a map from user_id to list of unique reference book reviews."""
    user_reviews = {}
    seen = set()
    for entry in entries:
        user_id = entry["user_id"]
        # Handle both old format (reference_book) and new format (reference_books)
        ref_books = entry.get("reference_books",
                              [entry.get("reference_book")] if "reference_book" in entry else [])
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
                "review_text": ref.get("review_text", "") or "",
            }
            user_reviews.setdefault(user_id, []).append(review)
    return user_reviews


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _tfidf_similarity(reviews, query_docs):
    """Score each review by TF-IDF cosine against the concatenation of query_docs.

    Simple pure-Python implementation to avoid scikit-learn dependency.
    """
    docs = [_tokenize(r["review_text"] + " " + r["title"]) for r in reviews]
    query_tokens = []
    for qd in query_docs:
        query_tokens.extend(_tokenize(qd))
    if not reviews:
        return []
    # Document frequency over the review pool plus the query as one extra doc.
    df = Counter()
    for doc in docs + [query_tokens]:
        for tok in set(doc):
            df[tok] += 1
    n_docs = len(docs) + 1
    def tfidf(tokens):
        tf = Counter(tokens)
        return {t: (c / max(len(tokens), 1)) * math.log((1 + n_docs) / (1 + df[t])) for t, c in tf.items()}
    q_vec = tfidf(query_tokens)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
    scores = []
    for doc in docs:
        d_vec = tfidf(doc)
        d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
        dot = sum(q_vec.get(t, 0.0) * v for t, v in d_vec.items())
        scores.append(dot / (q_norm * d_norm))
    return scores


def select_user_reviews(user_reviews, y, strategy, entry, rng):
    if not user_reviews or y <= 0:
        return []
    if strategy == "first":
        return user_reviews[:y]
    if strategy == "random":
        if len(user_reviews) <= y:
            return list(user_reviews)
        return rng.sample(user_reviews, y)
    if strategy == "tfidf":
        query = [
            entry["book_a"]["title"],
            entry["book_b"]["title"],
            entry["book_a"].get("genres") or "",
            entry["book_b"].get("genres") or "",
        ]
        sample_texts = [r["review_text"] for r in entry["book_a"]["sample_reviews"][:3]]
        sample_texts += [r["review_text"] for r in entry["book_b"]["sample_reviews"][:3]]
        query.extend(sample_texts)
        scores = _tfidf_similarity(user_reviews, query)
        ranked = sorted(range(len(user_reviews)), key = lambda i: scores[i], reverse = True)
        return [user_reviews[i] for i in ranked[:y]]
    raise ValueError(f"Unknown reference_selection: {strategy}")


def select_community_reviews(sample_reviews, x, strategy, rng):
    if x <= 0 or not sample_reviews:
        return []
    if strategy == "first":
        return sample_reviews[:x]
    if strategy == "random":
        if len(sample_reviews) <= x:
            return list(sample_reviews)
        return rng.sample(sample_reviews, x)
    if strategy == "top_upvoted":
        ranked = sorted(
            sample_reviews,
            key = lambda r: r.get("n_votes", 0) if r.get("n_votes") is not None else 0,
            reverse = True,
        )
        return ranked[:x]
    raise ValueError(f"Unknown community_selection: {strategy}")


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_user_reviews_block(reviews, mode):
    if mode == "none":
        return "(No past reviews provided for this user.)"
    if not reviews:
        return "(No past reviews available.)"
    parts = []
    for i, r in enumerate(reviews, start = 1):
        if mode == "ratings_only":
            parts.append(f'Review {i}: rated "{r["title"]}" {r["rating"]}/5')
        else:
            parts.append(f'Review {i} of "{r["title"]}" (rated {r["rating"]}/5): "{r["review_text"]}"')
    return "\n\n".join(parts)


def format_sample_reviews_block(sample_reviews):
    if not sample_reviews:
        return "(No community reviews available.)"
    parts = []
    for i, r in enumerate(sample_reviews, start = 1):
        parts.append(f'Review {i} (rated {r["rating"]}/5): "{r["review_text"]}"')
    return "\n\n".join(parts)


def build_prompt(template, entry, user_reviews_block, community_a, community_b):
    return template.format_map({
        "user_reviews_block": user_reviews_block,
        "book_a_title": entry["book_a"]["title"],
        "book_b_title": entry["book_b"]["title"],
        "book_a_reviews_block": format_sample_reviews_block(community_a),
        "book_b_reviews_block": format_sample_reviews_block(community_b),
    })


def resolve_template_path(prompt_mode, explicit):
    if explicit:
        return explicit
    if prompt_mode == "answer_only":
        return "util/templates/answer_only_prompt.txt"
    return "util/templates/strict_format_prompt.txt"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
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
    import torch
    inputs = tokenizer(
        prompts,
        return_tensors = "pt",
        padding = True,
        truncation = True,
        max_length = 8192,  # Increased from 3072 to accommodate more reviews
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Log token lengths to detect truncation
    token_lengths = [len(ids) for ids in inputs["input_ids"]]
    max_tokens = max(token_lengths)
    if max_tokens >= 8192:
        print(f"  WARNING: Truncation occurring! Max tokens in batch: {max_tokens}")

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
    match = re.search(r"ANSWER:\s*\[?(A|B)\]?", raw_response, re.IGNORECASE)
    if match:
        return match.group(1).upper(), True
    return None, False


# ---------------------------------------------------------------------------
# Metrics + I/O
# ---------------------------------------------------------------------------

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
    """Save all results (both parsed and unparsed) to the output file."""
    path = os.path.join(output_dir, "raw_outputs.jsonl")
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    parsed_count = sum(1 for r in results if r["parse_success"])
    failed_count = len(results) - parsed_count
    print(f"  Saved {len(results)} total results ({parsed_count} parsed, {failed_count} parse failed)")


def save_metrics(metrics, output_dir):
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent = 2)


def save_config(args, output_dir):
    path = os.path.join(output_dir, "config.json")
    with open(path, "w") as f:
        json.dump(vars(args), f, indent = 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import torch
    args = parse_args()
    torch.manual_seed(args.random_seed)
    rng = random.Random(args.random_seed)

    template_path = resolve_template_path(args.prompt_mode, args.template)
    with open(template_path) as f:
        template = f.read()

    all_entries = load_entries(args.input)
    entries, orig_indices, swap_map = apply_subset(all_entries, args.subset, args.splits)

    if args.num_entries is not None:
        entries = entries[:args.num_entries]
        orig_indices = orig_indices[:args.num_entries]
        if swap_map is not None:
            swap_map = {k: v for k, v in swap_map.items() if k < args.num_entries}

    print(f"Subset={args.subset}: processing {len(entries)}/{len(all_entries)} entries")

    # Build user-review map over the FULL dataset so swap_map can index any user.
    full_user_review_map = build_user_review_map(all_entries)

    prompts = []
    effective_user_ids = []
    for pos, entry in enumerate(entries):
        if swap_map is not None:
            src_entry = all_entries[swap_map[pos]]
            source_user_id = src_entry["user_id"]
        else:
            source_user_id = entry["user_id"]
        effective_user_ids.append(source_user_id)

        user_reviews_all = full_user_review_map.get(source_user_id, [])
        selected_user_reviews = select_user_reviews(
            user_reviews_all, args.num_user_reviews, args.reference_selection, entry, rng,
        )
        user_block = format_user_reviews_block(selected_user_reviews, args.user_context_mode)

        community_a = select_community_reviews(
            entry["book_a"]["sample_reviews"], args.num_book_reviews,
            args.community_selection, rng,
        )
        community_b = select_community_reviews(
            entry["book_b"]["sample_reviews"], args.num_book_reviews,
            args.community_selection, rng,
        )

        prompts.append(build_prompt(template, entry, user_block, community_a, community_b))

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    wrapped_prompts = [apply_chat_wrapper(p, tokenizer) for p in prompts]

    os.makedirs(args.output_dir, exist_ok = True)
    save_config(args, args.output_dir)

    results = []
    for batch_start in range(0, len(wrapped_prompts), args.batch_size):
        batch = wrapped_prompts[batch_start : batch_start + args.batch_size]
        if args.debug:
            for i, prompt in enumerate(batch):
                print(f"\n{'=' * 60}\n[DEBUG] Entry {batch_start + i} prompt:\n{'=' * 60}\n{prompt}\n{'=' * 60}\n")
        raw_outputs = run_batch_inference(batch, model, tokenizer, args.max_new_tokens, args.device)
        for i, raw_response in enumerate(raw_outputs):
            pos = batch_start + i
            entry = entries[pos]
            predicted, parse_success = parse_response(raw_response)
            results.append({
                "entry_idx": orig_indices[pos],
                "user_id": entry["user_id"],
                "effective_user_id": effective_user_ids[pos],
                "book_a_id": entry["book_a"]["book_id"],
                "book_b_id": entry["book_b"]["book_id"],
                "book_a_title": entry["book_a"]["title"],
                "book_b_title": entry["book_b"]["title"],
                "ground_truth": entry["preferred"],
                "predicted": predicted,
                "rating_difference": entry["rating_difference"],
                "raw_input": wrapped_prompts[pos],
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
