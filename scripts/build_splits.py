"""
Build analysis subsets for the pairwise preference dataset.

Produces a single JSON file (data/splits/splits.json) containing:
- subsets: named entry-index lists for hardness / popularity / genre analyses
- swap_pairs: entry i -> entry j where j's user context is spliced in for the
              swapped-user personalization sanity check (a derangement)
- meta: seeds, dataset path, hash

There is no train/test split: the LLM has no trainable parameters here and
prompts/selection strategies are frozen, so the full dataset is evaluated as
one test set.
"""

import argparse
import hashlib
import json
import os
import random


def load_entries(path):
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def popularity_score(entry):
    # Use total n_votes across sampled community reviews of A and B as popularity proxy.
    # sample_reviews are capped at a fixed number in dataset construction, so the
    # number of reviews is not discriminative -- summed n_votes is.
    def book_score(book):
        return sum(int(r["n_votes"]) for r in book["sample_reviews"] if "n_votes" in r)
    return book_score(entry["book_a"]) + book_score(entry["book_b"])


def build_subsets(entries, rating_diff_thresh, popularity_quantile):
    hard_rating_matched = []
    same_genre = []
    for i, e in enumerate(entries):
        avg_a = e["book_a"]["average_rating"]
        avg_b = e["book_b"]["average_rating"]
        if avg_a is not None and avg_b is not None:
            if abs(avg_a - avg_b) <= rating_diff_thresh:
                hard_rating_matched.append(i)
        if e["book_a"]["genres"] is not None and e["book_b"]["genres"] is not None:
            if e["book_a"]["genres"] == e["book_b"]["genres"]:
                same_genre.append(i)

    scores = [(i, popularity_score(e)) for i, e in enumerate(entries)]
    scores_sorted = sorted(scores, key = lambda x: x[1])
    k = int(len(scores_sorted) * popularity_quantile)
    # Lowest-popularity quantile -- where consensus signal is weakest.
    low_popularity = sorted(i for i, _ in scores_sorted[:k])
    high_popularity = sorted(i for i, _ in scores_sorted[-k:])

    return {
        "hard_rating_matched": hard_rating_matched,
        "same_genre": same_genre,
        "low_popularity": low_popularity,
        "high_popularity": high_popularity,
    }


def build_swap_pairs(entries, seed):
    # Deterministic derangement: pair each entry i with a partner j whose user
    # differs from entry i's user. When --subset swapped_user is used at eval
    # time, entry i's user-context block is built from entry j's reference_books.
    n = len(entries)
    users = [e["user_id"] for e in entries]
    rng = random.Random(seed)

    order = list(range(n))
    for _ in range(100):
        rng.shuffle(order)
        if all(users[order[i]] != users[i] for i in range(n)):
            break
    else:
        # Fallback: swap any remaining same-user pairs with a neighbor.
        for i in range(n):
            if users[order[i]] == users[i]:
                for k in range(n):
                    if users[order[k]] != users[i] and users[order[i]] != users[k]:
                        order[i], order[k] = order[k], order[i]
                        break
    return {i: order[i] for i in range(n)}


def main():
    parser = argparse.ArgumentParser(description = "Build analysis subsets.")
    parser.add_argument("--input", type = str, default = "data/book_preference_dataset.jsonl")
    parser.add_argument("--output", type = str, default = "data/splits/splits.json")
    parser.add_argument("--rating_diff_thresh", type = float, default = 0.1,
                        help = "|avg_A - avg_B| <= thresh is hard_rating_matched.")
    parser.add_argument("--popularity_quantile", type = float, default = 0.25)
    parser.add_argument("--seed", type = int, default = 86)
    args = parser.parse_args()

    entries = load_entries(args.input)
    n = len(entries)
    n_users = len({e["user_id"] for e in entries})
    print(f"Loaded {n} entries from {n_users} distinct users")

    subsets = build_subsets(entries, args.rating_diff_thresh, args.popularity_quantile)
    swap_pairs = build_swap_pairs(entries, args.seed)

    out = {
        "meta": {
            "input_path": os.path.abspath(args.input),
            "input_sha256": file_sha256(args.input),
            "n_entries": n,
            "n_users": n_users,
            "rating_diff_thresh": args.rating_diff_thresh,
            "popularity_quantile": args.popularity_quantile,
            "seed": args.seed,
        },
        "subsets": subsets,
        "swap_pairs": {str(k): v for k, v in swap_pairs.items()},
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok = True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent = 2)

    print(f"Saved splits -> {args.output}")
    for name, idxs in subsets.items():
        print(f"  subset {name}: {len(idxs)} entries")


if __name__ == "__main__":
    main()
