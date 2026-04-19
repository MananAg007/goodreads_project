"""Compare two LLM evaluation runs with paired McNemar test and Wilson CIs.

Reads two raw_outputs.jsonl files produced by util/run_llm_eval.py and prints:
  - per-run accuracy and Wilson 95% CI
  - McNemar test on paired correctness
  - subgroup Wilson CIs by rating_difference

Entries are aligned by entry_idx. Entries missing from either run or with
parse_success=False in either run are dropped from paired tests (reported).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import stats as S


def load_run(path):
    items = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            items[r["entry_idx"]] = r
    return items


def align_pair(run_a, run_b):
    common = sorted(set(run_a.keys()) & set(run_b.keys()))
    rows = []
    dropped_parse = 0
    for idx in common:
        ra, rb = run_a[idx], run_b[idx]
        if not ra["parse_success"] or not rb["parse_success"]:
            dropped_parse += 1
            continue
        rows.append({
            "entry_idx": idx,
            "user_id": ra["user_id"],
            "rating_difference": ra["rating_difference"],
            "ground_truth": ra["ground_truth"],
            "correct_a": int(ra["predicted"] == ra["ground_truth"]),
            "correct_b": int(rb["predicted"] == rb["ground_truth"]),
        })
    only_a = set(run_a.keys()) - set(run_b.keys())
    only_b = set(run_b.keys()) - set(run_a.keys())
    return rows, {
        "n_common": len(common),
        "dropped_parse_fail": dropped_parse,
        "only_a": len(only_a),
        "only_b": len(only_b),
    }


def main():
    parser = argparse.ArgumentParser(description = "Compare two raw_outputs.jsonl runs.")
    parser.add_argument("--a", required = True, help = "Path to first run raw_outputs.jsonl")
    parser.add_argument("--b", required = True, help = "Path to second run raw_outputs.jsonl")
    parser.add_argument("--label_a", default = "A")
    parser.add_argument("--label_b", default = "B")
    parser.add_argument("--output", default = None, help = "Optional JSON path to save the full report.")
    args = parser.parse_args()

    run_a = load_run(args.a)
    run_b = load_run(args.b)
    rows, align = align_pair(run_a, run_b)
    if not rows:
        print("No paired entries with parse_success=True in both runs.")
        print(f"Alignment info: {align}")
        return

    correct_a = np.array([r["correct_a"] for r in rows])
    correct_b = np.array([r["correct_b"] for r in rows])
    rating_diffs = [str(r["rating_difference"]) for r in rows]

    print(f"Paired on {len(rows)} entries; alignment: {align}")
    print()

    ci_a = S.wilson_ci(correct_a)
    ci_b = S.wilson_ci(correct_b)
    print(f"{args.label_a}: accuracy = {ci_a['mean']:.4f}  95% Wilson CI = [{ci_a['ci_lo']:.4f}, {ci_a['ci_hi']:.4f}]  (n={ci_a['n']})")
    print(f"{args.label_b}: accuracy = {ci_b['mean']:.4f}  95% Wilson CI = [{ci_b['ci_lo']:.4f}, {ci_b['ci_hi']:.4f}]  (n={ci_b['n']})")
    print()

    mc = S.mcnemar_test(correct_a, correct_b)
    print(f"McNemar ({mc.method}): b={mc.b}, c={mc.c}, stat={mc.statistic:.4f}, p={mc.pvalue:.4g}")
    print()

    print("Per-rating-difference subgroup Wilson CIs:")
    sg_a = S.subgroup_wilson(correct_a, rating_diffs)
    sg_b = S.subgroup_wilson(correct_b, rating_diffs)
    keys = sorted(set(sg_a.keys()) | set(sg_b.keys()))
    for k in keys:
        a_str = (f"{sg_a[k]['mean']:.3f} [{sg_a[k]['ci_lo']:.3f},{sg_a[k]['ci_hi']:.3f}] n={sg_a[k]['n']}"
                 if k in sg_a else "-")
        b_str = (f"{sg_b[k]['mean']:.3f} [{sg_b[k]['ci_lo']:.3f},{sg_b[k]['ci_hi']:.3f}] n={sg_b[k]['n']}"
                 if k in sg_b else "-")
        print(f"  rating_diff={k}:  {args.label_a}: {a_str}   {args.label_b}: {b_str}")

    report = {
        "alignment": align,
        "labels": {"a": args.label_a, "b": args.label_b},
        "paths": {"a": os.path.abspath(args.a), "b": os.path.abspath(args.b)},
        "accuracy": {args.label_a: ci_a, args.label_b: ci_b},
        "mcnemar": {"b": mc.b, "c": mc.c, "stat": mc.statistic, "pvalue": mc.pvalue, "method": mc.method},
        "subgroup_rating_diff": {args.label_a: sg_a, args.label_b: sg_b},
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok = True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent = 2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
