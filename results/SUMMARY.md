# LLM Evaluation Sweep Summary

## Overview

This report summarizes the 16-run LLM evaluation sweep launched via [`scripts/slurm/run_llm_eval_sweep.sh`](/home/saksham3/courses/Semester%202/10-718-project/goodreads_project/scripts/slurm/run_llm_eval_sweep.sh:1) and evaluated by [`util/run_llm_eval.py`](/home/saksham3/courses/Semester%202/10-718-project/goodreads_project/util/run_llm_eval.py:394).

The sweep crossed:

- 2 prompt variants:
  - `answer_only`
  - `answer_reason`
- 8 evaluation configurations

## Experimental Setup

- Model: `Qwen/Qwen2.5-14B-Instruct`
- Dataset: `data/book_preference_dataset.jsonl`
- Entries per run: `100`
- User reviews shown: `1`
- Community reviews shown: `8` per candidate book
- Output directories: `runs/<tag>_<prompt_tag>/`

The prompt variants differ only in output format:

- `answer_only`: requires exactly `ANSWER: [A or B]`
- `answer_reason`: requires
  - `ANSWER: [A or B]`
  - `REASON: [1-2 sentences ...]`

The 8 base configurations were:

1. `baseline_no_user`
   `subset=full`, `user_context=none`, `ref_sel=random`, `comm_sel=random`
2. `ratings_only`
   `subset=full`, `user_context=ratings_only`, `ref_sel=random`, `comm_sel=random`
3. `full_reviews_random_ref`
   `subset=full`, `user_context=full_reviews`, `ref_sel=random`, `comm_sel=random`
4. `full_reviews_tfidf_ref`
   `subset=full`, `user_context=full_reviews`, `ref_sel=tfidf`, `comm_sel=random`
5. `full_reviews_random_comm`
   `subset=full`, `user_context=full_reviews`, `ref_sel=random`, `comm_sel=random`
6. `full_reviews_top_comm`
   `subset=full`, `user_context=full_reviews`, `ref_sel=random`, `comm_sel=top_upvoted`
7. `full_reviews_tfidf_top`
   `subset=full`, `user_context=full_reviews`, `ref_sel=tfidf`, `comm_sel=top_upvoted`
8. `swapped_user_tfidf_top`
   `subset=swapped_user`, `user_context=full_reviews`, `ref_sel=tfidf`, `comm_sel=top_upvoted`

`tfidf` selection means the evaluator ranks the user's past reviews by TF-IDF cosine similarity against the current candidate-book query and keeps the most relevant one.

## Metrics

Accuracy is computed on parsed outputs only.

Wilson 95% confidence intervals are uncertainty intervals for a run's binary accuracy. Wider intervals mean more uncertainty.

McNemar's test is the paired run-vs-run significance test used here. It compares two runs on the same evaluation items and only looks at disagreement cases: where run A is correct and run B is wrong, versus the reverse.

## Top-Line Results

| Run | Accuracy | Wilson 95% CI | Parsed | Parse failed |
| --- | --- | --- | --- | --- |
| baseline_no_user_answer_only | 60.0% | [50.2%, 69.1%] | 100/100 | 0 |
| ratings_only_answer_only | 62.0% | [52.2%, 70.9%] | 100/100 | 0 |
| full_reviews_random_ref_answer_only | 61.0% | [51.2%, 70.0%] | 100/100 | 0 |
| full_reviews_tfidf_ref_answer_only | 64.0% | [54.2%, 72.7%] | 100/100 | 0 |
| full_reviews_random_comm_answer_only | 61.0% | [51.2%, 70.0%] | 100/100 | 0 |
| full_reviews_top_comm_answer_only | 58.3% | [48.3%, 67.7%] | 96/100 | 4 |
| full_reviews_tfidf_top_answer_only | 63.2% | [53.1%, 72.2%] | 95/100 | 5 |
| swapped_user_tfidf_top_answer_only | 61.1% | [51.0%, 70.3%] | 95/100 | 5 |
| baseline_no_user_answer_reason | 61.0% | [51.2%, 70.0%] | 100/100 | 0 |
| ratings_only_answer_reason | 59.0% | [49.2%, 68.1%] | 100/100 | 0 |
| full_reviews_random_ref_answer_reason | 61.0% | [51.2%, 70.0%] | 100/100 | 0 |
| full_reviews_tfidf_ref_answer_reason | 64.0% | [54.2%, 72.7%] | 100/100 | 0 |
| full_reviews_random_comm_answer_reason | 61.0% | [51.2%, 70.0%] | 100/100 | 0 |
| full_reviews_top_comm_answer_reason | 57.3% | [47.3%, 66.7%] | 96/100 | 4 |
| full_reviews_tfidf_top_answer_reason | 64.2% | [54.2%, 73.1%] | 95/100 | 5 |
| swapped_user_tfidf_top_answer_reason | 55.8% | [45.8%, 65.4%] | 95/100 | 5 |

## Best Configurations

The best accuracy in the sweep was a tie:

- `full_reviews_tfidf_ref_answer_only`: `64.0%`
- `full_reviews_tfidf_ref_answer_reason`: `64.0%`

The next-best run was:

- `full_reviews_tfidf_top_answer_reason`: `64.2%` on `95` parsed examples

Among fully parsed 100-example runs, the strongest configuration was `full_reviews_tfidf_ref` for both prompt variants.

## Prompt Comparison

Comparing `answer_only` vs `answer_reason` within the same configuration:

| Config | Answer only | Answer + reason | Paired n | McNemar p |
| --- | --- | --- | --- | --- |
| baseline_no_user | 60.0% | 61.0% | 100 | 1.0000 |
| ratings_only | 62.0% | 59.0% | 100 | 0.3750 |
| full_reviews_random_ref | 61.0% | 61.0% | 100 | 1.0000 |
| full_reviews_tfidf_ref | 64.0% | 64.0% | 100 | 1.0000 |
| full_reviews_top_comm | 58.3% | 57.3% | 96 | 1.0000 |
| full_reviews_tfidf_top | 63.2% | 64.2% | 95 | 1.0000 |
| swapped_user_tfidf_top | 61.1% | 55.8% | 95 | 0.1250 |

Main takeaway: adding a required reason did not materially change performance in this sweep. None of the prompt-paired comparisons were statistically significant.

## Personalization Effects

For the best random-community setting, TF-IDF selection helped relative to random review selection:

- `full_reviews_random_ref_answer_only`: `61.0%`
- `full_reviews_tfidf_ref_answer_only`: `64.0%`
- `full_reviews_random_ref_answer_reason`: `61.0%`
- `full_reviews_tfidf_ref_answer_reason`: `64.0%`

However, compared against the no-user baseline, the improvement was modest and not significant in paired testing:

| Comparison | Paired n | Acc A | Acc B | McNemar p |
| --- | --- | --- | --- | --- |
| baseline_no_user_answer_only vs full_reviews_tfidf_ref_answer_only | 100 | 60.0% | 64.0% | 0.6265 |
| baseline_no_user_answer_reason vs full_reviews_tfidf_ref_answer_reason | 100 | 61.0% | 64.0% | 0.7277 |

So the evidence for a real personalization gain in this sweep is weak.

## Swapped-User Falsification

The swapped-user test checks whether using another user's history degrades performance.

| Comparison | Paired n | Acc A | Acc B | McNemar p |
| --- | --- | --- | --- | --- |
| full_reviews_tfidf_top_answer_only vs swapped_user_tfidf_top_answer_only | 94 | 62.8% | 60.6% | 0.7266 |
| full_reviews_tfidf_top_answer_reason vs swapped_user_tfidf_top_answer_reason | 94 | 63.8% | 55.3% | 0.0768 |

The direction is as expected: the swapped-user condition is worse, especially for `answer_reason`, but the evidence is still not strong enough to call it conclusive on this sample.

## Accuracy By Rating Difference

Selected subgroup results by `rating_difference`:

| Run | diff=1 | diff=2 | diff=3 | diff=4 |
| --- | --- | --- | --- | --- |
| baseline_no_user_answer_only | 47.9% (48) | 80.0% (25) | 56.2% (16) | 72.7% (11) |
| baseline_no_user_answer_reason | 50.0% (48) | 76.0% (25) | 62.5% (16) | 72.7% (11) |
| full_reviews_tfidf_ref_answer_only | 56.2% (48) | 72.0% (25) | 68.8% (16) | 72.7% (11) |
| full_reviews_tfidf_ref_answer_reason | 56.2% (48) | 76.0% (25) | 62.5% (16) | 72.7% (11) |
| full_reviews_tfidf_top_answer_only | 53.2% (47) | 75.0% (24) | 62.5% (16) | 87.5% (8) |
| full_reviews_tfidf_top_answer_reason | 51.1% (47) | 79.2% (24) | 62.5% (16) | 100.0% (8) |

Counts are shown in parentheses.

## Main Findings

1. `full_reviews_tfidf_ref` was the strongest stable configuration.
   It matched the best top-line accuracy in both prompt variants and did so with perfect parse rate (`100/100` parsed).

2. The prompt format change had little effect.
   `answer_only` and `answer_reason` were nearly identical across the sweep, with no significant prompt-paired differences.

3. TF-IDF user-review selection was directionally better than random selection.
   This held for both prompt variants in the `full_reviews` + random-community setting.

4. The swapped-user falsification degraded performance, but not decisively.
   The effect was larger for `answer_reason`, but still only borderline in the paired test.

5. Parse failures were concentrated in the `top_upvoted` configurations.
   The random-community and random-reference runs parsed perfectly, while the `top_comm` and `tfidf_top` variants dropped a few examples.

## Bottom Line

On this sweep, the best-performing configuration was `full_reviews_tfidf_ref`, and forcing the model to also output a reason did not materially help or hurt. The strongest evidence in the results is for using TF-IDF to choose the single user review; the evidence for a larger benefit from personalization beyond that remains limited.
