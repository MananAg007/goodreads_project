"""Statistical tests for pairwise preference evaluation.

  - mcnemar_test: paired binary test for model-vs-model accuracy comparison.
  - wilson_ci: closed-form 95% CI on a binomial proportion (accuracy).

Each eval entry has a distinct user (1 item per user), so cluster-level
resampling / sign-flipping collapses to the item-level case. The paired
permutation test on binary correctness is equivalent to McNemar's exact
binomial. Those redundant tests were removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class McNemarResult:
    b: int   # model A correct, model B wrong
    c: int   # model B correct, model A wrong
    statistic: float
    pvalue: float
    method: str


def mcnemar_test(
    correct_a: Sequence[int],
    correct_b: Sequence[int],
    exact_threshold: int = 25,
) -> McNemarResult:
    """McNemar's test for two paired binary classifiers on the same items.

    Args:
      correct_a, correct_b: 0/1 sequences, same length, aligned by item.
      exact_threshold: use the exact binomial form when b + c <= this,
        else the chi-squared approximation with continuity correction.
    """
    a = np.asarray(correct_a, dtype = int)
    b_arr = np.asarray(correct_b, dtype = int)
    if a.shape != b_arr.shape:
        raise ValueError("correct_a and correct_b must have identical shape")

    b_cnt = int(np.sum((a == 1) & (b_arr == 0)))
    c_cnt = int(np.sum((a == 0) & (b_arr == 1)))

    if b_cnt + c_cnt == 0:
        return McNemarResult(b_cnt, c_cnt, statistic = 0.0, pvalue = 1.0, method = "exact")

    if b_cnt + c_cnt <= exact_threshold:
        # Two-sided exact binomial test against p = 0.5.
        result = stats.binomtest(min(b_cnt, c_cnt), n = b_cnt + c_cnt, p = 0.5, alternative = "two-sided")
        return McNemarResult(b_cnt, c_cnt, statistic = float(min(b_cnt, c_cnt)), pvalue = float(result.pvalue), method = "exact")

    # Chi-squared with continuity correction, 1 dof.
    chi2 = (abs(b_cnt - c_cnt) - 1) ** 2 / (b_cnt + c_cnt)
    pvalue = stats.chi2.sf(chi2, df = 1)
    return McNemarResult(b_cnt, c_cnt, statistic = float(chi2), pvalue = float(pvalue), method = "chi2_cc")


def wilson_ci(correct: Sequence[int], alpha: float = 0.05) -> dict:
    """Wilson score interval for a binomial proportion (accuracy).

    Closed form, exact for binary 0/1 outcomes, well-behaved near 0/1 and
    for small n (unlike the normal-approximation interval).
    """
    arr = np.asarray(correct, dtype = int)
    n = int(arr.size)
    if n == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": 0, "alpha": alpha}
    k = int(arr.sum())
    p_hat = k / n
    z = float(stats.norm.ppf(1 - alpha / 2))
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / denom
    return {
        "mean": float(p_hat),
        "ci_lo": float(max(0.0, center - half)),
        "ci_hi": float(min(1.0, center + half)),
        "n": n,
        "alpha": alpha,
    }


def subgroup_wilson(
    correct: Sequence[int],
    groups: Sequence,
    alpha: float = 0.05,
) -> dict[str, dict]:
    """Wilson CI per group label. Returns group -> {mean, ci_lo, ci_hi, n}."""
    arr = np.asarray(correct, dtype = int)
    groups = np.asarray(groups)
    out: dict[str, dict] = {}
    for g in sorted({str(x) for x in groups.tolist()}):
        mask = groups.astype(str) == g
        if mask.sum() == 0:
            continue
        out[g] = wilson_ci(arr[mask], alpha = alpha)
    return out
