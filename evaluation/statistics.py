"""
Statistical Rigor methods for the evaluating LLM pipeline architectures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from evaluation.models import EvalReport


def compute_bootstrap_ci(
    scores: list[float], num_bootstraps: int = 1000, alpha: float = 0.05, seed: int | None = 42
) -> tuple[float, float]:
    """
    Computes the (lower, upper) Confidence Interval bounds for the mean of the scores
    using Bootstrap sampling with replacement.

    Args:
        scores: List of scores (e.g. [0.5, 0.8, 1.0, 0.6]).
        num_bootstraps: Number of bootstrap iterations.
        alpha: Significance level (0.05 -> 95% CI).
        seed: Random seed for deterministic bootstrap sampling.

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not scores:
        return (0.0, 0.0)
    if len(scores) == 1:
        return (scores[0], scores[0])

    rng = np.random.default_rng(seed)
    scores_array = np.array(scores)
    # Generate bootstrap samples: shape (num_bootstraps, len(scores_array))
    bootstraps = rng.choice(scores_array, size=(num_bootstraps, len(scores_array)), replace=True)
    # Compute the mean over each sample
    bootstrap_means = np.mean(bootstraps, axis=1)

    # Calculate percentiles to capture the middle (1-alpha) region
    lower = float(np.percentile(bootstrap_means, 100 * (alpha / 2)))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return lower, upper


def compare_models(report_a: EvalReport, report_b: EvalReport) -> dict[str, dict[str, float]]:
    """
    Compares two evaluation reports by running paired Student's t-test and
    Wilcoxon signed-rank test on their shared metrics across identical samples.

    Matches by `sample_id` to ensure correct paired hypothesis testing.

    Args:
        report_a: The baseline evaluation report.
        report_b: The new feature evaluation report.

    Returns:
        A dictionary mapping metric names to a dictionary of p-values.
        E.g. {"faithfulness": {"ttest_p_value": 0.03, "wilcoxon_p_value": 0.02}}
    """
    # ── Map sample_id to its dict of scores
    scores_a = {
        r.sample.sample_id: r.score_for for r in report_a.sample_results if not r.pipeline_failed
    }
    scores_b = {
        r.sample.sample_id: r.score_for for r in report_b.sample_results if not r.pipeline_failed
    }

    common_samples = set(scores_a.keys()).intersection(set(scores_b.keys()))
    if not common_samples:
        return {}

    all_metrics = set()
    for s_id in common_samples:
        all_metrics.update(scores_a[s_id].keys())
        all_metrics.update(scores_b[s_id].keys())

    results = {}
    for metric in all_metrics:
        arr_a = []
        arr_b = []
        for s_id in common_samples:
            if metric in scores_a[s_id] and metric in scores_b[s_id]:
                arr_a.append(scores_a[s_id][metric])
                arr_b.append(scores_b[s_id][metric])

        if len(arr_a) < 2:
            continue

        # If perfectly strictly identical, tests are unnecessary and may fail via zero variance
        if np.allclose(arr_a, arr_b):
            results[metric] = {"ttest_p_value": 1.0, "wilcoxon_p_value": 1.0}
            continue

        # Perform Student's paired t-test
        ttest_res = stats.ttest_rel(arr_a, arr_b)
        ttest_p = float(ttest_res.pvalue) if not np.isnan(ttest_res.pvalue) else 1.0

        # Perform Wilcoxon signed-rank test (non-parametric, robust to non-normal differences)
        try:
            wilcoxon_res = stats.wilcoxon(arr_a, arr_b)
            wilcoxon_p = float(wilcoxon_res.pvalue)
        except ValueError:
            wilcoxon_p = 1.0

        results[metric] = {
            "ttest_p_value": ttest_p,
            "wilcoxon_p_value": wilcoxon_p,
        }

    return results
