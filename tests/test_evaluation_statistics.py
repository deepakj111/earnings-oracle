from evaluation.models import EvalReport, EvalSample, EvalSampleResult, MetricScore
from evaluation.statistics import compare_models, compute_bootstrap_ci


def test_compute_bootstrap_ci() -> None:
    # Setup deterministic scores
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Run boostrap
    lower, upper = compute_bootstrap_ci(scores, num_bootstraps=1000, alpha=0.05, seed=42)

    # with 10 values evenly spaced 0.1-1.0, mean is 0.55
    # The bootstrap CI should be roughly centered around 0.55
    assert 0.3 < lower < 0.55
    assert 0.55 < upper < 0.8
    assert lower < upper


def test_compute_bootstrap_ci_edge_cases() -> None:
    assert compute_bootstrap_ci([]) == (0.0, 0.0)
    assert compute_bootstrap_ci([0.5]) == (0.5, 0.5)

    # identical elements should return identical CI
    lower, upper = compute_bootstrap_ci([0.7, 0.7, 0.7, 0.7])
    assert lower == 0.7
    assert upper == 0.7


def _build_mock_report(scores_dict):
    """Helper to build a mock EvalReport. scores_dict is {sample_id: {metric: score}}"""
    results = []
    for s_id, sm in scores_dict.items():
        sample = EvalSample(question="q", ground_truth="a", sample_id=s_id)
        m_scores = {k: MetricScore(k, v, "test") for k, v in sm.items()}
        results.append(
            EvalSampleResult(
                sample=sample, generated_answer="a", context_chunks=[], metric_scores=m_scores
            )
        )

    return EvalReport(
        dataset_name="test",
        n_samples=len(scores_dict),
        n_failed=0,
        metric_averages={},
        sample_results=results,
        total_latency_seconds=0.0,
    )


def test_compare_models_significant_difference() -> None:
    # Report A has generally lower scores than Report B
    report_a = _build_mock_report(
        {
            "1": {"faithfulness": 0.5},
            "2": {"faithfulness": 0.4},
            "3": {"faithfulness": 0.6},
            "4": {"faithfulness": 0.3},
            "5": {"faithfulness": 0.5},
        }
    )

    report_b = _build_mock_report(
        {
            "1": {"faithfulness": 0.9},
            "2": {"faithfulness": 0.8},
            "3": {"faithfulness": 1.0},
            "4": {"faithfulness": 0.9},
            "5": {"faithfulness": 0.9},
        }
    )

    results = compare_models(report_a, report_b)

    assert "faithfulness" in results
    # We expect a very low p-value because B is strictly higher
    assert results["faithfulness"]["ttest_p_value"] < 0.05
    assert results["faithfulness"]["wilcoxon_p_value"] <= 0.0625


def test_compare_models_identical() -> None:
    report_a = _build_mock_report(
        {
            "1": {"relevancy": 0.8},
            "2": {"relevancy": 0.7},
        }
    )

    # pass identical so p-vals should be 1.0
    results = compare_models(report_a, report_a)

    assert "relevancy" in results
    assert results["relevancy"]["ttest_p_value"] == 1.0
    assert results["relevancy"]["wilcoxon_p_value"] == 1.0
