# tests/test_evaluation_harness.py
"""Tests for evaluation/harness.py — evaluation harness and EvalReport."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evaluation.dataset import GOLDEN_DATASET, get_dataset_by_ticker, get_dataset_subset
from evaluation.harness import EvaluationHarness
from evaluation.models import EvalReport, EvalSample, EvalSampleResult, MetricScore
from generation.models import GenerationResult

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_pipeline(sample_generation_result: GenerationResult) -> MagicMock:
    pipeline = MagicMock()
    pipeline.ask_verbose.return_value = (
        sample_generation_result,
        "query summary",
        "retrieval summary",
    )
    return pipeline


@pytest.fixture
def sample_eval_sample() -> EvalSample:
    return EvalSample(
        sample_id="test_sample_001",
        question="What was Apple's total net sales in Q4 2024?",
        ground_truth="Apple reported total net sales of $94.9 billion in Q4 FY2024.",
        ticker="AAPL",
        year=2024,
        quarter="Q4",
    )


# ── Dataset ────────────────────────────────────────────────────────────────────


def test_golden_dataset_not_empty() -> None:
    assert len(GOLDEN_DATASET) > 0


def test_golden_dataset_all_have_sample_ids() -> None:
    for s in GOLDEN_DATASET:
        assert s.sample_id, f"EvalSample missing sample_id: {s.question[:40]}"


def test_golden_dataset_all_have_ground_truth() -> None:
    for s in GOLDEN_DATASET:
        assert s.ground_truth, f"EvalSample missing ground_truth: {s.question[:40]}"


def test_get_dataset_by_ticker() -> None:
    aapl = get_dataset_by_ticker("AAPL")
    assert all(s.ticker == "AAPL" for s in aapl)
    assert len(aapl) >= 1


def test_get_dataset_subset_limits_count() -> None:
    subset = get_dataset_subset(3)
    assert len(subset) == 3


def test_get_dataset_subset_zero_returns_empty() -> None:
    subset = get_dataset_subset(0)
    assert subset == []


def test_eval_sample_auto_generates_id() -> None:
    s = EvalSample(question="What was revenue?", ground_truth="$100B")
    assert s.sample_id != ""
    assert len(s.sample_id) == 12


# ── EvaluationHarness.run ──────────────────────────────────────────────────────


@patch("evaluation.harness.score_all")
def test_harness_run_single_sample(
    mock_score_all: MagicMock,
    mock_pipeline: MagicMock,
    sample_eval_sample: EvalSample,
) -> None:
    mock_score_all.return_value = [
        MetricScore("faithfulness", 0.9, "good"),
        MetricScore("answer_relevancy", 0.85, "relevant"),
    ]

    harness = EvaluationHarness(mock_pipeline)
    report = harness.run(
        dataset=[sample_eval_sample],
        metrics=["faithfulness", "answer_relevancy"],
    )

    assert report.n_samples == 1
    assert report.n_failed == 0
    assert "faithfulness" in report.metric_averages
    assert "answer_relevancy" in report.metric_averages
    assert report.pass_rate == 1.0


@patch("evaluation.harness.score_all")
def test_harness_run_empty_dataset(
    mock_score_all: MagicMock,
    mock_pipeline: MagicMock,
) -> None:
    harness = EvaluationHarness(mock_pipeline)
    report = harness.run(dataset=[], metrics=["faithfulness"])
    assert report.n_samples == 0
    assert report.metric_averages == {}


@patch("evaluation.harness.score_all")
def test_harness_pipeline_failure_marked(
    mock_score_all: MagicMock,
    sample_eval_sample: EvalSample,
) -> None:
    """When the pipeline raises, the sample should be marked as failed."""
    broken_pipeline = MagicMock()
    broken_pipeline.ask_verbose.side_effect = RuntimeError("Qdrant offline")

    harness = EvaluationHarness(broken_pipeline)
    report = harness.run(dataset=[sample_eval_sample], metrics=["faithfulness"])

    assert report.n_failed == 1
    assert report.sample_results[0].pipeline_failed is True
    assert "Qdrant offline" in report.sample_results[0].error_message


@patch("evaluation.harness.score_all")
def test_harness_metric_averages_correct(
    mock_score_all: MagicMock,
    mock_pipeline: MagicMock,
) -> None:
    """Average should be computed only across non-failed samples."""
    samples = [EvalSample(f"q{i}", f"gt{i}", sample_id=f"s{i}") for i in range(4)]
    call_count = [0]

    def _score_side_effect(*args, **kwargs):
        call_count[0] += 1
        return [MetricScore("faithfulness", 0.8 if call_count[0] % 2 == 0 else 1.0, "ok")]

    mock_score_all.side_effect = _score_side_effect

    harness = EvaluationHarness(mock_pipeline)
    report = harness.run(dataset=samples, metrics=["faithfulness"])

    assert (
        abs(report.metric_averages["faithfulness"] - 0.9) < 0.01
    )  # mean of 1.0 and 0.8 alternating
    assert report.n_samples == 4


@patch("evaluation.harness.score_all")
def test_harness_pass_rate(
    mock_score_all: MagicMock,
    sample_eval_sample: EvalSample,
) -> None:
    broken = MagicMock()
    broken.ask_verbose.side_effect = RuntimeError("fail")
    mock_score_all.return_value = []

    harness = EvaluationHarness(broken)
    report = harness.run(dataset=[sample_eval_sample] * 3, metrics=["faithfulness"])
    assert report.pass_rate == 0.0


# ── EvalReport serialization ───────────────────────────────────────────────────


def test_eval_report_to_json_valid() -> None:
    report = EvalReport(
        dataset_name="test",
        n_samples=2,
        n_failed=0,
        metric_averages={"faithfulness": 0.88},
        sample_results=[],
        total_latency_seconds=5.0,
    )
    data = json.loads(report.to_json())
    assert data["n_samples"] == 2
    assert data["metric_averages"]["faithfulness"] == pytest.approx(0.88, abs=0.001)


def test_eval_report_to_csv_has_header() -> None:
    report = EvalReport(
        dataset_name="test",
        n_samples=0,
        n_failed=0,
        metric_averages={},
        sample_results=[],
        total_latency_seconds=0.0,
    )
    csv_text = report.to_csv()
    assert "sample_id" in csv_text
    assert "metric" in csv_text


def test_eval_report_to_csv_includes_failed() -> None:
    failed = EvalSampleResult(
        sample=EvalSample("Q?", "GT.", sample_id="s1"),
        generated_answer="",
        context_chunks=[],
        pipeline_failed=True,
        error_message="timeout",
    )
    report = EvalReport(
        dataset_name="test",
        n_samples=1,
        n_failed=1,
        metric_averages={},
        sample_results=[failed],
        total_latency_seconds=1.0,
    )
    csv_text = report.to_csv()
    assert "timeout" in csv_text or "True" in csv_text


def test_eval_report_summary_contains_metrics() -> None:
    report = EvalReport(
        dataset_name="test_run",
        n_samples=5,
        n_failed=1,
        metric_averages={"faithfulness": 0.82, "answer_relevancy": 0.91},
        sample_results=[],
        total_latency_seconds=30.0,
    )
    summary = report.summary()
    assert "faithfulness" in summary
    assert "0.82" in summary
    assert "test_run" in summary


# ── EvaluationHarness.save_report ─────────────────────────────────────────────


@patch("evaluation.harness.score_all")
def test_save_report_writes_json_and_csv(
    mock_score_all: MagicMock,
    tmp_path: Path,
    mock_pipeline: MagicMock,
    sample_eval_sample: EvalSample,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("evaluation.harness._eval_cfg.output_dir", str(tmp_path))
    mock_score_all.return_value = [MetricScore("faithfulness", 0.9, "ok")]

    harness = EvaluationHarness(mock_pipeline)
    report = harness.run(dataset=[sample_eval_sample], metrics=["faithfulness"])
    json_path, csv_path = harness.save_report(report)

    assert json_path.exists()
    assert csv_path.exists()
    data = json.loads(json_path.read_text())
    assert data["n_samples"] == 1
