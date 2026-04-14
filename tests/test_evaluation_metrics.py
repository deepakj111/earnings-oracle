# tests/test_evaluation_metrics.py
"""Tests for evaluation/metrics.py — LLM-based evaluation metrics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from evaluation.metrics import (
    _parse_score,
    score_all,
    score_answer_relevancy,
    score_context_precision,
    score_context_recall,
    score_faithfulness,
)
from evaluation.models import MetricScore

# ── _parse_score ───────────────────────────────────────────────────────────────


def test_parse_score_valid() -> None:
    raw = '{"score": 0.85, "reasoning": "all claims are supported"}'
    score, reasoning = _parse_score(raw, "faithfulness")
    assert abs(score - 0.85) < 0.001
    assert "supported" in reasoning


def test_parse_score_clamps_above_one() -> None:
    raw = '{"score": 1.8, "reasoning": "test"}'
    score, _ = _parse_score(raw, "test")
    assert score == 1.0


def test_parse_score_clamps_below_zero() -> None:
    raw = '{"score": -0.5, "reasoning": "test"}'
    score, _ = _parse_score(raw, "test")
    assert score == 0.0


def test_parse_score_embedded_json() -> None:
    raw = 'Sure! {"score": 0.6, "reasoning": "partial"} That is my evaluation.'
    score, _ = _parse_score(raw, "test")
    assert abs(score - 0.6) < 0.001


def test_parse_score_no_json_fallback() -> None:
    score, reasoning = _parse_score("not json", "test")
    assert score == 0.5
    assert "error" in reasoning.lower() or "parse" in reasoning.lower()


def test_parse_score_malformed_json_fallback() -> None:
    score, _ = _parse_score("{score: invalid}", "test")
    assert score == 0.5


# ── score_faithfulness ─────────────────────────────────────────────────────────


@patch("evaluation.metrics._call")
def test_score_faithfulness_high(mock_call: MagicMock) -> None:
    mock_call.return_value = '{"score": 0.95, "reasoning": "all claims supported by context"}'
    result = score_faithfulness(
        question="What was Apple's Q4 revenue?",
        answer="Apple reported $94.9B [1].",
        context_chunks=["Apple Q4 2024 revenue was $94.9 billion."],
    )
    assert isinstance(result, MetricScore)
    assert result.metric == "faithfulness"
    assert result.score == pytest.approx(0.95, abs=0.001)
    assert "supported" in result.reasoning


@patch("evaluation.metrics._call")
def test_score_faithfulness_low(mock_call: MagicMock) -> None:
    mock_call.return_value = '{"score": 0.1, "reasoning": "answer contains claims not in context"}'
    result = score_faithfulness("Q?", "Answer not in context.", ["Unrelated text."])
    assert result.score < 0.3


@patch("evaluation.metrics._call")
def test_score_faithfulness_api_failure_returns_default(mock_call: MagicMock) -> None:
    mock_call.side_effect = RuntimeError("API error")
    result = score_faithfulness("Q?", "A.", ["C."])
    assert result.score == 0.5
    assert "error" in result.reasoning.lower()


def test_score_faithfulness_no_context_chunks() -> None:
    """Empty context should still call the metric — grader handles it."""
    with patch("evaluation.metrics._call") as mock_call:
        mock_call.return_value = '{"score": 0.0, "reasoning": "no context"}'
        result = score_faithfulness("Q?", "A.", [])
    assert result.score == 0.0


# ── score_answer_relevancy ─────────────────────────────────────────────────────


@patch("evaluation.metrics._call")
def test_score_answer_relevancy_on_topic(mock_call: MagicMock) -> None:
    mock_call.return_value = '{"score": 1.0, "reasoning": "answer directly addresses question"}'
    result = score_answer_relevancy(
        question="What was Apple's Q4 revenue?",
        answer="Apple reported $94.9B in Q4 FY2024 [1].",
    )
    assert result.metric == "answer_relevancy"
    assert result.score == pytest.approx(1.0, abs=0.001)


@patch("evaluation.metrics._call")
def test_score_answer_relevancy_off_topic(mock_call: MagicMock) -> None:
    mock_call.return_value = '{"score": 0.0, "reasoning": "answer discusses wrong company"}'
    result = score_answer_relevancy("AAPL revenue?", "Microsoft reported $65B.")
    assert result.score < 0.2


# ── score_context_precision ────────────────────────────────────────────────────


@patch("evaluation.metrics._call")
def test_score_context_precision_all_relevant(mock_call: MagicMock) -> None:
    mock_call.return_value = '{"score": 1.0, "reasoning": "all chunks directly relevant"}'
    result = score_context_precision(
        question="What was Apple's Q4 revenue?",
        context_chunks=["AAPL Q4 revenue $94.9B", "AAPL Q4 EPS $1.64"],
    )
    assert result.metric == "context_precision"
    assert result.score == pytest.approx(1.0, abs=0.001)


def test_score_context_precision_empty_returns_zero() -> None:
    result = score_context_precision("Q?", [])
    assert result.score == 0.0
    assert "no context" in result.reasoning.lower()


@patch("evaluation.metrics._call")
def test_score_context_precision_mixed_chunks(mock_call: MagicMock) -> None:
    mock_call.return_value = '{"score": 0.5, "reasoning": "half the chunks are relevant"}'
    result = score_context_precision("AAPL revenue?", ["AAPL revenue", "NVDA boilerplate"])
    assert abs(result.score - 0.5) < 0.001


# ── score_context_recall ───────────────────────────────────────────────────────


@patch("evaluation.metrics._call")
def test_score_context_recall_full_coverage(mock_call: MagicMock) -> None:
    mock_call.return_value = '{"score": 1.0, "reasoning": "all ground truth facts present"}'
    result = score_context_recall(
        question="What was AAPL Q4 revenue?",
        context_chunks=["AAPL Q4 2024 total net sales $94.9B up 6% YoY"],
        ground_truth="Apple reported $94.9B in Q4 2024, up 6% year-over-year.",
    )
    assert result.metric == "context_recall"
    assert result.score == pytest.approx(1.0, abs=0.001)


def test_score_context_recall_empty_returns_zero() -> None:
    result = score_context_recall("Q?", [], "ground truth text")
    assert result.score == 0.0


# ── score_all ──────────────────────────────────────────────────────────────────


@patch("evaluation.metrics._call")
def test_score_all_returns_all_four_metrics(mock_call: MagicMock) -> None:
    mock_call.return_value = '{"score": 0.8, "reasoning": "good"}'
    results = score_all(
        question="Q?",
        answer="A.",
        context_chunks=["C."],
        ground_truth="GT.",
    )
    metric_names = {r.metric for r in results}
    assert metric_names == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    }


@patch("evaluation.metrics._call")
def test_score_all_subset_of_metrics(mock_call: MagicMock) -> None:
    mock_call.return_value = '{"score": 0.7, "reasoning": "ok"}'
    results = score_all(
        question="Q?",
        answer="A.",
        context_chunks=["C."],
        ground_truth="GT.",
        metrics=["faithfulness", "answer_relevancy"],
    )
    assert len(results) == 2
    assert all(r.metric in {"faithfulness", "answer_relevancy"} for r in results)


@patch("evaluation.metrics._call")
def test_score_all_unknown_metric_ignored(mock_call: MagicMock) -> None:
    mock_call.return_value = '{"score": 0.7, "reasoning": "ok"}'
    results = score_all("Q?", "A.", ["C."], "GT.", metrics=["faithfulness", "nonexistent_metric"])
    assert len(results) == 1
    assert results[0].metric == "faithfulness"
