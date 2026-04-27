# tests/test_crag_grader.py
"""Tests for crag/grader.py — LLM-based chunk relevance grading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from crag.grader import RelevanceGrader, _grade_one, _parse_response
from crag.models import RelevanceGrade
from retrieval.models import SearchResult

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_chunk(chunk_id: str = "c1", text: str = "Apple Q4 revenue $94.9B") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        parent_id=None,
        text=text,
        parent_text=text,
        rrf_score=0.5,
        rerank_score=0.8,
        ticker="AAPL",
        company="Apple",
        date="2024-10-31",
        year=2024,
        quarter="Q4",
        fiscal_period="Q4 2024",
        section_title="Revenue",
        doc_type="earnings_release",
        source="dense",
    )


def _mock_openai_response(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── _parse_response ────────────────────────────────────────────────────────────


def test_parse_response_valid_relevant() -> None:
    """Parse response valid relevant."""
    raw = '{"relevant": true, "score": 0.95, "reasoning": "chunk contains Q4 revenue"}'
    relevant, score, reasoning = _parse_response(raw, "c1")
    assert relevant is True
    assert abs(score - 0.95) < 0.001
    assert "revenue" in reasoning


def test_parse_response_valid_irrelevant() -> None:
    """Parse response valid irrelevant."""
    raw = '{"relevant": false, "score": 0.1, "reasoning": "wrong company"}'
    relevant, score, reasoning = _parse_response(raw, "c1")
    assert relevant is False
    assert abs(score - 0.1) < 0.001


def test_parse_response_score_clamped() -> None:
    """Parse response score clamped."""
    raw = '{"relevant": true, "score": 1.5, "reasoning": "test"}'
    _, score, _ = _parse_response(raw, "c1")
    assert score <= 1.0


def test_parse_response_score_clamped_negative() -> None:
    """Parse response score clamped negative."""
    raw = '{"relevant": true, "score": -0.3, "reasoning": "test"}'
    _, score, _ = _parse_response(raw, "c1")
    assert score >= 0.0


def test_parse_response_embedded_json() -> None:
    """Parse response embedded json."""
    raw = 'Here is my assessment: {"relevant": true, "score": 0.8, "reasoning": "relevant"} done.'
    relevant, score, _ = _parse_response(raw, "c1")
    assert relevant is True
    assert abs(score - 0.8) < 0.001


def test_parse_response_no_json_falls_back() -> None:
    """Parse response no json falls back."""
    relevant, score, reasoning = _parse_response("not json at all", "c1")
    assert relevant is True  # fail-open
    assert score == 0.5
    assert "parse" in reasoning.lower()


def test_parse_response_malformed_json_falls_back() -> None:
    """Parse response malformed json falls back."""
    relevant, score, _ = _parse_response("{broken json", "c1")
    assert relevant is True
    assert score == 0.5


# ── _grade_one ─────────────────────────────────────────────────────────────────


@patch("crag.grader._get_client")
def test_grade_one_relevant(mock_get_client: MagicMock) -> None:
    """Grade one relevant."""
    mock_get_client.return_value.chat.completions.create.return_value = _mock_openai_response(
        '{"relevant": true, "score": 0.92, "reasoning": "contains the revenue figure"}'
    )
    chunk = _make_chunk()
    grade = _grade_one("What was Apple's Q4 revenue?", chunk)

    assert isinstance(grade, RelevanceGrade)
    assert grade.chunk_id == "c1"
    assert grade.relevant is True
    assert grade.score == pytest.approx(0.92, abs=0.001)


@patch("crag.grader._get_client")
def test_grade_one_irrelevant(mock_get_client: MagicMock) -> None:
    """Grade one irrelevant."""
    mock_get_client.return_value.chat.completions.create.return_value = _mock_openai_response(
        '{"relevant": false, "score": 0.05, "reasoning": "wrong fiscal period"}'
    )
    chunk = _make_chunk()
    grade = _grade_one("What was Apple's Q1 2023 revenue?", chunk)

    assert grade.relevant is False
    assert grade.score < 0.2


@patch("crag.grader._get_client")
def test_grade_one_api_failure_fails_open(mock_get_client: MagicMock) -> None:
    """Any API error should produce a lenient grade (relevant=True, score=0.5)."""
    mock_get_client.return_value.chat.completions.create.side_effect = RuntimeError("API down")
    chunk = _make_chunk()
    grade = _grade_one("What was revenue?", chunk)

    assert grade.relevant is True
    assert grade.score == 0.5
    assert "error" in grade.reasoning.lower()


@patch("crag.grader._get_client")
def test_grade_one_empty_response_falls_back(mock_get_client: MagicMock) -> None:
    """Grade one empty response falls back."""
    mock_get_client.return_value.chat.completions.create.return_value = _mock_openai_response("")
    chunk = _make_chunk()
    grade = _grade_one("What was revenue?", chunk)
    assert grade.relevant is True  # fail-open


# ── RelevanceGrader.grade_chunks ───────────────────────────────────────────────


@patch("crag.grader._grade_one")
def test_grade_chunks_preserves_order(mock_grade_one: MagicMock) -> None:
    """Grades should be returned in the same order as input chunks."""
    chunks = [_make_chunk(f"c{i}") for i in range(5)]
    mock_grade_one.side_effect = lambda q, c: RelevanceGrade(
        chunk_id=c.chunk_id, relevant=True, score=0.9, reasoning="ok"
    )
    grader = RelevanceGrader()
    grades = grader.grade_chunks("What was revenue?", chunks)

    assert len(grades) == 5
    for i, grade in enumerate(grades):
        assert grade.chunk_id == f"c{i}"


@patch("crag.grader._grade_one")
def test_grade_chunks_empty_input(mock_grade_one: MagicMock) -> None:
    """Grade chunks empty input."""
    grader = RelevanceGrader()
    grades = grader.grade_chunks("test", [])
    assert grades == []
    mock_grade_one.assert_not_called()


@patch("crag.grader._grade_one")
def test_grade_chunks_all_relevant(mock_grade_one: MagicMock) -> None:
    """Grade chunks all relevant."""
    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    mock_grade_one.return_value = RelevanceGrade("x", True, 0.9, "relevant")
    grader = RelevanceGrader()
    grades = grader.grade_chunks("What was revenue?", chunks)
    assert all(g.relevant for g in grades)


@patch("crag.grader._grade_one")
def test_grade_chunks_none_relevant(mock_grade_one: MagicMock) -> None:
    """Grade chunks none relevant."""
    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    mock_grade_one.return_value = RelevanceGrade("x", False, 0.1, "irrelevant")
    grader = RelevanceGrader()
    grades = grader.grade_chunks("What was revenue?", chunks)
    assert not any(g.relevant for g in grades)


def test_grade_single_delegates_to_grade_one() -> None:
    """Grade single delegates to grade one."""
    chunk = _make_chunk()
    grader = RelevanceGrader()
    with patch("crag.grader._grade_one") as mock:
        mock.return_value = RelevanceGrade("c1", True, 0.9, "ok")
        grade = grader.grade_single("test?", chunk)
    mock.assert_called_once_with("test?", chunk)
    assert grade.chunk_id == "c1"
