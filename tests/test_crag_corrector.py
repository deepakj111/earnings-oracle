# tests/test_crag_corrector.py
"""Tests for crag/corrector.py — CRAG orchestrator logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from crag.corrector import CRAGCorrector, _build_corrected_result, _web_to_search_result
from crag.models import CRAGAction, CRAGResult, RelevanceGrade, WebSearchResult
from generation.models import GenerationResult
from retrieval.models import RetrievalResult, SearchResult

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_search_result(chunk_id: str, ticker: str = "AAPL") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        parent_id=None,
        text=f"chunk text for {chunk_id}",
        parent_text=f"parent text for {chunk_id}",
        rrf_score=0.5,
        rerank_score=0.8,
        ticker=ticker,
        company="Apple",
        date="2024-10-31",
        year=2024,
        quarter="Q4",
        fiscal_period="Q4 2024",
        section_title="Revenue",
        doc_type="earnings_release",
        source="dense",
    )


def _make_retrieval_result(
    chunks: list[SearchResult] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        query="test question",
        results=chunks or [_make_search_result("c1"), _make_search_result("c2")],
        reranked=True,
        total_candidates=10,
    )


def _make_gen_result(grounded: bool = True, answer: str = "Test answer [1].") -> GenerationResult:
    return GenerationResult(
        question="Test question",
        answer=answer,
        citations=[],
        model="gpt-4.1-nano",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        context_chunks_used=2,
        context_tokens_used=512,
        latency_seconds=1.0,
        grounded=grounded,
        retrieval_failed=False,
    )


def _make_grades(relevant_flags: list[bool]) -> list[RelevanceGrade]:
    return [
        RelevanceGrade(
            chunk_id=f"c{i}",
            relevant=flag,
            score=0.9 if flag else 0.1,
            reasoning="test",
        )
        for i, flag in enumerate(relevant_flags)
    ]


# ── _web_to_search_result ──────────────────────────────────────────────────────


def test_web_to_search_result_structure() -> None:
    web = WebSearchResult(
        title="AAPL Q4", url="https://example.com", snippet="Revenue $94.9B", score=0.8
    )
    sr = _web_to_search_result(web, 0)
    assert sr.chunk_id.startswith("web:")
    assert sr.doc_type == "web"
    assert sr.source == "web"
    assert "Revenue $94.9B" in sr.text
    assert sr.parent_text == sr.text


def test_web_to_search_result_deterministic() -> None:
    """Same URL should always produce the same chunk_id."""
    web = WebSearchResult(title="T", url="https://a.com/b", snippet="x")
    r1 = _web_to_search_result(web, 0)
    r2 = _web_to_search_result(web, 0)
    assert r1.chunk_id == r2.chunk_id


# ── _build_corrected_result ────────────────────────────────────────────────────


def test_build_corrected_incorrect_uses_web_only() -> None:
    original = _make_retrieval_result()
    web = [WebSearchResult(title="X", url="u1", snippet="s1")]
    result = _build_corrected_result(original, [], web, CRAGAction.INCORRECT)
    assert len(result.results) == 1
    assert result.results[0].source == "web"


def test_build_corrected_ambiguous_combines_local_and_web() -> None:
    original = _make_retrieval_result()
    local = [_make_search_result("c1")]
    web = [WebSearchResult(title="X", url="u1", snippet="s1")]
    result = _build_corrected_result(original, local, web, CRAGAction.AMBIGUOUS)
    assert len(result.results) == 2
    sources = {r.source for r in result.results}
    assert "dense" in sources and "web" in sources


def test_build_corrected_correct_keeps_original() -> None:
    original = _make_retrieval_result()
    result = _build_corrected_result(original, [], [], CRAGAction.CORRECT)
    assert result.results is original.results


# ── CRAGCorrector.correct ──────────────────────────────────────────────────────


@patch("crag.corrector.RelevanceGrader")
@patch("crag.corrector.WebSearchClient")
@patch("crag.corrector.Generator")
def test_correct_fast_path_when_grounded(
    mock_gen_cls: MagicMock,
    mock_web_cls: MagicMock,
    mock_grader_cls: MagicMock,
) -> None:
    """grounded=True + grade_even_if_grounded=False → CORRECT, no grading."""
    gen_result = _make_gen_result(grounded=True)
    retrieval = _make_retrieval_result()

    corrector = CRAGCorrector()
    corrector._grader = mock_grader_cls.return_value
    corrector._web = mock_web_cls.return_value

    with patch("crag.corrector._cfg") as mock_cfg:
        mock_cfg.enabled = True
        mock_cfg.grade_even_if_grounded = False
        mock_cfg.high_relevance_threshold = 0.6
        mock_cfg.low_relevance_threshold = 0.2
        mock_cfg.web_search_max_results = 4
        result = corrector.correct("test question", gen_result, retrieval)

    assert result.action == CRAGAction.CORRECT
    assert result.was_corrected is False
    assert result.web_search_triggered is False
    corrector._grader.grade_chunks.assert_not_called()


@patch("crag.corrector.RelevanceGrader")
@patch("crag.corrector.WebSearchClient")
@patch("crag.corrector.Generator")
def test_correct_all_relevant_returns_correct(
    mock_gen_cls: MagicMock,
    mock_web_cls: MagicMock,
    mock_grader_cls: MagicMock,
) -> None:
    gen_result = _make_gen_result(grounded=False)
    chunks = [_make_search_result(f"c{i}") for i in range(3)]
    retrieval = _make_retrieval_result(chunks)

    corrector = CRAGCorrector()
    corrector._grader = MagicMock()
    corrector._grader.grade_chunks.return_value = _make_grades([True, True, True])
    corrector._web = MagicMock()
    corrector._generator = MagicMock()

    with patch("crag.corrector._cfg") as mock_cfg:
        mock_cfg.enabled = True
        mock_cfg.grade_even_if_grounded = True
        mock_cfg.high_relevance_threshold = 0.6
        mock_cfg.low_relevance_threshold = 0.2
        mock_cfg.web_search_max_results = 4
        result = corrector.correct("test", gen_result, retrieval)

    assert result.action == CRAGAction.CORRECT
    assert result.was_corrected is False
    corrector._web.search.assert_not_called()


@patch("crag.corrector.RelevanceGrader")
@patch("crag.corrector.WebSearchClient")
@patch("crag.corrector.Generator")
def test_correct_none_relevant_triggers_web(
    mock_gen_cls: MagicMock,
    mock_web_cls: MagicMock,
    mock_grader_cls: MagicMock,
) -> None:
    gen_result = _make_gen_result(grounded=False)
    chunks = [_make_search_result(f"c{i}") for i in range(3)]
    retrieval = _make_retrieval_result(chunks)

    new_gen = _make_gen_result(grounded=True, answer="Web answer [1].")
    corrector = CRAGCorrector()
    corrector._grader = MagicMock()
    corrector._grader.grade_chunks.return_value = _make_grades([False, False, False])
    corrector._web = MagicMock()
    corrector._web.search.return_value = [WebSearchResult("T", "u", "web snippet", 0.8)]
    corrector._generator = MagicMock()
    corrector._generator.generate.return_value = new_gen

    with patch("crag.corrector._cfg") as mock_cfg:
        mock_cfg.enabled = True
        mock_cfg.grade_even_if_grounded = True
        mock_cfg.high_relevance_threshold = 0.6
        mock_cfg.low_relevance_threshold = 0.2
        mock_cfg.web_search_max_results = 4
        result = corrector.correct("test", gen_result, retrieval)

    assert result.action == CRAGAction.INCORRECT
    assert result.was_corrected is True
    assert result.web_search_triggered is True
    corrector._web.search.assert_called_once()
    corrector._generator.generate.assert_called_once()


@patch("crag.corrector.RelevanceGrader")
@patch("crag.corrector.WebSearchClient")
@patch("crag.corrector.Generator")
def test_correct_partial_relevant_triggers_ambiguous(
    mock_gen_cls: MagicMock,
    mock_web_cls: MagicMock,
    mock_grader_cls: MagicMock,
) -> None:
    gen_result = _make_gen_result(grounded=False)
    chunks = [_make_search_result(f"c{i}") for i in range(4)]
    retrieval = _make_retrieval_result(chunks)

    new_gen = _make_gen_result(grounded=True)
    corrector = CRAGCorrector()
    corrector._grader = MagicMock()
    # 1 out of 4 relevant → ratio = 0.25, between low (0.2) and high (0.6) → AMBIGUOUS
    corrector._grader.grade_chunks.return_value = _make_grades([True, False, False, False])
    corrector._web = MagicMock()
    corrector._web.search.return_value = [WebSearchResult("T", "u", "s", 0.7)]
    corrector._generator = MagicMock()
    corrector._generator.generate.return_value = new_gen

    with patch("crag.corrector._cfg") as mock_cfg:
        mock_cfg.enabled = True
        mock_cfg.grade_even_if_grounded = True
        mock_cfg.high_relevance_threshold = 0.6
        mock_cfg.low_relevance_threshold = 0.2
        mock_cfg.web_search_max_results = 4
        result = corrector.correct("test", gen_result, retrieval)

    assert result.action == CRAGAction.AMBIGUOUS
    corrector._web.search.assert_called_once()


def test_correct_crag_disabled_returns_original() -> None:
    gen_result = _make_gen_result(grounded=False)
    retrieval = _make_retrieval_result()

    corrector = CRAGCorrector.__new__(CRAGCorrector)
    corrector._grader = MagicMock()
    corrector._web = MagicMock()
    corrector._generator = MagicMock()

    with patch("crag.corrector._cfg") as mock_cfg:
        mock_cfg.enabled = False
        result = corrector.correct("test", gen_result, retrieval)

    assert result.action == CRAGAction.CORRECT
    assert result.final_result is gen_result
    corrector._grader.grade_chunks.assert_not_called()


# ── CRAGResult properties ──────────────────────────────────────────────────────


def test_crag_result_relevance_ratio() -> None:
    gen = _make_gen_result()
    crag = CRAGResult(
        question="q",
        action=CRAGAction.AMBIGUOUS,
        original_result=gen,
        final_result=gen,
        relevance_grades=_make_grades([True, False, True, True]),
    )
    assert crag.relevance_ratio == pytest.approx(0.75, abs=0.01)
    assert crag.relevant_chunk_count == 3


def test_crag_result_was_corrected() -> None:
    gen = _make_gen_result()
    for action, expected in [
        (CRAGAction.CORRECT, False),
        (CRAGAction.AMBIGUOUS, True),
        (CRAGAction.INCORRECT, True),
    ]:
        r = CRAGResult(question="q", action=action, original_result=gen, final_result=gen)
        assert r.was_corrected is expected


def test_crag_result_to_json_round_trips() -> None:
    import json

    gen = _make_gen_result()
    r = CRAGResult(question="q?", action=CRAGAction.CORRECT, original_result=gen, final_result=gen)
    data = json.loads(r.to_json())
    assert data["action"] == "correct"
    assert data["was_corrected"] is False
