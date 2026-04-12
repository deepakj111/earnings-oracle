"""Tests for retrieval/reranker.py — FlashRank cross-encoder reranking."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from retrieval.models import SearchResult
from retrieval.reranker import rerank


def _make_result(chunk_id: str, rrf_score: float, text: str = "dummy text") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        parent_id=None,
        text=text,
        parent_text=text,
        rrf_score=rrf_score,
        rerank_score=float("-inf"),
        ticker="AAPL",
        company="Apple",
        date="2024-10-31",
        year=2024,
        quarter="Q1",
        fiscal_period="Q1 2024",
        section_title="Revenue",
        doc_type="earnings_release",
        source="dense",
    )


class TestRerankDisabled:
    def test_returns_rrf_sorted_when_disabled(self):
        candidates = [
            _make_result("a", rrf_score=0.01),
            _make_result("b", rrf_score=0.05),
            _make_result("c", rrf_score=0.03),
        ]
        with patch("retrieval.reranker.settings") as mock_settings:
            mock_settings.reranker.enabled = False
            mock_settings.retrieval.top_k_final = 3
            results = rerank("test query", candidates)

        scores = [r.rerank_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_truncates_to_top_k_when_disabled(self):
        candidates = [_make_result(f"chunk_{i}", rrf_score=float(i)) for i in range(10)]
        with patch("retrieval.reranker.settings") as mock_settings:
            mock_settings.reranker.enabled = False
            mock_settings.retrieval.top_k_final = 3
            results = rerank("test", candidates)
        assert len(results) == 3

    def test_rerank_score_set_to_rrf_when_disabled(self):
        c = _make_result("x", rrf_score=0.042)
        with patch("retrieval.reranker.settings") as mock_settings:
            mock_settings.reranker.enabled = False
            mock_settings.retrieval.top_k_final = 5
            results = rerank("q", [c])
        assert results[0].rerank_score == 0.042


class TestRerankEnabled:
    def _mock_ranker(self, scores: list[float]) -> tuple:
        ranker = MagicMock()
        ranker.rerank.return_value = [{"id": i, "score": s} for i, s in enumerate(scores)]
        mock_rerank_request_cls = MagicMock()
        return ranker, mock_rerank_request_cls

    def test_returns_top_k_final_results(self):
        candidates = [_make_result(f"c{i}", rrf_score=0.01) for i in range(10)]
        mock_ranker, mock_rrc = self._mock_ranker([float(i) for i in range(10)])
        with (
            patch("retrieval.reranker.settings") as mock_settings,
            patch("retrieval.reranker._get_ranker", return_value=(mock_ranker, mock_rrc)),
        ):
            mock_settings.reranker.enabled = True
            mock_settings.retrieval.top_k_final = 3
            results = rerank("test", candidates)
        assert len(results) == 3

    def test_results_sorted_by_rerank_score_descending(self):
        candidates = [_make_result(f"c{i}", rrf_score=0.01) for i in range(4)]
        scores = [0.3, 0.9, 0.1, 0.7]
        mock_ranker, mock_rrc = self._mock_ranker(scores)
        with (
            patch("retrieval.reranker.settings") as mock_settings,
            patch("retrieval.reranker._get_ranker", return_value=(mock_ranker, mock_rrc)),
        ):
            mock_settings.reranker.enabled = True
            mock_settings.retrieval.top_k_final = 4
            results = rerank("test", candidates)
        result_scores = [r.rerank_score for r in results]
        assert result_scores == sorted(result_scores, reverse=True)

    def test_rerank_score_populated_from_flashrank(self):
        candidates = [_make_result("only_one", rrf_score=0.02)]
        mock_ranker, mock_rrc = self._mock_ranker([0.88])
        with (
            patch("retrieval.reranker.settings") as mock_settings,
            patch("retrieval.reranker._get_ranker", return_value=(mock_ranker, mock_rrc)),
        ):
            mock_settings.reranker.enabled = True
            mock_settings.retrieval.top_k_final = 5
            results = rerank("q", candidates)
        assert results[0].rerank_score == pytest.approx(0.88)

    def test_flashrank_failure_falls_back_to_rrf_order(self):
        candidates = [
            _make_result("a", rrf_score=0.05),
            _make_result("b", rrf_score=0.02),
        ]
        mock_ranker = MagicMock()
        mock_ranker.rerank.side_effect = RuntimeError("model exploded")
        with (
            patch("retrieval.reranker.settings") as mock_settings,
            patch("retrieval.reranker._get_ranker", return_value=(mock_ranker, MagicMock())),
        ):
            mock_settings.reranker.enabled = True
            mock_settings.retrieval.top_k_final = 5
            results = rerank("q", candidates)
        assert len(results) == 2
        assert results[0].rerank_score >= results[1].rerank_score
