"""
Tests for generation/generator.py

Coverage:
  _extract_citations  — valid citations, out-of-range, no citations, multi-citation claims
  _is_grounded        — grounded answers, all ungrounded phrases, partial grounding
  Generator.generate  — empty retrieval fast path, mocked LLM call, full result structure
  Generator.generate_streaming — streaming token yield with mocked OpenAI stream
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from generation.generator import Generator, _extract_citations, _is_grounded
from generation.models import GenerationResult
from generation.prompts import UNGROUNDED_PHRASES
from retrieval.models import RetrievalResult, SearchResult


@pytest.fixture(autouse=True)
def caplog_loguru(caplog):
    """Bridge loguru to pytest's caplog."""

    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropagateHandler(), format="{message}")
    yield caplog
    logger.remove(handler_id)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(
    chunk_id: str = "c1",
    parent_id: str | None = "p1",
    ticker: str = "AAPL",
    fiscal_period: str = "Q4 2024",
    section_title: str = "Revenue",
    rerank_score: float = 0.9,
    text: str = "Apple reported $100B in revenue.",
    parent_text: str = "Apple reported $100B in revenue for Q4 2024.",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        parent_id=parent_id,
        text=text,
        parent_text=parent_text,
        rrf_score=0.05,
        rerank_score=rerank_score,
        ticker=ticker,
        company="Apple",
        date="2024-10-31",
        year=2024,
        quarter="Q4",
        fiscal_period=fiscal_period,
        section_title=section_title,
        doc_type="earnings_release",
        source="both",
    )


def _make_retrieval_result(
    results: list[SearchResult] | None = None,
    reranked: bool = True,
) -> RetrievalResult:
    return RetrievalResult(
        query="What was Apple's revenue?",
        results=results or [_make_result()],
        reranked=reranked,
        total_candidates=len(results) if results else 1,
    )


def _make_empty_retrieval() -> RetrievalResult:
    return RetrievalResult(
        query="What was Apple's revenue?",
        results=[],
        reranked=False,
        total_candidates=0,
    )


def _mock_openai_response(answer: str, prompt_tokens: int = 150, completion_tokens: int = 80):
    """Build a mock OpenAI ChatCompletion response object."""
    choice = MagicMock()
    choice.message.content = answer

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


# ── _extract_citations ────────────────────────────────────────────────────────


class TestExtractCitations:
    def _make_results(self, n: int) -> list[SearchResult]:
        return [_make_result(chunk_id=f"c{i}", rerank_score=1.0 - i * 0.1) for i in range(n)]

    def test_single_citation(self) -> None:
        results = self._make_results(3)
        citations = _extract_citations("Revenue was $100B [1].", results)
        assert len(citations) == 1
        assert citations[0].index == 1
        assert citations[0].chunk_id == "c0"

    def test_multiple_distinct_citations(self) -> None:
        results = self._make_results(3)
        citations = _extract_citations("Revenue [1]. EPS [2]. Guidance [3].", results)
        assert len(citations) == 3
        assert [c.index for c in citations] == [1, 2, 3]

    def test_duplicate_citation_deduplicated(self) -> None:
        """Same citation number appearing twice → only one Citation object."""
        results = self._make_results(3)
        citations = _extract_citations("Revenue [1] was good [1].", results)
        assert len(citations) == 1
        assert citations[0].index == 1

    def test_multi_citation_claim(self) -> None:
        """[1][2] on a single claim → two separate Citation objects."""
        results = self._make_results(3)
        citations = _extract_citations("Revenue [1][2] shows strength.", results)
        assert len(citations) == 2
        indices = {c.index for c in citations}
        assert indices == {1, 2}

    def test_no_citations_returns_empty(self) -> None:
        results = self._make_results(3)
        citations = _extract_citations("Revenue was strong.", results)
        assert citations == []

    def test_out_of_range_citation_skipped(self, caplog) -> None:
        """[99] with only 2 results → skipped with a warning, no crash."""
        import logging

        results = self._make_results(2)
        with caplog.at_level(logging.WARNING):
            citations = _extract_citations("Revenue [99].", results)
        assert citations == []
        assert "99" in caplog.text or "Hallucinated" in caplog.text

    def test_citation_metadata_populated(self) -> None:
        r = _make_result(
            chunk_id="aapl_c1",
            ticker="AAPL",
            fiscal_period="Q4 2024",
            section_title="Services Revenue",
            rerank_score=0.95,
            parent_text="Apple Services hit $26B.",
        )
        citations = _extract_citations("Services revenue grew [1].", [r])
        assert len(citations) == 1
        c = citations[0]
        assert c.chunk_id == "aapl_c1"
        assert c.ticker == "AAPL"
        assert c.fiscal_period == "Q4 2024"
        assert c.section_title == "Services Revenue"
        assert c.rerank_score == pytest.approx(0.95)
        assert "Apple Services" in c.excerpt

    def test_excerpt_truncated_to_250_chars(self) -> None:
        long_text = "x" * 500
        r = _make_result(parent_text=long_text)
        citations = _extract_citations("[1]", [r])
        assert len(citations[0].excerpt) <= 250

    def test_citations_sorted_by_index(self) -> None:
        """Citations must be returned in ascending index order."""
        results = self._make_results(5)
        citations = _extract_citations("[3] then [1] then [5].", results)
        indices = [c.index for c in citations]
        assert indices == sorted(indices)

    def test_empty_answer_returns_empty(self) -> None:
        results = self._make_results(3)
        assert _extract_citations("", results) == []

    def test_empty_results_with_citation_returns_empty(self, caplog) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            citations = _extract_citations("Revenue [1].", [])
        assert citations == []


# ── _is_grounded ──────────────────────────────────────────────────────────────


class TestIsGrounded:
    def test_normal_answer_is_grounded(self) -> None:
        answer = "Apple reported $100B in revenue for Q4 2024 [1]."
        assert _is_grounded(answer) is True

    def test_empty_answer_is_grounded(self) -> None:
        # Empty doesn't trigger ungrounded phrases — caller handles empty separately
        assert _is_grounded("") is True

    @pytest.mark.parametrize("phrase", UNGROUNDED_PHRASES)
    def test_each_ungrounded_phrase_detected(self, phrase: str) -> None:
        answer = f"The {phrase} to answer this question."
        assert _is_grounded(answer) is False

    def test_case_insensitive_detection(self) -> None:
        answer = "The PROVIDED DOCUMENTS DO NOT CONTAIN SUFFICIENT INFORMATION."
        assert _is_grounded(answer) is False

    def test_phrase_embedded_in_sentence_detected(self) -> None:
        answer = (
            "While I can provide some context, the documents do not contain "
            "sufficient information about the specific metric you asked for."
        )
        assert _is_grounded(answer) is False

    def test_partial_phrase_not_detected(self) -> None:
        """'cannot' alone shouldn't trigger 'cannot determine'."""
        answer = "I cannot confirm the exact figures without more context."
        # 'cannot determine' is not present — should be grounded
        assert _is_grounded(answer) is True


# ── Generator.generate ────────────────────────────────────────────────────────


class TestGeneratorGenerate:
    @patch("generation.generator._get_client")
    def test_empty_retrieval_returns_no_context_answer(self, mock_get_client) -> None:
        """Empty RetrievalResult must return the no-context fallback immediately."""
        generator = Generator()
        result = generator.generate(
            question="What was Apple's revenue?",
            retrieval_result=_make_empty_retrieval(),
        )
        assert result.retrieval_failed is True
        assert result.grounded is False
        assert result.citations == []
        assert result.prompt_tokens == 0
        assert result.total_tokens == 0
        assert "No relevant documents" in result.answer
        # LLM must NOT be called for empty retrieval
        mock_get_client.assert_not_called()

    @patch("generation.generator._call_llm")
    def test_generate_returns_generation_result(self, mock_call_llm) -> None:
        """Happy path: mocked LLM returns an answer with citations."""
        mock_call_llm.return_value = (
            "Apple reported revenue of $94.9B [1], up 6% YoY [1].",
            200,
            60,
        )
        generator = Generator()
        result = generator.generate(
            question="What was Apple's Q4 2024 revenue?",
            retrieval_result=_make_retrieval_result(),
        )

        assert isinstance(result, GenerationResult)
        assert result.retrieval_failed is False
        assert result.grounded is True
        assert result.prompt_tokens == 200
        assert result.completion_tokens == 60
        assert result.total_tokens == 260
        assert result.latency_seconds >= 0
        assert result.model is not None
        assert "Apple" in result.answer

    @patch("generation.generator._call_llm")
    def test_citations_extracted_from_answer(self, mock_call_llm) -> None:
        mock_call_llm.return_value = (
            "Revenue was $94.9B [1]. Services was $26.3B [1].",
            200,
            40,
        )
        r = _make_result(chunk_id="aapl_c1")
        result = Generator().generate(
            question="Revenue?",
            retrieval_result=_make_retrieval_result([r]),
        )
        assert len(result.citations) == 1
        assert result.citations[0].index == 1
        assert result.citations[0].chunk_id == "aapl_c1"

    @patch("generation.generator._call_llm")
    def test_ungrounded_answer_sets_grounded_false(self, mock_call_llm) -> None:
        mock_call_llm.return_value = (
            "The provided documents do not contain sufficient information to answer this question.",
            150,
            25,
        )
        result = Generator().generate(
            question="What was the 5-year CAGR?",
            retrieval_result=_make_retrieval_result(),
        )
        assert result.grounded is False
        assert result.citations == []

    @patch("generation.generator._call_llm")
    def test_context_chunks_used_populated(self, mock_call_llm) -> None:
        mock_call_llm.return_value = ("Revenue [1].", 100, 20)
        results = [_make_result(chunk_id=f"c{i}") for i in range(5)]
        result = Generator().generate(
            question="Revenue?",
            retrieval_result=_make_retrieval_result(results),
        )
        assert 1 <= result.context_chunks_used <= 5

    @patch("generation.generator._call_llm")
    def test_context_tokens_used_populated(self, mock_call_llm) -> None:
        mock_call_llm.return_value = ("Revenue [1].", 100, 20)
        result = Generator().generate(
            question="Revenue?",
            retrieval_result=_make_retrieval_result(),
        )
        assert result.context_tokens_used > 0

    @patch("generation.generator._call_llm")
    def test_format_answer_with_citations_output(self, mock_call_llm) -> None:
        mock_call_llm.return_value = ("Revenue was $100B [1].", 100, 20)
        r = _make_result(ticker="AAPL", fiscal_period="Q4 2024", section_title="Revenue")
        result = Generator().generate(
            question="Revenue?",
            retrieval_result=_make_retrieval_result([r]),
        )
        formatted = result.format_answer_with_citations()
        assert "Revenue was $100B [1]." in formatted
        assert "Sources:" in formatted
        assert "[1]" in formatted

    @patch("generation.generator._call_llm")
    def test_to_dict_structure(self, mock_call_llm) -> None:
        mock_call_llm.return_value = ("Revenue [1].", 100, 20)
        result = Generator().generate(
            question="Revenue?",
            retrieval_result=_make_retrieval_result(),
        )
        d = result.to_dict()
        assert "question" in d
        assert "answer" in d
        assert "citations" in d
        assert "usage" in d
        assert "context" in d
        assert "latency_seconds" in d
        assert "grounded" in d
        assert "retrieval_failed" in d
        assert "unique_sources" in d

    @patch("generation.generator._call_llm")
    def test_unique_sources_populated(self, mock_call_llm) -> None:
        mock_call_llm.return_value = ("Revenue [1][2].", 200, 40)
        r1 = _make_result("c1", ticker="AAPL", fiscal_period="Q4 2024", parent_id="p1")
        r2 = _make_result("c2", ticker="NVDA", fiscal_period="Q3 2024", parent_id="p2")
        result = Generator().generate(
            question="Compare AAPL and NVDA revenue?",
            retrieval_result=_make_retrieval_result([r1, r2]),
        )
        sources = result.unique_sources
        assert "AAPL Q4 2024" in sources
        assert "NVDA Q3 2024" in sources


# ── Generator.generate_streaming ─────────────────────────────────────────────


class TestGeneratorStreaming:
    def test_empty_retrieval_yields_no_context_message(self) -> None:
        generator = Generator()
        tokens = list(
            generator.generate_streaming(
                question="Revenue?",
                retrieval_result=_make_empty_retrieval(),
            )
        )
        full_text = "".join(tokens)
        assert "No relevant documents" in full_text

    @patch("generation.generator._get_client")
    def test_streaming_yields_tokens(self, mock_get_client) -> None:
        """Mock the streaming API and verify tokens are yielded correctly."""

        # Build fake streaming chunks
        def _fake_chunks():
            for word in ["Apple ", "reported ", "$100B ", "revenue [1]."]:
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = word
                yield chunk

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=_fake_chunks())
        mock_stream.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream
        mock_get_client.return_value = mock_client

        generator = Generator()
        tokens = list(
            generator.generate_streaming(
                question="Revenue?",
                retrieval_result=_make_retrieval_result(),
            )
        )
        full_text = "".join(tokens)
        assert "Apple" in full_text
        assert "$100B" in full_text
