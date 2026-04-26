"""Tests for retrieval/models.py — MetadataFilter, SearchResult, RetrievalResult."""

from __future__ import annotations

from retrieval.models import MetadataFilter, RetrievalResult, SearchResult


def _make_result(**kwargs) -> SearchResult:
    defaults = {
        "chunk_id": "AAPL_2024-10-31_abc_0",
        "parent_id": "AAPL_2024-10-31_abc",
        "text": "Apple reported revenue of $94.9 billion.",
        "parent_text": "Apple reported revenue of $94.9 billion for Q1 2024.",
        "rrf_score": 0.032,
        "rerank_score": 0.91,
        "ticker": "AAPL",
        "company": "Apple",
        "date": "2024-10-31",
        "year": 2024,
        "quarter": "Q1",
        "fiscal_period": "Q1 2024",
        "section_title": "Revenue",
        "doc_type": "earnings_release",
        "source": "both",
    }
    defaults.update(kwargs)
    return SearchResult(**defaults)


class TestMetadataFilter:
    def test_all_fields_optional(self) -> None:
        mf = MetadataFilter()
        assert mf.ticker is None
        assert mf.year is None
        assert mf.quarter is None

    def test_ticker_only(self) -> None:
        mf = MetadataFilter(ticker="NVDA")
        assert mf.ticker == "NVDA"
        assert mf.year is None

    def test_all_fields_set(self) -> None:
        mf = MetadataFilter(ticker="MSFT", year=2024, quarter="Q2")
        assert mf.ticker == "MSFT"
        assert mf.year == 2024
        assert mf.quarter == "Q2"


class TestSearchResultFromPayload:
    def test_constructs_from_full_payload(self) -> None:
        payload = {
            "chunk_id": "AAPL_2024-10-31_abc_0",
            "parent_id": "AAPL_2024-10-31_abc",
            "text": "Revenue grew 6% YoY.",
            "ticker": "AAPL",
            "company": "Apple",
            "date": "2024-10-31",
            "year": 2024,
            "quarter": "Q1",
            "fiscal_period": "Q1 2024",
            "section_title": "Revenue",
            "doc_type": "earnings_release",
        }
        r = SearchResult.from_payload(payload, rrf_score=0.02, source="dense")
        assert r.chunk_id == "AAPL_2024-10-31_abc_0"
        assert r.ticker == "AAPL"
        assert r.rrf_score == 0.02
        assert r.source == "dense"

    def test_rerank_score_initialised_to_neg_inf(self) -> None:
        r = SearchResult.from_payload({}, rrf_score=0.01, source="bm25")
        assert r.rerank_score == float("-inf")

    def test_missing_payload_fields_use_defaults(self) -> None:
        r = SearchResult.from_payload({}, rrf_score=0.0, source="dense")
        assert r.chunk_id == ""
        assert r.ticker == ""
        assert r.year == 0
        assert r.parent_id is None

    def test_parent_text_initially_same_as_text(self) -> None:
        payload = {"chunk_id": "x", "text": "Some child text."}
        r = SearchResult.from_payload(payload, rrf_score=0.01, source="dense")
        assert r.parent_text == r.text

    def test_source_stored_correctly(self) -> None:
        for src in ("dense", "bm25", "both"):
            r = SearchResult.from_payload({}, rrf_score=0.0, source=src)
            assert r.source == src


class TestRetrievalResult:
    def test_is_empty_true_when_no_results(self) -> None:
        rr = RetrievalResult(query="test", results=[], reranked=False, total_candidates=0)
        assert rr.is_empty is True

    def test_is_empty_false_when_results_present(self) -> None:
        rr = RetrievalResult(
            query="test", results=[_make_result()], reranked=True, total_candidates=20
        )
        assert rr.is_empty is False

    def test_summary_contains_query(self) -> None:
        rr = RetrievalResult(
            query="Apple revenue Q1 2024", results=[], reranked=False, total_candidates=0
        )
        assert "Apple revenue Q1 2024" in rr.summary()

    def test_summary_contains_candidate_count(self) -> None:
        rr = RetrievalResult(query="q", results=[], reranked=False, total_candidates=17)
        assert "17" in rr.summary()

    def test_summary_contains_result_count(self) -> None:
        rr = RetrievalResult(
            query="q",
            results=[_make_result(), _make_result(chunk_id="other")],
            reranked=True,
            total_candidates=20,
        )
        assert "2" in rr.summary()

    def test_summary_shows_filter_when_set(self) -> None:
        rr = RetrievalResult(
            query="q",
            results=[],
            reranked=False,
            total_candidates=0,
            metadata_filter=MetadataFilter(ticker="TSLA"),
        )
        assert "TSLA" in rr.summary()

    def test_summary_omits_filter_when_none(self) -> None:
        rr = RetrievalResult(query="q", results=[], reranked=False, total_candidates=0)
        assert "Filter" not in rr.summary()

    def test_failed_techniques_defaults_to_empty(self) -> None:
        rr = RetrievalResult(query="q", results=[], reranked=False, total_candidates=0)
        assert rr.failed_techniques == []
