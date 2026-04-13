# tests/test_api_query.py
"""
Tests for api/routes/query.py

Coverage:
  POST /query         — structured response (ask)
    · Happy path: grounded answer with citations
    · Verbose mode: query_summary / retrieval_summary included
    · Metadata filter round-trip (ticker, year, quarter)
    · Pydantic validation failures: short question, invalid ticker, bad quarter
    · Pipeline error propagation
    · _to_metadata_filter helper logic
    · _serialise helper output shape

  POST /query/stream  — SSE streaming
    · Response headers (content-type, X-Request-ID, Cache-Control)
    · SSE body: token events + [DONE] sentinel
    · Correct tokens forwarded from mock pipeline
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api.models import AskRequest, MetadataFilterIn
from api.routes.query import _serialise, _to_metadata_filter
from generation.models import GenerationResult
from retrieval.models import MetadataFilter

# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestToMetadataFilter:
    def test_none_when_filter_is_none(self) -> None:
        req = AskRequest(question="What is Apple revenue?", filter=None)
        assert _to_metadata_filter(req) is None

    def test_none_when_all_filter_fields_empty(self) -> None:
        req = AskRequest(
            question="What is Apple revenue?",
            filter=MetadataFilterIn(ticker=None, year=None, quarter=None),
        )
        assert _to_metadata_filter(req) is None

    def test_builds_filter_with_ticker_only(self) -> None:
        req = AskRequest(
            question="Apple revenue?",
            filter=MetadataFilterIn(ticker="AAPL"),
        )
        result = _to_metadata_filter(req)
        assert isinstance(result, MetadataFilter)
        assert result.ticker == "AAPL"
        assert result.year is None
        assert result.quarter is None

    def test_builds_filter_with_all_fields(self) -> None:
        req = AskRequest(
            question="Apple Q4 2024?",
            filter=MetadataFilterIn(ticker="NVDA", year=2024, quarter="Q3"),
        )
        result = _to_metadata_filter(req)
        assert result is not None
        assert result.ticker == "NVDA"
        assert result.year == 2024
        assert result.quarter == "Q3"

    def test_ticker_lowercased_normalised(self) -> None:
        """MetadataFilterIn validator uppercases ticker; _to_metadata_filter preserves it."""
        req = AskRequest(
            question="Apple revenue?",
            filter=MetadataFilterIn(ticker="aapl"),  # validator uppercases
        )
        result = _to_metadata_filter(req)
        assert result is not None
        assert result.ticker == "AAPL"


class TestSerialise:
    def test_question_preserved(self, sample_generation_result: GenerationResult) -> None:
        resp = _serialise(
            sample_generation_result, verbose=False, query_summary=None, retrieval_summary=None
        )
        assert resp.question == sample_generation_result.question

    def test_answer_preserved(self, sample_generation_result: GenerationResult) -> None:
        resp = _serialise(
            sample_generation_result, verbose=False, query_summary=None, retrieval_summary=None
        )
        assert resp.answer == sample_generation_result.answer

    def test_citations_mapped(self, sample_generation_result: GenerationResult) -> None:
        resp = _serialise(
            sample_generation_result, verbose=False, query_summary=None, retrieval_summary=None
        )
        assert len(resp.citations) == 1
        c = resp.citations[0]
        assert c.ticker == "AAPL"
        assert c.fiscal_period == "Q4 2024"
        assert c.rerank_score == round(0.9821, 4)

    def test_verbose_false_hides_summaries(
        self, sample_generation_result: GenerationResult
    ) -> None:
        resp = _serialise(
            sample_generation_result, verbose=False, query_summary="qs", retrieval_summary="rs"
        )
        assert resp.query_summary is None
        assert resp.retrieval_summary is None

    def test_verbose_true_exposes_summaries(
        self, sample_generation_result: GenerationResult
    ) -> None:
        resp = _serialise(
            sample_generation_result, verbose=True, query_summary="qs", retrieval_summary="rs"
        )
        assert resp.query_summary == "qs"
        assert resp.retrieval_summary == "rs"

    def test_usage_tokens_rounded(self, sample_generation_result: GenerationResult) -> None:
        resp = _serialise(
            sample_generation_result, verbose=False, query_summary=None, retrieval_summary=None
        )
        assert resp.usage.total_tokens == 1280
        assert resp.usage.prompt_tokens == 1200
        assert resp.usage.completion_tokens == 80

    def test_latency_rounded_to_3_dp(self, sample_generation_result: GenerationResult) -> None:
        resp = _serialise(
            sample_generation_result, verbose=False, query_summary=None, retrieval_summary=None
        )
        assert resp.latency_seconds == round(2.34, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests — POST /query
# ─────────────────────────────────────────────────────────────────────────────


class TestAskEndpoint:
    def test_returns_200_on_valid_question(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "What was Apple revenue Q4 2024?"})
        assert resp.status_code == 200

    def test_response_shape(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "What was Apple revenue Q4 2024?"})
        data = resp.json()
        # Top-level keys
        for key in (
            "question",
            "answer",
            "citations",
            "grounded",
            "model",
            "usage",
            "context",
            "latency_seconds",
            "unique_tickers",
            "unique_sources",
            "retrieval_failed",
        ):
            assert key in data, f"Missing key: {key}"

    def test_answer_text_present(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "What was Apple revenue Q4 2024?"})
        assert resp.json()["answer"] != ""

    def test_citations_list_populated(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "What was Apple revenue Q4 2024?"})
        citations = resp.json()["citations"]
        assert len(citations) == 1
        assert citations[0]["ticker"] == "AAPL"

    def test_grounded_true_for_mocked_result(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "What was Apple revenue Q4 2024?"})
        assert resp.json()["grounded"] is True

    def test_usage_block_present(self, client: TestClient) -> None:
        data = client.post("/query/", json={"question": "Apple revenue?"}).json()
        usage = data["usage"]
        assert usage["total_tokens"] == 1280
        assert "cost_estimate_usd" in usage

    def test_context_block_present(self, client: TestClient) -> None:
        data = client.post("/query/", json={"question": "Apple revenue?"}).json()
        assert data["context"]["chunks_used"] == 3
        assert data["context"]["tokens_used"] == 2048

    def test_unique_tickers_populated(self, client: TestClient) -> None:
        data = client.post("/query/", json={"question": "Apple revenue?"}).json()
        assert "AAPL" in data["unique_tickers"]

    def test_question_too_short_returns_422(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "Hi"})
        assert resp.status_code == 422

    def test_question_too_long_returns_422(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "A" * 2001})
        assert resp.status_code == 422

    def test_missing_question_returns_422(self, client: TestClient) -> None:
        resp = client.post("/query/", json={})
        assert resp.status_code == 422

    def test_filter_with_valid_ticker_passes(self, client: TestClient) -> None:
        resp = client.post(
            "/query/",
            json={"question": "Apple revenue Q4 2024?", "filter": {"ticker": "AAPL"}},
        )
        assert resp.status_code == 200

    def test_filter_with_invalid_ticker_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/query/",
            json={"question": "Revenue Q4?", "filter": {"ticker": "INVALID"}},
        )
        assert resp.status_code == 422

    def test_filter_with_invalid_quarter_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/query/",
            json={"question": "Revenue Q4?", "filter": {"quarter": "Q5"}},
        )
        assert resp.status_code == 422

    def test_filter_passed_to_pipeline_as_metadata_filter(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        client.post(
            "/query/",
            json={
                "question": "Apple revenue Q4 2024?",
                "filter": {"ticker": "AAPL", "year": 2024, "quarter": "Q4"},
            },
        )
        mock_pipeline.ask.assert_called_once()
        _, kwargs = mock_pipeline.ask.call_args
        mf = kwargs["metadata_filter"]
        assert mf.ticker == "AAPL"
        assert mf.year == 2024
        assert mf.quarter == "Q4"

    def test_no_filter_passes_none_to_pipeline(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        client.post("/query/", json={"question": "What is Apple revenue?"})
        mock_pipeline.ask.assert_called_once()
        _, kwargs = mock_pipeline.ask.call_args
        assert kwargs["metadata_filter"] is None

    def test_verbose_false_hides_summaries(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "Apple revenue?", "verbose": False})
        data = resp.json()
        assert data["query_summary"] is None
        assert data["retrieval_summary"] is None

    def test_verbose_true_returns_summaries(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        resp = client.post("/query/", json={"question": "Apple revenue?", "verbose": True})
        data = resp.json()
        assert data["query_summary"] is not None
        assert data["retrieval_summary"] is not None
        # ask_verbose was called, not ask
        mock_pipeline.ask_verbose.assert_called_once()
        mock_pipeline.ask.assert_not_called()

    def test_request_id_header_present(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "Apple revenue?"})
        assert "x-request-id" in resp.headers

    def test_response_time_header_present(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "Apple revenue?"})
        assert "x-response-time-ms" in resp.headers
        assert int(resp.headers["x-response-time-ms"]) >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests — POST /query/stream
# ─────────────────────────────────────────────────────────────────────────────


class TestAskStreamEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.post("/query/stream", json={"question": "Apple revenue Q4 2024?"})
        assert resp.status_code == 200

    def test_content_type_is_event_stream(self, client: TestClient) -> None:
        resp = client.post("/query/stream", json={"question": "Apple revenue Q4 2024?"})
        assert "text/event-stream" in resp.headers["content-type"]

    def test_cache_control_no_cache(self, client: TestClient) -> None:
        resp = client.post("/query/stream", json={"question": "Apple revenue Q4 2024?"})
        assert resp.headers.get("cache-control") == "no-cache"

    def test_sse_body_contains_done_sentinel(self, client: TestClient) -> None:
        resp = client.post("/query/stream", json={"question": "Apple revenue Q4 2024?"})
        assert "data: [DONE]" in resp.text

    def test_sse_body_contains_token_events(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        mock_pipeline.ask_streaming.return_value = iter(["Hello ", "World"])
        resp = client.post("/query/stream", json={"question": "Apple revenue Q4 2024?"})
        body = resp.text
        assert 'data: {"token": "Hello "}' in body
        assert 'data: {"token": "World"}' in body

    def test_sse_events_ordered_before_done(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        mock_pipeline.ask_streaming.return_value = iter(["Token1"])
        resp = client.post("/query/stream", json={"question": "Apple revenue?"})
        body = resp.text
        token_pos = body.find("Token1")
        done_pos = body.find("[DONE]")
        assert token_pos < done_pos

    def test_question_too_short_returns_422(self, client: TestClient) -> None:
        resp = client.post("/query/stream", json={"question": "Hi"})
        assert resp.status_code == 422

    def test_x_request_id_in_response_headers(self, client: TestClient) -> None:
        resp = client.post("/query/stream", json={"question": "Apple revenue?"})
        assert "x-request-id" in resp.headers

    def test_pipeline_ask_streaming_called(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        mock_pipeline.ask_streaming.return_value = iter([])
        client.post("/query/stream", json={"question": "NVDA data center revenue?"})
        mock_pipeline.ask_streaming.assert_called_once()
        _, kwargs = mock_pipeline.ask_streaming.call_args
        assert "NVDA" in kwargs["question"]
