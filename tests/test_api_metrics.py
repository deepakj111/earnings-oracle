# tests/test_api_metrics.py
"""
Tests for api/metrics.py — Prometheus counters, histograms, and middleware.

Design:
  TestPrometheusMiddleware  — mocked pipeline; no model loading, no real LLM/Qdrant calls.
                              Only validates that middleware increments counters.
  TestRecordGenerationResult / TestRecordRetrievalResult / TestRecordCragResult
                            — call record_*() helpers directly with MagicMocks.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from api.main import app
from api.metrics import RAG_REGISTRY
from generation.models import GenerationResult

# ── Helper ────────────────────────────────────────────────────────────────────


def _sample_value(metric_name: str, labels: dict[str, str] | None = None) -> float:
    """
    Read the current value of a metric sample from RAG_REGISTRY.

    Family-level match if ANY of:
      a. family.name == metric_name  (exact)
      b. family.name == metric_name with _total stripped
      c. metric_name starts with family.name + "_"
         (histograms: family "rag_context_tokens_used" emits
          samples "rag_context_tokens_used_count", "_sum", "_bucket")

    Label matching is partial — only keys present in `labels` must match.
    Returns 0.0 if no matching sample is found.
    """
    for metric_family in RAG_REGISTRY.collect():
        base = metric_family.name
        family_matches = (
            base == metric_name
            or base == metric_name.removesuffix("_total")
            or metric_name.startswith(base + "_")
        )
        if not family_matches:
            continue

        for sample in metric_family.samples:
            if sample.name != metric_name:
                continue
            if labels is None:
                return sample.value
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value

    return 0.0


# ── Mock factories ────────────────────────────────────────────────────────────


def _make_mock_generation_result(
    answer: str = "Mock answer",
    grounded: bool = True,
    citations: list | None = None,
    total_tokens: int = 100,
    prompt_tokens: int = 60,
    completion_tokens: int = 40,
    model: str = "gpt-4.1-nano",
    context_tokens_used: int = 512,
    retrieval_failed: bool = False,
) -> MagicMock:
    result = MagicMock()
    result.answer = answer
    result.grounded = grounded
    result.citations = citations or []
    result.total_tokens = total_tokens
    result.prompt_tokens = prompt_tokens
    result.completion_tokens = completion_tokens
    result.model = model
    result.context_tokens_used = context_tokens_used
    result.retrieval_failed = retrieval_failed
    result.to_dict.return_value = {
        "answer": answer,
        "citations": [],
        "grounded": grounded,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "model": model,
        "context_tokens_used": context_tokens_used,
        "retrieval_failed": retrieval_failed,
    }
    return result


def _make_generation_result(
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    grounded: bool = True,
    retrieval_failed: bool = False,
    context_tokens_used: int = 1024,
    model: str = "gpt-4.1-nano",
) -> MagicMock:
    return _make_mock_generation_result(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        grounded=grounded,
        retrieval_failed=retrieval_failed,
        context_tokens_used=context_tokens_used,
        model=model,
    )


def _make_retrieval_result(
    total_candidates: int = 20,
    results: list | None = None,
) -> MagicMock:
    result = MagicMock()
    result.total_candidates = total_candidates
    result.results = results if results is not None else [MagicMock()] * 5
    return result


def _make_crag_result(action_value: str = "correct") -> MagicMock:
    result = MagicMock()
    result.action = MagicMock()
    result.action.value = action_value
    return result


def _make_mock_result() -> MagicMock:
    """
    Build a MagicMock that has real Python types for every attribute
    that the route handler uses in f-strings with format specs like :.5f, :.2f.
    A plain MagicMock() crashes on those format specs.
    """
    mock = MagicMock(spec=GenerationResult)
    mock.question = "What was Apple's revenue in Q4 2024?"
    mock.answer = "Apple reported revenue of $90.1B in Q4 2024 [1]."
    mock.citations = []
    mock.model = "gpt-4.1-nano"
    mock.prompt_tokens = 500
    mock.completion_tokens = 120
    mock.total_tokens = 620
    mock.context_chunks_used = 3
    mock.context_tokens_used = 1024
    mock.latency_seconds = 0.42
    mock.grounded = True
    mock.retrieval_failed = False
    mock.cost_estimate_usd = 0.000062
    mock.unique_tickers = ["AAPL"]
    mock.unique_sources = ["AAPL Q4 2024"]
    mock.to_dict.return_value = {
        "question": "What was Apple's revenue in Q4 2024?",
        "answer": "Apple reported revenue of $90.1B in Q4 2024 [1].",
        "citations": [],
        "model": "gpt-4.1-nano",
        "usage": {
            "prompt_tokens": 500,
            "completion_tokens": 120,
            "total_tokens": 620,
            "cost_estimate_usd": 0.000062,
        },
        "context": {
            "chunks_used": 3,
            "tokens_used": 1024,
        },
        "latency_seconds": 0.42,
        "grounded": True,
        "retrieval_failed": False,
        "unique_tickers": ["AAPL"],
        "unique_sources": ["AAPL Q4 2024"],
    }
    return mock


@pytest.fixture
def client() -> TestClient:
    """
    TestClient with the full pipeline mocked at the api.main level.

    The pipeline lives in app.state (set during lifespan). The correct
    patch target is api.main.FinancialRAGPipeline — this intercepts
    the constructor call inside lifespan() so app.state.pipeline becomes
    our mock instead of a real pipeline that tries to load models.

    QdrantClient is also patched so the lifespan does not attempt a
    real TCP connection to localhost:6333 during test setup.
    """
    mock_result = _make_mock_result()
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.ask.return_value = mock_result
    mock_pipeline_instance.ask_streaming.return_value = iter(
        ["Apple reported ", "$90.1B ", "in Q4 2024."]
    )

    with patch("api.main.FinancialRAGPipeline", return_value=mock_pipeline_instance):
        with patch("api.main.QdrantClient"):
            with TestClient(app) as client:
                yield client


# ── PrometheusMiddleware tests ────────────────────────────────────────────────


class TestPrometheusMiddleware:
    def test_successful_query_increments_http_request_counter(self, client: TestClient) -> None:
        before = _sample_value(
            "rag_http_requests_total",
            {"endpoint": "/query", "method": "POST", "status_code": "200"},
        )
        client.post(
            "/query/",
            json={"question": "What was Apple's revenue in Q4 2024?"},
        )
        after = _sample_value(
            "rag_http_requests_total",
            {"endpoint": "/query", "method": "POST", "status_code": "200"},
        )
        assert after > before, (
            "HTTP request counter should increment after a successful POST /query"
        )

    def test_health_liveness_increments_counter(self, client: TestClient) -> None:
        before = _sample_value(
            "rag_http_requests_total",
            {"endpoint": "/health/live", "method": "GET", "status_code": "200"},
        )
        client.get("/health/live")
        after = _sample_value(
            "rag_http_requests_total",
            {"endpoint": "/health/live", "method": "GET", "status_code": "200"},
        )
        assert after > before

    def test_duration_histogram_records_observation(self, client: TestClient) -> None:
        before = _sample_value(
            "rag_http_request_duration_seconds_count",
            {"endpoint": "/query"},
        )
        client.post(
            "/query/",
            json={"question": "What was Apple's revenue?"},
        )
        after = _sample_value(
            "rag_http_request_duration_seconds_count",
            {"endpoint": "/query"},
        )
        assert after > before

    def test_unknown_path_bucketed_as_other(self, client: TestClient) -> None:
        before = _sample_value(
            "rag_http_requests_total",
            {"endpoint": "/other", "method": "GET"},
        )
        client.get("/this/path/does/not/exist")
        after = _sample_value(
            "rag_http_requests_total",
            {"endpoint": "/other", "method": "GET"},
        )
        assert after > before


# ── record_generation_result tests ───────────────────────────────────────────


class TestRecordGenerationResult:
    def test_increments_prompt_tokens(self) -> None:
        from api.metrics import record_generation_result

        result = _make_generation_result(prompt_tokens=300, completion_tokens=50)
        before = _sample_value(
            "rag_llm_tokens_total", {"model": "gpt-4.1-nano", "token_type": "prompt"}
        )
        record_generation_result(result)
        after = _sample_value(
            "rag_llm_tokens_total", {"model": "gpt-4.1-nano", "token_type": "prompt"}
        )
        assert after - before == pytest.approx(300, abs=1)

    def test_increments_completion_tokens(self) -> None:
        from api.metrics import record_generation_result

        result = _make_generation_result(prompt_tokens=0, completion_tokens=75)
        before = _sample_value(
            "rag_llm_tokens_total", {"model": "gpt-4.1-nano", "token_type": "completion"}
        )
        record_generation_result(result)
        after = _sample_value(
            "rag_llm_tokens_total", {"model": "gpt-4.1-nano", "token_type": "completion"}
        )
        assert after - before == pytest.approx(75, abs=1)

    def test_increments_grounded_true_counter(self) -> None:
        from api.metrics import record_generation_result

        result = _make_generation_result(grounded=True)
        before = _sample_value("rag_grounded_responses_total", {"grounded": "true"})
        record_generation_result(result)
        after = _sample_value("rag_grounded_responses_total", {"grounded": "true"})
        assert after - before == pytest.approx(1.0)

    def test_increments_grounded_false_counter(self) -> None:
        from api.metrics import record_generation_result

        result = _make_generation_result(grounded=False)
        before = _sample_value("rag_grounded_responses_total", {"grounded": "false"})
        record_generation_result(result)
        after = _sample_value("rag_grounded_responses_total", {"grounded": "false"})
        assert after - before == pytest.approx(1.0)

    def test_increments_retrieval_failed_counter(self) -> None:
        from api.metrics import record_generation_result

        result = _make_generation_result(retrieval_failed=True)
        before = _sample_value("rag_retrieval_failed_total")
        record_generation_result(result)
        after = _sample_value("rag_retrieval_failed_total")
        assert after - before == pytest.approx(1.0)

    def test_records_context_tokens_histogram(self) -> None:
        from api.metrics import record_generation_result

        result = _make_generation_result(context_tokens_used=2048)
        before = _sample_value("rag_context_tokens_used_count")
        record_generation_result(result)
        after = _sample_value("rag_context_tokens_used_count")
        assert after - before == pytest.approx(1.0)


# ── record_retrieval_result tests ─────────────────────────────────────────────


class TestRecordRetrievalResult:
    def test_records_candidates_histogram(self) -> None:
        from api.metrics import record_retrieval_result

        result = _make_retrieval_result(total_candidates=35, results=[MagicMock()] * 5)
        before = _sample_value("rag_retrieval_candidates_count")
        record_retrieval_result(result)
        after = _sample_value("rag_retrieval_candidates_count")
        assert after - before == pytest.approx(1.0)

    def test_records_results_returned_histogram(self) -> None:
        from api.metrics import record_retrieval_result

        result = _make_retrieval_result(total_candidates=20, results=[MagicMock()] * 3)
        before = _sample_value("rag_retrieval_results_returned_count")
        record_retrieval_result(result)
        after = _sample_value("rag_retrieval_results_returned_count")
        assert after - before == pytest.approx(1.0)


# ── record_crag_result tests ──────────────────────────────────────────────────


class TestRecordCragResult:
    @pytest.mark.parametrize("action", ["correct", "ambiguous", "incorrect"])
    def test_increments_crag_action_counter(self, action: str) -> None:
        from api.metrics import record_crag_result

        result = _make_crag_result(action_value=action)
        before = _sample_value("rag_crag_actions_total", {"action": action})
        record_crag_result(result)
        after = _sample_value("rag_crag_actions_total", {"action": action})
        assert after - before == pytest.approx(1.0)
