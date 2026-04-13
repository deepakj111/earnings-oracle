# tests/conftest.py
"""
Shared pytest fixtures for the Financial RAG test suite.

Provides:
  - Environment setup (fake API keys so settings.validate() does not crash)
  - Mock Citation / GenerationResult instances for stable assertions
  - Mock FinancialRAGPipeline with all three call variants pre-configured
  - Mock QdrantClient with realistic collection scaffolding
  - FastAPI TestClient wired to a fresh app whose lifespan is replaced with
    a lightweight test lifespan that sets app.state from fixtures

Lifespan strategy
-----------------
api/main.py defines the real lifespan which calls settings.validate() and
downloads ML models — both unacceptable in unit tests.  We patch
api.main.lifespan with a no-op coroutine that merely sets app.state from
the fixture mocks, then call create_app() inside the patch context so the
fresh FastAPI instance captures the test lifespan.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from generation.models import Citation, GenerationResult
from rag_pipeline import FinancialRAGPipeline

# ── Environment ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Inject fake env vars so the Settings singleton does not raise on import
    and so settings.validate() does not fail if it is called.
    These values never reach real external services — all network calls are mocked.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0000000000000000000000000000000000000000000000")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")


# ── Domain object factories ────────────────────────────────────────────────────


@pytest.fixture
def sample_citation() -> Citation:
    """A fully populated Citation for use in assertions."""
    return Citation(
        index=1,
        chunk_id="AAPL_2024-10-31_abc12345_0_c0",
        parent_id="AAPL_2024-10-31_abc12345_0",
        ticker="AAPL",
        company="Apple",
        date="2024-10-31",
        fiscal_period="Q4 2024",
        section_title="Revenue",
        doc_type="earnings_release",
        source="both",
        rerank_score=0.9821,
        excerpt="Apple reported total net sales of $94.9 billion for Q4 2024.",
    )


@pytest.fixture
def sample_generation_result(sample_citation: Citation) -> GenerationResult:
    """A fully populated GenerationResult that mirrors what the real pipeline returns."""
    return GenerationResult(
        question="What was Apple's revenue in Q4 2024?",
        answer="Apple reported total net sales of $94.9 billion in Q4 2024 [1].",
        citations=[sample_citation],
        model="gpt-4.1-nano",
        prompt_tokens=1200,
        completion_tokens=80,
        total_tokens=1280,
        context_chunks_used=3,
        context_tokens_used=2048,
        latency_seconds=2.34,
        grounded=True,
        retrieval_failed=False,
    )


@pytest.fixture
def sample_generation_result_ungrounded() -> GenerationResult:
    """A GenerationResult where the model could not ground its answer."""
    return GenerationResult(
        question="What was Berkshire Hathaway's Q4 2024 revenue?",
        answer="The provided documents do not contain sufficient information to answer this question.",
        citations=[],
        model="gpt-4.1-nano",
        prompt_tokens=900,
        completion_tokens=20,
        total_tokens=920,
        context_chunks_used=0,
        context_tokens_used=0,
        latency_seconds=1.1,
        grounded=False,
        retrieval_failed=True,
    )


# ── Mock pipeline ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_pipeline(
    sample_generation_result: GenerationResult,
    sample_generation_result_ungrounded: GenerationResult,
) -> MagicMock:
    """
    MagicMock of FinancialRAGPipeline with all three call variants pre-wired.

    ask()         → sample_generation_result  (grounded, has citations)
    ask_verbose() → (result, query_summary, retrieval_summary)
    ask_streaming → yields three token strings
    """
    pipeline = MagicMock(spec=FinancialRAGPipeline)

    pipeline.ask.return_value = sample_generation_result

    pipeline.ask_verbose.return_value = (
        sample_generation_result,
        "Original: What was Apple's revenue?\nHyDE doc: Apple reported...",
        "Query: ...\nCandidates: 20\nReturned: 5",
    )

    # ask_streaming must be an iterable each time it is called
    pipeline.ask_streaming.return_value = iter(
        ["Apple ", "reported ", "$94.9B ", "in ", "revenue [1]."]
    )

    return pipeline


# ── Mock Qdrant ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_qdrant() -> MagicMock:
    """
    MagicMock of QdrantClient configured to report the earnings_transcripts
    collection as present with 5,000 points.
    """
    qdrant = MagicMock(spec=QdrantClient)

    # get_collections() response
    mock_coll_desc = MagicMock()
    mock_coll_desc.name = "earnings_transcripts"
    mock_get_collections_resp = MagicMock()
    mock_get_collections_resp.collections = [mock_coll_desc]
    qdrant.get_collections.return_value = mock_get_collections_resp

    # get_collection() response — points_count is Optional[int] in qdrant-client 1.7+
    mock_coll_info = MagicMock()
    mock_coll_info.points_count = 5000
    qdrant.get_collection.return_value = mock_coll_info

    return qdrant


# ── FastAPI TestClient ─────────────────────────────────────────────────────────


@pytest.fixture
def client(mock_pipeline: MagicMock, mock_qdrant: MagicMock) -> TestClient:  # type: ignore[type-arg]
    """
    Return a synchronous TestClient whose app has:
      • A lightweight test lifespan (no model downloads, no settings.validate())
      • app.state.pipeline  = mock_pipeline
      • app.state.qdrant    = mock_qdrant
      • app.state.startup_time set 60 s in the past

    The TestClient is yielded as a context manager so lifespan startup/shutdown
    runs correctly.
    """
    import api.main  # ensure the module is cached before patching

    @asynccontextmanager
    async def _test_lifespan(app):  # type: ignore[no-untyped-def]
        app.state.pipeline = mock_pipeline
        app.state.qdrant = mock_qdrant
        app.state.startup_time = time.time() - 60.0
        yield

    with patch("api.main.lifespan", _test_lifespan):
        test_app = api.main.create_app()

    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def client_no_pipeline(mock_qdrant: MagicMock) -> TestClient:  # type: ignore[type-arg]
    """
    TestClient where app.state.pipeline is intentionally NOT set.
    Used to test 503 readiness probe behaviour during cold-start.
    """
    import api.main

    @asynccontextmanager
    async def _no_pipeline_lifespan(app):  # type: ignore[no-untyped-def]
        # Deliberately omit pipeline — simulates mid-startup state
        app.state.qdrant = mock_qdrant
        app.state.startup_time = time.time()
        yield

    with patch("api.main.lifespan", _no_pipeline_lifespan):
        test_app = api.main.create_app()

    with TestClient(test_app, raise_server_exceptions=False) as c:
        yield c
