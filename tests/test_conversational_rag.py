"""
Unit tests for Conversational RAG functionality:
- Contextual query condensation via QueryTransformer
- UserContextMiddleware & RateLimitMiddleware user scoping
- Session persistence in query endpoints
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.chat_store import ChatSessionStore
from api.main import app
from query.transformer import QueryTransformer


class TestContextualizeQuery:
    @pytest.mark.asyncio
    async def test_contextualize_query_empty_history(self):
        transformer = QueryTransformer()
        q = "What was the operating margin?"
        res = await transformer.contextualize_query(q, None)
        assert res == q

        res2 = await transformer.contextualize_query(q, [])
        assert res2 == q

    @pytest.mark.asyncio
    async def test_contextualize_query_with_history(self):
        transformer = QueryTransformer()
        history = [
            {"role": "user", "content": "How much revenue did Apple make in Q4 2024?"},
            {
                "role": "assistant",
                "content": "Apple reported $94.9 billion in revenue for Q4 2024.",
            },
        ]
        follow_up = "What about net income?"

        with patch("query.transformer._call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "What was Apple's net income in Q4 2024?"
            condensed = await transformer.contextualize_query(follow_up, history)
            assert condensed == "What was Apple's net income in Q4 2024?"
            assert mock_llm.called

    @pytest.mark.asyncio
    async def test_contextualize_query_fallback_on_error(self):
        transformer = QueryTransformer()
        history = [{"role": "user", "content": "Tell me about MSFT."}]
        follow_up = "And cloud revenue?"

        with patch("query.transformer._call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("LLM timeout")
            condensed = await transformer.contextualize_query(follow_up, history)
            assert condensed == follow_up


class TestUserContextMiddleware:
    def test_user_headers_parsed(self):
        client = TestClient(app)
        res = self._check_request(client, user_id="analyst_alice", tenant_id="tenant_beta")
        assert res.status_code == 200

    def _check_request(self, client: TestClient, user_id: str, tenant_id: str):
        return client.get(
            "/sessions",
            headers={"X-User-ID": user_id, "X-Tenant-ID": tenant_id},
        )

    def test_default_user_context(self):
        client = TestClient(app)
        # Without headers, defaults to default_user
        res = client.get("/sessions")
        assert res.status_code == 200


class TestConversationalPersistenceInQuery:
    @pytest.fixture(autouse=True)
    def setup_mocks(self, tmp_path, monkeypatch):
        test_store = ChatSessionStore(db_path=tmp_path / "conv_query_test.db")
        monkeypatch.setattr("api.routes.query.get_chat_store", lambda: test_store)
        monkeypatch.setattr("api.routes.sessions.get_chat_store", lambda: test_store)
        self.store = test_store
        self.client = TestClient(app)

    def test_query_persists_to_session(self, monkeypatch):
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.question = "What was Nvidia data center revenue?"
        mock_result.answer = "Nvidia data center revenue was $30.8 billion in Q3 2025."
        mock_result.citations = []
        mock_result.confidence_score = 0.98
        mock_result.numerical_hallucination_warnings = []
        mock_result.citation_integrity_warnings = []
        mock_result.grounded = True
        mock_result.retrieval_summary = None
        mock_result.query_summary = None
        mock_result.calculations = []
        mock_result.model = "gpt-4o-mini"
        mock_result.latency_seconds = 0.45
        mock_result.tokens = 150
        mock_result.prompt_tokens = 100
        mock_result.completion_tokens = 50
        mock_result.total_tokens = 150
        mock_result.unique_tickers = ["NVDA"]
        mock_result.unique_sources = ["NVDA Q3 2025"]
        mock_result.grounding_score = 0.95
        mock_result.context_tokens_used = 120
        mock_result.retrieval_failed = False
        mock_result.latency_breakdown = {}
        mock_result.estimated_cost = 0.0005
        mock_result.trace_id = "tr-123"
        mock_result.cache_hit = False
        mock_result.reflexion_attempts = 0

        async def fake_ask(*args, **kwargs):
            return mock_result

        mock_pipeline.ask = fake_ask
        from api.dependencies import get_pipeline

        app.dependency_overrides[get_pipeline] = lambda: mock_pipeline

        sid = self.store.create_session(user_id="user_fin", title="Nvidia Analysis")

        try:
            resp = self.client.post(
                "/query",
                json={
                    "question": "What was Nvidia data center revenue?",
                    "conversation_id": sid,
                },
                headers={"X-User-ID": "user_fin"},
            )
            assert resp.status_code == 200
        finally:
            app.dependency_overrides.clear()

        # Verify session has both user query and assistant response
        session_detail = self.store.get_session(sid)
        assert session_detail is not None
        assert len(session_detail.messages) == 2
        assert session_detail.messages[0].role == "user"
        assert session_detail.messages[0].content == "What was Nvidia data center revenue?"
        assert session_detail.messages[1].role == "assistant"
        assert "Nvidia data center revenue was $30.8 billion" in session_detail.messages[1].content
