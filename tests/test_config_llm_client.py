"""
Unit tests for config/llm_client.py — universal model routing and client helpers.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from config.llm_client import (
    LLMResponse,
    _augment_messages_with_schema,
    _clean_vertex_model_name,
    _extract_embeddings,
    _get_adc_paths,
    _get_vertex_location,
    _get_vertex_project,
    _inject_api_key,
    _is_openai_model,
    _is_vertex_call,
    _should_include_temperature,
    _vertex_acomplete,
    _vertex_aembed,
    _vertex_astream,
    acomplete,
    aembed,
    aparse,
    astream,
    complete,
    embed,
    get_async_semaphore,
    get_embedding_token_ceiling,
    get_sync_semaphore,
    parse,
    reset_clients,
)


class SampleSchema(BaseModel):
    summary: str
    score: float


class TestLLMClientHelpers:
    def test_embedding_token_ceilings(self) -> None:
        assert get_embedding_token_ceiling("text-embedding-004") == 2048
        assert get_embedding_token_ceiling("text-embedding-3-small") == 8000
        assert get_embedding_token_ceiling("voyage-finance-2") == 16000
        assert get_embedding_token_ceiling("custom-unknown-model") == 8000

    def test_get_adc_paths(self) -> None:
        paths = _get_adc_paths()
        assert isinstance(paths, list)
        assert len(paths) >= 2

    def test_get_vertex_project_and_location(self) -> None:
        proj = _get_vertex_project()
        loc = _get_vertex_location()
        assert isinstance(proj, str) and len(proj) > 0
        assert isinstance(loc, str) and len(loc) > 0

    def test_clean_vertex_model_name(self) -> None:
        assert _clean_vertex_model_name("vertex_ai/gemini-2.5-flash") == "gemini-2.5-flash"
        assert _clean_vertex_model_name("gemini-2.0-flash") == "gemini-2.5-flash"
        assert _clean_vertex_model_name("gpt-4.1-nano") == "gemini-2.5-flash"
        assert _clean_vertex_model_name("text-embedding-3-small") == "text-embedding-004"
        assert _clean_vertex_model_name("custom-model") == "custom-model"

    def test_should_include_temperature(self) -> None:
        assert _should_include_temperature("o1-mini", 0.5) is False
        assert _should_include_temperature("o3-mini", 0.5) is False
        assert _should_include_temperature("gpt-5-mini", 0.5) is False
        assert _should_include_temperature("gemini-2.5-flash", 1.0) is False
        assert _should_include_temperature("gemini-2.5-flash", 0.7) is True

    def test_is_vertex_call(self) -> None:
        assert _is_vertex_call("vertex_ai/gemini-2.5-flash") is True
        with (
            patch("config.llm_client._get_adc_token", return_value=None),
            patch("config.llm_client.settings") as mock_settings,
        ):
            mock_settings.infra.provider = "openai"
            mock_settings.infra.gemini_api_key = "key"
            assert _is_vertex_call("gpt-4.1-nano") is False

    def test_is_openai_model(self) -> None:
        assert _is_openai_model("gpt-4o") is True
        assert _is_openai_model("openai/gpt-4o") is True
        assert _is_openai_model("vertex_ai/gemini-2.5-flash") is False

    def test_inject_api_key(self) -> None:
        kwargs: dict[str, Any] = {}
        with patch("config.llm_client.settings") as mock_settings:
            mock_settings.infra.gemini_api_key = "gemini-test-key"
            mock_settings.infra.openai_api_key = "openai-test-key"

            _inject_api_key(kwargs, "gemini/gemini-2.5-flash")
            assert kwargs.get("api_key") == "gemini-test-key"

            kwargs.clear()
            _inject_api_key(kwargs, "gpt-4o")
            assert kwargs.get("api_key") == "openai-test-key"

    def test_augment_messages_with_schema(self) -> None:
        # Case 1: System prompt exists
        msgs = [{"role": "system", "content": "You are a bot."}, {"role": "user", "content": "Hi"}]
        aug = _augment_messages_with_schema(msgs, "\nRespond in JSON")
        assert aug[0]["content"] == "You are a bot.\nRespond in JSON"

        # Case 2: No system prompt
        msgs2 = [{"role": "user", "content": "Hi"}]
        aug2 = _augment_messages_with_schema(msgs2, "Respond in JSON")
        assert aug2[0]["role"] == "system"
        assert aug2[0]["content"] == "Respond in JSON"

    def test_extract_embeddings_helper(self) -> None:
        # None response
        assert _extract_embeddings(None, 2) == [[], []]

        # Valid mock response with items
        item0 = MagicMock(index=1, embedding=[0.3, 0.4])
        item1 = MagicMock(index=0, embedding=[0.1, 0.2])
        resp = MagicMock(data=[item0, item1])
        extracted = _extract_embeddings(resp, 2)
        assert extracted == [[0.1, 0.2], [0.3, 0.4]]

    def test_semaphore_singletons(self) -> None:
        reset_clients()
        sem1 = get_async_semaphore(5)
        sem2 = get_async_semaphore()
        assert sem1 is sem2

        sync_sem1 = get_sync_semaphore(5)
        sync_sem2 = get_sync_semaphore()
        assert sync_sem1 is sync_sem2
        reset_clients()


@pytest.mark.asyncio
class TestVertexDirectEngine:
    async def test_vertex_acomplete_success(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": "Apple reported Q4 revenue of $94.9B."}]}}
            ],
            "usageMetadata": {
                "promptTokenCount": 50,
                "candidatesTokenCount": 20,
            },
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with (
            patch("config.llm_client._get_adc_token", return_value="fake_adc_token"),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            resp = await _vertex_acomplete(
                messages=[
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": "What was Apple revenue?"},
                ],
                model="gemini-2.5-flash",
                temperature=0.2,
                max_tokens=256,
                json_mode=True,
            )

            assert isinstance(resp, LLMResponse)
            assert resp.content == "Apple reported Q4 revenue of $94.9B."
            assert resp.prompt_tokens == 50
            assert resp.completion_tokens == 20

    async def test_vertex_acomplete_error_handling(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with (
            patch("config.llm_client._get_adc_token", return_value="fake_adc_token"),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            with pytest.raises(RuntimeError, match="Vertex AI API error"):
                await _vertex_acomplete(
                    messages=[{"role": "user", "content": "Test"}],
                    model="gemini-2.5-flash",
                )

    async def test_vertex_aembed_success(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [
                {"embeddings": {"values": [0.1, 0.2, 0.3]}},
                {"embeddings": {"values": [0.4, 0.5, 0.6]}},
            ]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with (
            patch("config.llm_client._get_adc_token", return_value="fake_adc_token"),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            vectors = await _vertex_aembed(["apple revenue", "google profit"], "text-embedding-004")
            assert len(vectors) == 2
            assert vectors[0] == [0.1, 0.2, 0.3]
            assert vectors[1] == [0.4, 0.5, 0.6]

    async def test_vertex_aembed_empty(self) -> None:
        vectors = await _vertex_aembed([], "text-embedding-004")
        assert vectors == []

    async def test_vertex_astream_success(self) -> None:
        line1 = 'data: {"candidates": [{"content": {"parts": [{"text": "Hello "}]}}]}'
        line2 = 'data: {"candidates": [{"content": {"parts": [{"text": "World!"}]}}]}'

        async def fake_aiter_lines():
            yield line1
            yield line2

        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 200
        mock_stream_resp.aiter_lines = fake_aiter_lines

        class FakeStreamContext:
            async def __aenter__(self):
                return mock_stream_resp

            async def __aexit__(self, exc_type, exc, tb):
                pass

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream.return_value = FakeStreamContext()

        with (
            patch("config.llm_client._get_adc_token", return_value="fake_adc_token"),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            tokens = []
            async for token in _vertex_astream(
                messages=[{"role": "user", "content": "Hi"}],
                model="gemini-2.5-flash",
            ):
                tokens.append(token)
            assert "".join(tokens) == "Hello World!"


class TestUniversalLLMDispatch:
    @pytest.mark.asyncio
    async def test_acomplete_litellm_success(self) -> None:
        mock_msg = MagicMock()
        mock_msg.content = "Net income was $24B."
        mock_choice = MagicMock(message=mock_msg)
        mock_resp = MagicMock(
            choices=[mock_choice],
            usage=MagicMock(prompt_tokens=40, completion_tokens=15),
            model="gpt-4.1-nano",
        )

        with (
            patch("config.llm_client._is_vertex_call", return_value=False),
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp),
        ):
            resp = await acomplete(
                messages=[{"role": "user", "content": "What was net income?"}],
                model="gpt-4.1-nano",
            )
            assert resp.content == "Net income was $24B."
            assert resp.prompt_tokens == 40
            assert resp.completion_tokens == 15

    def test_complete_litellm_success(self) -> None:
        mock_msg = MagicMock()
        mock_msg.content = "Operating margin was 30%."
        mock_choice = MagicMock(message=mock_msg)
        mock_resp = MagicMock(
            choices=[mock_choice],
            usage=MagicMock(prompt_tokens=35, completion_tokens=10),
            model="gpt-4.1-nano",
        )

        with (
            patch("config.llm_client._is_vertex_call", return_value=False),
            patch("litellm.completion", return_value=mock_resp),
        ):
            resp = complete(
                messages=[{"role": "user", "content": "Operating margin?"}],
                model="gpt-4.1-nano",
            )
            assert resp.content == "Operating margin was 30%."
            assert resp.prompt_tokens == 35

    @pytest.mark.asyncio
    async def test_aembed_litellm_success(self) -> None:
        item1 = MagicMock(index=0, embedding=[0.1, 0.2])
        item2 = MagicMock(index=1, embedding=[0.3, 0.4])
        mock_resp = MagicMock(data=[item1, item2])

        with (
            patch("config.llm_client._is_vertex_call", return_value=False),
            patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_resp),
        ):
            vectors = await aembed(["hello", "world"], model="text-embedding-3-small")
            assert len(vectors) == 2
            assert vectors[0] == [0.1, 0.2]
            assert vectors[1] == [0.3, 0.4]

    def test_embed_litellm_success(self) -> None:
        item = MagicMock(index=0, embedding=[0.5, 0.6])
        mock_resp = MagicMock(data=[item])

        with (
            patch("config.llm_client._is_vertex_call", return_value=False),
            patch("litellm.embedding", return_value=mock_resp),
        ):
            vectors = embed(["query text"], model="text-embedding-3-small")
            assert len(vectors) == 1
            assert vectors[0] == [0.5, 0.6]

    @pytest.mark.asyncio
    async def test_aparse_success(self) -> None:
        mock_resp = LLMResponse(
            content='{"summary": "Strong growth", "score": 0.95}',
            prompt_tokens=50,
            completion_tokens=20,
            model="gpt-4.1-nano",
        )

        with (
            patch("config.llm_client._is_vertex_call", return_value=False),
            patch("config.llm_client.acomplete", new_callable=AsyncMock, return_value=mock_resp),
        ):
            parsed = await aparse(
                messages=[{"role": "user", "content": "Analyze performance"}],
                schema=SampleSchema,
                model="gpt-4.1-nano",
            )
            assert isinstance(parsed, SampleSchema)
            assert parsed.summary == "Strong growth"
            assert parsed.score == 0.95

    def test_parse_success(self) -> None:
        mock_resp = LLMResponse(
            content='```json\n{"summary": "Solid quarter", "score": 0.88}\n```',
            prompt_tokens=45,
            completion_tokens=25,
            model="gpt-4.1-nano",
        )

        with (
            patch("config.llm_client._is_vertex_call", return_value=False),
            patch("config.llm_client.complete", return_value=mock_resp),
        ):
            parsed = parse(
                messages=[{"role": "user", "content": "Analyze Q3"}],
                schema=SampleSchema,
                model="gpt-4.1-nano",
            )
            assert isinstance(parsed, SampleSchema)
            assert parsed.summary == "Solid quarter"
            assert parsed.score == 0.88

    @pytest.mark.asyncio
    async def test_astream_litellm_success(self) -> None:
        chunk1 = MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello "))])
        chunk2 = MagicMock(choices=[MagicMock(delta=MagicMock(content="World!"))])

        async def fake_stream():
            yield chunk1
            yield chunk2

        with (
            patch("config.llm_client._is_vertex_call", return_value=False),
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=fake_stream()),
        ):
            tokens = []
            async for token in astream(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4.1-nano",
            ):
                tokens.append(token)
            assert "".join(tokens) == "Hello World!"
