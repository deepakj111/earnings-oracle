# tests/test_query_transformer.py

from unittest.mock import MagicMock, patch

import pytest

import query.transformer as transformer_module
from query.models import TransformedQuery
from query.prompts import HYDE_SYSTEM, STEPBACK_SYSTEM
from query.transformer import (
    _TEMP_HYDE,
    _TEMP_MULTI,
    _TEMP_STEPBACK,
    QueryTransformer,
    _cache,
    _cache_key,
    _run_hyde,
    _run_multi_query,
    _run_stepback,
)

# ── Test helpers ──────────────────────────────────────────────────────────────


def _make_openai_response(content: str) -> MagicMock:
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 50
    mock_usage.completion_tokens = 30
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    return mock_response


def _mock_openai_client(content: str = "mocked response text") -> MagicMock:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(content)
    return mock_client


def _make_transformed(**kwargs) -> TransformedQuery:
    defaults = {
        "original": "test query",
        "hyde_document": "a hypothetical passage",
        "multi_queries": ["test query"],
        "stepback_query": "broader test question",
    }
    defaults.update(kwargs)
    return TransformedQuery(**defaults)


# ── _cache_key ────────────────────────────────────────────────────────────────


class TestCacheKey:
    def test_returns_hex_string(self) -> None:
        key = _cache_key("Apple revenue Q4")
        assert isinstance(key, str)
        assert all(c in "0123456789abcdef" for c in key)

    def test_length_is_64_chars_sha256(self) -> None:
        assert len(_cache_key("any query")) == 64

    def test_deterministic_for_same_input(self) -> None:
        assert _cache_key("test query") == _cache_key("test query")

    def test_case_insensitive(self) -> None:
        assert _cache_key("Apple Revenue Q4") == _cache_key("apple revenue q4")

    def test_strips_whitespace(self) -> None:
        assert _cache_key("  test  ") == _cache_key("test")

    def test_different_queries_produce_different_keys(self) -> None:
        assert _cache_key("query one") != _cache_key("query two")


# ── cache get / put ───────────────────────────────────────────────────────────


class TestCacheOperations:
    def setup_method(self):
        _cache.clear()

    def test_miss_returns_none(self) -> None:
        assert _cache.get("nonexistent_key_xyz") is None

    def test_put_then_get_returns_same_object(self) -> None:
        tq = _make_transformed()
        _cache["k1"] = tq
        assert _cache.get("k1") is tq

    def test_cache_grows_after_put(self) -> None:
        before = len(_cache)
        _cache["unique_abc"] = _make_transformed()
        assert len(_cache) == before + 1

    def test_lru_eviction_removes_oldest_on_overflow(self) -> None:
        from cachetools import LRUCache

        old_cache = transformer_module._cache
        try:
            transformer_module._cache = LRUCache(maxsize=2)
            tq = _make_transformed()
            transformer_module._cache["evict_k1"] = tq
            transformer_module._cache["evict_k2"] = tq
            transformer_module._cache["evict_k3"] = tq
            assert "evict_k1" not in transformer_module._cache
            assert "evict_k2" in transformer_module._cache
        finally:
            transformer_module._cache = old_cache


# ── _run_hyde ─────────────────────────────────────────────────────────────────


class TestRunHyde:
    def test_returns_llm_output(self) -> None:
        with patch("query.transformer._call_llm", return_value="hypothetical passage"):
            assert _run_hyde("some query") == "hypothetical passage"

    def test_passes_hyde_system_prompt(self) -> None:
        with patch("query.transformer._call_llm") as mock_llm:
            mock_llm.return_value = "passage"
            _run_hyde("some query")
        assert mock_llm.call_args.kwargs["system"] == HYDE_SYSTEM

    def test_user_prompt_contains_query(self) -> None:
        with patch("query.transformer._call_llm") as mock_llm:
            mock_llm.return_value = "passage"
            _run_hyde("Apple Q4 2024 revenue?")
        assert "Apple Q4 2024 revenue?" in mock_llm.call_args.kwargs["user"]

    def test_uses_hyde_temperature(self) -> None:
        with patch("query.transformer._call_llm") as mock_llm:
            mock_llm.return_value = "passage"
            _run_hyde("some query")
        assert mock_llm.call_args.kwargs["temperature"] == _TEMP_HYDE

    def test_label_is_hyde(self) -> None:
        with patch("query.transformer._call_llm") as mock_llm:
            mock_llm.return_value = "passage"
            _run_hyde("query")
        assert mock_llm.call_args.kwargs["label"] == "HyDE"


# ── _run_stepback ─────────────────────────────────────────────────────────────


class TestRunStepback:
    def test_returns_llm_output(self) -> None:
        with patch("query.transformer._call_llm", return_value="broader question"):
            assert _run_stepback("specific metric question") == "broader question"

    def test_user_prompt_contains_query(self) -> None:
        with patch("query.transformer._call_llm") as mock_llm:
            mock_llm.return_value = "broader"
            _run_stepback("specific Q4 AAPL metric")
        assert "specific Q4 AAPL metric" in mock_llm.call_args.kwargs["user"]

    def test_uses_stepback_temperature(self) -> None:
        with patch("query.transformer._call_llm") as mock_llm:
            mock_llm.return_value = "broader"
            _run_stepback("query")
        assert mock_llm.call_args.kwargs["temperature"] == _TEMP_STEPBACK

    def test_passes_stepback_system_prompt(self) -> None:
        with patch("query.transformer._call_llm") as mock_llm:
            mock_llm.return_value = "broader"
            _run_stepback("query")
        assert mock_llm.call_args.kwargs["system"] == STEPBACK_SYSTEM


# ── _run_multi_query ──────────────────────────────────────────────────────────


class TestRunMultiQuery:
    def test_original_is_always_first(self) -> None:
        with patch("query.transformer._call_llm") as mock_llm:
            mock_llm.return_value = (
                "Rephrasing one sentence\nRephrasing two sentence\nRephrasing three"
            )
            result = _run_multi_query("original query here")
        assert result[0] == "original query here"

    def test_returns_list_of_strings(self) -> None:
        with patch("query.transformer._call_llm") as mock_llm:
            mock_llm.return_value = "How much revenue?\nWhat were earnings?\nQ4 results detail?"
            result = _run_multi_query("Apple Q4 revenue")
        assert isinstance(result, list)
        assert all(isinstance(q, str) for q in result)

    def test_strips_numbering_from_rephrasings(self) -> None:
        with patch("query.transformer._call_llm") as mock_llm:
            mock_llm.return_value = (
                "1. What was Apple's total revenue?\n"
                "2. How did Apple perform financially?\n"
                "3. Apple quarterly earnings results?"
            )
            result = _run_multi_query("Apple revenue")
        for q in result[1:]:
            assert not q[0].isdigit()

    def test_filters_lines_fewer_than_three_words(self) -> None:
        with patch("query.transformer._call_llm") as mock_llm:
            mock_llm.return_value = "yes\nA proper rephrasing of the original query\nno"
            result = _run_multi_query("test query for filtering")
        for q in result[1:]:
            assert len(q.split()) >= 3

    def test_deduplicates_rephrasings(self) -> None:
        with patch("query.transformer._call_llm") as mock_llm:
            mock_llm.return_value = "Apple revenue Q4\nApple revenue Q4\nApple revenue Q4"
            result = _run_multi_query("Apple revenue Q4")
        lower = [q.lower() for q in result]
        assert len(lower) == len(set(lower))

    def test_max_four_queries_returned(self) -> None:
        with patch("query.transformer._call_llm") as mock_llm:
            mock_llm.return_value = (
                "Rephrasing one full line\n"
                "Rephrasing two full line\n"
                "Rephrasing three full line\n"
                "Rephrasing four full line\n"
                "Rephrasing five full line"
            )
            result = _run_multi_query("original query")
        assert len(result) <= 4

    def test_uses_multi_query_temperature(self) -> None:
        with patch("query.transformer._call_llm") as mock_llm:
            mock_llm.return_value = "Rephrasing one full sentence"
            _run_multi_query("Apple Q4 revenue")
        assert mock_llm.call_args.kwargs["temperature"] == _TEMP_MULTI

    def test_empty_lines_excluded_from_result(self) -> None:
        with patch("query.transformer._call_llm") as mock_llm:
            mock_llm.return_value = "\nWhat was Apple revenue?\n\nHow did Apple perform?\n"
            result = _run_multi_query("Apple Q4")
        assert all(q.strip() != "" for q in result)


# ── _call_llm ─────────────────────────────────────────────────────────────────


class TestCallLlm:
    def setup_method(self):
        # We don't clear the shared singleton here as it affects other tests
        pass

    def test_returns_stripped_text_on_success(self) -> None:
        mock_client = _mock_openai_client("  The revenue was $94.9 billion.  ")
        with patch("query.transformer.get_openai_client", return_value=mock_client):
            from query.transformer import _call_llm

            result = _call_llm(
                system="sys", user="usr", temperature=0.3, max_tokens=100, label="test"
            )
        assert result == "The revenue was $94.9 billion."

    def test_correct_model_passed_to_client(self) -> None:
        mock_client = _mock_openai_client("ok")
        with patch("query.transformer.get_openai_client", return_value=mock_client):
            with patch("query.transformer.QUERY_TRANSFORM_MODEL", "test-model-xyz"):
                from query.transformer import _call_llm

                _call_llm("sys", "usr", 0.3, 100, "test")
        assert mock_client.chat.completions.create.call_args.kwargs["model"] == "test-model-xyz"

    # def test_temperature_passed_to_client(self) -> None:
    #     mock_client = _mock_openai_client("ok")
    #     with patch("query.transformer._get_client", return_value=mock_client):
    #         from query.transformer import _call_llm

    #         _call_llm("sys", "usr", 0.42, 100, "test")
    #     assert mock_client.chat.completions.create.call_args.kwargs["temperature"] == 0.42

    def test_max_tokens_passed_to_client(self) -> None:
        mock_client = _mock_openai_client("ok")
        with patch("query.transformer.get_openai_client", return_value=mock_client):
            from query.transformer import _call_llm

            _call_llm("sys", "usr", 0.3, 999, "test")
        assert mock_client.chat.completions.create.call_args.kwargs["max_completion_tokens"] == 999

    def test_system_and_user_messages_structured_correctly(self) -> None:
        mock_client = _mock_openai_client("ok")
        with patch("query.transformer.get_openai_client", return_value=mock_client):
            from query.transformer import _call_llm

            _call_llm("MY SYSTEM PROMPT", "MY USER MSG", 0.3, 100, "test")
        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]

        # Verify it's a single merged user message
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "MY SYSTEM PROMPT" in messages[0]["content"]
        assert "MY USER MSG" in messages[0]["content"]

    def test_empty_response_raises_exception(self) -> None:
        mock_client = _mock_openai_client("")
        with patch("query.transformer.get_openai_client", return_value=mock_client):
            from query.transformer import _call_llm

            with pytest.raises(ValueError):
                _call_llm("sys", "usr", 0.3, 100, "test")


# ── QueryTransformer.__init__ ─────────────────────────────────────────────────


class TestQueryTransformerInit:
    def test_cache_enabled_by_default(self) -> None:
        t = QueryTransformer()
        assert t.enable_cache is True

    def test_cache_can_be_disabled(self) -> None:
        t = QueryTransformer(enable_cache=False)
        assert t.enable_cache is False


# ── QueryTransformer.transform ────────────────────────────────────────────────


class TestQueryTransformerTransform:
    def setup_method(self):
        transformer_module._cache.clear()
        # No more direct client reset here

    def _patched_transform(self, query, *, hyde="hyde doc", multi=None, stepback="broader q"):
        if multi is None:
            multi = [query, "rephrasing one", "rephrasing two"]
        t = QueryTransformer(enable_cache=False)
        with (
            patch("query.transformer._run_hyde", return_value=hyde),
            patch("query.transformer._run_multi_query", return_value=multi),
            patch("query.transformer._run_stepback", return_value=stepback),
        ):
            return t.transform(query)

    def test_returns_transformed_query_instance(self) -> None:
        result = self._patched_transform("Apple Q4 revenue?")
        assert isinstance(result, TransformedQuery)

    def test_original_query_preserved_in_result(self) -> None:
        result = self._patched_transform("Apple Q4 revenue?")
        assert result.original == "Apple Q4 revenue?"

    def test_hyde_document_populated(self) -> None:
        result = self._patched_transform("Apple revenue?", hyde="Apple earned $94.9B")
        assert result.hyde_document == "Apple earned $94.9B"

    def test_multi_queries_populated(self) -> None:
        result = self._patched_transform("q?", multi=["q?", "alt1", "alt2"])
        assert result.multi_queries == ["q?", "alt1", "alt2"]

    def test_stepback_query_populated(self) -> None:
        result = self._patched_transform("q?", stepback="broader abstract q")
        assert result.stepback_query == "broader abstract q"

    def test_empty_query_raises_value_error(self) -> None:
        t = QueryTransformer(enable_cache=False)
        with pytest.raises(ValueError, match="empty"):
            t.transform("")

    def test_whitespace_only_query_raises_value_error(self) -> None:
        t = QueryTransformer(enable_cache=False)
        with pytest.raises(ValueError):
            t.transform("   ")

    def test_query_stripped_before_processing(self) -> None:
        result = self._patched_transform("  Apple revenue?  ")
        assert result.original == "Apple revenue?"

    def test_failed_techniques_empty_on_full_success(self) -> None:
        result = self._patched_transform("test query")
        assert result.failed_techniques == []

    def test_graceful_degradation_when_hyde_fails(self) -> None:
        t = QueryTransformer(enable_cache=False)
        with (
            patch("query.transformer._run_hyde", side_effect=RuntimeError("LLM down")),
            patch("query.transformer._run_multi_query", return_value=["q1", "q2"]),
            patch("query.transformer._run_stepback", return_value="broader"),
        ):
            result = t.transform("Apple revenue?")
        assert isinstance(result, TransformedQuery)
        assert "hyde" in result.failed_techniques

    def test_graceful_degradation_when_stepback_fails(self) -> None:
        t = QueryTransformer(enable_cache=False)
        with (
            patch("query.transformer._run_hyde", return_value="hyde"),
            patch("query.transformer._run_multi_query", return_value=["q"]),
            patch("query.transformer._run_stepback", side_effect=RuntimeError("fail")),
        ):
            result = t.transform("Apple revenue Q4?")
        assert isinstance(result, TransformedQuery)
        assert "stepback" in result.failed_techniques

    def test_graceful_degradation_when_multi_query_fails(self) -> None:
        t = QueryTransformer(enable_cache=False)
        with (
            patch("query.transformer._run_hyde", return_value="hyde"),
            patch("query.transformer._run_multi_query", side_effect=RuntimeError("fail")),
            patch("query.transformer._run_stepback", return_value="broader"),
        ):
            result = t.transform("Apple revenue Q4?")
        assert isinstance(result, TransformedQuery)
        assert "multi" in result.failed_techniques

    def test_hyde_fallback_is_original_query(self) -> None:
        t = QueryTransformer(enable_cache=False)
        with (
            patch("query.transformer._run_hyde", side_effect=RuntimeError("fail")),
            patch("query.transformer._run_multi_query", return_value=["q"]),
            patch("query.transformer._run_stepback", return_value="s"),
        ):
            result = t.transform("Apple revenue Q4?")
        assert result.hyde_document == "Apple revenue Q4?"

    def test_stepback_fallback_is_original_query(self) -> None:
        t = QueryTransformer(enable_cache=False)
        with (
            patch("query.transformer._run_hyde", return_value="hyde"),
            patch("query.transformer._run_multi_query", return_value=["q"]),
            patch("query.transformer._run_stepback", side_effect=RuntimeError("fail")),
        ):
            result = t.transform("Apple revenue Q4?")
        assert result.stepback_query == "Apple revenue Q4?"

    def test_multi_fallback_is_list_with_original(self) -> None:
        t = QueryTransformer(enable_cache=False)
        with (
            patch("query.transformer._run_hyde", return_value="hyde"),
            patch("query.transformer._run_multi_query", side_effect=RuntimeError("fail")),
            patch("query.transformer._run_stepback", return_value="s"),
        ):
            result = t.transform("Apple revenue Q4?")
        assert result.multi_queries == ["Apple revenue Q4?"]

    def test_cache_hit_returns_same_object(self) -> None:
        t = QueryTransformer(enable_cache=True)
        with (
            patch("query.transformer._run_hyde", return_value="hyde"),
            patch("query.transformer._run_multi_query", return_value=["q"]),
            patch("query.transformer._run_stepback", return_value="s"),
        ):
            r1 = t.transform("Apple revenue?")
            r2 = t.transform("Apple revenue?")
        assert r1 is r2

    def test_cache_disabled_calls_llm_each_time(self) -> None:
        t = QueryTransformer(enable_cache=False)
        call_count = {"n": 0}

        def counting_hyde(q):
            call_count["n"] += 1
            return "hyde"

        with (
            patch("query.transformer._run_hyde", side_effect=counting_hyde),
            patch("query.transformer._run_multi_query", return_value=["q"]),
            patch("query.transformer._run_stepback", return_value="s"),
        ):
            t.transform("Apple revenue?")
            t.transform("Apple revenue?")
        assert call_count["n"] == 2

    def test_cache_is_case_insensitive(self) -> None:
        t = QueryTransformer(enable_cache=True)
        with (
            patch("query.transformer._run_hyde", return_value="hyde"),
            patch("query.transformer._run_multi_query", return_value=["q"]),
            patch("query.transformer._run_stepback", return_value="s"),
        ):
            r1 = t.transform("Apple Revenue Q4?")
            r2 = t.transform("apple revenue q4?")
        assert r1 is r2

    def test_all_three_techniques_invoked(self) -> None:
        t = QueryTransformer(enable_cache=False)
        calls = set()

        with (
            patch("query.transformer._run_hyde", side_effect=lambda q: calls.add("hyde") or "h"),
            patch(
                "query.transformer._run_multi_query",
                side_effect=lambda q: calls.add("multi") or ["q"],
            ),
            patch(
                "query.transformer._run_stepback",
                side_effect=lambda q: calls.add("stepback") or "s",
            ),
        ):
            t.transform("Apple revenue?")

        assert calls == {"hyde", "multi", "stepback"}
