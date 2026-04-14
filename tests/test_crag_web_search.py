# tests/test_crag_web_search.py
"""Tests for crag/web_search.py — web search abstraction."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from crag.models import WebSearchResult
from crag.web_search import WebSearchClient, _trim

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_web_result(**kwargs) -> WebSearchResult:
    defaults = {
        "title": "Test Title",
        "url": "https://example.com",
        "snippet": "Test snippet",
        "score": 0.7,
    }
    return WebSearchResult(**{**defaults, **kwargs})


# ── _trim ──────────────────────────────────────────────────────────────────────


def test_trim_short_text_unchanged() -> None:
    text = "short text"
    assert _trim(text) == text


def test_trim_long_text_truncated() -> None:
    text = "x" * 2000
    result = _trim(text)
    assert len(result) <= 1500


def test_trim_strips_whitespace() -> None:
    assert _trim("  hello  ") == "hello"


# ── WebSearchResult ────────────────────────────────────────────────────────────


def test_web_search_result_to_context_block() -> None:
    r = _make_web_result(title="AAPL Q4", snippet="Revenue $94.9B")
    block = r.to_context_block(3)
    assert "[3]" in block
    assert "WEB" in block
    assert "Revenue $94.9B" in block


def test_web_search_result_to_dict_truncates_snippet() -> None:
    r = _make_web_result(snippet="x" * 1000)
    d = r.to_dict()
    assert len(d["snippet"]) <= 500


# ── WebSearchClient provider detection ────────────────────────────────────────


def test_provider_tavily_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test123")
    client = WebSearchClient()
    assert client.provider == "tavily"


def test_provider_duckduckgo_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    client = WebSearchClient()
    assert client.provider == "duckduckgo"


# ── WebSearchClient.search — Tavily ────────────────────────────────────────────


def test_search_tavily_returns_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    _ = {
        "results": [
            {"title": "AAPL Q4", "url": "https://a.com", "content": "Revenue $94.9B", "score": 0.9},
            {
                "title": "Apple Earnings",
                "url": "https://b.com",
                "content": "EPS $1.64",
                "score": 0.7,
            },
        ]
    }

    with patch(
        "crag.web_search._search_tavily",
        return_value=[
            WebSearchResult("AAPL Q4", "https://a.com", "Revenue $94.9B", 0.9),
            WebSearchResult("Apple Earnings", "https://b.com", "EPS $1.64", 0.7),
        ],
    ):
        client = WebSearchClient()
        results = client.search("Apple Q4 2024 revenue")

    assert len(results) == 2
    assert results[0].score >= results[1].score  # sorted by score


def test_search_duckduckgo_returns_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with patch(
        "crag.web_search._search_duckduckgo",
        return_value=[
            WebSearchResult("DDG Result", "https://c.com", "Some text", 0.5),
        ],
    ):
        client = WebSearchClient()
        results = client.search("Apple revenue")

    assert len(results) == 1
    assert results[0].title == "DDG Result"


def test_search_returns_empty_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with patch("crag.web_search._search_duckduckgo", side_effect=RuntimeError("network error")):
        client = WebSearchClient()
        results = client.search("test")
    assert results == []


def test_search_max_results_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with patch("crag.web_search._search_duckduckgo", return_value=[]) as mock:
        client = WebSearchClient()
        client.search("test", max_results=999)
        _, kwargs = mock.call_args
        assert mock.call_args[0][1] <= 10  # max_results positional arg


def test_search_results_sorted_by_score(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    unsorted = [
        WebSearchResult("C", "u3", "s", 0.3),
        WebSearchResult("A", "u1", "s", 0.9),
        WebSearchResult("B", "u2", "s", 0.6),
    ]
    with patch("crag.web_search._search_duckduckgo", return_value=unsorted):
        client = WebSearchClient()
        results = client.search("test")

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
