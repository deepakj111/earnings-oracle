# crag/web_search.py
"""
Layer 5b — Web search abstraction for CRAG fallback.

Provider hierarchy (auto-detected at init time):
  1. Tavily  — set TAVILY_API_KEY in .env (best quality for financial queries)
  2. DuckDuckGo — free fallback, no API key required (rate-limited)

Both providers return normalised WebSearchResult objects.
Snippets are capped at _MAX_SNIPPET_CHARS to control generation context size.

Install providers:
    poetry add tavily-python          # Tavily (recommended)
    poetry add duckduckgo-search      # DuckDuckGo fallback
"""

from __future__ import annotations

import os

from loguru import logger

from crag.models import WebSearchResult

_MAX_SNIPPET_CHARS = 1_500


def _trim(text: str) -> str:
    return text[:_MAX_SNIPPET_CHARS].strip()


# ── Tavily ─────────────────────────────────────────────────────────────────────


def _search_tavily(query: str, max_results: int, api_key: str) -> list[WebSearchResult]:
    try:
        from tavily import TavilyClient  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("Run: poetry add tavily-python") from exc

    client = TavilyClient(api_key=api_key)
    resp = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_answer=False,
        include_raw_content=False,
    )
    results: list[WebSearchResult] = []
    for item in resp.get("results", []):
        snippet = _trim(item.get("content", "") or item.get("snippet", ""))
        if snippet:
            results.append(
                WebSearchResult(
                    title=(item.get("title", "") or "Untitled")[:200],
                    url=item.get("url", ""),
                    snippet=snippet,
                    score=float(item.get("score", 0.5)),
                )
            )
    return results


# ── DuckDuckGo ─────────────────────────────────────────────────────────────────


def _search_duckduckgo(query: str, max_results: int) -> list[WebSearchResult]:
    try:
        from duckduckgo_search import DDGS  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("Run: poetry add duckduckgo-search") from exc

    results: list[WebSearchResult] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results, timelimit="y"):
            snippet = _trim(item.get("body", ""))
            if snippet:
                results.append(
                    WebSearchResult(
                        title=(item.get("title", "") or "Untitled")[:200],
                        url=item.get("href", ""),
                        snippet=snippet,
                        score=0.5,  # DuckDuckGo doesn't expose relevance scores
                    )
                )
    return results


# ── Public client ──────────────────────────────────────────────────────────────


class WebSearchClient:
    """
    Unified web-search abstraction for CRAG.

    Auto-selects provider:
      Tavily   if TAVILY_API_KEY env var is set (recommended)
      DuckDuckGo otherwise (free, no API key, rate-limited)

    Never raises — returns empty list on any failure.

    Usage:
        client = WebSearchClient()
        results = client.search("Apple Q4 2024 revenue earnings", max_results=4)
    """

    def __init__(self) -> None:
        self._tavily_key: str | None = (os.getenv("TAVILY_API_KEY") or "").strip() or None
        logger.info(f"WebSearchClient | provider={self.provider}")

    @property
    def provider(self) -> str:
        return "tavily" if self._tavily_key else "duckduckgo"

    def search(self, query: str, max_results: int = 4) -> list[WebSearchResult]:
        """
        Search the web. Returns up to max_results results, sorted by score.
        Returns [] on any failure — never raises.
        """
        max_results = max(1, min(10, max_results))
        try:
            if self._tavily_key:
                raw = _search_tavily(query, max_results, self._tavily_key)
            else:
                raw = _search_duckduckgo(query, max_results)
            results = sorted(raw, key=lambda r: r.score, reverse=True)
            logger.info(f"WebSearch [{self.provider}]: {len(results)} results for q={query!r:.60}")
            return results
        except Exception as exc:
            logger.error(f"WebSearch failed ({type(exc).__name__}): {exc}")
            return []

    def is_available(self) -> bool:
        """Return True if the selected provider library is importable."""
        try:
            if self._tavily_key:
                from tavily import TavilyClient  # noqa: F401  # type: ignore[import-untyped]
            else:
                from duckduckgo_search import DDGS  # noqa: F401  # type: ignore[import-untyped]
            return True
        except ImportError:
            return False
