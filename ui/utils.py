# ui/utils.py
"""
Pure utility functions for the Financial RAG Streamlit UI.

All functions in this module are stateless and have no Streamlit imports,
making them fully testable with pytest without a running Streamlit server.

Functions:
  parse_sse_line        — parse a single SSE text line into a dict or None
  stream_query          — generator that yields text tokens from the SSE stream
  fetch_structured      — blocking POST /query call, returns parsed JSON dict
  build_metadata_filter — convert sidebar widget values into API filter dict
  format_latency        — human-readable latency string
  format_token_count    — human-readable token count string
  citation_badge_label  — short label for a citation badge in the UI
  group_citations_by_source — group citations by ticker + fiscal period
  health_status_emoji   — map component status string to emoji
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Generator
from typing import Any

import requests

# ── SSE helpers ────────────────────────────────────────────────────────────────


def parse_sse_line(line: str | bytes) -> dict[str, Any] | None:
    """
    Parse a single SSE data line into a Python dict.

    Returns:
        dict  — if line is a valid "data: {...}" event
        None  — if line is empty, a comment, or the terminal "[DONE]" sentinel

    Examples:
        >>> parse_sse_line('data: {"token": "Apple"}')
        {"token": "Apple"}

        >>> parse_sse_line("data: [DONE]")
        None

        >>> parse_sse_line("")
        None
    """
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")

    line = line.strip()
    if not line or line.startswith(":"):
        return None

    if not line.startswith("data: "):
        return None

    payload = line[6:]  # strip "data: " prefix

    if payload == "[DONE]":
        return None

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def stream_query(
    base_url: str,
    question: str,
    metadata_filter: dict[str, Any] | None = None,
    timeout: int = 90,
) -> Generator[str, None, None]:
    """
    Consume the POST /query/stream SSE endpoint and yield text tokens.

    Yields text strings as they arrive.  Raises on HTTP errors or connection
    failures so the Streamlit caller can display an error message.

    Args:
        base_url       : API base URL, e.g. "http://localhost:8000"
        question       : user question
        metadata_filter: optional {"ticker": "AAPL", "year": 2024, "quarter": "Q4"}
        timeout        : total seconds before giving up

    Yields:
        str — text token deltas from the LLM stream

    Raises:
        requests.HTTPError    — non-2xx response from the API
        requests.Timeout      — server did not respond within `timeout` seconds
        requests.ConnectionError — cannot reach the API server
    """
    url = f"{base_url.rstrip('/')}/query/stream"
    payload: dict[str, Any] = {"question": question}
    if metadata_filter:
        payload["filter"] = metadata_filter

    with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            event = parse_sse_line(raw_line)
            if event is None:
                continue
            if "token" in event:
                yield event["token"]
            elif "error" in event:
                # Surface API-side errors as a final token with error formatting
                yield f"\n\n⚠️ **API error:** {event['error']}"
                return


def fetch_structured(
    base_url: str,
    question: str,
    metadata_filter: dict[str, Any] | None = None,
    verbose: bool = False,
    timeout: int = 90,
) -> dict[str, Any]:
    """
    Call POST /query (structured, non-streaming) and return the JSON response.

    Args:
        base_url       : API base URL
        question       : user question
        metadata_filter: optional filter dict
        verbose        : include query_summary and retrieval_summary
        timeout        : request timeout seconds

    Returns:
        Parsed JSON dict matching the AskResponse schema.

    Raises:
        requests.HTTPError — on non-2xx responses
    """
    url = f"{base_url.rstrip('/')}/query/"
    payload: dict[str, Any] = {
        "question": question,
        "verbose": verbose,
    }
    if metadata_filter:
        payload["filter"] = metadata_filter

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ── Filter helpers ─────────────────────────────────────────────────────────────


def build_metadata_filter(
    ticker: str | None,
    year: int | None,
    quarter: str | None,
) -> dict[str, Any] | None:
    """
    Build a metadata filter dict for the API request.

    Returns None if none of the fields are set (meaning: no filter).
    The API accepts partial filters — e.g. only ticker without year.

    Examples:
        >>> build_metadata_filter("AAPL", 2024, "Q4")
        {"ticker": "AAPL", "year": 2024, "quarter": "Q4"}

        >>> build_metadata_filter(None, None, None)
        None

        >>> build_metadata_filter("NVDA", None, None)
        {"ticker": "NVDA"}
    """
    parts: dict[str, Any] = {}
    if ticker:
        parts["ticker"] = ticker.upper().strip()
    if year:
        parts["year"] = int(year)
    if quarter:
        parts["quarter"] = quarter.upper().strip()

    return parts if parts else None


# ── Formatting helpers ─────────────────────────────────────────────────────────


def format_latency(seconds: float) -> str:
    """
    Human-readable latency with appropriate unit.

    Examples:
        >>> format_latency(0.095)
        "95ms"
        >>> format_latency(2.45)
        "2.45s"
        >>> format_latency(62.0)
        "62.0s"
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def format_token_count(n: int) -> str:
    """
    Human-readable token count with K suffix for large numbers.

    Examples:
        >>> format_token_count(950)
        "950"
        >>> format_token_count(12400)
        "12.4K"
    """
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}K"


# ── Citation helpers ───────────────────────────────────────────────────────────


def citation_badge_label(citation: dict[str, Any]) -> str:
    """
    Short label for a citation reference badge.

    Examples:
        >>> citation_badge_label({"index": 1, "ticker": "AAPL", "fiscal_period": "Q4 2024"})
        "[1] AAPL Q4 2024"
    """
    idx = citation.get("index", "?")
    ticker = citation.get("ticker", "")
    period = citation.get("fiscal_period", "")
    return f"[{idx}] {ticker} {period}".strip()


def group_citations_by_source(
    citations: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Group citation dicts by their "TICKER fiscal_period" key.

    Returns:
        Ordered dict mapping "AAPL Q4 2024" → [citation, ...]
        Order matches first appearance in citations list.

    Example:
        citations = [
            {"ticker": "AAPL", "fiscal_period": "Q4 2024", "index": 1},
            {"ticker": "NVDA", "fiscal_period": "Q3 2024", "index": 2},
            {"ticker": "AAPL", "fiscal_period": "Q4 2024", "index": 3},
        ]
        → {"AAPL Q4 2024": [c1, c3], "NVDA Q3 2024": [c2]}
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in citations:
        key = f"{c.get('ticker', '')} {c.get('fiscal_period', '')}".strip()
        groups[key].append(c)
    return dict(groups)


# ── Health check helpers ───────────────────────────────────────────────────────


def health_status_emoji(status: str) -> str:
    """
    Map a component status string to a display emoji.

    Examples:
        >>> health_status_emoji("ok")
        "✅"
        >>> health_status_emoji("error")
        "❌"
        >>> health_status_emoji("degraded")
        "⚠️"
    """
    mapping = {
        "ok": "✅",
        "healthy": "✅",
        "error": "❌",
        "unhealthy": "❌",
        "degraded": "⚠️",
    }
    return mapping.get(status.lower(), "❓")


def fetch_health(base_url: str, timeout: int = 5) -> dict[str, Any]:
    """
    Call GET /health and return the parsed JSON response.
    Returns a minimal error dict if the API is unreachable.
    """
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return {"status": "unhealthy", "components": {}, "error": "Cannot connect to API"}
    except requests.HTTPError as exc:
        return {"status": "unhealthy", "components": {}, "error": str(exc)}
    except Exception as exc:
        return {"status": "unhealthy", "components": {}, "error": str(exc)}
