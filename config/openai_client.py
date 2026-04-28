# config/openai_client.py
"""
Shared OpenAI client factory — single source of truth for API client creation.

Previously, 5 separate modules each maintained their own lazy-singleton
OpenAI client with duplicated "check API key, create client" boilerplate.
This wastes TCP connections and scatters configuration logic.

Thread-safety:
  Both clients are safe for concurrent use after first initialisation.
  The OpenAI SDK uses httpx internally, which manages connection pooling.

Usage:
    from config.openai_client import get_openai_client, get_async_openai_client

    client = get_openai_client()
    response = client.chat.completions.create(...)

    async_client = get_async_openai_client()
    response = await async_client.chat.completions.create(...)
"""

from __future__ import annotations

from loguru import logger
from openai import AsyncOpenAI, OpenAI

from config import settings

# ── Sync client ────────────────────────────────────────────────────────────────

_sync_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """
    Return a shared synchronous OpenAI client (lazy singleton).

    Raises:
        OSError: if OPENAI_API_KEY is not configured.
    """
    global _sync_client
    if _sync_client is None:
        api_key = settings.infra.openai_api_key
        if not api_key:
            raise OSError("OPENAI_API_KEY is not set. Add it to your .env file.")
        _sync_client = OpenAI(api_key=api_key, max_retries=2)
        logger.info("Shared OpenAI sync client initialised.")
    return _sync_client


# ── Async client ───────────────────────────────────────────────────────────────

_async_client: AsyncOpenAI | None = None


def get_async_openai_client() -> AsyncOpenAI:
    """
    Return a shared asynchronous OpenAI client (lazy singleton).

    Raises:
        OSError: if OPENAI_API_KEY is not configured.
    """
    global _async_client
    if _async_client is None:
        api_key = settings.infra.openai_api_key
        if not api_key:
            raise OSError("OPENAI_API_KEY is not set. Add it to your .env file.")
        _async_client = AsyncOpenAI(api_key=api_key, max_retries=2)
        logger.info("Shared OpenAI async client initialised.")
    return _async_client


def reset_clients() -> None:
    """Reset both client singletons. Used in tests to force re-creation."""
    global _sync_client, _async_client
    _sync_client = None
    _async_client = None
