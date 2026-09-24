# config/openai_client.py
"""
Backward-compatibility shim for config.openai_client.

The canonical LLM client is config.llm_client, which provides a
provider-agnostic interface supporting Vertex AI (ADC), Gemini, OpenAI,
Anthropic, and more.

This module re-exports the original function signatures so that any existing
code, tests, or scripts that import from config.openai_client continue to work
without modification.

New code should import from config.llm_client directly:
    from config.llm_client import acomplete, embed, get_async_semaphore
"""

from __future__ import annotations

import asyncio
import threading

from loguru import logger
from openai import AsyncOpenAI, OpenAI

from config import settings
from config.llm_client import get_async_semaphore as get_async_openai_semaphore  # noqa: F401
from config.llm_client import get_sync_semaphore as get_sync_openai_semaphore  # noqa: F401
from config.llm_client import reset_clients as _reset_llm_clients

_sync_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None
_sync_lock = threading.Lock()
_async_lock = threading.Lock()


def get_openai_client() -> OpenAI:
    """
    Return a shared synchronous OpenAI client (lazy singleton).
    Thread-safe via double-checked locking.
    """
    global _sync_client
    if _sync_client is None:
        with _sync_lock:
            if _sync_client is None:
                api_key = getattr(settings.infra, "openai_api_key", "")
                if not api_key:
                    raise OSError("OPENAI_API_KEY is not set. Add it to your .env file.")
                _sync_client = OpenAI(api_key=api_key, max_retries=2)
                logger.info("Shared OpenAI sync client initialised.")
    return _sync_client


def get_async_openai_client() -> AsyncOpenAI:
    """
    Return a shared asynchronous OpenAI client (lazy singleton).
    Thread-safe via double-checked locking.
    """
    global _async_client
    if _async_client is None:
        with _async_lock:
            if _async_client is None:
                api_key = getattr(settings.infra, "openai_api_key", "")
                if not api_key:
                    raise OSError("OPENAI_API_KEY is not set. Add it to your .env file.")
                _async_client = AsyncOpenAI(api_key=api_key, max_retries=2)
                logger.info("Shared OpenAI async client initialised.")
    return _async_client


def reset_clients() -> None:
    """Reset both client singletons and semaphore singletons."""
    global _sync_client, _async_client
    _sync_client = None
    _async_client = None
    _reset_llm_clients()


def _get_async_semaphore(max_concurrency: int | None = None) -> asyncio.Semaphore:
    return get_async_openai_semaphore(max_concurrency)


def _get_sync_semaphore(max_concurrency: int | None = None) -> threading.BoundedSemaphore:
    return get_sync_openai_semaphore(max_concurrency)
