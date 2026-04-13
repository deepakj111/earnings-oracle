"""
FastAPI dependency injection — typed accessors for app.state singletons.

All heavy objects (pipeline, qdrant client) are initialised once in the
lifespan context manager (api/main.py) and stored on app.state.  Route
handlers receive them via FastAPI's Depends() mechanism, which makes the
dependency graph explicit and testable (you can override these in tests).

Usage in route handlers:
    from api.dependencies import get_pipeline, get_qdrant
    from typing import Annotated
    from fastapi import Depends

    Pipeline = Annotated[FinancialRAGPipeline, Depends(get_pipeline)]

    @router.post("/")
    async def my_route(pipeline: Pipeline) -> ...:
        ...
"""

from __future__ import annotations

import time

from fastapi import Request
from qdrant_client import QdrantClient

from rag_pipeline import FinancialRAGPipeline


def get_pipeline(request: Request) -> FinancialRAGPipeline:
    """
    Inject the shared FinancialRAGPipeline singleton.

    The pipeline is stateless between calls — it is safe to share across
    concurrent requests.  All heavy models (embedder, BM25, reranker) are
    pre-loaded during startup and cached inside the pipeline instance.
    """
    return request.app.state.pipeline  # type: ignore[no-any-return]


def get_qdrant(request: Request) -> QdrantClient:
    """
    Inject the shared QdrantClient singleton.

    Used directly by health checks; route handlers go through the pipeline.
    """
    return request.app.state.qdrant  # type: ignore[no-any-return]


def get_uptime_seconds(request: Request) -> float:
    """Return how many seconds this process has been running."""
    startup: float = request.app.state.startup_time
    return round(time.time() - startup, 1)
