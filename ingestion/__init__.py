from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ingestion.chunker import Chunk, create_parent_child_chunks
from ingestion.facts_store import FactStore
from ingestion.indexer import index_document, init_qdrant

if TYPE_CHECKING:
    from ingestion.pipeline import run_pipeline, run_pipeline_async


def __getattr__(name: str) -> Any:
    if name in ("run_pipeline", "run_pipeline_async"):
        import ingestion.pipeline as _pipeline

        return getattr(_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Chunk",
    "create_parent_child_chunks",
    "FactStore",
    "init_qdrant",
    "index_document",
    "run_pipeline",
    "run_pipeline_async",
]
