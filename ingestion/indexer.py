"""
Embedding, Qdrant indexing, and BM25 corpus construction for the ingestion pipeline.

Two structures are built in lockstep for every child chunk:
  bm25_texts[i]  — tokenised text list (consumed by BM25Okapi)
  bm25_corpus[i] — metadata dict (maps BM25 rank index → chunk ID / Qdrant payload)

These two lists MUST share the same index — bm25_texts[i] and bm25_corpus[i]
always describe the same chunk. The retrieval layer uses bm25_corpus to resolve
BM25 result indices back to chunk IDs and document metadata.
"""

import asyncio
import uuid

import numpy as np
from fastembed import TextEmbedding
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import settings as _settings
from ingestion.chunker import Chunk
from ingestion.metadata_extractor import DocumentMetadata

_cfg = _settings.embedding
COLLECTION_NAME: str = _cfg.collection_name
VECTOR_DIM: int = _cfg.vector_dim
EMBEDDING_MODEL: str = _cfg.model
UPSERT_BATCH_SIZE: int = _cfg.upsert_batch_size

_embed_model: TextEmbedding | None = None


def setup_embedder() -> None:
    """
    Load the ONNX embedding model into memory.
    First call downloads the model (~340MB) to ~/.cache/fastembed/
    and caches it permanently — subsequent runs are instant.
    """
    global _embed_model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    _embed_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    logger.info("Embedding model ready.")


def _get_embeddings(texts: list[str]) -> list[list[float]]:
    if _embed_model is None:
        raise RuntimeError(
            "Embedding model is not loaded. Call setup_embedder() before calling _get_embeddings()."
        )
    vectors = list(_embed_model.embed(texts))
    result = []
    for vec in vectors:
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        result.append(arr.tolist())
    return result


def _ensure_payload_indices(client: QdrantClient) -> None:
    """
    Create payload field indices on the Qdrant collection for fast metadata filtering.

    Called on every init_qdrant() — fully idempotent, safe to repeat on restarts.
    Without these indices Qdrant performs a full collection scan on every filtered
    query, which is functionally correct but O(n) instead of O(log n).

    Index types:
      ticker   → keyword  (exact match: ticker == "AAPL")
      year     → integer  (range: year >= 2023)
      quarter  → keyword  (exact match: quarter == "Q4")
      date     → keyword  (exact match or range via string comparison)
    """
    index_fields: list[tuple[str, str]] = [
        ("ticker", "keyword"),
        ("year", "integer"),
        ("quarter", "keyword"),
        ("date", "keyword"),
    ]
    for field_name, schema_type in index_fields:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema_type,  # type: ignore[arg-type]
            )
            logger.debug(f"Payload index ensured: {field_name} ({schema_type})")
        except Exception as exc:
            # Qdrant raises if the index already exists — this is expected on
            # every non-first run. Log at debug level and continue.
            logger.debug(f"Payload index '{field_name}' already present or skipped: {exc}")


def init_qdrant(url: str) -> QdrantClient:
    client = QdrantClient(url=url)
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection '{COLLECTION_NAME}' (dim={VECTOR_DIM})")
    else:
        logger.info(f"Qdrant collection '{COLLECTION_NAME}' already exists — reusing.")

    # Always ensure indices exist — idempotent, costs nothing on subsequent calls
    _ensure_payload_indices(client)
    return client


async def index_document(
    chunks: list[Chunk],
    metadata: DocumentMetadata,
    qdrant: QdrantClient,
) -> tuple[list[list[str]], list[dict]]:
    child_chunks = [c for c in chunks if c.chunk_type == "child"]
    points: list[PointStruct] = []
    new_bm25_texts: list[list[str]] = []
    new_bm25_corpus: list[dict] = []

    if not child_chunks:
        return new_bm25_texts, new_bm25_corpus

    texts = [chunk.text for chunk in child_chunks]

    # Run heavy embedding CPU work in a thread pool
    embeddings = await asyncio.to_thread(_get_embeddings, texts)

    for chunk, embedding in zip(child_chunks, embeddings, strict=False):
        # --- FIXED: deterministic ID based on chunk_id ---
        # uuid5 hashes chunk_id → same chunk always gets the same Qdrant point ID.
        # upsert is then idempotent even if the checkpoint is bypassed.
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
        # --------------------------------------------------

        payload = {
            "chunk_id": chunk.chunk_id,
            "parent_id": chunk.parent_id,
            "text": chunk.text,
            "ticker": metadata.ticker,
            "company": metadata.company,
            "date": metadata.date,
            "year": metadata.year,
            "quarter": metadata.quarter,
            "fiscal_period": metadata.fiscal_period,
            "section_title": chunk.section_title,
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        )

        new_bm25_texts.append(chunk.text.lower().split())
        new_bm25_corpus.append(payload)

    def _sync_upsert() -> None:
        for i in range(0, len(points), UPSERT_BATCH_SIZE):
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i : i + UPSERT_BATCH_SIZE],
            )

    # Run blocking IO Qdrant insert in a thread
    await asyncio.to_thread(_sync_upsert)

    return new_bm25_texts, new_bm25_corpus
