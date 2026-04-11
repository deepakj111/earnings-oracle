import uuid

import numpy as np
from fastembed import TextEmbedding
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ingestion.chunker import Chunk
from ingestion.metadata_extractor import DocumentMetadata

COLLECTION_NAME = "earnings_transcripts"
VECTOR_DIM = 1024
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
UPSERT_BATCH_SIZE = 50

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


def _get_embedding(text: str) -> list[float]:
    assert _embed_model is not None, "Call setup_embedder() before indexing"
    vec = next(iter(_embed_model.embed([text])))
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


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
    return client


def index_document(
    chunks: list[Chunk],
    metadata: DocumentMetadata,
    qdrant: QdrantClient,
    bm25_texts: list[list[str]],
) -> list[list[str]]:
    child_chunks = [c for c in chunks if c.chunk_type == "child"]
    points: list[PointStruct] = []

    for chunk in child_chunks:
        embedding = _get_embedding(chunk.text)

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "parent_id": chunk.parent_id,
                    "text": chunk.text,
                    "ticker": metadata.ticker,
                    "company": metadata.company,
                    "date": metadata.date,
                    "year": metadata.year,
                    "quarter": metadata.quarter,
                    "fiscal_period": metadata.fiscal_period,
                },
            )
        )
        bm25_texts.append(chunk.text.lower().split())

    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + UPSERT_BATCH_SIZE],
        )

    return bm25_texts
