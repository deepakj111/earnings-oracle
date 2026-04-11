import os
import time
import uuid

import numpy as np
from google import genai
from google.genai import types as genai_types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ingestion.chunker import Chunk
from ingestion.metadata_extractor import DocumentMetadata

COLLECTION_NAME = "earnings_transcripts"
VECTOR_DIM = 768
EMBEDDING_MODEL = "gemini-embedding-001"
UPSERT_BATCH_SIZE = 50

_genai_client: genai.Client | None = None


def setup_genai() -> None:
    global _genai_client
    _genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def _get_embedding(text: str) -> list[float]:
    assert _genai_client is not None, "Call setup_genai() before indexing"
    result = _genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=VECTOR_DIM,
        ),
    )
    vec = result.embeddings[0].values
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
        time.sleep(0.05)  # respect Gemini free-tier rate limits

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
