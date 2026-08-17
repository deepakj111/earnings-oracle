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
import re
import threading
import time
import uuid
from typing import Any

import tiktoken
from loguru import logger
from openai import RateLimitError
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from config import settings as _settings
from config.openai_client import get_openai_client
from ingestion.chunker import Chunk
from ingestion.metadata_extractor import DocumentMetadata

_cfg = _settings.embedding
COLLECTION_NAME: str = _cfg.collection_name
VECTOR_DIM: int = _cfg.vector_dim
EMBEDDING_MODEL: str = _cfg.model
UPSERT_BATCH_SIZE: int = _cfg.upsert_batch_size

_ENC = tiktoken.get_encoding("cl100k_base")
_MAX_EMBEDDING_TOKENS = (
    8000  # Hard ceiling for OpenAI text-embedding-3-small (8192 tokens max per item)
)

# BM25 tokenizer that preserves financial tokens like "$94.9b", "6.7%", "q4", "2024"
# while stripping trailing punctuation that naive .split() would attach.
_BM25_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.$%/-]*")
_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_NAMES = {
    "01": "january",
    "02": "february",
    "03": "march",
    "04": "april",
    "05": "may",
    "06": "june",
    "07": "july",
    "08": "august",
    "09": "september",
    "10": "october",
    "11": "november",
    "12": "december",
}


def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for BM25 indexing and querying.

    Preserves:
      - Clean alphanumeric and financial tokens ('$94.9b', '6.7%', 'q4', '10-k')
      - Dual date representation: ISO dates ('2024-12-31') expand to include
        year, month name ('december'), and day ('31') so that natural queries
        match ISO formatted dates in tables.
    """
    text_lower = text.lower()
    tokens = _BM25_TOKEN_RE.findall(text_lower)

    # Expand ISO dates to enable natural query date matching
    extra_date_tokens: list[str] = []
    for match in _DATE_ISO_RE.finditer(text_lower):
        year, month, day = match.group(1), match.group(2), match.group(3)
        month_name = _MONTH_NAMES.get(month)
        if month_name:
            extra_date_tokens.extend([year, month_name, str(int(day))])

    if extra_date_tokens:
        tokens.extend(extra_date_tokens)

    return tokens


def _truncate_for_embedding(text: str, max_tokens: int = _MAX_EMBEDDING_TOKENS) -> str:
    """Truncate text to stay strictly under OpenAI's 8192 token per-item limit.

    The previous character-length heuristic (len(text) <= 24_000 → skip) was
    unsafe: dense financial tables or CJK/multi-byte content can produce far
    more tokens per character than ASCII prose, causing 400 errors from the
    embeddings API. We now always tokenise and only decode-truncate when needed.
    """
    tokens = _ENC.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return _ENC.decode(tokens[:max_tokens])


def setup_embedder(threads: int | None = None) -> None:
    """
    Initialise OpenAI API embedding configuration.
    """
    logger.info(f"OpenAI embedding model ready: {EMBEDDING_MODEL} (dim={VECTOR_DIM})")


# A global lock to prevent concurrent documents from bursting OpenAI simultaneously
_openai_rate_lock = threading.Lock()


# Create a helper function decorated with retry logic
@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(RateLimitError),
)
def _call_openai_with_retry(client: Any, batch: list[str], model: str) -> Any:
    """Calls OpenAI API and automatically retries if rate limited."""
    return client.embeddings.create(input=batch, model=model)


def _get_embeddings(
    texts: list[str],
    max_batch_size: int = 50,
    max_batch_tokens: int = 50000,  # Lowered to 50k for smoother pacing
) -> list[list[float]]:
    if not texts:
        return []
    client = get_openai_client()
    embeddings: list[list[float]] = []

    current_batch: list[str] = []
    current_tokens = 0

    for text in texts:
        safe_text = _truncate_for_embedding(text)
        est_tokens = max(1, len(safe_text) // 3)

        if current_batch and (
            len(current_batch) >= max_batch_size or current_tokens + est_tokens > max_batch_tokens
        ):
            # Enforce global pacing across ALL concurrent documents
            with _openai_rate_lock:
                res = _call_openai_with_retry(client, current_batch, EMBEDDING_MODEL)
                # 50k tokens max per batch -> max 20 batches per minute to stay under 1M TPM.
                # Sleeping 3 seconds guarantees a maximum of 20 batches/minute globally.
                time.sleep(3.0)

            for item in res.data:
                embeddings.append(item.embedding)

            current_batch = []
            current_tokens = 0

        current_batch.append(safe_text)
        current_tokens += est_tokens

    if current_batch:
        with _openai_rate_lock:
            res = _call_openai_with_retry(client, current_batch, EMBEDDING_MODEL)
            time.sleep(3.0)

        for item in res.data:
            embeddings.append(item.embedding)

    return embeddings


def _ensure_payload_indices(client: QdrantClient) -> None:
    """
    Create payload field indices on the Qdrant collection for fast metadata filtering.

    Called on every init_qdrant() — fully idempotent, safe to repeat on restarts.
    Without these indices Qdrant performs a full collection scan on every filtered
    query, which is functionally correct but O(n) instead of O(log n).

    Index types:
      ticker    → keyword  (exact match: ticker == "AAPL")
      year      → integer  (range: year >= 2023)
      quarter   → keyword  (exact match: quarter == "Q4")
      date      → keyword  (exact match or range via string comparison)
      parent_id → keyword  (exact match for fast parent context reconstruction)
    """
    index_fields: list[tuple[str, str]] = [
        ("ticker", "keyword"),
        ("year", "integer"),
        ("quarter", "keyword"),
        ("date", "keyword"),
        ("parent_id", "keyword"),
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
    """Initialize Qdrant client and optionally create the target collection if missing."""
    try:
        client = QdrantClient(url=url, timeout=5, check_compatibility=False)
        existing = {c.name for c in client.get_collections().collections}
    except Exception as exc:
        logger.warning(
            f"Could not connect to Qdrant at {url} ({exc}). Using local path storage 'data/qdrant_user_storage'..."
        )
        client = QdrantClient(path="data/qdrant_user_storage")
        existing = {c.name for c in client.get_collections().collections}

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            on_disk_payload=True,
            hnsw_config=HnswConfigDiff(on_disk=True),
        )
        logger.info(
            f"Created Qdrant collection '{COLLECTION_NAME}' (dim={VECTOR_DIM}, on_disk=True)"
        )

    # Always ensure indices exist — idempotent, costs nothing on subsequent calls
    _ensure_payload_indices(client)
    return client


async def index_document(
    chunks: list[Chunk],
    metadata: DocumentMetadata,
    qdrant: QdrantClient,
    timings: dict | None = None,
) -> tuple[list[list[str]], list[dict]]:
    """Embed chunks and index them into Qdrant, returning data for the BM25 corpus."""
    indexable_chunks = [c for c in chunks if c.chunk_type in ("child", "table")]
    points: list[PointStruct] = []
    new_bm25_texts: list[list[str]] = []
    new_bm25_corpus: list[dict] = []

    if not indexable_chunks:
        return new_bm25_texts, new_bm25_corpus

    texts = [chunk.text for chunk in indexable_chunks]

    # Run heavy embedding CPU work in a thread pool
    t0 = time.perf_counter()
    embeddings = await asyncio.to_thread(_get_embeddings, texts)
    t1 = time.perf_counter()
    if timings is not None:
        timings["embedding"] = round(t1 - t0, 4)

    logger.debug(
        f"[QDRANT INDEX] Embedding and preparing {len(indexable_chunks)} points for {metadata.ticker} ({metadata.fiscal_period})..."
    )
    for chunk, embedding in zip(indexable_chunks, embeddings, strict=False):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

        payload = {
            "chunk_id": chunk.chunk_id,
            "parent_id": chunk.parent_id or chunk.chunk_id,
            "file_name": getattr(metadata, "file_name", ""),
            "text": chunk.text,
            "ticker": metadata.ticker,
            "company": metadata.company,
            "date": metadata.date,
            "year": metadata.year,
            "quarter": metadata.quarter,
            "fiscal_period": metadata.fiscal_period,
            "form_type": metadata.form_type,
            "section_title": chunk.section_title or "Financial Table",
            "chunk_type": chunk.chunk_type,
            "is_table": (chunk.chunk_type == "table"),
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        )

        logger.debug(
            f"  └─ [QDRANT POINT] Point ID: {point_id} | chunk_id: {chunk.chunk_id} | "
            f"ticker: {metadata.ticker} | quarter: {metadata.quarter} | "
            f"section: '{chunk.section_title}' | text preview: {chunk.text[:80]!r}"
        )

        new_bm25_texts.append(_tokenize_for_bm25(chunk.text))
        new_bm25_corpus.append(payload)

    def _sync_upsert() -> None:
        if not qdrant.collection_exists(COLLECTION_NAME):
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
                on_disk_payload=True,
                hnsw_config=HnswConfigDiff(on_disk=True),
            )
            _ensure_payload_indices(qdrant)
        for i in range(0, len(points), UPSERT_BATCH_SIZE):
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i : i + UPSERT_BATCH_SIZE],
            )
        logger.debug(
            f"[QDRANT UPSERT] Successfully upserted {len(points)} points into collection '{COLLECTION_NAME}'"
        )

    # Run blocking IO Qdrant insert in a thread
    t2 = time.perf_counter()
    await asyncio.to_thread(_sync_upsert)
    t3 = time.perf_counter()
    if timings is not None:
        timings["qdrant_upsert"] = round(t3 - t2, 4)

    return new_bm25_texts, new_bm25_corpus
