# scripts/reset_index.py
"""
Script to wipe Qdrant vector index collection and clean local ingestion caches/indexes.

Cleans:
  1. Qdrant collection (`earnings_call_chunks`)
  2. Local BM25 index & corpus pickles (`data/bm25_index.pkl`, `data/bm25_corpus.pkl`)
  3. Knowledge Graph JSON store (`data/knowledge_graph.json`)
  4. Ingestion checkpoint file (`data/ingested_filings_checkpoint.txt`)
  5. Ingestion metrics file (`data/ingestion_metrics.json`)
"""

from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient

from config import settings

FILES_TO_REMOVE = [
    Path("data/bm25_index.pkl"),
    Path("data/bm25_corpus.pkl"),
    Path("data/ingested_filings_checkpoint.txt"),
    Path("data/ingestion_metrics.json"),
    Path("data/knowledge_graph.json"),
]


def reset_all() -> None:
    # 1. Clean Qdrant Collection
    qdrant_url = settings.infra.qdrant_url
    collection_name = settings.embedding.collection_name
    logger.info(f"Connecting to Qdrant at {qdrant_url}...")

    try:
        client = QdrantClient(url=qdrant_url, timeout=30.0, check_compatibility=False)
        existing = {c.name for c in client.get_collections().collections}
        if collection_name in existing:
            client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted Qdrant collection '{collection_name}' successfully.")
        else:
            logger.info(f"Qdrant collection '{collection_name}' does not exist — skipping.")
    except Exception as exc:
        logger.error(f"Failed to clear Qdrant collection: {exc}")

    # 2. Clean Local Files
    for file_path in FILES_TO_REMOVE:
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Removed local index file: {file_path}")
            except Exception as exc:
                logger.warning(f"Failed to remove {file_path}: {exc}")
        else:
            logger.debug(f"File not found (already clean): {file_path}")

    logger.info("Cleanup complete. System is clean and ready for fresh ingestion!")


if __name__ == "__main__":
    reset_all()
