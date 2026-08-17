# scripts/reset_index.py
"""
Script to wipe Qdrant vector index collection and clean local ingestion caches/indexes.

Cleans:
  1. Qdrant collection (`company_filings`)
  2. Local BM25 index & corpus pickles (`data/bm25_index.pkl`, `data/bm25_corpus.pkl`)
  3. Legacy ingestion checkpoint file (`data/ingested_filings_checkpoint.txt` if present)
  4. Ingestion metrics file (`data/ingestion_metrics.json`)
  5. Knowledge Graph (ONLY if explicitly requested via --wipe-kg)
"""

import argparse
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient

from config import settings

FILES_TO_REMOVE = [
    Path("data/bm25_index.pkl"),
    Path("data/bm25_corpus.pkl"),
    Path("data/ingested_filings_checkpoint.txt"),
    Path("data/ingestion_metrics.json"),
]
KG_FILE = Path("data/knowledge_graph.json")


def reset_all(wipe_kg: bool = False) -> None:
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

    # 2. Clean Local Storage Folders
    import shutil

    for local_dir in [Path("data/qdrant_user_storage"), Path("data/qdrant_storage")]:
        if local_dir.exists():
            try:
                shutil.rmtree(local_dir)
                logger.info(f"Removed local Qdrant storage directory: {local_dir}")
            except Exception as exc:
                logger.warning(f"Failed to remove {local_dir}: {exc}")

    # 3. Clean Local Index Files
    files_to_clean = list(FILES_TO_REMOVE)
    if wipe_kg:
        files_to_clean.append(KG_FILE)
    else:
        logger.info(
            f"Preserving existing Knowledge Graph ({KG_FILE}) to save LLM extraction costs."
        )

    for file_path in files_to_clean:
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
    parser = argparse.ArgumentParser(description="Reset Qdrant and BM25 search indices.")
    parser.add_argument(
        "--wipe-kg",
        action="store_true",
        default=False,
        help="Wipe the Knowledge Graph JSON store as well (default: False to preserve KG)",
    )
    args = parser.parse_args()
    reset_all(wipe_kg=args.wipe_kg)
