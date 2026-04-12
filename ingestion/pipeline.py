import pickle  # nosec B403
from pathlib import Path

from loguru import logger
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from config import settings as _settings
from ingestion.chunker import create_parent_child_chunks
from ingestion.indexer import index_document, init_qdrant, setup_embedder
from ingestion.metadata_extractor import extract_metadata
from ingestion.parser import parse_html

TRANSCRIPTS_DIR = Path("data/transcripts")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
CHECKPOINT_PATH = Path("data/pipeline_checkpoint.txt")


def _load_checkpoint() -> set[str]:
    if CHECKPOINT_PATH.exists():
        names = {line.strip() for line in CHECKPOINT_PATH.read_text().splitlines() if line.strip()}
        logger.info(f"Checkpoint loaded — {len(names)} files already indexed, skipping.")
        return names
    return set()


def _mark_done(filename: str) -> None:
    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
    with open(CHECKPOINT_PATH, "a") as f:
        f.write(filename + "\n")


def _save_bm25(bm25_texts: list[list[str]]) -> None:
    if not bm25_texts:
        return
    bm25 = BM25Okapi(bm25_texts)
    BM25_INDEX_PATH.parent.mkdir(exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)  # nosec B403
    logger.info(f"BM25 index saved → {BM25_INDEX_PATH} ({len(bm25_texts)} chunks)")


def run_pipeline() -> None:
    setup_embedder()

    qdrant = init_qdrant(_settings.infra.qdrant_url)

    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.htm"))
    logger.info(f"Found {len(transcript_files)} .htm files in {TRANSCRIPTS_DIR}")

    already_done = _load_checkpoint()
    pending = [f for f in transcript_files if f.name not in already_done]
    logger.info(f"{len(already_done)} skipped (checkpoint) | {len(pending)} to process")

    bm25_texts: list[list[str]] = []
    indexed_count: int = 0
    skipped_count: int = len(already_done)

    for file_path in tqdm(pending, desc="Ingesting"):
        doc = parse_html(file_path)
        if doc is None:
            skipped_count += 1
            logger.debug(f"Skipped (not earnings content): {file_path.name}")
            continue

        metadata = extract_metadata(doc.ticker, doc.date, doc.raw_text)
        chunks = create_parent_child_chunks(doc.ticker, doc.date, doc.sections)
        child_count = sum(1 for c in chunks if c.chunk_type == "child")

        bm25_texts = index_document(chunks, metadata, qdrant, bm25_texts)

        _mark_done(file_path.name)
        indexed_count += 1
        logger.info(f"{file_path.name} | {metadata.fiscal_period} | {child_count} child chunks")

    _save_bm25(bm25_texts)
    logger.info(f"Pipeline complete: {indexed_count} indexed this run, {skipped_count} skipped")


if __name__ == "__main__":
    run_pipeline()
