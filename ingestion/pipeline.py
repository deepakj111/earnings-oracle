import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from ingestion.chunker import create_parent_child_chunks
from ingestion.indexer import index_document, init_qdrant, setup_genai
from ingestion.metadata_extractor import extract_metadata
from ingestion.parser import parse_html

load_dotenv()

TRANSCRIPTS_DIR = Path("data/transcripts")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")


def run_pipeline() -> None:
    setup_genai()
    qdrant = init_qdrant(os.getenv("QDRANT_URL", "http://localhost:6333"))

    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.htm"))
    logger.info(f"Found {len(transcript_files)} .htm files in {TRANSCRIPTS_DIR}")

    bm25_texts: list[list[str]] = []
    indexed_count = 0
    skipped_count = 0

    for file_path in tqdm(transcript_files, desc="Ingesting"):
        doc = parse_html(file_path)
        if doc is None:
            skipped_count += 1
            logger.debug(f"Skipped (not earnings call): {file_path.name}")
            continue

        metadata = extract_metadata(doc.ticker, doc.date, doc.raw_text)
        chunks = create_parent_child_chunks(doc.ticker, doc.date, doc.sections)
        child_count = sum(1 for c in chunks if c.chunk_type == "child")

        bm25_texts = index_document(chunks, metadata, qdrant, bm25_texts)

        indexed_count += 1
        logger.info(
            f"{file_path.name} | {metadata.fiscal_period} | {child_count} child chunks"
        )

    if bm25_texts:
        bm25 = BM25Okapi(bm25_texts)
        BM25_INDEX_PATH.parent.mkdir(exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(bm25, f)
        logger.info(f"BM25 index saved → {BM25_INDEX_PATH} ({len(bm25_texts)} chunks)")

    logger.info(
        f"Pipeline complete: {indexed_count} docs indexed, {skipped_count} skipped"
    )


if __name__ == "__main__":
    run_pipeline()