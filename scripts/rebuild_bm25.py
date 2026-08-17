#!/usr/bin/env python3
"""
Rebuild the BM25 index from the existing bm25_corpus.pkl.

Use this script whenever BM25 hyperparameters change (k1, b) without
needing to re-embed or re-ingest documents. No API calls needed.

Usage:
    poetry run python scripts/rebuild_bm25.py
"""

import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rank_bm25 import BM25Okapi

BM25_INDEX_PATH = Path("data/bm25_index.pkl")
BM25_CORPUS_PATH = Path("data/bm25_corpus.pkl")

# BM25 hyperparameters tuned for financial document corpora:
#   k1=1.5 : term frequency saturation (standard default, range 1.2-2.0)
#   b=0.5  : reduced from 0.75 default — financial chunks vary wildly in length
#             (3-row tables vs 500-word MD&A sections). Lower b reduces
#             over-penalization of longer, information-rich chunks.
K1 = 1.5
B = 0.5


def rebuild_bm25() -> None:
    if not BM25_CORPUS_PATH.exists():
        logger.error(f"BM25 corpus not found at {BM25_CORPUS_PATH}. Run ingestion first.")
        sys.exit(1)

    logger.info(f"Loading BM25 corpus from {BM25_CORPUS_PATH} ...")
    with open(BM25_CORPUS_PATH, "rb") as f:
        corpus: list[dict] = pickle.load(f)  # nosec B301

    logger.info(f"Corpus loaded: {len(corpus)} entries")

    # Re-tokenize from stored chunk text
    from ingestion.indexer import _tokenize_for_bm25

    logger.info("Re-tokenizing corpus texts ...")
    bm25_texts: list[list[str]] = []
    for entry in corpus:
        text = entry.get("text", "")
        bm25_texts.append(_tokenize_for_bm25(text))

    logger.info(f"Building BM25Okapi index with k1={K1}, b={B} ...")
    bm25 = BM25Okapi(bm25_texts, k1=K1, b=B)

    # Back up old index
    if BM25_INDEX_PATH.exists():
        backup_path = BM25_INDEX_PATH.with_suffix(".pkl.bak")
        shutil.copy2(BM25_INDEX_PATH, backup_path)
        logger.info(f"Old index backed up -> {backup_path}")

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)  # nosec B403

    logger.info(
        f"BM25 index rebuilt -> {BM25_INDEX_PATH} ({len(bm25_texts)} chunks, k1={K1}, b={B})"
    )


if __name__ == "__main__":
    rebuild_bm25()
