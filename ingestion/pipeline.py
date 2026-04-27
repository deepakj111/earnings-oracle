import asyncio
import pickle  # nosec B403
from pathlib import Path
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from config import settings as _settings
from ingestion.chunker import create_parent_child_chunks
from ingestion.indexer import index_document, init_qdrant, setup_embedder
from ingestion.metadata_extractor import extract_metadata
from ingestion.parser import parse_html

TRANSCRIPTS_DIR = Path("data/transcripts")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
BM25_CORPUS_PATH = Path("data/bm25_corpus.pkl")
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


def _save_bm25(bm25_texts: list[list[str]], bm25_corpus: list[dict]) -> None:
    """
    Persist the BM25 index and its parallel corpus metadata file.

    Two files are always written together:
      bm25_index.pkl  — BM25Okapi object (token weights for scoring)
      bm25_corpus.pkl — list[dict], same length and order as bm25_index's corpus

    The retrieval layer loads both. BM25 search returns integer rank indices;
    those indices are resolved to chunk metadata via bm25_corpus[index].
    """
    if not bm25_texts:
        logger.warning("No BM25 texts to save — skipping index write.")
        return

    if len(bm25_texts) != len(bm25_corpus):
        raise RuntimeError(
            f"BM25 invariant violated: bm25_texts has {len(bm25_texts)} entries "
            f"but bm25_corpus has {len(bm25_corpus)}. They must be equal length."
        )

    bm25 = BM25Okapi(bm25_texts)
    BM25_INDEX_PATH.parent.mkdir(exist_ok=True)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)  # nosec B403
    logger.info(f"BM25 index saved → {BM25_INDEX_PATH} ({len(bm25_texts)} chunks)")

    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump(bm25_corpus, f)  # nosec B403
    logger.info(f"BM25 corpus saved → {BM25_CORPUS_PATH} ({len(bm25_corpus)} entries)")


def _load_existing_bm25() -> tuple[list[list[str]], list[dict]]:
    """
    Load the existing BM25 corpus from disk.
    Gracefully handles corrupt/truncated pickle files by starting fresh.
    """
    if not BM25_CORPUS_PATH.exists():
        logger.info("No existing BM25 corpus found — starting fresh.")
        return [], []

    try:
        with open(BM25_CORPUS_PATH, "rb") as f:  # nosec B403
            bm25_corpus: list[dict] = pickle.load(f)  # nosec B301
        bm25_texts = [entry["text"].lower().split() for entry in bm25_corpus]
        logger.info(f"Loaded existing BM25 corpus — {len(bm25_corpus)} chunks carried forward.")
        return bm25_texts, bm25_corpus
    except (pickle.UnpicklingError, EOFError, Exception) as exc:
        logger.warning(
            f"BM25 corpus at {BM25_CORPUS_PATH} is corrupt or unreadable ({exc}). "
            "Starting fresh — all files will be re-indexed this run."
        )
        return [], []


async def _process_document(
    file_path: Path,
    qdrant: QdrantClient,
    semaphore: asyncio.Semaphore,
    kg_enabled: bool,
    kg_graph: Any,
) -> tuple[int, list[list[str]], list[dict]]:
    """Process a single document concurrently."""
    async with semaphore:
        doc = parse_html(file_path)
        if doc is None:
            logger.debug(f"Skipped (not earnings content): {file_path.name}")
            return 0, [], []

        metadata = extract_metadata(doc.ticker, doc.date, doc.raw_text)
        chunks = create_parent_child_chunks(doc.ticker, doc.date, doc.sections)
        child_count = sum(1 for c in chunks if c.chunk_type == "child")

        new_bm25_texts, new_bm25_corpus = await index_document(chunks, metadata, qdrant)

        # ── Knowledge Graph extraction ─────────────────────────────────
        if kg_enabled:
            from knowledge_graph.extractor import extract_entities_from_chunks

            parent_chunks = [c for c in chunks if c.chunk_type == "parent"]
            try:
                entities, relationships = await extract_entities_from_chunks(
                    parent_chunks, metadata.ticker, metadata.fiscal_period
                )
                for entity in entities:
                    kg_graph.add_entity(entity)
                for rel in relationships:
                    kg_graph.add_relationship(rel)
            except Exception as exc:
                logger.warning(f"KG extraction failed for {file_path.name}: {exc}")

        await asyncio.to_thread(_mark_done, file_path.name)
        logger.info(f"{file_path.name} | {metadata.fiscal_period} | {child_count} child chunks")

        return child_count, new_bm25_texts, new_bm25_corpus


async def run_pipeline_async() -> None:
    """Run the ingestion indexing pipeline asynchronously for pending transcript files."""
    setup_embedder()
    qdrant = init_qdrant(_settings.infra.qdrant_url)

    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.htm"))
    logger.info(f"Found {len(transcript_files)} .htm files in {TRANSCRIPTS_DIR}")

    already_done = _load_checkpoint()
    pending = [f for f in transcript_files if f.name not in already_done]
    logger.info(f"{len(already_done)} skipped (checkpoint) | {len(pending)} to process")

    # --- FIXED: seed bm25 with previously indexed docs ---
    bm25_texts, bm25_corpus = _load_existing_bm25()
    # -----------------------------------------------------

    # ── Knowledge Graph setup ──────────────────────────────────────────────
    kg_enabled = _settings.knowledge_graph.extraction_enabled
    kg_store, kg_graph = None, None
    if kg_enabled:
        from knowledge_graph.entity_store import EntityStore

        kg_store = EntityStore()
        kg_graph = kg_store.load()

    indexed_count: int = 0
    skipped_count: int = len(already_done)

    semaphore = asyncio.Semaphore(5)

    tasks = [_process_document(f, qdrant, semaphore, kg_enabled, kg_graph) for f in pending]

    if tasks:
        logger.info(f"Launching {len(tasks)} document ingestion tasks concurrently...")
        results = await asyncio.gather(*tasks)

        for child_count, new_bm25_texts, new_bm25_corpus in results:
            if child_count == 0 and not new_bm25_texts:
                skipped_count += 1
            else:
                indexed_count += 1
                bm25_texts.extend(new_bm25_texts)
                bm25_corpus.extend(new_bm25_corpus)

    _save_bm25(bm25_texts, bm25_corpus)

    # Persist knowledge graph
    if kg_enabled and kg_store and kg_graph:
        kg_store.save(kg_graph)
        logger.info(f"Knowledge graph: {kg_graph.summary()}")

    logger.info(f"Pipeline complete: {indexed_count} indexed this run, {skipped_count} skipped")


def run_pipeline() -> None:
    """Synchronous entry point to run the ingestion pipeline."""
    asyncio.run(run_pipeline_async())


if __name__ == "__main__":
    run_pipeline()
