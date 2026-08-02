import asyncio
import json
import os
import pickle  # nosec B403
import time
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

TRANSCRIPTS_DIR = Path("data/company_filings")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
BM25_CORPUS_PATH = Path("data/bm25_corpus.pkl")
CHECKPOINT_PATH = Path("data/ingested_filings_checkpoint.txt")
INGESTION_METRICS_PATH = Path("data/ingestion_metrics.json")


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
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)  # nosec B403 — trusted local data only
    logger.info(f"BM25 index saved → {BM25_INDEX_PATH} ({len(bm25_texts)} chunks)")

    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump(bm25_corpus, f, protocol=pickle.HIGHEST_PROTOCOL)  # nosec B403 — trusted local data only
    logger.info(f"BM25 corpus saved → {BM25_CORPUS_PATH} ({len(bm25_corpus)} entries)")


def _load_existing_metrics() -> list[dict]:
    """Load existing per-document ingestion metrics from JSON file if present."""
    if not INGESTION_METRICS_PATH.exists():
        return []
    try:
        with open(INGESTION_METRICS_PATH, encoding="utf-8") as f:
            data = json.load(f)
            if (
                isinstance(data, dict)
                and "documents" in data
                and isinstance(data["documents"], list)
            ):
                logger.info(
                    f"Loaded existing metrics — {len(data['documents'])} document records carried forward."
                )
                return data["documents"]
    except Exception as exc:
        logger.warning(
            f"Failed to load existing ingestion metrics ({exc}). Starting with empty metrics records."
        )
    return []


def _save_ingestion_metrics(metrics_records: list[dict], total_pipeline_time: float) -> None:
    """Persist structured per-step and per-document ingestion timing metrics to JSON."""
    if not metrics_records:
        return

    step_totals: dict[str, float] = {}
    for record in metrics_records:
        for step, duration in record.get("timings_seconds", {}).items():
            if step != "total_document":
                step_totals[step] = round(step_totals.get(step, 0.0) + duration, 4)

    doc_totals = [
        r["timings_seconds"]["total_document"]
        for r in metrics_records
        if "total_document" in r.get("timings_seconds", {})
    ]
    avg_doc_time = round(sum(doc_totals) / len(doc_totals), 4) if doc_totals else 0.0

    report = {
        "summary": {
            "total_documents_processed": len(metrics_records),
            "total_pipeline_time_seconds": round(total_pipeline_time, 4),
            "average_document_time_seconds": avg_doc_time,
            "step_totals_seconds": step_totals,
        },
        "documents": metrics_records,
    }

    INGESTION_METRICS_PATH.parent.mkdir(exist_ok=True)
    temp_path = INGESTION_METRICS_PATH.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temp_path.replace(INGESTION_METRICS_PATH)

    logger.info(f"Ingestion timing metrics saved → {INGESTION_METRICS_PATH.resolve()}")


def _load_existing_bm25() -> tuple[list[list[str]], list[dict]]:
    """
    Load the existing BM25 corpus from disk.
    Gracefully handles corrupt/truncated pickle files by starting fresh.
    """
    if not BM25_CORPUS_PATH.exists():
        logger.info("No existing BM25 corpus found — starting fresh.")
        return [], []

    try:
        with open(BM25_CORPUS_PATH, "rb") as f:  # nosec B403 — trusted local data
            bm25_corpus: list[dict] = pickle.load(f)  # nosec B301 — trusted local data only
        bm25_texts = [entry["text"].lower().split() for entry in bm25_corpus]
        logger.info(f"Loaded existing BM25 corpus — {len(bm25_corpus)} chunks carried forward.")
        return bm25_texts, bm25_corpus
    except (pickle.UnpicklingError, EOFError, Exception) as exc:
        logger.warning(
            f"BM25 corpus at {BM25_CORPUS_PATH} is corrupt or unreadable ({exc}). "
            "Starting fresh — all files will be re-indexed this run."
        )
        return [], []


LOG_FILE_PATH = Path("logs/ingestion_debug.log")


def setup_ingestion_logging() -> Path:
    """Configure detailed debug log file for ingestion pipeline execution."""
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        LOG_FILE_PATH,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
    )
    logger.info(f"Detailed ingestion log file active → {LOG_FILE_PATH.resolve()}")
    return LOG_FILE_PATH


async def _process_document(
    file_path: Path,
    qdrant: QdrantClient,
    semaphore: asyncio.Semaphore,
    kg_enabled: bool,
    kg_graph: Any,
    checkpoint_lock: asyncio.Lock | None = None,
    kg_lock: asyncio.Lock | None = None,
    metrics_lock: asyncio.Lock | None = None,
    metrics_records: list[dict] | None = None,
    pipeline_start_time: float | None = None,
) -> tuple[int, list[list[str]], list[dict], dict | None]:
    """Process a single document concurrently while profiling execution steps."""
    t_start = time.perf_counter()
    doc_timings: dict[str, float] = {}

    try:
        async with semaphore:
            t0 = time.perf_counter()
            doc = parse_html(file_path)
            t1 = time.perf_counter()
            doc_timings["parse_html"] = round(t1 - t0, 4)

            if doc is None:
                logger.debug(f"Skipped (not earnings content): {file_path.name}")
                return 0, [], [], None

            t0 = time.perf_counter()
            metadata = extract_metadata(doc.ticker, doc.date, doc.raw_text)
            t1 = time.perf_counter()
            doc_timings["extract_metadata"] = round(t1 - t0, 4)

            t0 = time.perf_counter()
            chunks = create_parent_child_chunks(doc.ticker, doc.date, doc.sections)
            t1 = time.perf_counter()
            doc_timings["create_chunks"] = round(t1 - t0, 4)

            parent_count = sum(1 for c in chunks if c.chunk_type == "parent")
            child_count = sum(1 for c in chunks if c.chunk_type == "child")

            logger.debug(
                f"[PARSE] File: {file_path.name} | Ticker: {metadata.ticker} | "
                f"Period: {metadata.fiscal_period} | Sections: {len(doc.sections)} | "
                f"Raw text length: {len(doc.raw_text)} chars"
            )
            logger.debug(
                f"[CHUNK SUMMARY] {file_path.name} | {parent_count} parent chunks, {child_count} child chunks"
            )

            indexer_timings: dict[str, float] = {}
            new_bm25_texts, new_bm25_corpus = await index_document(
                chunks, metadata, qdrant, timings=indexer_timings
            )
            doc_timings["embedding"] = indexer_timings.get("embedding", 0.0)
            doc_timings["qdrant_upsert"] = indexer_timings.get("qdrant_upsert", 0.0)

            # ── Knowledge Graph extraction ─────────────────────────────────
            t_kg_start = time.perf_counter()
            if kg_enabled and kg_graph is not None:
                from knowledge_graph.extractor import extract_entities_from_chunks

                parent_chunks = [c for c in chunks if c.chunk_type == "parent"]
                try:
                    entities, relationships = await extract_entities_from_chunks(
                        parent_chunks, metadata.ticker, metadata.fiscal_period
                    )
                    if kg_lock:
                        async with kg_lock:
                            for entity in entities:
                                kg_graph.add_entity(entity)
                            for rel in relationships:
                                kg_graph.add_relationship(rel)
                    else:
                        for entity in entities:
                            kg_graph.add_entity(entity)
                        for rel in relationships:
                            kg_graph.add_relationship(rel)
                except Exception as exc:
                    logger.warning(f"KG extraction failed for {file_path.name}: {exc}")
            t_kg_end = time.perf_counter()
            doc_timings["kg_extraction"] = round(t_kg_end - t_kg_start, 4)

            t_cp_start = time.perf_counter()
            if checkpoint_lock:
                async with checkpoint_lock:
                    await asyncio.to_thread(_mark_done, file_path.name)
            else:
                await asyncio.to_thread(_mark_done, file_path.name)
            t_cp_end = time.perf_counter()
            doc_timings["checkpoint_mark"] = round(t_cp_end - t_cp_start, 4)

            t_total = time.perf_counter() - t_start
            doc_timings["total_document"] = round(t_total, 4)

            doc_metric = {
                "file_name": file_path.name,
                "ticker": metadata.ticker,
                "fiscal_period": metadata.fiscal_period,
                "raw_text_length": len(doc.raw_text),
                "parent_chunks": parent_count,
                "child_chunks": child_count,
                "timings_seconds": doc_timings,
            }

            if (
                metrics_lock is not None
                and metrics_records is not None
                and pipeline_start_time is not None
            ):
                async with metrics_lock:
                    existing_idx = next(
                        (
                            i
                            for i, r in enumerate(metrics_records)
                            if r.get("file_name") == doc_metric["file_name"]
                        ),
                        None,
                    )
                    if existing_idx is not None:
                        metrics_records[existing_idx] = doc_metric
                    else:
                        metrics_records.append(doc_metric)
                    _save_ingestion_metrics(
                        metrics_records, time.perf_counter() - pipeline_start_time
                    )

            logger.info(
                f"{file_path.name} | {metadata.fiscal_period} | {child_count} child chunks | {t_total:.3f}s"
            )

            return child_count, new_bm25_texts, new_bm25_corpus, doc_metric

    except Exception as exc:
        logger.error(f"Error processing {file_path.name}: {exc}")
        return 0, [], [], None


async def run_pipeline_async(
    fast: bool = False,
    concurrency_override: int | None = None,
    threads_override: int | None = None,
) -> None:
    """Run the ingestion indexing pipeline asynchronously for pending transcript files."""
    pipeline_start_time = time.perf_counter()
    setup_ingestion_logging()
    setup_embedder(threads=threads_override)
    qdrant = init_qdrant(_settings.infra.qdrant_url)

    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.htm"))
    logger.info(f"Found {len(transcript_files)} .htm files in {TRANSCRIPTS_DIR}")

    already_done = _load_checkpoint()
    pending = [f for f in transcript_files if f.name not in already_done]
    logger.info(f"{len(already_done)} skipped (checkpoint) | {len(pending)} to process")

    # --- seed bm25 with previously indexed docs ---
    bm25_texts, bm25_corpus = _load_existing_bm25()

    # ── Knowledge Graph setup ──────────────────────────────────────────────
    kg_enabled = False if fast else _settings.knowledge_graph.extraction_enabled
    kg_store, kg_graph = None, None
    if kg_enabled:
        from knowledge_graph.entity_store import EntityStore

        kg_store = EntityStore()
        kg_graph = kg_store.load()

    indexed_count: int = 0
    skipped_count: int = len(already_done)

    concurrency = concurrency_override or _settings.embedding.max_concurrency
    batch_size = _settings.embedding.ingestion_batch_size
    semaphore = asyncio.Semaphore(concurrency)
    checkpoint_lock = asyncio.Lock()
    kg_lock = asyncio.Lock()
    metrics_lock = asyncio.Lock()

    existing_chunk_ids = {entry["chunk_id"] for entry in bm25_corpus if "chunk_id" in entry}
    metrics_records: list[dict] = _load_existing_metrics()

    if pending:
        logger.info(
            f"Processing {len(pending)} pending documents in batches of {batch_size} "
            f"with concurrency limit {concurrency} (llm_kg={'off' if fast else 'on'})..."
        )
        import gc

        from tqdm.asyncio import tqdm

        total_batches = (len(pending) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            batch_files = pending[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_tasks = [
                _process_document(
                    f,
                    qdrant,
                    semaphore,
                    kg_enabled,
                    kg_graph,
                    checkpoint_lock,
                    kg_lock,
                    metrics_lock,
                    metrics_records,
                    pipeline_start_time,
                )
                for f in batch_files
            ]

            results = await tqdm.gather(
                *batch_tasks, desc=f"Ingesting batch {batch_idx + 1}/{total_batches}"
            )
            for res in results:
                child_count, new_bm25_texts, new_bm25_corpus = res[0], res[1], res[2]
                doc_metric = res[3] if len(res) > 3 else None

                if child_count == 0 and not new_bm25_texts:
                    skipped_count += 1
                else:
                    indexed_count += 1
                    if doc_metric and not any(
                        r.get("file_name") == doc_metric["file_name"] for r in metrics_records
                    ):
                        metrics_records.append(doc_metric)
                    for text, corpus_entry in zip(new_bm25_texts, new_bm25_corpus, strict=False):
                        cid = corpus_entry.get("chunk_id")
                        if cid and cid in existing_chunk_ids:
                            continue
                        if cid:
                            existing_chunk_ids.add(cid)
                        bm25_texts.append(text)
                        bm25_corpus.append(corpus_entry)

            del batch_tasks
            del results
            gc.collect()

            # Save metrics live after each batch
            _save_ingestion_metrics(metrics_records, time.perf_counter() - pipeline_start_time)

    _save_bm25(bm25_texts, bm25_corpus)

    # Persist knowledge graph
    if kg_enabled and kg_store and kg_graph:
        kg_store.save(kg_graph)
        logger.info(f"Knowledge graph: {kg_graph.summary()}")

    pipeline_total_time = time.perf_counter() - pipeline_start_time
    _save_ingestion_metrics(metrics_records, pipeline_total_time)

    logger.info(f"Pipeline complete: {indexed_count} indexed this run, {skipped_count} skipped")


def run_pipeline(
    fast: bool = False, concurrency: int | None = None, threads: int | None = None
) -> None:
    """Synchronous entry point to run the ingestion pipeline."""
    asyncio.run(
        run_pipeline_async(fast=fast, concurrency_override=concurrency, threads_override=threads)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingestion indexing pipeline")
    parser.add_argument(
        "--fast",
        "--no-kg",
        action="store_true",
        help="Run fast ingestion (use instant regex entity extraction, skip LLM network calls)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Override worker task concurrency limit",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Override embedding model ONNX CPU thread count (0 for auto multi-threading)",
    )
    args = parser.parse_args()
    run_pipeline(fast=args.fast, concurrency=args.concurrency, threads=args.threads)
