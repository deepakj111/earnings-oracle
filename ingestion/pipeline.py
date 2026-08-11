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
from ingestion.indexer import COLLECTION_NAME, index_document, init_qdrant, setup_embedder
from ingestion.metadata_extractor import extract_metadata
from ingestion.parser import parse_html

TRANSCRIPTS_DIR = Path("data/company_filings")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
BM25_CORPUS_PATH = Path("data/bm25_corpus.pkl")
INGESTION_METRICS_PATH = Path("data/ingestion_metrics.json")
KG_GRAPH_PATH = Path("data/knowledge_graph.json")


class IngestionStoreState:
    """
    In-memory representation of indexed chunk IDs across Vector DB (Qdrant),
    BM25 corpus, and Knowledge Graph.

    Replaces external sidecar text checkpoint files with direct, automatic state
    inspection derived from the actual underlying data stores.
    """

    def __init__(
        self,
        qdrant_chunk_ids: set[str],
        bm25_chunk_ids: set[str],
        kg_chunk_ids: set[str],
    ) -> None:
        self.qdrant_chunk_ids = qdrant_chunk_ids
        self.bm25_chunk_ids = bm25_chunk_ids
        self.kg_chunk_ids = kg_chunk_ids

    @classmethod
    def load(
        cls,
        qdrant: QdrantClient | None,
        bm25_corpus: list[dict],
        kg_graph: Any | None,
    ) -> "IngestionStoreState":
        qdrant_chunk_ids: set[str] = set()
        if qdrant is not None:
            try:
                if qdrant.collection_exists(COLLECTION_NAME):
                    offset = None
                    while True:
                        batch, offset = qdrant.scroll(
                            COLLECTION_NAME,
                            limit=500,
                            offset=offset,
                            with_payload=["chunk_id"],
                            with_vectors=False,
                        )
                        for point in batch:
                            if point.payload and "chunk_id" in point.payload:
                                qdrant_chunk_ids.add(point.payload["chunk_id"])
                        if offset is None:
                            break
            except Exception as exc:
                logger.warning(f"Failed to scroll Qdrant collection for store state: {exc}")

        bm25_chunk_ids: set[str] = {
            entry["chunk_id"] for entry in bm25_corpus if "chunk_id" in entry
        }

        kg_chunk_ids: set[str] = set()
        if kg_graph is not None:
            try:
                kg_chunk_ids.update(getattr(kg_graph, "processed_chunk_ids", set()))
                for entity in kg_graph.entities.values():
                    kg_chunk_ids.update(entity.chunk_ids)
                for rel in kg_graph.relationships:
                    if rel.chunk_id:
                        kg_chunk_ids.add(rel.chunk_id)
            except Exception as exc:
                logger.warning(f"Failed to inspect Knowledge Graph for store state: {exc}")

        logger.info(
            f"Store state loaded — Qdrant: {len(qdrant_chunk_ids)} chunks | "
            f"BM25: {len(bm25_chunk_ids)} chunks | "
            f"KG: {len(kg_chunk_ids)} chunk references"
        )
        return cls(qdrant_chunk_ids, bm25_chunk_ids, kg_chunk_ids)


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
    store_state: IngestionStoreState,
    kg_store: Any | None = None,
    store_lock: asyncio.Lock | None = None,
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
            stem_parts = file_path.stem.split("_")
            form_type = stem_parts[1] if len(stem_parts) >= 2 else "unknown"
            metadata = extract_metadata(
                doc.ticker,
                doc.date,
                doc.raw_text,
                form_type=form_type,
                file_name=file_path.name,
            )
            t1 = time.perf_counter()
            doc_timings["extract_metadata"] = round(t1 - t0, 4)

            t0 = time.perf_counter()
            chunks = create_parent_child_chunks(doc.ticker, doc.date, doc.sections)
            t1 = time.perf_counter()
            doc_timings["create_chunks"] = round(t1 - t0, 4)

            parent_count = sum(1 for c in chunks if c.chunk_type == "parent")
            child_count = sum(1 for c in chunks if c.chunk_type == "child")

            indexable_chunks = [c for c in chunks if c.chunk_type in ("child", "table")]
            parent_chunks = [c for c in chunks if c.chunk_type == "parent"]

            missing_qdrant = [
                c for c in indexable_chunks if c.chunk_id not in store_state.qdrant_chunk_ids
            ]
            missing_bm25 = [
                c for c in indexable_chunks if c.chunk_id not in store_state.bm25_chunk_ids
            ]
            unindexed_kg_chunks = [
                c for c in parent_chunks if c.chunk_id not in store_state.kg_chunk_ids
            ]
            kg_needed = kg_enabled and kg_graph is not None and bool(unindexed_kg_chunks)

            if not missing_qdrant and not missing_bm25 and not kg_needed:
                logger.info(
                    f"Skipped (already fully indexed across Qdrant, BM25, KG): {file_path.name}"
                )
                return 0, [], [], None

            logger.debug(
                f"[PROCESS] {file_path.name} | missing Qdrant: {len(missing_qdrant)}/{len(indexable_chunks)} | "
                f"missing BM25: {len(missing_bm25)}/{len(indexable_chunks)} | missing KG: {len(unindexed_kg_chunks)}/{len(parent_chunks)}"
            )

            indexer_timings: dict[str, float] = {}
            new_bm25_texts: list[list[str]] = []
            new_bm25_corpus: list[dict] = []

            if missing_qdrant:
                q_bm25_texts, q_bm25_corpus = await index_document(
                    missing_qdrant, metadata, qdrant, timings=indexer_timings
                )
                doc_timings["embedding"] = indexer_timings.get("embedding", 0.0)
                doc_timings["qdrant_upsert"] = indexer_timings.get("qdrant_upsert", 0.0)

                if store_lock:
                    async with store_lock:
                        for c in missing_qdrant:
                            store_state.qdrant_chunk_ids.add(c.chunk_id)
                else:
                    for c in missing_qdrant:
                        store_state.qdrant_chunk_ids.add(c.chunk_id)

                for text, corpus_entry in zip(q_bm25_texts, q_bm25_corpus, strict=False):
                    cid = corpus_entry.get("chunk_id")
                    if cid and cid not in store_state.bm25_chunk_ids:
                        new_bm25_texts.append(text)
                        new_bm25_corpus.append(corpus_entry)

            bm25_only_chunks = [c for c in missing_bm25 if c not in missing_qdrant]
            if bm25_only_chunks:
                from ingestion.indexer import _tokenize_for_bm25

                for c in bm25_only_chunks:
                    if c.chunk_id not in store_state.bm25_chunk_ids:
                        new_bm25_texts.append(_tokenize_for_bm25(c.text))
                        new_bm25_corpus.append(
                            {
                                "chunk_id": c.chunk_id,
                                "parent_id": c.parent_id or c.chunk_id,
                                "file_name": file_path.name,
                                "text": c.text,
                                "ticker": metadata.ticker,
                                "company": metadata.company,
                                "date": metadata.date,
                                "year": metadata.year,
                                "quarter": metadata.quarter,
                                "fiscal_period": metadata.fiscal_period,
                                "form_type": metadata.form_type,
                                "section_title": c.section_title or "Financial Table",
                                "chunk_type": c.chunk_type,
                                "is_table": (c.chunk_type == "table"),
                            }
                        )

            if store_lock:
                async with store_lock:
                    for corpus_entry in new_bm25_corpus:
                        cid = corpus_entry.get("chunk_id")
                        if cid:
                            store_state.bm25_chunk_ids.add(cid)
            else:
                for corpus_entry in new_bm25_corpus:
                    cid = corpus_entry.get("chunk_id")
                    if cid:
                        store_state.bm25_chunk_ids.add(cid)

            # ── Knowledge Graph extraction ─────────────────────────────────
            t_kg_start = time.perf_counter()
            if kg_needed:
                from knowledge_graph.extractor import extract_entities_from_chunks

                try:
                    await extract_entities_from_chunks(
                        unindexed_kg_chunks,
                        metadata.ticker,
                        metadata.fiscal_period,
                        kg_graph=kg_graph,
                        kg_store=kg_store,
                        store_state=store_state,
                        kg_lock=kg_lock,
                        store_lock=store_lock,
                    )
                except Exception as exc:
                    logger.warning(f"KG extraction failed for {file_path.name}: {exc}")
            t_kg_end = time.perf_counter()
            doc_timings["kg_extraction"] = round(t_kg_end - t_kg_start, 4)

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
                f"{file_path.name} | {metadata.fiscal_period} | {len(new_bm25_corpus)} new chunks | {t_total:.3f}s"
            )

            return child_count, new_bm25_texts, new_bm25_corpus, doc_metric

    except Exception as exc:
        logger.error(f"Error processing {file_path.name}: {exc}")
        return 0, [], [], None


async def _run_kg_only_async(threads_override: int | None = None) -> None:
    """Re-run KG extraction only on all indexed files.

    Use this to recover a blank knowledge graph without re-embedding documents.
    Parses and chunks each file then calls the LLM extractor and saves the result.
    """
    from knowledge_graph.entity_store import EntityStore
    from knowledge_graph.extractor import extract_entities_from_chunks

    setup_ingestion_logging()
    setup_embedder(threads=threads_override)
    qdrant = init_qdrant(_settings.infra.qdrant_url)
    _, bm25_corpus = _load_existing_bm25()

    kg_store = EntityStore()
    kg_graph = kg_store.load()
    store_state = IngestionStoreState.load(qdrant, bm25_corpus, kg_graph)

    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.htm"))
    target_files = []
    for f in transcript_files:
        doc = parse_html(f)
        if doc is None:
            continue
        chunks = create_parent_child_chunks(doc.ticker, doc.date, doc.sections)
        indexable_chunks = [c for c in chunks if c.chunk_type in ("child", "table")]
        if any(
            c.chunk_id in store_state.qdrant_chunk_ids or c.chunk_id in store_state.bm25_chunk_ids
            for c in indexable_chunks
        ):
            target_files.append(f)

    if not target_files:
        logger.warning("No indexed files found in store state — nothing to re-extract KG from.")
        return

    logger.info(f"KG-only mode: re-extracting from {len(target_files)} indexed file(s)")

    kg_lock = asyncio.Lock()

    for file_path in target_files:
        doc = parse_html(file_path)
        if doc is None:
            logger.debug(f"Skipped (not earnings content): {file_path.name}")
            continue
        metadata = extract_metadata(
            doc.ticker,
            doc.date,
            doc.raw_text,
            form_type=file_path.stem.split("_")[1]
            if len(file_path.stem.split("_")) >= 2
            else "unknown",
            file_name=file_path.name,
        )
        chunks = create_parent_child_chunks(doc.ticker, doc.date, doc.sections)
        parent_chunks = [c for c in chunks if c.chunk_type == "parent"]
        unindexed_kg_chunks = [
            c for c in parent_chunks if c.chunk_id not in store_state.kg_chunk_ids
        ]
        if not unindexed_kg_chunks:
            logger.info(f"[KG-only] Skipped (all KG chunks present): {file_path.name}")
            continue

        logger.info(
            f"[KG-only] {file_path.name} | {len(unindexed_kg_chunks)}/{len(parent_chunks)} missing parent chunks | "
            f"ticker={metadata.ticker} period={metadata.fiscal_period}"
        )
        try:
            await extract_entities_from_chunks(
                unindexed_kg_chunks,
                metadata.ticker,
                metadata.fiscal_period,
                kg_graph=kg_graph,
                kg_store=kg_store,
                store_state=store_state,
                kg_lock=kg_lock,
            )
        except Exception as exc:
            logger.warning(f"KG extraction failed for {file_path.name}: {exc}")

    kg_store.save(kg_graph)
    logger.info(f"KG-only extraction complete. {kg_graph.summary()}")


async def run_pipeline_async(
    fast: bool = False,
    concurrency_override: int | None = None,
    threads_override: int | None = None,
    kg_only: bool = False,
) -> None:
    """Run the ingestion indexing pipeline asynchronously using automatic store-state inspection."""
    if kg_only:
        await _run_kg_only_async(threads_override=threads_override)
        return
    pipeline_start_time = time.perf_counter()
    setup_ingestion_logging()
    setup_embedder(threads=threads_override)
    qdrant = init_qdrant(_settings.infra.qdrant_url)

    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.htm"))
    logger.info(f"Found {len(transcript_files)} .htm files in {TRANSCRIPTS_DIR}")

    # --- seed bm25 with previously indexed docs ---
    bm25_texts, bm25_corpus = _load_existing_bm25()

    # ── Knowledge Graph setup ──────────────────────────────────────────────
    kg_enabled = False if fast else _settings.knowledge_graph.extraction_enabled
    kg_store, kg_graph = None, None
    if kg_enabled:
        from knowledge_graph.entity_store import EntityStore

        kg_store = EntityStore()
        kg_graph = kg_store.load()

    store_state = IngestionStoreState.load(qdrant, bm25_corpus, kg_graph)

    indexed_count: int = 0
    skipped_count: int = 0

    concurrency = concurrency_override or _settings.embedding.max_concurrency
    batch_size = _settings.embedding.ingestion_batch_size
    semaphore = asyncio.Semaphore(concurrency)
    store_lock = asyncio.Lock()
    kg_lock = asyncio.Lock()
    metrics_lock = asyncio.Lock()

    existing_chunk_ids = {entry["chunk_id"] for entry in bm25_corpus if "chunk_id" in entry}
    metrics_records: list[dict] = _load_existing_metrics()

    if transcript_files:
        logger.info(
            f"Evaluating {len(transcript_files)} documents in batches of {batch_size} "
            f"with concurrency limit {concurrency} (llm_kg={'off' if fast else 'on'})..."
        )
        import gc

        from tqdm.asyncio import tqdm

        total_batches = (len(transcript_files) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            batch_files = transcript_files[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_tasks = [
                _process_document(
                    f,
                    qdrant,
                    semaphore,
                    kg_enabled,
                    kg_graph,
                    store_state,
                    kg_store,
                    store_lock,
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

            _save_ingestion_metrics(metrics_records, time.perf_counter() - pipeline_start_time)

    _save_bm25(bm25_texts, bm25_corpus)

    if kg_enabled and kg_store and kg_graph:
        kg_store.save(kg_graph)
        logger.info(f"Knowledge graph: {kg_graph.summary()}")

    pipeline_total_time = time.perf_counter() - pipeline_start_time
    _save_ingestion_metrics(metrics_records, pipeline_total_time)

    logger.info(f"Pipeline complete: {indexed_count} indexed this run, {skipped_count} skipped")


def run_pipeline(
    fast: bool = False,
    concurrency: int | None = None,
    threads: int | None = None,
    kg_only: bool = False,
) -> None:
    """Synchronous entry point to run the ingestion pipeline."""
    asyncio.run(
        run_pipeline_async(
            fast=fast,
            concurrency_override=concurrency,
            threads_override=threads,
            kg_only=kg_only,
        )
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
    parser.add_argument(
        "--kg-only",
        action="store_true",
        help=(
            "Re-run KG extraction only on already-indexed files without re-embedding. "
            "Use to recover a blank knowledge graph after a failed extraction."
        ),
    )
    args = parser.parse_args()
    run_pipeline(
        fast=args.fast,
        concurrency=args.concurrency,
        threads=args.threads,
        kg_only=args.kg_only,
    )
