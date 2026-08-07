"""
scripts/repair_fiscal_period.py
─────────────────────────────────────────────────────────────────────────────
Retroactive repair script for `fiscal_period`, `year`, `quarter`, and
`form_type` metadata fields in Qdrant + the BM25 corpus pickle.

Background
──────────
The original `metadata_extractor._detect_quarter()` used the filing *month*
to infer the fiscal quarter, which is systematically wrong:

  • A 10-K filed in January covers the *prior* fiscal year (FY 2024), not Q1.
  • A 10-Q filed in April covers Q1 of the current year, not Q2.

This script:
  1. Scans all .htm files in data/company_filings/ to build a ground-truth
     map: chunk_id prefix → correct (fiscal_period, year, quarter, form_type)
  2. Scrolls every Qdrant point, recomputes the correct metadata, and batches
     SetPayload calls to Qdrant.
  3. Rebuilds the BM25 corpus pickle with the corrected metadata fields.

Usage
──────
  # Dry-run (show what would change, make no writes):
  poetry run python scripts/repair_fiscal_period.py --dry-run

  # Live run (writes to Qdrant + BM25 corpus):
  poetry run python scripts/repair_fiscal_period.py

  # Limit to a specific ticker:
  poetry run python scripts/repair_fiscal_period.py --ticker NFLX
"""

from __future__ import annotations

import argparse
import pickle  # nosec B403
import re
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Project imports
from config import settings as _settings
from ingestion.metadata_extractor import _derive_fiscal_period

COLLECTION = _settings.embedding.collection_name
BM25_CORPUS_PATH = Path("data/bm25_corpus.pkl")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
FILINGS_DIR = Path("data/company_filings")

SCROLL_BATCH = 500  # points per Qdrant scroll page
UPSERT_BATCH = 200  # points per SetPayload batch


# ─── Step 1: Build ground-truth map from filenames ───────────────────────────


def _build_ground_truth() -> dict[str, dict]:
    """
    Scan all .htm filenames to derive the correct fiscal metadata.

    Filename pattern: {TICKER}_{FORM}_{DATE}_{CIK}.htm
    e.g. NFLX_10-K_2025-01-27_0001065280.htm

    Returns:
        mapping from "{TICKER}_{DATE}" key → corrected metadata dict
    """
    ground_truth: dict[str, dict] = {}

    for htm_file in sorted(FILINGS_DIR.glob("*.htm")):
        stem_parts = htm_file.stem.split("_")
        if len(stem_parts) < 3:
            logger.warning(f"Unexpected filename format, skipping: {htm_file.name}")
            continue

        ticker = stem_parts[0]
        form_type = stem_parts[1]  # "10-K" or "10-Q"

        date_match = re.search(r"\d{4}-\d{2}-\d{2}", htm_file.stem)
        if not date_match:
            logger.warning(f"No date found in filename, skipping: {htm_file.name}")
            continue
        filing_date = date_match.group(0)

        fiscal_period, fiscal_year, quarter = _derive_fiscal_period(form_type, filing_date)

        key = f"{ticker}_{filing_date}"
        ground_truth[key] = {
            "fiscal_period": fiscal_period,
            "year": fiscal_year,
            "quarter": quarter,
            "form_type": form_type,
        }
        logger.debug(
            f"  {htm_file.name} → {fiscal_period} | year={fiscal_year} | "
            f"quarter={quarter} | form={form_type}"
        )

    logger.info(f"Ground-truth map built: {len(ground_truth)} file entries.")
    return ground_truth


# ─── Step 2: Repair Qdrant ────────────────────────────────────────────────────


def _get_correct_meta(
    payload: dict,
    ground_truth: dict[str, dict],
) -> dict | None:
    """
    Return the corrected metadata fields for a Qdrant point, or None if already correct.
    """
    ticker = payload.get("ticker", "")
    date = payload.get("date", "")
    key = f"{ticker}_{date}"

    correct = ground_truth.get(key)
    if correct is None:
        return None  # No ground-truth file found for this point

    needs_update = (
        payload.get("fiscal_period") != correct["fiscal_period"]
        or payload.get("year") != correct["year"]
        or payload.get("quarter") != correct["quarter"]
        or payload.get("form_type") != correct.get("form_type")
    )
    return correct if needs_update else None


def repair_qdrant(
    client: QdrantClient,
    ground_truth: dict[str, dict],
    ticker_filter: str | None,
    dry_run: bool,
) -> tuple[int, int]:
    """
    Scroll all Qdrant points, find outdated metadata, and update in batches.

    Returns:
        (total_scanned, total_updated)
    """
    total_scanned = 0
    total_updated = 0
    offset = None

    pending_updates: list[tuple[str, dict]] = []  # (point_id, new_fields)

    logger.info(f"Scanning Qdrant collection '{COLLECTION}'...")

    while True:
        scroll_filter = None
        if ticker_filter:
            scroll_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="ticker",
                        match=qmodels.MatchValue(value=ticker_filter),
                    )
                ]
            )

        result, next_offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=scroll_filter,
            offset=offset,
            limit=SCROLL_BATCH,
            with_payload=True,
            with_vectors=False,
        )

        for point in result:
            total_scanned += 1
            if not point.payload:
                continue

            correct = _get_correct_meta(point.payload, ground_truth)
            if correct is not None:
                pending_updates.append((str(point.id), correct))

        # Flush batch
        if len(pending_updates) >= UPSERT_BATCH or (next_offset is None and pending_updates):
            total_updated += _flush_qdrant_updates(client, pending_updates, dry_run)
            pending_updates = []

        if next_offset is None:
            break
        offset = next_offset

    # Final flush
    if pending_updates:
        total_updated += _flush_qdrant_updates(client, pending_updates, dry_run)

    return total_scanned, total_updated


def _flush_qdrant_updates(
    client: QdrantClient,
    updates: list[tuple[str, dict]],
    dry_run: bool,
) -> int:
    if not updates:
        return 0

    # All updates in this batch share the same corrected metadata keyed by point id
    # We must update each point individually (different dates → different corrections)
    for point_id, new_fields in updates:
        if dry_run:
            logger.info(f"  [DRY-RUN] Would update point {point_id}: {new_fields}")
        else:
            client.set_payload(
                collection_name=COLLECTION,
                payload=new_fields,
                points=[point_id],
            )

    verb = "Would update" if dry_run else "Updated"
    logger.info(f"  {verb} {len(updates)} Qdrant points.")
    return len(updates)


# ─── Step 3: Repair BM25 corpus ──────────────────────────────────────────────


def repair_bm25(
    ground_truth: dict[str, dict],
    ticker_filter: str | None,
    dry_run: bool,
) -> tuple[int, int]:
    """
    Load the BM25 corpus pickle, patch fiscal metadata, and re-save.

    Returns:
        (total_entries, total_updated)
    """
    if not BM25_CORPUS_PATH.exists():
        logger.warning(f"BM25 corpus not found at {BM25_CORPUS_PATH} — skipping.")
        return 0, 0

    with open(BM25_CORPUS_PATH, "rb") as f:
        corpus: list[dict] = pickle.load(f)  # nosec B301

    total_entries = len(corpus)
    total_updated = 0

    for entry in corpus:
        ticker = entry.get("ticker", "")
        date = entry.get("date", "")

        if ticker_filter and ticker != ticker_filter:
            continue

        key = f"{ticker}_{date}"
        correct = ground_truth.get(key)
        if correct is None:
            continue

        needs_update = (
            entry.get("fiscal_period") != correct["fiscal_period"]
            or entry.get("year") != correct["year"]
            or entry.get("quarter") != correct["quarter"]
            or entry.get("form_type") != correct.get("form_type")
        )

        if needs_update:
            if dry_run:
                logger.info(
                    f"  [DRY-RUN] BM25 entry {entry.get('chunk_id', '?')}: "
                    f"{entry.get('fiscal_period')} → {correct['fiscal_period']}"
                )
            else:
                entry.update(correct)
            total_updated += 1

    if not dry_run and total_updated > 0:
        # Rebuild BM25 index from corrected corpus (tokenisation unchanged — just metadata)
        from rank_bm25 import BM25Okapi

        bm25_texts = [e.get("text", "").lower().split() for e in corpus]
        bm25 = BM25Okapi(bm25_texts)

        tmp_corpus = BM25_CORPUS_PATH.with_suffix(".pkl.tmp")
        tmp_index = BM25_INDEX_PATH.with_suffix(".pkl.tmp")

        with open(tmp_corpus, "wb") as f:
            pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)  # nosec B301
        with open(tmp_index, "wb") as f:
            pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)  # nosec B301

        tmp_corpus.replace(BM25_CORPUS_PATH)
        tmp_index.replace(BM25_INDEX_PATH)
        logger.info(f"BM25 corpus + index rebuilt: {total_updated} entries corrected.")

    return total_entries, total_updated


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair fiscal_period / year / quarter metadata in Qdrant and BM25 corpus."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing anything.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Limit repair to a specific ticker (e.g. NFLX).",
    )
    parser.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Skip Qdrant repair (only repair BM25 corpus).",
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Skip BM25 corpus repair (only repair Qdrant).",
    )
    args = parser.parse_args()

    dry_run = args.dry_run
    if dry_run:
        logger.info("DRY-RUN mode — no changes will be written.")

    # ── Build ground truth ────────────────────────────────────────────────────
    logger.info("=== Step 1: Building ground-truth metadata from filenames ===")
    ground_truth = _build_ground_truth()

    if not ground_truth:
        logger.error("No filing files found in data/company_filings/. Aborting.")
        return

    # Show summary of expected corrections
    logger.info("\nExpected fiscal period corrections:")
    for key, meta in sorted(ground_truth.items()):
        logger.info(
            f"  {key:<35} → {meta['fiscal_period']:<12} "
            f"year={meta['year']}  quarter={meta['quarter']}  form={meta['form_type']}"
        )

    # ── Repair Qdrant ─────────────────────────────────────────────────────────
    if not args.skip_qdrant:
        logger.info("\n=== Step 2: Repairing Qdrant metadata ===")
        client = QdrantClient(url=_settings.infra.qdrant_url, timeout=60, check_compatibility=False)
        scanned, updated = repair_qdrant(client, ground_truth, args.ticker, dry_run)
        logger.info(
            f"Qdrant repair complete: {scanned} points scanned, "
            f"{updated} {'would be ' if dry_run else ''}updated."
        )

    # ── Repair BM25 ───────────────────────────────────────────────────────────
    if not args.skip_bm25:
        logger.info("\n=== Step 3: Repairing BM25 corpus ===")
        total, updated = repair_bm25(ground_truth, args.ticker, dry_run)
        logger.info(
            f"BM25 repair complete: {total} entries, "
            f"{updated} {'would be ' if dry_run else ''}updated."
        )

    logger.info("\n=== Repair complete ===")
    if dry_run:
        logger.info("Re-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
