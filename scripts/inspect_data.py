#!/usr/bin/env python3
"""
scripts/inspect_data.py
=======================
Comprehensive inspection of all indexed data stores:
  - Qdrant vector DB (what files, tickers, years, chunk counts)
  - BM25 keyword index (corpus size, files represented)
  - Knowledge Graph (entities, relationships, tickers, years)
  - Cross-store consistency check (mismatches flagged clearly)

Usage:
    poetry run python scripts/inspect_data.py
    poetry run python scripts/inspect_data.py --verbose
    poetry run python scripts/inspect_data.py --json      # output as JSON
"""

from __future__ import annotations

import argparse
import json
import os
import pickle  # nosec B403
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── ensure repo root is on sys.path ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # resolve relative paths used by settings

from config import settings  # noqa: E402  (after sys.path setup)

# ── Paths ─────────────────────────────────────────────────────────────────────
BM25_INDEX_PATH = ROOT / "data" / "bm25_index.pkl"
BM25_CORPUS_PATH = ROOT / "data" / "bm25_corpus.pkl"
CHECKPOINT_PATH = ROOT / "data" / "ingested_filings_checkpoint.txt"
KG_PATH = ROOT / "data" / "knowledge_graph.json"
FILINGS_DIR = ROOT / "data" / "company_filings"

QDRANT_URL = settings.infra.qdrant_url
COLLECTION = settings.embedding.collection_name

SEP = "─" * 68
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"


def header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{SEP}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{SEP}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def err(msg: str) -> None:
    print(f"  {RED}✗{RESET}  {msg}")


def info(msg: str) -> None:
    print(f"     {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Source files
# ─────────────────────────────────────────────────────────────────────────────


def inspect_source_files() -> dict:
    header("Source Files  (data/company_filings/)")
    files = sorted(FILINGS_DIR.glob("*.htm"))
    result = {"count": len(files), "files": []}
    if not files:
        warn("No .htm files found — run ingestion.download_filings first.")
        return result
    for f in files:
        size_kb = f.stat().st_size // 1024
        info(f"{f.name}  ({size_kb:,} KB)")
        result["files"].append({"name": f.name, "size_kb": size_kb})
    ok(f"{len(files)} source filing(s) on disk")

    # Checkpoint
    if CHECKPOINT_PATH.exists():
        checkpointed = {
            line.strip() for line in CHECKPOINT_PATH.read_text().splitlines() if line.strip()
        }

        missing = {f.name for f in files} - checkpointed
        extra = checkpointed - {f.name for f in files}
        if not missing and not extra:
            ok(f"Checkpoint matches source files ({len(checkpointed)} entries)")
        else:
            if missing:
                warn(f"NOT in checkpoint (not yet ingested): {sorted(missing)}")
            if extra:
                warn(f"In checkpoint but file MISSING on disk: {sorted(extra)}")
        result["checkpoint"] = {
            "entries": sorted(checkpointed),
            "missing": sorted(missing),
            "extra": sorted(extra),
        }
    else:
        warn("No checkpoint file found — run ingestion.pipeline first.")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Qdrant
# ─────────────────────────────────────────────────────────────────────────────


def inspect_qdrant(verbose: bool = False) -> dict:
    header(f"Qdrant Vector DB  ({QDRANT_URL}  /  collection: {COLLECTION})")
    result: dict = {"url": QDRANT_URL, "collection": COLLECTION}

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(QDRANT_URL, timeout=10, check_compatibility=False)
    except Exception as e:
        err(f"Cannot connect to Qdrant: {e}")
        result["error"] = str(e)
        return result

    try:
        col_info = client.get_collection(COLLECTION)
        total = col_info.points_count
        indexed = col_info.indexed_vectors_count
        vector_dim = col_info.config.params.vectors.size  # type: ignore[union-attr]
        vector_size_kb = (vector_dim * 4) / 1024  # 4 bytes per float32
        ok(f"Collection exists | {total:,} points")
        ok(
            f"Vector Embeddings: {vector_dim} dimensions ({vector_size_kb:.2f} KB float32 per vector)"
        )
        if indexed == 0 and total > 0:
            ok(
                f"Vector Search Mode: Exact flat search active ({total:,} points stored, under HNSW 20k threshold)"
            )
        elif indexed < total:
            warn(f"HNSW indexing in progress ({indexed:,}/{total:,} vectors indexed)")
        else:
            ok(f"HNSW Graph Index: 100% indexed ({indexed:,}/{total:,} vectors)")
        result.update(
            {
                "total_points": total,
                "indexed_vectors": indexed,
                "vector_dim": vector_dim,
                "vector_bytes_per_point": vector_dim * 4,
            }
        )
    except Exception as e:
        err(f"Collection not found or error: {e}")
        result["error"] = str(e)
        return result

    # Scroll all points metadata
    offset = None
    all_pts: list = []
    try:
        while True:
            batch, offset = client.scroll(
                COLLECTION,
                limit=250,
                offset=offset,
                with_payload=["ticker", "year", "date", "fiscal_period", "chunk_id", "parent_id"],
            )
            all_pts.extend(batch)
            if offset is None:
                break
    except Exception as e:
        warn(f"Could not scroll collection: {e}")
        return result

    by_filing: dict[str, list] = defaultdict(list)
    for pt in all_pts:
        p = pt.payload or {}
        key = f"{p.get('ticker', '?')} | {p.get('date', '?')} | {p.get('fiscal_period', '?')}"
        by_filing[key].append(pt.id)

    tickers = Counter(pt.payload.get("ticker", "?") for pt in all_pts if pt.payload)
    years = Counter(pt.payload.get("year", "?") for pt in all_pts if pt.payload)
    dates = Counter(pt.payload.get("date", "?") for pt in all_pts if pt.payload)

    print(f"\n  {'Filing (ticker | date | period)':<45} {'chunks':>8}")
    print(f"  {'─' * 45} {'─' * 8}")
    filing_summary = []
    for key in sorted(by_filing):
        cnt = len(by_filing[key])
        print(f"  {key:<45} {cnt:>8,}")
        filing_summary.append({"key": key, "chunks": cnt})
    print()
    ok(f"Tickers: {dict(sorted(tickers.items()))}")
    ok(f"Years:   {dict(sorted(years.items()))}")

    if verbose:
        ok(f"Dates:   {dict(sorted(dates.items()))}")

    # Parent/child breakdown
    parents = sum(1 for pt in all_pts if pt.payload and not pt.payload.get("parent_id"))
    children = sum(1 for pt in all_pts if pt.payload and pt.payload.get("parent_id"))
    ok(
        f"Chunk types: {parents:,} parent-level | {children:,} child-level (child chunks carry parent_id links)"
    )

    result.update(
        {
            "filings": filing_summary,
            "tickers": dict(tickers),
            "years": dict(years),
            "dates": dict(dates),
            "parent_chunks": parents,
            "child_chunks": children,
            "chunk_ids": {pt.id: pt.payload.get("chunk_id", "") for pt in all_pts if pt.payload},
        }
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. BM25
# ─────────────────────────────────────────────────────────────────────────────


def inspect_bm25(verbose: bool = False) -> dict:
    header("BM25 Keyword Index  (data/bm25_index.pkl + bm25_corpus.pkl)")
    result: dict = {}

    if not BM25_CORPUS_PATH.exists():
        err(f"BM25 corpus not found at {BM25_CORPUS_PATH}")
        result["error"] = "corpus missing"
        return result
    if not BM25_INDEX_PATH.exists():
        err(f"BM25 index not found at {BM25_INDEX_PATH}")
        result["error"] = "index missing"
        return result

    try:
        with open(BM25_CORPUS_PATH, "rb") as f:
            corpus: list[dict] = pickle.load(f)  # nosec B301
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25 = pickle.load(f)  # nosec B301
    except Exception as e:
        err(f"Failed to load BM25 files: {e}")
        result["error"] = str(e)
        return result

    ok(f"BM25 corpus: {len(corpus):,} entries  |  index vocab size: {len(bm25.idf):,} terms")

    tickers = Counter(e.get("ticker", "?") for e in corpus)
    years = Counter(e.get("year", "?") for e in corpus)
    dates = Counter(e.get("date", "?") for e in corpus)

    by_filing: dict[str, int] = Counter(
        f"{e.get('ticker', '?')} | {e.get('date', '?')}" for e in corpus
    )

    print(f"\n  {'Filing (ticker | date)':<35} {'chunks':>8}")
    print(f"  {'─' * 35} {'─' * 8}")
    for key in sorted(by_filing):
        print(f"  {key:<35} {by_filing[key]:>8,}")
    print()
    ok(f"Tickers: {dict(sorted(tickers.items()))}")
    ok(f"Years:   {dict(sorted(years.items()))}")
    if verbose:
        ok(f"Dates:   {dict(sorted(dates.items()))}")

    size_index = BM25_INDEX_PATH.stat().st_size // 1024
    size_corpus = BM25_CORPUS_PATH.stat().st_size // 1024
    ok(f"File sizes: index={size_index} KB  |  corpus={size_corpus} KB")

    result.update(
        {
            "total_entries": len(corpus),
            "vocab_size": len(bm25.idf),
            "filings": dict(by_filing),
            "tickers": dict(tickers),
            "years": dict(years),
            "dates": dict(dates),
            "chunk_ids": [e.get("chunk_id", "") for e in corpus],
        }
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Knowledge Graph
# ─────────────────────────────────────────────────────────────────────────────


def inspect_knowledge_graph() -> dict:
    header("Knowledge Graph  (data/knowledge_graph.json)")
    result: dict = {}

    if not KG_PATH.exists():
        warn("knowledge_graph.json not found — KG extraction may not have run.")
        result["error"] = "file missing"
        return result

    try:
        with open(KG_PATH, encoding="utf-8") as f:
            kg_data = json.load(f)
    except Exception as e:
        err(f"Failed to load knowledge_graph.json: {e}")
        result["error"] = str(e)
        return result

    raw_entities = kg_data.get("entities", [])
    relationships = kg_data.get("relationships", [])

    if isinstance(raw_entities, dict):
        entity_list = list(raw_entities.values())
    else:
        entity_list = list(raw_entities) if isinstance(raw_entities, list) else []

    ok(f"Entities: {len(entity_list):,}  |  Relationships: {len(relationships):,}")

    # Entity breakdown by type
    entity_types: Counter = Counter()
    entity_tickers: Counter = Counter()
    entity_years: Counter = Counter()

    for edata in entity_list:
        if not isinstance(edata, dict):
            continue
        entity_types[edata.get("entity_type", edata.get("type", "unknown"))] += 1
        tickers = edata.get("tickers", [edata.get("ticker")] if edata.get("ticker") else [])
        for ticker in tickers:
            if ticker:
                entity_tickers[ticker] += 1
        years = edata.get("years", [edata.get("year")] if edata.get("year") else [])
        for year in years:
            if year:
                entity_years[year] += 1

    if entity_types:
        print("\n  Entity types:")
        for etype, cnt in sorted(entity_types.items(), key=lambda x: -x[1]):
            print(f"    {etype:<25} {cnt:>6,}")

    if entity_tickers:
        ok(f"Tickers referenced: {dict(sorted(entity_tickers.items()))}")
    if entity_years:
        ok(f"Years referenced:   {dict(sorted(entity_years.items()))}")

    # Relationship types
    rel_types: Counter = Counter(
        r.get("relationship_type", r.get("relation", "?"))
        for r in relationships
        if isinstance(r, dict)
    )
    if rel_types:
        print("\n  Top relationship types:")
        for rtype, cnt in rel_types.most_common(10):
            print(f"    {rtype:<30} {cnt:>5,}")

    result.update(
        {
            "entity_count": len(entity_list),
            "relationship_count": len(relationships),
            "entity_types": dict(entity_types),
            "tickers": dict(entity_tickers),
            "years": dict(entity_years),
            "rel_types": dict(rel_types),
        }
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cross-store consistency check
# ─────────────────────────────────────────────────────────────────────────────


def cross_check(qdrant_result: dict, bm25_result: dict, source_result: dict) -> None:
    header("Cross-Store Consistency Check")

    issues_found = False

    # ── Qdrant vs BM25 chunk counts ───────────────────────────────────────────
    q_total = qdrant_result.get("total_points", 0)
    b_total = bm25_result.get("total_entries", 0)

    if q_total == 0 and b_total == 0:
        warn("Both Qdrant and BM25 are empty — run ingestion.pipeline")
        return

    if q_total != b_total:
        warn(f"Chunk count MISMATCH: Qdrant has {q_total:,} points, BM25 has {b_total:,} entries")
        info(
            "Expected: BM25 corpus covers CHILD chunks only; "
            "Qdrant stores both parent AND child chunks."
        )
        info(
            f"  Qdrant children: {qdrant_result.get('child_chunks', '?'):,} "
            f"  BM25 entries: {b_total:,}"
        )
        q_children = qdrant_result.get("child_chunks", 0)
        if q_children == b_total:
            ok(f"Qdrant child count ({q_children:,}) == BM25 corpus ({b_total:,})  ✓")
        else:
            err(
                f"Qdrant child chunks ({q_children:,}) ≠ BM25 entries ({b_total:,}) — "
                "re-run ingestion.pipeline to rebuild both"
            )
            issues_found = True
    else:
        ok(f"Qdrant total ({q_total:,}) == BM25 total ({b_total:,})")

    # ── Ticker consistency ────────────────────────────────────────────────────
    q_tickers = set(qdrant_result.get("tickers", {}).keys())
    b_tickers = set(bm25_result.get("tickers", {}).keys())
    if q_tickers == b_tickers:
        ok(f"Tickers match: {sorted(q_tickers)}")
    else:
        err(f"Ticker MISMATCH — Qdrant: {sorted(q_tickers)}, BM25: {sorted(b_tickers)}")
        issues_found = True

    # ── Year consistency ──────────────────────────────────────────────────────
    q_years = set(qdrant_result.get("years", {}).keys())
    b_years = set(bm25_result.get("years", {}).keys())
    if q_years == b_years:
        ok(f"Years match: {sorted(q_years)}")
    else:
        err(f"Year MISMATCH — Qdrant: {sorted(q_years)}, BM25: {sorted(b_years)}")
        issues_found = True

    # ── Source files vs checkpoint ────────────────────────────────────────────
    src_files = {f["name"] for f in source_result.get("files", [])}
    checkpoint = set(source_result.get("checkpoint", {}).get("entries", []))
    if checkpoint and src_files:
        if src_files == checkpoint:
            ok(f"Source files == Checkpoint ({len(src_files)} files)")
        else:
            missing_in_ckpt = src_files - checkpoint
            extra_in_ckpt = checkpoint - src_files
            if missing_in_ckpt:
                err(f"Files NOT ingested yet: {sorted(missing_in_ckpt)}")
                issues_found = True
            if extra_in_ckpt:
                warn(f"Checkpoint references DELETED files: {sorted(extra_in_ckpt)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    if not issues_found:
        ok("All stores are consistent — no mismatches detected.")
    else:
        err("Issues found above — run `poetry run python -m ingestion.pipeline` to fix.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect RAG data stores (Qdrant, BM25, Knowledge Graph)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show extra detail")
    parser.add_argument("--json", action="store_true", help="Output full report as JSON")
    args = parser.parse_args()

    source = inspect_source_files()
    qdrant = inspect_qdrant(verbose=args.verbose)
    bm25 = inspect_bm25(verbose=args.verbose)
    kg = inspect_knowledge_graph()
    cross_check(qdrant, bm25, source)

    if args.json:
        report = {
            "source_files": source,
            "qdrant": {k: v for k, v in qdrant.items() if k != "chunk_ids"},
            "bm25": {k: v for k, v in bm25.items() if k != "chunk_ids"},
            "knowledge_graph": kg,
        }
        print("\n" + json.dumps(report, indent=2, default=str))

    print()


if __name__ == "__main__":
    main()
