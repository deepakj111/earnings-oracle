import os
import pickle
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("RAG_QDRANT_COLLECTION", "earnings_transcripts")
TRANSCRIPTS_DIR = Path("data/transcripts")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
BM25_CORPUS_PATH = Path("data/bm25_corpus.pkl")
CHECKPOINT_PATH = Path("data/pipeline_checkpoint.txt")

TICKERS = ["AAPL", "NVDA", "MSFT", "AMZN", "META", "JPM", "XOM", "UNH", "TSLA", "WMT"]

SEP = "-" * 68


def fmt(n: int | float) -> str:
    """Format an integer with comma thousands separators."""
    return f"{n:,}"


def section(title: str) -> None:
    """Print a prominent section header for the report."""
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ─── 1. FILESYSTEM ────────────────────────────────────────────────────────────
section("1. FILESYSTEM")

htm_files = sorted(TRANSCRIPTS_DIR.glob("*.htm")) if TRANSCRIPTS_DIR.exists() else []
print(f"  Transcripts dir  : {TRANSCRIPTS_DIR}")
print(f"  .htm files found : {fmt(len(htm_files))}")

if htm_files:
    ticker_file_counts = Counter(f.stem.split("_")[0] for f in htm_files)
    print("\n  Files per ticker:")
    for ticker in TICKERS:
        count = ticker_file_counts.get(ticker, 0)
        bar = "█" * count
        print(f"    {ticker:<6} {count:>3}  {bar}")
    other = {k: v for k, v in ticker_file_counts.items() if k not in TICKERS}
    if other:
        print(f"    Other: {dict(other)}")

for path in [BM25_INDEX_PATH, BM25_CORPUS_PATH, CHECKPOINT_PATH]:
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"\n  {path.name:<30} {size_mb:.2f} MB")
    else:
        print(f"\n  {path.name:<30} NOT FOUND")

if CHECKPOINT_PATH.exists():
    lines = [line.strip() for line in CHECKPOINT_PATH.read_text().splitlines() if line.strip()]
    print(f"\n  Checkpoint entries (already indexed): {fmt(len(lines))}")


# ─── 2. BM25 INDEX ────────────────────────────────────────────────────────────
section("2. BM25 INDEX")

if not BM25_CORPUS_PATH.exists():
    print("  bm25_corpus.pkl not found — run ingestion first.")
else:
    with open(BM25_CORPUS_PATH, "rb") as f:
        corpus = pickle.load(f)  # noqa: S301

    print(f"  Total chunks in BM25 corpus : {fmt(len(corpus))}")

    ticker_counts = Counter(e.get("ticker", "UNKNOWN") for e in corpus)
    quarter_counts = Counter(e.get("fiscal_period", "UNKNOWN") for e in corpus)
    year_counts = Counter(str(e.get("year", "?")) for e in corpus)
    section_counts = Counter(e.get("section_title", "") for e in corpus)

    print("\n  Chunks per ticker:")
    for ticker in TICKERS:
        count = ticker_counts.get(ticker, 0)
        pct = count / len(corpus) * 100 if corpus else 0
        bar = "█" * (count // 50)
        print(f"    {ticker:<6} {fmt(count):>7}  ({pct:4.1f}%)  {bar}")

    print("\n  Chunks per year:")
    for year, count in sorted(year_counts.items()):
        print(f"    {year}  {fmt(count):>7}")

    print("\n  Top 10 fiscal periods:")
    for period, count in quarter_counts.most_common(10):
        print(f"    {period:<20} {fmt(count):>7}")

    print("\n  Top 10 section titles:")
    for section_title, count in section_counts.most_common(10):
        label = section_title[:50] if section_title else "(empty)"
        print(f"    {label:<52} {fmt(count):>6}")

    # token length stats
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    lengths = [len(enc.encode(e.get("text", ""), disallowed_special=())) for e in corpus]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    print(f"\n  Avg chunk tokens : {avg_len:.1f}")
    print(f"  Min chunk tokens : {min(lengths)}")
    print(f"  Max chunk tokens : {max(lengths)}")

    if BM25_INDEX_PATH.exists():
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25 = pickle.load(f)  # noqa: S301
        print(f"\n  BM25 vocab size  : {fmt(len(bm25.idf))}")
        print(f"  BM25 avgdl       : {bm25.avgdl:.2f} tokens")


# ─── 3. QDRANT VECTOR DATABASE ────────────────────────────────────────────────
section("3. QDRANT VECTOR DATABASE")
total_points: int = 0

try:
    client = QdrantClient(url=QDRANT_URL)
    server_info = client.get_collections()
    all_collections = [c.name for c in server_info.collections]
    print(f"  Qdrant URL         : {QDRANT_URL}")
    print(f"  Collections found  : {all_collections}")

    if COLLECTION not in all_collections:
        print(f"\n  Collection '{COLLECTION}' does NOT exist.")
        print("  Run: poetry run python -m ingestion.pipeline")
        sys.exit(1)

    info = client.get_collection(COLLECTION)
    total_points = info.points_count or 0

    vectors_config = info.config.params.vectors
    if isinstance(vectors_config, dict):
        first_vector = list(vectors_config.values())[0]
        vector_dim = first_vector.size
        distance = first_vector.distance.name
    else:
        vector_dim = vectors_config.size
        distance = vectors_config.distance.name

    indexed_vectors = info.indexed_vectors_count or 0

    print(f"\n  Collection         : {COLLECTION}")
    print(f"  Total points       : {fmt(total_points)}")
    print(f"  Indexed vectors    : {fmt(indexed_vectors)}")
    print(f"  Vector dimensions  : {vector_dim}")
    print(f"  Distance metric    : {distance}")
    print(f"  Segments           : {info.segments_count}")
    print(f"  Disk usage (bytes) : {fmt(getattr(info, 'disk_data_size', 0))}")
    print(f"  RAM  usage (bytes) : {fmt(getattr(info, 'ram_data_size', 0))}")

    if total_points == 0:
        print("\n  Collection is EMPTY — ingestion did not upsert any vectors.")
        print("  Delete checkpoint and re-run: rm data/pipeline_checkpoint.txt")
        print("  Then: poetry run python -m ingestion.pipeline")
        sys.exit(0)

    # Per-ticker breakdown via Qdrant count()
    print("\n  Points per ticker (Qdrant):")
    for ticker in TICKERS:
        count_result = client.count(
            collection_name=COLLECTION,
            count_filter=Filter(
                must=[FieldCondition(key="ticker", match=MatchValue(value=ticker))]
            ),
            exact=True,
        )
        count = count_result.count
        pct = count / total_points * 100 if total_points else 0
        bar = "█" * (count // 100)
        print(f"    {ticker:<6} {fmt(count):>7}  ({pct:4.1f}%)  {bar}")

    # Per-quarter breakdown
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    print("\n  Points per quarter (Qdrant):")
    for q in quarters:
        count_result = client.count(
            collection_name=COLLECTION,
            count_filter=Filter(must=[FieldCondition(key="quarter", match=MatchValue(value=q))]),
            exact=True,
        )
        print(f"    {q}  {fmt(count_result.count):>7}")

    # Sample 3 random points to show payload shape
    print("\n  Sample point payloads (3 random):")
    sample, _ = client.scroll(
        collection_name=COLLECTION,
        limit=3,
        with_payload=True,
        with_vectors=False,
    )
    for i, pt in enumerate(sample, 1):
        p = pt.payload or {}
        text_preview = (p.get("text") or "")[:80].replace("\n", " ")
        print(f"\n    [{i}] chunk_id    : {p.get('chunk_id', '?')}")
        print(f"        ticker      : {p.get('ticker')}  |  company    : {p.get('company')}")
        print(f"        date        : {p.get('date')}  |  year       : {p.get('year')}")
        print(
            f"        quarter     : {p.get('quarter')}  |  fiscal_period: {p.get('fiscal_period')}"
        )
        print(f"        section     : {p.get('section_title')}")
        print(f"        text preview: {text_preview}...")

except Exception as exc:
    print(f"  Qdrant connection failed: {exc}")
    print("  Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")


# ─── 4. CONSISTENCY CHECK ─────────────────────────────────────────────────────
section("4. CONSISTENCY CHECK")

# Initialize with safe defaults before checking
bm25_count: int = 0
qdrant_count: int = 0
checkpoint_count: int = 0

if BM25_CORPUS_PATH.exists():
    bm25_count = len(corpus)  # corpus is already loaded in section 2

if CHECKPOINT_PATH.exists():
    checkpoint_count = len(lines)  # lines already loaded in section 1

# total_points is set inside the Qdrant try/except block above
# Use a module-level sentinel instead of dir() hack:
try:
    qdrant_count = total_points  # type: ignore[possibly-undefined]
except NameError:
    qdrant_count = 0

print(f"  .htm files on disk   : {fmt(len(htm_files))}")
print(f"  Checkpoint entries   : {fmt(checkpoint_count)}")
print(f"  BM25 corpus chunks   : {fmt(bm25_count)}")
print(f"  Qdrant vector points : {fmt(qdrant_count)}")

if qdrant_count == 0 and bm25_count > 0:
    print("\n  MISMATCH: BM25 has chunks but Qdrant is empty.")
    print("  This means ingestion wrote BM25 but did NOT upsert to Qdrant,")
    print("  OR Qdrant was recreated with empty storage after ingestion.")
    print("  Fix: rm data/pipeline_checkpoint.txt && poetry run python -m ingestion.pipeline")
elif qdrant_count > 0 and bm25_count > 0:
    ratio = qdrant_count / bm25_count
    if 0.95 <= ratio <= 1.05:
        print("\n  OK: BM25 and Qdrant counts are consistent.")
    else:
        print(
            f"\n  WARNING: BM25 has {fmt(bm25_count)} chunks but Qdrant has {fmt(qdrant_count)} points."
        )
        print("  They should be equal. Re-run ingestion with a fresh checkpoint.")
else:
    print("\n  Both indexes appear empty — run ingestion pipeline first.")

print(f"\n{SEP}\n")
