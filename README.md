# 📊 Financial RAG System

A production-grade Retrieval-Augmented Generation (RAG) system for querying financial earnings releases and 8-K filings from the SEC EDGAR database.

Built with a hybrid retrieval approach — dense vector search (fastembed + Qdrant) combined with sparse keyword search (BM25) — to answer precise financial questions over structured earnings documents.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                                 │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │  INGESTION   │───▶│  RETRIEVAL   │───▶│     GENERATION       │   │
│  │              │    │              │    │                      │   │
│  │ SEC EDGAR    │    │ BM25 (sparse)│    │  LLM Answer          │   │
│  │ 8-K Filings  │    │   +          │    │  + Source Citations  │   │
│  │     ↓        │    │ Qdrant dense │    │                      │   │
│  │ Parse HTML   │    │   =          │    └──────────────────────┘   │
│  │     ↓        │    │ Hybrid RRF   │                               │
│  │ Parent/Child │    └──────────────┘                               │
│  │  Chunking    │                                                    │
│  │     ↓        │    ┌──────────────┐    ┌──────────────────────┐   │
│  │ fastembed    │    │  EVALUATION  │    │      CRAG            │   │
│  │ BAAI/bge     │    │  (Ragas)     │    │  Corrective RAG      │   │
│  │     ↓        │    └──────────────┘    └──────────────────────┘   │
│  │ Qdrant +     │                                                    │
│  │ BM25 Index   │                                                    │
│  └──────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Ingestion** | `requests`, `beautifulsoup4` | SEC EDGAR HTTP scraping |
| **Parsing** | `lxml`, `beautifulsoup4` | HTML → clean text |
| **Chunking** | `tiktoken` | Token-aware parent/child splitting |
| **Embedding** | `fastembed` + `BAAI/bge-large-en-v1.5` | 1024-dim ONNX embeddings (local, free) |
| **Vector DB** | `qdrant-client` | Dense vector storage + ANN search |
| **Keyword Search** | `rank-bm25` | Sparse BM25 index |
| **Logging** | `loguru` | Structured pipeline logging |
| **Config** | `python-dotenv` | Environment variable management |
| **Linting** | `ruff` | Linting + import sorting (replaces flake8 + isort) |
| **Type Checking** | `mypy` | Static type checking |
| **Security** | `bandit` | Security vulnerability scanning |
| **Testing** | `pytest` | 151 tests across all ingestion modules |

---

## Project Structure

```
rag-project/
│
├── ingestion/                  # ✅ Complete — SEC data pipeline
│   ├── __init__.py
│   ├── download_filings.py     # SEC EDGAR 8-K scraper
│   ├── parser.py               # HTML → ParsedDocument
│   ├── chunker.py              # Parent/child chunk architecture
│   ├── metadata_extractor.py   # Ticker, date, quarter detection
│   ├── indexer.py              # fastembed + Qdrant + BM25 indexing
│   └── pipeline.py             # End-to-end orchestrator
│
├── retrieval/                  # 🔜 Hybrid BM25 + vector search
├── generation/                 # 🔜 LLM answer synthesis
├── query/                      # 🔜 Query preprocessing + routing
├── crag/                       # 🔜 Corrective RAG loop
├── api/                        # 🔜 FastAPI endpoints
├── ui/                         # 🔜 Streamlit / Gradio interface
├── evaluation/                 # 🔜 Ragas evaluation harness
│
├── tests/                      # 151 tests, 100% ingestion coverage
│   ├── test_chunker.py
│   ├── test_download_filings.py
│   ├── test_indexer.py
│   ├── test_metadata_extractor.py
│   ├── test_parser.py
│   └── test_pipeline.py
│
├── data/                       # Runtime data (gitignored)
│   ├── transcripts/            # Downloaded .htm filings
│   ├── bm25_index.pkl          # Serialized BM25 index
│   └── pipeline_checkpoint.txt # Incremental ingestion progress
│
├── .pre-commit-config.yaml     # 4-stage code quality gates
├── pyproject.toml              # Poetry deps + ruff/mypy/bandit config
└── README.md
```

---

## Supported Companies

| Ticker | Company |
|--------|---------|
| AAPL | Apple |
| NVDA | NVIDIA |
| MSFT | Microsoft |
| AMZN | Amazon |
| META | Meta Platforms |
| JPM | JPMorgan Chase |
| XOM | ExxonMobil |
| UNH | UnitedHealth Group |
| TSLA | Tesla |
| WMT | Walmart |

---

## Setup

### Prerequisites

- Python 3.11 or 3.12
- [Poetry](https://python-poetry.org/docs/#installation)
- [Qdrant](https://qdrant.tech/documentation/quick-start/) running locally or via Docker

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/rag-project.git
cd rag-project
poetry install
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# SEC EDGAR — required for downloading filings
# Format: "Your Name your@email.com" (SEC fair-use policy)
SEC_USER_AGENT="Your Name your@email.com"

# Qdrant vector database URL
QDRANT_URL=http://localhost:6333
```

### 3. Start Qdrant

```bash
# Using Docker (recommended)
docker run -p 6333:6333 qdrant/qdrant

# Or use Qdrant Cloud — set QDRANT_URL to your cluster URL
```

### 4. Install pre-commit hooks

```bash
poetry run pre-commit install
```

---

## Usage

### Step 1 — Download 8-K filings from SEC EDGAR

```bash
poetry run python -m ingestion.download_filings
```

Downloads earnings press releases (EX-99.1 exhibits) for all 10 companies from 2023-01-01 to today. Files land in `data/transcripts/` as `{TICKER}_{DATE}_{ACCESSION}.htm`.

### Step 2 — Run the ingestion pipeline

```bash
poetry run python -m ingestion.pipeline
```

This will:
1. Load the `BAAI/bge-large-en-v1.5` embedding model (~340MB, cached after first run)
2. Parse each `.htm` file into clean text sections
3. Split into parent chunks (~512 tokens) and child chunks (~128 tokens)
4. Embed all child chunks and upsert into Qdrant
5. Build a BM25 index and save to `data/bm25_index.pkl`
6. Write a checkpoint file — re-running skips already-indexed files

**Incremental re-runs are safe** — the checkpoint file tracks completed files.

---

## Chunking Architecture

The chunker uses a three-stage parent/child approach designed for financial documents:

```
Raw HTML
   │
   ▼
Stage 1: Structure-aware section splitting
   │  - Financial section headers as hard boundaries
   │    ("Revenue", "Segment Results", "Outlook", etc.)
   │  - Markdown tables detected and kept atomic (never split)
   │
   ▼
Stage 2: Parent chunks (~512 tokens)
   │  - 64-token overlap between consecutive parents
   │  - Each parent carries a contextual prefix:
   │    [Context: AAPL | earnings_release | 2024-10-31 | Section: Revenue]
   │
   ▼
Stage 3: Child chunks (~128 tokens)
   │  - Sentence boundaries respected (no mid-sentence splits)
   │  - 32-token overlap between consecutive children
   │  - Contextual prefix re-applied to every child
   │
   ▼
Qdrant (child embeddings) + BM25 (child token lists)
```

**Why parent/child?**
- Children are small enough for precise dense retrieval (128 tokens ≈ 1-2 sentences)
- When a child chunk is retrieved, the full parent is fetched for context before generation
- This gives the LLM enough surrounding context without bloating the embedding window

---

## Running Tests

```bash
# Full test suite with timing
poetry run pytest tests/ -v --durations=10

# Quiet mode (just pass/fail counts)
poetry run pytest tests/ -q
```

Current status: **151 passed in ~8 seconds** — full coverage of all ingestion modules.

```
tests/test_chunker.py             67 tests
tests/test_download_filings.py    18 tests
tests/test_indexer.py             14 tests
tests/test_metadata_extractor.py  24 tests
tests/test_parser.py              14 tests
tests/test_pipeline.py             7 tests
                            ──────────────
                           151 total tests
```

---

## Code Quality

This project uses four pre-commit hooks that run automatically on every `git commit`:

```bash
# Run all hooks manually against all files
poetry run pre-commit run --all-files
```

| Hook | Tool | What it checks |
|------|------|----------------|
| File hygiene | pre-commit-hooks | Trailing whitespace, EOF newline, YAML/TOML validity, large files, private keys |
| Linting + formatting | `ruff` | PEP8, unused imports, bugbear patterns, import order, code style |
| Type checking | `mypy` | Type annotations on all `ingestion/` functions |
| Security scanning | `bandit` | Known insecure patterns (pickle, subprocess, hardcoded passwords, etc.) |

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SEC_USER_AGENT` | `"Your Name your@email.com"` | Sent in HTTP headers to SEC EDGAR. Required by SEC fair-use policy. |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL. Can be a cloud cluster URL. |

---

## Roadmap

- [x] SEC EDGAR 8-K downloader
- [x] HTML parser with noise removal
- [x] Parent/child chunker (token-aware, sentence-boundary-safe)
- [x] Metadata extractor (ticker, company, quarter, fiscal period)
- [x] Fastembed indexer (BAAI/bge-large-en-v1.5 + Qdrant + BM25)
- [x] End-to-end ingestion pipeline with checkpointing
- [x] 151-test suite with full ingestion coverage
- [ ] Hybrid retrieval — BM25 + Qdrant with Reciprocal Rank Fusion (RRF)
- [ ] Query preprocessing — query expansion, HyDE
- [ ] LLM generation layer with source citations
- [ ] Corrective RAG (CRAG) — retrieval quality scoring + web fallback
- [ ] FastAPI REST endpoints
- [ ] Ragas evaluation harness
- [ ] Streamlit UI

---



## License

MIT
