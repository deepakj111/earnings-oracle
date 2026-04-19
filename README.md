# 📊 Financial RAG System

> A production-grade Retrieval-Augmented Generation system for querying SEC 8-K earnings filings — built with hybrid retrieval, corrective RAG, and full LLMOps observability.

[![CI](https://github.com/deepakj111/earnings-oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/deepakj111/earnings-oracle/actions/workflows/ci.yml)
[![CD](https://github.com/deepakj111/earnings-oracle/actions/workflows/cd.yml/badge.svg)](https://github.com/deepakj111/earnings-oracle/actions/workflows/cd.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Overview

The Financial RAG System answers precise financial questions over SEC 8-K earnings press releases using a **seven-layer pipeline**:

| Layer | Component | Purpose |
|-------|-----------|---------|
| L0 | **Semantic Cache** | Instant retrieval for repeated queries via Qdrant embedding distances |
| L1 | **Ingestion & Extraction** | SEC Scrape → HTML parse → Parent/Child chunking → **GraphRAG Entity Extraction** |
| L2 | **Query Transformation** | HyDE + Multi-Query + Step-Back prompting (concurrent) |
| L3 | **Vector Retrieval** | BM25 sparse + Qdrant dense search → RRF fusion |
| L4 | **Graph Fused Retrieval** | Knowledge Graph traversal (`data/knowledge_graph.json`) context injection |
| L5 | **Answer Generation** | FlashRank reranking → LLM synthesis with grounded `[N]` inline citations |
| L6 | **CRAG** | Corrective RAG loop: relevance grading → web-search fallback on poor retrieval |

Served via **FastAPI** with Server-Sent Events streaming, **Prometheus** metrics, and a **Streamlit** chat UI.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           FINANCIAL RAG PIPELINE                              │
│                                                                               │
│                      ┌─────────────────────────────────┐                      │
│                      │ L0 SEMANTIC CACHE (Qdrant)      │                      │
│                      └────────────────┬────────────────┘                      │
│                                       │ (Cache Miss)                          │
│                                       ▼                                       │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌───────────┐  │
│  │L1 INGEST/EXT│   │  L2 QUERY XFORM  │   │  L3 VECTOR   │   │  L5 GEN   │  │
│  │             │   │                  │   │   RETRIEVAL  │   │           │  │
│  │ SEC EDGAR   │   │ HyDE             │   │ BM25 sparse  │   │ OpenAI    │  │
│  │ 8-K scrape  │   │ Multi-Query (3x) │   │ + Qdrant     │   │ gpt-4.1-  │  │
│  │ HTML parse  │──▶│ Step-Back        │──▶│ dense → RRF  │──▶│ nano      │  │
│  │ Parent/child│   │                  │   │              │   │           │  │
│  │ fastembed   │   │     ┌────────────┴───┴─────┐        │   │ Citations │  │
│  │ Graph Extr. │   │     │ L4 GRAPH RETRIEVAL   │        │   │ Grounding │  │
│  └─────────────┘   └─────┤ KG context injection ├────────┘   └─────┬─────┘  │
│                          └──────────────────────┘                  │        │
│  ┌─────────────────────────────────────────────────────────────────▼───────┐│
│  │                         L6 CRAG (Corrective RAG)                        ││
│  │  Grade chunks → CORRECT / AMBIGUOUS / INCORRECT → Web fallback + regen  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐   ┌──────────────────┐   ┌───────────────────────────────────┐
│  FastAPI     │   │  Streamlit UI    │   │  Observability                    │
│  REST + SSE  │   │  Chat interface  │   │  Prometheus metrics + Grafana     │
│  /query      │   │  Streaming mode  │   │  Custom RAG_REGISTRY              │
│  /health     │   │  Citation cards  │   │  LLMOps counters + histograms     │
│  /metrics    │   │  Sidebar filters │   │  Per-layer pipeline latency       │
└──────────────┘   └──────────────────┘   └───────────────────────────────────┘
```

### Chunking Architecture

```
Raw HTML Filing
     │
     ▼
Stage 1: Structure-aware section splitting
     │  ├── Financial section headers as hard boundaries
     │  └── Markdown tables detected and kept atomic (never split)
     │
     ▼
Stage 2: Parent chunks  (~512 tokens, 64-token overlap)
     │  └── Contextual prefix: [Context: AAPL | earnings_release | 2024-10-31 | Section: Revenue]
     │
     ▼
Stage 3: Child chunks   (~128 tokens, 32-token overlap, sentence-boundary-safe)
     │  └── Contextual prefix re-applied to every child
     │
     ├──▶ Qdrant  (child embeddings via fastembed BAAI/bge-large-en-v1.5)
     └──▶ BM25    (child token lists via rank-bm25)

Retrieval: fetch small children for precision → fetch full parents for generation context
```

### Lost-in-the-Middle Mitigation

Results are valley-reordered before LLM context assembly. With `k=5` candidates ranked `[r1..r5]`:

```
Standard order:  [r1, r2, r3, r4, r5]
Valley order:    [r1, r3, r5, r4, r2]   ← r1 at position 0, r2 at position -1
LLM attention:    HIGH  ↑         HIGH   (U-shaped attention pattern)
```

---

## Tech Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Runtime | Python | 3.11 / 3.12 | Application runtime |
| Web Framework | FastAPI | ≥0.115 | REST API + SSE streaming |
| ASGI Server | uvicorn + uvloop + httptools | ≥0.34 | Production-grade async server |
| Embedding | fastembed + BAAI/bge-large-en-v1.5 | ^0.4.0 | 1024-dim ONNX embeddings (local, free) |
| Vector DB | Qdrant | ^1.17 (client) | Dense ANN vector search + **Semantic Caching** |
| Graph DB | JSON | — | Lightweight **GraphRAG Entity/Relationship store** |
| Keyword Search | rank-bm25 | ^0.2.2 | Sparse BM25 index |
| Reranker | FlashRank (ms-marco-MiniLM-L-12-v2) | ^0.2.9 | Local cross-encoder reranking |
| LLM | OpenAI gpt-4.1-nano | ≥2.0.0 (SDK) | Query transform + generation + evaluation |
| Token Counting | tiktoken | ^0.12 | Token-aware chunking |
| Monitoring | prometheus-client | ^0.20 | Custom LLMOps metrics |
| Tracing | loguru | ^0.7.3 | Structured pipeline logging |
| UI | Streamlit | ≥1.35 | Chat interface |
| Config | python-dotenv | ^1.1.0 | Environment management |
| Retry | tenacity | ^9.1.4 | Exponential backoff on LLM calls |
| HTML Parsing | beautifulsoup4 + lxml | ^4.12 | SEC filing parsing |
| Linting | ruff | ^0.9.0 | PEP8 + import sorting |
| Type Checking | mypy | ^1.20 | Static type annotations |
| Security | bandit | ^1.9.4 | Vulnerability scanning → SARIF |
| Testing | pytest + pytest-cov | ^8.3.0 | 151+ tests, ≥80% coverage gate |
| Containerisation | Docker (multi-stage) | — | Builder → runtime image |
| Orchestration | docker-compose | — | Full stack local dev |

---

## Supported Companies

| Ticker | Company | Exchange |
|--------|---------|---------|
| AAPL | Apple Inc. | NASDAQ |
| NVDA | NVIDIA Corporation | NASDAQ |
| MSFT | Microsoft Corporation | NASDAQ |
| AMZN | Amazon.com, Inc. | NASDAQ |
| META | Meta Platforms, Inc. | NASDAQ |
| JPM | JPMorgan Chase & Co. | NYSE |
| XOM | ExxonMobil Corporation | NYSE |
| UNH | UnitedHealth Group | NYSE |
| TSLA | Tesla, Inc. | NASDAQ |
| WMT | Walmart Inc. | NYSE |

Data range: **2023-01-01 → present** (incremental, checkpoint-based)

---

## Project Structure

```
rag-project/
│
├── ingestion/                    # ✅ L1 — SEC EDGAR data pipeline
│   ├── download_filings.py       #    SEC EDGAR 8-K scraper (EX-99.1 exhibits)
│   ├── parser.py                 #    HTML → ParsedDocument (BeautifulSoup + lxml)
│   ├── chunker.py                #    Parent/child chunk architecture (tiktoken)
│   ├── metadata_extractor.py     #    Ticker, date, quarter, fiscal period detection
│   ├── indexer.py                #    fastembed + Qdrant upsert + BM25 corpus build
│   └── pipeline.py               #    End-to-end orchestrator with checkpointing
│
├── cache/                        # ✅ L0 — Semantic Caching
│   └── semantic_cache.py         #    Qdrant-backed zero-latency embedding text cache
│
├── knowledge_graph/              # ✅ L4 — GraphRAG
│   ├── models.py                 #    Entity, Relationship, KnowledgeGraph dataclasses
│   ├── extractor.py              #    LLM-based context extractor (`gpt-4.1-nano`)
│   └── graph_retriever.py        #    Traverse nodes -> context injection
│
├── query/                        # ✅ L2 — Query Transformation
│   ├── models.py                 #    TransformedQuery dataclass
│   ├── prompts.py                #    HyDE / Multi-Query / Step-Back prompt templates
│   └── transformer.py            #    Concurrent LLM calls + in-memory LRU cache
│
├── retrieval/                    # ✅ L3 — Hybrid Retrieval
│   ├── models.py                 #    MetadataFilter, SearchResult, RetrievalResult
│   ├── searcher.py               #    BM25 + Qdrant dense search + RRF fusion
│   ├── reranker.py               #    FlashRank cross-encoder reranking
│   └── __init__.py               #    retrieve() public API + warmup functions
│
├── generation/                   # ✅ L4 — Answer Generation
│   ├── models.py                 #    Citation, GenerationResult dataclasses
│   ├── context_builder.py        #    Dedup + valley reorder + token budget + formatting
│   ├── generator.py              #    LLM synthesis + citation extraction + grounding check
│   ├── prompts.py                #    System/user prompt templates + ungrounded phrases
│   └── __init__.py               #    generate() module-level shortcut
│
├── crag/                         # ✅ L5 — Corrective RAG
│   ├── models.py                 #    CRAGAction, CRAGResult, RelevanceGrade, WebSearchResult
│   ├── corrector.py              #    CRAG orchestration: grade → decide → web search → regen
│   ├── grader.py                 #    LLM-based chunk relevance grading (concurrent)
│   ├── web_search.py             #    Tavily / DuckDuckGo web-search abstraction
│   └── __init__.py               #    Public CRAG API
│
├── api/                          # ✅ FastAPI application
│   ├── main.py                   #    App factory + lifespan + middleware stack
│   ├── models.py                 #    Pydantic v2 request/response schemas
│   ├── dependencies.py           #    FastAPI Depends() injection (pipeline, qdrant, uptime)
│   ├── errors.py                 #    Global exception → HTTP status handlers
│   ├── metrics.py                #    Prometheus custom registry + recording helpers
│   ├── middleware.py             #    RequestIDMiddleware + TimingMiddleware (pure ASGI)
│   └── routes/
│       ├── query.py              #    POST /query (structured) + POST /query/stream (SSE)
│       ├── health.py             #    GET /health, /health/live, /health/ready
│       └── metrics_route.py      #    GET /metrics (Prometheus scrape endpoint)
│
├── config/
│   └── settings.py               #    Centralised dataclass config (all env-overridable)
│
├── evaluation/                   # ✅ LLMOps evaluation harness
│   ├── dataset.py                #    16-sample golden QA dataset (SEC filings ground truth)
│   ├── harness.py                #    Parallel pipeline evaluation + report persistence
│   ├── metrics.py                #    faithfulness, answer_relevancy, context_precision, recall
│   ├── models.py                 #    EvalSample, MetricScore, EvalSampleResult, EvalReport
│   └── __init__.py
│
├── ui/
│   ├── app.py                    #    Streamlit chat UI (streaming + structured modes)
│   └── utils.py                  #    Pure utility functions (SSE parsing, formatting, health)
│
├── rag_pipeline.py               #    FinancialRAGPipeline — wires all 4+ layers
├── scripts/
│   ├── entrypoints.py            #    poetry run serve / serve-prod / ui shortcuts
│   └── inspect_index.py          #    Index diagnostic tool (filesystem + BM25 + Qdrant)
│
├── tests/                        #    151+ tests across all modules
│   ├── conftest.py               #    Shared fixtures (mock pipeline, qdrant, TestClient)
│   ├── test_api_*.py             #    API routes, errors, middleware, metrics, models
│   ├── test_chunker.py           #    67 chunker tests
│   ├── test_context_builder.py   #    Valley reorder, budget, dedup
│   ├── test_generator.py         #    Citation extraction, grounding, LLM mocking
│   ├── test_crag_*.py            #    CRAG corrector, grader, web search
│   ├── test_evaluation_*.py      #    Harness, metrics, dataset
│   ├── test_query_*.py           #    Transformer, prompts, models, cache
│   ├── test_retrieval_*.py       #    RRF fusion, BM25, reranker, models
│   ├── test_indexer.py           #    Embedding, upsert, BM25 corpus alignment
│   ├── test_metadata_extractor.py
│   ├── test_parser.py
│   ├── test_pipeline.py
│   ├── test_download_filings.py
│   └── test_ui_utils.py
│
├── data/                         # Runtime data (gitignored)
│   ├── transcripts/              #    Downloaded .htm 8-K filings
│   ├── bm25_index.pkl            #    Serialised BM25Okapi object
│   ├── bm25_corpus.pkl           #    Parallel metadata list (must match bm25_index)
│   ├── pipeline_checkpoint.txt   #    Incremental ingestion tracker
│   └── eval_reports/             #    JSON + CSV evaluation reports
│
├── grafana/provisioning/         # Grafana datasource auto-provisioning
├── .github/workflows/
│   ├── ci.yml                    #    6-job parallel CI (lint/type/security/test/validate/docker)
│   └── cd.yml                    #    Docker build → GHCR push → smoke test
├── Dockerfile                    #    Multi-stage: builder + runtime (non-root appuser)
├── docker-compose.yml            #    Full stack: Qdrant + API + UI + Prometheus + Grafana
├── prometheus.yml                #    Prometheus scrape config
├── pyproject.toml                #    Poetry deps + ruff/mypy/bandit/pytest config
└── .pre-commit-config.yaml       #    5-stage hooks: hygiene/ruff/mypy/bandit/trufflehog
```

---

## Quick Start

### Prerequisites

- Python 3.11 or 3.12
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker (for Qdrant)
- OpenAI API key

### 1. Clone and install

```bash
git clone https://github.com/deepakj111/earnings-oracle.git
cd rag-project
poetry install
```

### 2. Configure environment

```bash
cp .env.example .env
```

```dotenv
# .env
OPENAI_API_KEY=sk-...                          # Required — LLM calls
SEC_USER_AGENT="Your Name your@email.com"       # Required — SEC fair-use policy
QDRANT_URL=http://localhost:6333                # Qdrant instance URL

# Optional overrides (see config/settings.py for full reference)
RAG_GENERATION_MODEL=gpt-4.1-nano
RAG_RERANKER_ENABLED=true
RAG_CRAG_ENABLED=true
TAVILY_API_KEY=tvly-...                         # Optional — CRAG web search (Tavily)
```

### 3. Start Qdrant

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant:v1.9.2
```

### 4. Download SEC filings

```bash
poetry run python -m ingestion.download_filings
# Downloads EX-99.1 exhibits for 10 companies (2023-01-01 → today)
# Output: data/transcripts/*.htm
```

### 5. Run ingestion pipeline

```bash
poetry run python -m ingestion.pipeline
# Loads BAAI/bge-large-en-v1.5 (~340 MB, cached after first run)
# Embeds all child chunks → Qdrant
# Builds BM25 index → data/bm25_index.pkl
# Re-runs are safe — checkpoint skips already-indexed files
```

### 6. Start the API

```bash
poetry run serve          # Development (auto-reload)
poetry run serve-prod     # Production (4 workers)
```

### 7. Start the Streamlit UI

```bash
poetry run ui
# Open http://localhost:8501
```

---

## Usage Examples

### Python SDK

```python
from rag_pipeline import FinancialRAGPipeline
from qdrant_client import QdrantClient
from retrieval.models import MetadataFilter

client = QdrantClient(url="http://localhost:6333")
pipeline = FinancialRAGPipeline(qdrant_client=client)

# Simple question
result = pipeline.ask("What was Apple's revenue in Q4 2024?")
print(result.format_answer_with_citations())

# Scoped to a ticker + year
result = pipeline.ask(
    question="What was NVIDIA's data center gross margin?",
    metadata_filter=MetadataFilter(ticker="NVDA", year=2024),
)
print(result.to_json())

# Streaming (for UI layers)
for token in pipeline.ask_streaming("What was Meta's ad revenue?"):
    print(token, end="", flush=True)

# Verbose diagnostic mode
result, query_summary, retrieval_summary = pipeline.ask_verbose(
    "How did Apple's Services revenue trend across 2024?"
)

# With CRAG correction
crag_result = pipeline.ask_with_crag("What was Berkshire Hathaway's earnings?")
print(f"Action: {crag_result.action.value}")   # correct / ambiguous / incorrect
print(f"Web search triggered: {crag_result.web_search_triggered}")
```

### REST API

```bash
# Structured response
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Apple revenue Q4 2024?", "filter": {"ticker": "AAPL"}}'

# Streaming SSE
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What was NVIDIA Q3 2024 data center revenue?"}'

# Health check
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:8000/metrics
```

---

## Running the Full Stack

```bash
cp .env.example .env          # Fill in OPENAI_API_KEY and SEC_USER_AGENT
docker compose up -d

# Run ingestion (one-off)
docker compose exec api poetry run python -m ingestion.download_filings
docker compose exec api poetry run python -m ingestion.pipeline
```

| Service | URL |
|---------|-----|
| API (Swagger UI) | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |
| Qdrant Dashboard | http://localhost:6333/dashboard |

---

## Testing

```bash
# Full test suite with coverage
poetry run pytest tests/ -v --durations=10

# With coverage gate (CI mode)
poetry run pytest tests/ --cov-fail-under=80

# Specific module
poetry run pytest tests/test_chunker.py -v

# Skip integration tests
poetry run pytest tests/ -m "not integration"
```

Current status: **151+ passed in ~8 seconds** — full coverage of all ingestion modules, API, retrieval, generation, CRAG, evaluation, and UI utilities.

```
tests/test_api_errors.py           ~30 tests
tests/test_api_health.py           ~20 tests
tests/test_api_metrics.py          ~20 tests
tests/test_api_middleware.py       ~15 tests
tests/test_api_models.py           ~25 tests
tests/test_api_query.py            ~35 tests
tests/test_chunker.py               67 tests
tests/test_context_builder.py      ~25 tests
tests/test_crag_corrector.py       ~15 tests
tests/test_crag_grader.py          ~15 tests
tests/test_crag_web_search.py      ~15 tests
tests/test_download_filings.py      18 tests
tests/test_evaluation_harness.py   ~15 tests
tests/test_evaluation_metrics.py   ~20 tests
tests/test_generator.py            ~30 tests
tests/test_indexer.py               14 tests
tests/test_metadata_extractor.py    24 tests
tests/test_parser.py                14 tests
tests/test_pipeline.py               7 tests
tests/test_query_models.py         ~25 tests
tests/test_query_prompts.py        ~20 tests
tests/test_query_transformer.py    ~40 tests
tests/test_retrieval_models.py     ~20 tests
tests/test_retrieval_reranker.py   ~15 tests
tests/test_retrieval_searcher.py   ~15 tests
tests/test_ui_utils.py             ~40 tests
```

---

## Code Quality

```bash
# Run all pre-commit hooks
poetry run pre-commit run --all-files

# Individual tools
poetry run ruff check . --fix     # Lint + auto-fix
poetry run ruff format .           # Format (replaces Black)
poetry run mypy ingestion/         # Type checking
poetry run bandit -r ingestion/ generation/ retrieval/ query/ crag/ api/ -c pyproject.toml
```

| Gate | Tool | Scope |
|------|------|-------|
| File hygiene | pre-commit-hooks | Trailing whitespace, EOF, YAML/TOML/JSON validity, large files, private keys |
| Linting + formatting | ruff v0.9 | PEP8, unused imports, bugbear patterns, import order |
| Type checking | mypy 1.20 | All `ingestion/` functions require full type annotations |
| Security scanning | bandit | Insecure patterns → SARIF upload to GitHub Security tab |
| Secret scanning | TruffleHog | Scans staged git changes for verified leaked secrets |

---

## Evaluation

```bash
# Run evaluation harness (5 sample smoke test)
poetry run python -m evaluation.harness --n 5 --metrics faithfulness answer_relevancy

# Full golden dataset (16 samples)
poetry run python -m evaluation.harness

# Programmatic
from evaluation import EvaluationHarness
from evaluation.dataset import get_dataset_subset

harness = EvaluationHarness(pipeline)
report = harness.run(dataset=get_dataset_subset(5))
print(report.summary())
harness.save_report(report)   # → data/eval_reports/*.json + *.csv
```

Four LLM-based metrics (all via gpt-4.1-nano, structured JSON responses):

| Metric | Definition | Range |
|--------|-----------|-------|
| **faithfulness** | Fraction of answer claims directly supported by retrieved context | 0–1 |
| **answer_relevancy** | Does the answer directly address the question? | 0–1 |
| **context_precision** | Fraction of retrieved chunks relevant to the query | 0–1 |
| **context_recall** | Coverage of ground-truth facts in retrieved context | 0–1 |

---

## Observability

Prometheus metrics exposed at `GET /metrics`:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `rag_http_requests_total` | Counter | `endpoint, method, status_code` | Total HTTP requests |
| `rag_http_request_duration_seconds` | Histogram | `endpoint` | Request latency |
| `rag_llm_tokens_total` | Counter | `model, token_type` | Cumulative tokens (prompt/completion) |
| `rag_llm_cost_usd_total` | Counter | `model` | Estimated cost in USD |
| `rag_retrieval_candidates` | Histogram | — | Candidates entering reranker |
| `rag_retrieval_results_returned` | Histogram | — | Final results per query |
| `rag_context_tokens_used` | Histogram | — | Tokens in LLM context window |
| `rag_grounded_responses_total` | Counter | `grounded` | Grounded vs ungrounded |
| `rag_retrieval_failed_total` | Counter | — | Zero-result queries |
| `rag_crag_actions_total` | Counter | `action` | CRAG decision distribution |
| `rag_pipeline_latency_seconds` | Histogram | `layer` | Per-layer latency (L2/L3/L4) |

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key |
| `SEC_USER_AGENT` | `"Your Name your@email.com"` | **Required.** SEC EDGAR fair-use HTTP header |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `RAG_GENERATION_MODEL` | `gpt-4.1-nano` | LLM for answer generation |
| `RAG_QUERY_TRANSFORM_MODEL` | `gpt-4.1-nano` | LLM for HyDE/Multi-Query/Step-Back |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | fastembed model name |
| `RAG_QDRANT_COLLECTION` | `earnings_transcripts` | Qdrant collection name |
| `RAG_RETRIEVAL_TOP_K_DENSE` | `10` | Dense candidates per query variant |
| `RAG_RETRIEVAL_TOP_K_BM25` | `10` | BM25 candidates per query variant |
| `RAG_RETRIEVAL_TOP_K_FINAL` | `5` | Final chunks after reranking |
| `RAG_RERANKER_ENABLED` | `true` | Enable FlashRank cross-encoder |
| `RAG_CRAG_ENABLED` | `true` | Enable CRAG correction loop |
| `RAG_CRAG_HIGH_THRESHOLD` | `0.6` | Relevance ratio above which → CORRECT |
| `RAG_CRAG_LOW_THRESHOLD` | `0.2` | Relevance ratio below which → INCORRECT |
| `TAVILY_API_KEY` | — | Tavily API key (CRAG web search; falls back to DuckDuckGo) |
| `GRAFANA_ADMIN_PASSWORD` | — | Grafana admin password (docker-compose) |
| `RAG_EVAL_MODEL` | `gpt-4.1-nano` | LLM for evaluation metrics |
| `RAG_EVAL_OUTPUT_DIR` | `data/eval_reports` | Evaluation report output directory |

---

## Latency Profile

Measured on CPU-only (typical):

| Layer | Operation | Latency |
|-------|-----------|---------|
| L0 | **Semantic Cache hit (bypasses all other layers)** | **< 50 ms** |
| L2 | Query transformation (3 concurrent LLM calls) | 0.8–1.2 s |
| L3 | Hybrid vector retrieval + RRF + FlashRank reranking | 0.3–0.8 s |
| L3 | Parent fetch (batch Qdrant scroll) | ~50 ms |
| L4 | GraphRAG node traversal & context injection | 50–150 ms |
| L5 | Answer generation (single LLM call) | 0.8–2.0 s |
| **Total** | **End-to-end (Cache Miss)** | **~2–4 s** |

---

## Features (v1.0.0)

- [x] High-throughput Async ingestion pipeline (concurrent LLM eval + vector batching)
- [x] Scientific Evaluation Harness (Bootstrapped 95% CIs + Paired Significance Testing)
- [x] SEC EDGAR 8-K downloader (10 companies, 2023–present)
- [x] HTML parser with noise removal (script/style/nav stripping)
- [x] Parent/child chunker (token-aware, sentence-boundary-safe, table protection)
- [x] Metadata extractor (ticker, company, quarter, fiscal period)
- [x] fastembed indexer (BAAI/bge-large-en-v1.5 + Qdrant + BM25) with payload indices
- [x] Layer 0: Semantic Caching (Qdrant payload backed embedding cache)
- [x] Layer 2: Query transformation (HyDE + Multi-Query + Step-Back, concurrent)
- [x] Layer 3: Hybrid retrieval (BM25 + Qdrant + RRF + FlashRank + parent fetch)
- [x] Layer 4: Knowledge Graph Fused Retrieval (GraphRAG context injection)
- [x] Layer 5: Answer generation (valley ordering, citation extraction, grounding check)
- [x] Layer 6: Corrective RAG (relevance grading + Tavily/DuckDuckGo web fallback)
- [x] FastAPI REST + SSE streaming endpoints
- [x] Prometheus custom metrics + Grafana integration
- [x] Streamlit chat UI (streaming + structured modes, citation cards, health panel)
- [x] Multi-stage Docker build + docker-compose full stack
- [x] 6-job parallel CI + CD pipeline (GHCR push + smoke test)
- [x] 151+ test suite (≥80% coverage gate)

---

## License

MIT — see [LICENSE](LICENSE).
