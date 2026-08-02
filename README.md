# 📊 Financial Earnings Oracle: Production-Grade RAG System

> **A production-ready Retrieval-Augmented Generation (RAG) system for querying SEC 8-K earnings filings.** Built from the ground up for the modern AI/ML Engineer portfolio, demonstrating **Hybrid Retrieval**, **Corrective RAG (CRAG)**, **GraphRAG Entity Injection**, and **LLMOps Observability** with rigorous statistical evaluation.

[![CI](https://github.com/deepakj111/earnings-oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/deepakj111/earnings-oracle/actions/workflows/ci.yml)
[![CD](https://github.com/deepakj111/earnings-oracle/actions/workflows/cd.yml/badge.svg)](https://github.com/deepakj111/earnings-oracle/actions/workflows/cd.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 ML Engineering Highlights

This project is tailored to demonstrate senior-level competencies in **Applied AI, Machine Learning Engineering, and MLOps**:

- **Rigorous LLM Evaluation**: Uses an automated LLM-as-a-judge harness to measure *Faithfulness* and *Context Relevancy*. Employs **Bootstrap Resampling (95% CIs)** and paired **Wilcoxon signed-rank tests** to prove architectural improvements are statistically significant, avoiding "vibe checks".
- **Corrective RAG (CRAG)**: An autonomous meta-model grades chunk relevance. If local context is inadequate (e.g. data outside the 8-K corpus), it falls back to a web search aggregator (Tavily/DDG).
- **Advanced Context Engineering**: Mitigates the *Lost-in-the-Middle* phenomenon via U-shaped "valley reordering" of contexts. Utilizes token-aware parent-child chunking to guarantee bounded NLP context limits.
- **Production Observability**: Full asynchronous FastAPI deployment equipped with custom **Prometheus** endpoints (`RAG_REGISTRY`). Tracks LLM token usage, cost-in-USD, latency-by-layer, and cross-encoder RRF drift.

---

## 🏗 System Architecture

The pipeline consists of six distinct execution layers, parallelized via `asyncio` to bound P95 latencies under 3 seconds.

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                      FINANCIAL RAG EVALUATION & DEPLOYMENT                   │
│                                                                              │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌───────────┐   │
│  │L1 INGESTION │   │ L2 QUERY XFORM   │   │ L3 RETRIEVAL │   │ L4 SYNTH  │   │
│  │             │   │                  │   │              │   │           │   │
│  │ SEC EDGAR   │   │ HyDE             │   │ BM25 Sparse  │   │ OpenAI    │   │
│  │ Parser      │──▶│ Multi-Query (3x) │──▶│ Qdrant Dense │──▶│ Grounding │   │
│  │ Chunking    │   │ Step-Back Prompt │   │ FlashRank    │   │ Citations │   │
│  │ fastembed   │   │                  │   │              │   │           │   │
│  └─────────────┘   └──────────┬───────┘   └──────┬───────┘   └─────┬─────┘   │
│                               │                  │                 │         │
│                        ┌──────▼──────────────────▼────────┐        │         │
│                        │ L3.5: GRAPH-FUSED RETRIEVAL      │        │         │
│                        │ Knowledge Graph Traversal        │        │         │
│                        └──────────────────────────────────┘        │         │
│  ┌─────────────────────────────────────────────────────────────────▼───────┐ │
│  │                    L5: CRAG (Corrective Fallback)                       │ │
│  │  Grade context → CORRECT / AMBIGUOUS / INCORRECT → Web Search + Regen   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

- **ML Frameworks**: OpenAI Embeddings (`text-embedding-3-small`), `FlashRank` (`ms-marco-TinyBERT-L-2-v2` cross-encoder)
- **Vector Search**: `Qdrant` (Dense), `rank-bm25` (Sparse)
- **Generative AI**: `OpenAI SDK` (`gpt-5-mini` standardized across Query Routing, Query Transformation, Generation, CRAG Grading, KG Extraction, and Evaluation)
- **Infrastructure**: `FastAPI`, `Streamlit`, `Docker Compose`, `Prometheus`, `Grafana`
- **Code Quality**: Strict `mypy` typing, `ruff` checks, `bandit` security scanning, `pytest` suite (883+ tests with high coverage).

---

## 🚀 Quick Start (Local Reproduction)

### Prerequisites
- Python 3.11+
- Poetry
- Docker & Docker Compose

### 1. Setup Environment
```bash
git clone https://github.com/deepakj111/earnings-oracle.git
cd rag-project
poetry install
cp .env.example .env
```
*(Update `.env` with a placeholder `OPENAI_API_KEY` for evaluation)*

### 2. Standup Vector DB and UI
```bash
docker compose up -d
```

### 3. Run Ingestion Pipeline (SEC Scraping to Qdrant)
```bash
poetry run python -m ingestion.download_filings
poetry run python -m ingestion.pipeline
```

### 4. Serve API & Query Endpoints
Launch the production Uvicorn server:
```bash
poetry run serve-prod
```

Query the API via `curl` (use `-L` or trailing slash `/query/` to follow HTTP 307 redirects and pipe to `json.tool` for formatted output):

```bash
# Option A: Follow redirects (-L)
curl -L -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was Apple total revenue in 2024?",
    "filter": {"ticker": "AAPL"}
  }' | python3 -m json.tool

# Option B: Direct request with trailing slash (/query/)
curl -s -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was Apple total revenue in 2024?",
    "filter": {"ticker": "AAPL"}
  }' | python3 -m json.tool
```

### 5. Run E2E Evaluation Suite
Execute the automated MLOps statistical evaluation suite:
```bash
poetry run python -m evaluation.harness --metrics faithfulness answer_relevancy
```

---

## 🧪 Testing & CI/CD
This repository boasts a robust testing apparatus with **883+ unit tests** passing with excellent coverage.
```bash
poetry run pytest tests/
```
GitHub Actions orchestrates the CI/CD matrix: Python format enforcement (`ruff`), static analysis (`mypy`), security leak detection (`trufflehog`/`bandit`), and Docker smoke testing upon Main merges. See [CI/CD & Automation](docs/CI_CD.md) for details.

## 📄 License
MIT License.
