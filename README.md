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

## 🗂️ Data Strategy & Scope: Controlled Domain Specialization

Instead of scraping a massive, noisy assortment of random tickers, this system's ingestion pipeline is deliberately scoped to the **latest Annual Reports (SEC Form 10-K)** of two fundamentally contrasting Fortune 50 companies: **NVIDIA (NVDA)** and **Walmart (WMT)**.

This represents a deliberate ML engineering choice to optimize for **pipeline depth and architectural stress-testing over dataset width.**

A single SEC Form 10-K is a 150+ page behemoth of highly regulated financial prose, dense MD&A (Management's Discussion and Analysis), and complex HTML tables. Ingesting just two recent 10-Ks produces tens of thousands of tokens of high-entropy text.

This provides a rigorous stress-test to prove the efficacy of:
*   **Parent-Child Chunking:** Ensuring complex financial tables remain atomic while maintaining tight semantic retrieval windows.
*   **GraphRAG Entity Extraction:** Proving the LLM can map and traverse complex entity relationships across a massive single-document context.
*   **Lost-in-the-Middle Mitigation:** Testing if the system can accurately synthesize an answer using a single risk-factor footnote buried on page 104.

In production MLOps, managing compute budgets and evaluation variance is a core competency.

A tightly constrained dataset acts as a **controlled laboratory environment**. It allows the automated evaluation harness to run rigorous, repeated, and statistically significant A/B tests (e.g., paired t-tests and Wilcoxon signed-rank tests) on pipeline variants.

This proves the exact quantitative impact of adding CRAG or FlashRank to the architecture without introducing unmanageable data variance or unnecessary LLM token burn.


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
│  │ OpenAI Emb  │   │                  │   │              │   │           │   │
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

### 4. Serve API & Frontend Endpoints

#### Option A: Production Server (Multi-Worker)
Launch the production Uvicorn server with 4 worker processes:
```bash
poetry run serve-prod
```

#### Option B: Development Server (Auto-Reload)
Launch the single-worker development server with auto-reload:
```bash
poetry run serve
```

> **Note on Docker vs Local Port 8000**: Both `poetry run serve-prod` and Docker's `api` container bind host port `8000`. If you run `docker compose up -d` while running local `serve-prod`, stop the Docker API container (`docker stop rag_api`) to prevent port conflicts. Qdrant (`6333`), Prometheus (`9090`), and Grafana (`3000`) can remain running in Docker.

#### Access the Modern Web Frontend
When `serve-prod` or `serve` is running, access the single-page HTML chat interface with stateful conversation history, filter controls, and citation cards:
- **Web App**: [http://localhost:8000/app](http://localhost:8000/app) (or bare root `http://localhost:8000/`)
- **Streamlit UI** (Optional): `poetry run ui` (accessible at `http://localhost:8501`)

#### Query the API via `curl`
The endpoint directly accepts `POST /query` without redirects:

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was WALMART U.S. net sales and global total revenue for the fiscal year ended January 31, 2026?"
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
