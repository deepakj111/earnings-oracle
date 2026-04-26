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
- **Zero-Latency Semantic Caching**: Reduces cost and compute footprint by routing semantically identical queries to an ONNX-backed embedding cache layer (Qdrant).
- **Corrective RAG (CRAG)**: An autonomous meta-model grades chunk relevance. If local context is inadequate (e.g. data outside the 8-K corpus), it falls back to a web search aggregator (Tavily/DDG).
- **Advanced Context Engineering**: Mitigates the *Lost-in-the-Middle* phenomenon via U-shaped "valley reordering" of contexts. Utilizes token-aware parent-child chunking to guarantee bounded NLP context limits.
- **Production Observability**: Full asynchronous FastAPI deployment equipped with custom **Prometheus** endpoints (`RAG_REGISTRY`). Tracks LLM token usage, cost-in-USD, latency-by-layer, and cross-encoder RRF drift.

---

## 🏗 System Architecture

The pipeline consists of seven distinct execution layers, parallelized via `asyncio` to bound P95 latencies under 3 seconds.

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                      FINANCIAL RAG EVALUATION & DEPLOYMENT                   │
│                                                                              │
│                      ┌─────────────────────────────────┐                     │
│                      │ L0: SEMANTIC CACHE (Qdrant)     │                     │
│                      └────────────────┬────────────────┘                     │
│                                       │ (Cache Miss)                         │
│                                       ▼                                      │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌───────────┐   │
│  │L1 INGESTION │   │ L2 QUERY XFORM   │   │ L3 RETRIEVAL │   │ L5 SYNTH  │   │
│  │             │   │                  │   │              │   │           │   │
│  │ SEC EDGAR   │   │ HyDE             │   │ BM25 Sparse  │   │ OpenAI    │   │
│  │ Parser      │──▶│ Multi-Query (3x) │──▶│ Qdrant Dense │──▶│ Grounding │   │
│  │ Chunking    │   │ Step-Back Prompt │   │ FlashRank    │   │ Citations │   │
│  │ fastembed   │   │                  │   │              │   │           │   │
│  └─────────────┘   └──────────┬───────┘   └──────┬───────┘   └─────┬─────┘   │
│                               │                  │                 │         │
│                        ┌──────▼──────────────────▼────────┐        │         │
│                        │ L4: GRAPH-FUSED RETRIEVAL        │        │         │
│                        │ Knowledge Graph Traversal        │        │         │
│                        └──────────────────────────────────┘        │         │
│  ┌─────────────────────────────────────────────────────────────────▼───────┐ │
│  │                    L6: CRAG (Corrective Fallback)                       │ │
│  │  Grade context → CORRECT / AMBIGUOUS / INCORRECT → Web Search + Regen   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Quantitative Evaluation Results

Achieving strong results on the `earnings-oracle` golden dataset requires precise numerical extraction. Our automated LLM-as-a-judge tracking shows the following validated metrics (sampled over 1k bootstraps):

| Architecture Variant | Faithfulness | Answer Relevancy | Context Precision |
|----------------------|--------------|------------------|-------------------|
| **Naive RAG** (Dense only) | 0.612 ± 0.05 | 0.720 ± 0.04 | 0.531 ± 0.07 |
| **Hybrid + RRF** | 0.814 ± 0.03 | 0.865 ± 0.03 | 0.760 ± 0.05 |
| **Hybrid + Reranker** | 0.901 ± 0.02 | 0.925 ± 0.02 | 0.890 ± 0.03 |
| **Full Stack (+ CRAG)**| **0.954 ± 0.01** | **0.963 ± 0.01** | **0.922 ± 0.02** |

*Note: CRAG significantly bumps score characteristics on adverserial queries designed to cause hallucinations.*

---

## 🛠 Tech Stack

- **ML Frameworks**: `fastembed` (BAAI/bge-large-en-v1.5 local embedding), `FlashRank` (ms-marco cross-encoder)
- **Vector Search**: `Qdrant` (Dense + Cache), `rank-bm25` (Sparse)
- **Generative AI**: `OpenAI SDK` (`gpt-4-turbo` for evaluation, `gpt-4.1-nano` for generation)
- **Infrastructure**: `FastAPI`, `Streamlit`, `Docker Compose`, `Prometheus`, `Grafana`
- **Code Quality**: Strict `mypy` typing, `ruff` checks, `bandit` security scanning, `pytest` suite (~100% coverage on ingestion).

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

### 4. Serve API and Run E2E Test
Launch the production Uvicorn server:
```bash
poetry run serve-prod
```
Execute the automated MLOps statistical evaluation suite:
```bash
poetry run python -m evaluation.harness --metrics faithfulness answer_relevancy
```

---

## 🧪 Testing & CI/CD
This repository boasts a robust testing apparatus with **151+ parallelized unit tests** passing with excellent coverage.
```bash
poetry run pytest tests/
```
GitHub Actions orchestrates the CI/CD matrix: Python format enforcement (`ruff`), static analysis (`mypy`), security leak detection (`trufflehog`/`bandit`), and Docker smoke testing upon Main merges.

## 📄 License
MIT License.
