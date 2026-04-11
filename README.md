# Financial Earnings Call RAG System

A production-grade, 7-layer multi-document RAG system to query financial
earnings call transcripts across 10 companies, 8 quarters, and 2 years —
with inline citations, temporal reasoning, and hallucination guardrails.

## The Problem
Analysts waste hours manually reading 20–60 page earnings call transcripts.
They need answers with citations, temporal reasoning ("how did guidance change
from Q1 2024 to Q3 2025?"), and cross-company comparison at scale. Plain
keyword search, naive vector RAG, and LLM document upload all fail here.

## 7-Layer Architecture
1. **Ingestion & Parsing** — SEC EDGAR 8-K HTML → BeautifulSoup parsing, speaker segmentation
2. **Chunking** — Semantic chunking + parent-child hierarchy (LlamaIndex)
3. **Hybrid Retrieval** — Dense vector (Qdrant) + BM25 sparse search fused with RRF
4. **Query Transformation** — HyDE + multi-query expansion + step-back rephrasing
5. **Reranking** — Cross-encoder narrows 40 candidates → top 5
6. **CRAG** — LangGraph grades chunks; triggers Tavily web search fallback if all score INCORRECT
7. **Generation** — Gemini produces answer with inline citations and temporal disambiguation

## Repository Structure
- `/ingestion`   → SEC EDGAR downloader, HTML parsing, chunking, embedding, Qdrant indexing
- `/retrieval`   → hybrid search, RRF fusion, reranker
- `/query`       → HyDE, multi-query, step-back transformation
- `/crag`        → LangGraph state machine, relevance grader, web fallback
- `/generation`  → prompt templates, citation formatting, streaming
- `/evaluation`  → RAGAS pipeline, golden dataset, DeepEval CI gate
- `/cache`       → Redis semantic cache layer
- `/api`         → FastAPI routes (async)
- `/ui`          → Streamlit frontend
- `/data`        → raw transcripts only (gitignored, re-generated via downloader)
- `/tests`       → unit + integration tests

## Data Source
Transcripts sourced from SEC EDGAR's free public API (`data.sec.gov`) — no API
key required. 10 companies × 8 quarters = 80 transcripts (~1.3M tokens). Run
`ingestion/sec_downloader.py` to reproduce the full corpus locally.
