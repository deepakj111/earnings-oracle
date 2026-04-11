# Financial Earnings Call RAG System

A production-grade, 7-layer multi-document RAG system to query financial earnings call transcripts across multiple companies, quarters, and years with citations, temporal reasoning, and hallucination guardrails.

## The Problem
Analysts waste hours manually reading 20–60 page earnings call PDFs. They need answers with citations, temporal reasoning, and a scalable solution. Plain keyword search, naive vector RAG, and LLM document upload all fail at scale.

## 7-Layer Architecture
1. **Ingestion & Parsing**: Parse PDFs preserving tables, speaker turns (unstructured.io)
2. **Chunking**: Semantic chunking + parent-child hierarchy
3. **Hybrid Retrieval**: Dense vector + BM25 sparse search fused with RRF
4. **Query Transformation**: HyDE + multi-query + step-back rephrasing
5. **Reranking**: Cross-encoder narrows 40 candidates → top 5
6. **CRAG**: Grades retrieved chunks; triggers web search fallback if all score INCORRECT
7. **Generation**: Produces answer with inline citations, temporal disambiguation

## Repository Structure
- `/ingestion`        → PDF parsing, chunking, embedding, Qdrant indexing
- `/retrieval`        → hybrid search, RRF fusion, reranker
- `/query`            → HyDE, multi-query, step-back transformation
- `/crag`             → LangGraph state machine, relevance grader, web fallback
- `/generation`       → prompt templates, citation formatting, streaming
- `/evaluation`       → RAGAS pipeline, golden dataset, DeepEval CI gate
- `/cache`            → GPTCache + Redis semantic cache layer
- `/api`              → FastAPI routes (async)
- `/ui`               → Streamlit frontend
- `/data`             → transcript ingestion scripts
- `/tests`            → unit + integration tests
