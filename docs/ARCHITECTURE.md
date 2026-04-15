# System Architecture

> Deep technical reference for the Financial RAG System — design decisions, data flows, and component contracts.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Layer 1: Ingestion Pipeline](#layer-1-ingestion-pipeline)
3. [Layer 2: Query Transformation](#layer-2-query-transformation)
4. [Layer 3: Hybrid Retrieval](#layer-3-hybrid-retrieval)
5. [Layer 4: Answer Generation](#layer-4-answer-generation)
6. [Layer 5: Corrective RAG](#layer-5-corrective-rag)
7. [API Layer](#api-layer)
8. [Configuration System](#configuration-system)
9. [Data Contracts](#data-contracts)
10. [Concurrency Model](#concurrency-model)
11. [Design Decisions & Trade-offs](#design-decisions--trade-offs)

---

## System Overview

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  FinancialRAGPipeline  (rag_pipeline.py)                    │
│                                                             │
│   QueryTransformer.transform(question)                      │
│           │                                                 │
│           ▼                                                 │
│   retrieve(query, qdrant_client, metadata_filter)          │
│           │                                                 │
│           ▼                                                 │
│   Generator.generate(question, retrieval_result)           │
│           │                                                 │
│           ▼ (optional)                                      │
│   CRAGCorrector.correct(question, gen_result, ret_result)  │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
GenerationResult / CRAGResult
```

**Thread safety**: All pipeline components are stateless between calls. Internal model singletons (fastembed, BM25, FlashRank, OpenAI client) are safe for concurrent reads after first initialisation. Parallel `ask()` calls across threads are fully supported.

---

## Layer 1: Ingestion Pipeline

### Data Flow

```
SEC EDGAR API (JSON submissions endpoint)
        │
        │ CIK lookup → 8-K filing list → accession numbers
        ▼
Filing Index HTML (https://www.sec.gov/Archives/.../accession-index.htm)
        │
        │ EX-99.1 selection (earnings press release exhibit)
        ▼
Raw .htm file (download_document)
        │
        ▼  parse_html()
ParsedDocument {ticker, date, raw_text, sections[]}
        │
        │  extract_metadata()
        ▼
DocumentMetadata {ticker, company, date, year, quarter, fiscal_period}
        │
        │  create_parent_child_chunks()
        ▼
list[Chunk]  (parent + child + table chunks)
        │
        ├──▶  index_document()  ──▶  Qdrant (child embeddings)
        │                        ──▶  bm25_texts + bm25_corpus (in-memory)
        │
        └──▶  _mark_done()  ──▶  pipeline_checkpoint.txt
```

### Chunking Architecture

The chunker implements a **three-stage parent/child architecture** tuned for financial documents:

**Stage 1 — Structure-aware section splitting**

Financial section headers act as hard boundaries. A line qualifies as a header only if:
1. It matches a financial/markdown header pattern (Revenue, Segment Results, Outlook, etc.)
2. It contains ≤8 words (prevents long prose sentences matching header patterns)

Markdown tables are detected via `_is_table_block()` and kept **strictly atomic** — they are never split across parent or child chunks. This preserves tabular financial data integrity (revenue by segment, balance sheet items, etc.).

**Stage 2 — Parent chunks** (~512 tokens)
- 64-token overlap between consecutive parents
- Each parent carries a contextual prefix: `[Context: AAPL | earnings_release | 2024-10-31 | Section: Revenue]`
- Oversized sections are split into word-budget pages before accumulation, maintaining overlap continuity

**Stage 3 — Child chunks** (~128 tokens)
- Sentence boundaries are respected — no mid-sentence splits via `_split_into_sentences()`
- 32-token overlap between consecutive children
- Contextual prefix re-applied to every child (enables context injection at embedding time)
- Table parents produce exactly one child with identical text

**Why parent/child?**

| Concern | Solution |
|---------|---------|
| Embedding precision | Small 128-token children → more precise dense retrieval |
| Generation context | 512-token parents → richer context for LLM answer synthesis |
| Cost control | Embed children only; fetch parents lazily at retrieval time |
| Boilerplate dilution | Section boundaries prevent legal disclaimers contaminating financial content |

**Chunk ID determinism**: IDs are generated via `uuid5(NAMESPACE_DNS, f"{ticker}:{date}")` — the same chunk always produces the same Qdrant point ID, making upsert operations idempotent.

### BM25 Corpus Invariant

The two files `bm25_index.pkl` and `bm25_corpus.pkl` maintain a strict parallel-array invariant:

```
bm25_index.corpus[i]  ←→  bm25_corpus[i]  (always same length, same order)
```

The pipeline validates this invariant before writing. Breaking it would cause the retrieval layer to resolve BM25 rank indices to wrong chunk metadata.

---

## Layer 2: Query Transformation

### Motivation

The **query-document semantic gap** is the core challenge in RAG: a user's natural language question lives in a different part of embedding space than a formal 8-K press release passage. Layer 2 bridges this gap with three complementary techniques.

### Architecture

```
user question: "What was Apple's revenue in Q4 2024?"
        │
        ├──▶ [Thread 1] _run_hyde()         → hypothetical answer passage
        ├──▶ [Thread 2] _run_multi_query()  → 3 rephrasings
        └──▶ [Thread 3] _run_stepback()     → abstract question
                │
                │ as_completed()  — each result stored on return
                ▼
        TransformedQuery {
            original        = "What was Apple's revenue in Q4 2024?"
            hyde_document   = "Apple Inc. reported total net sales of $X billion for..."
            multi_queries   = [original, rephrasing1, rephrasing2, rephrasing3]
            stepback_query  = "What is Apple's revenue breakdown by segment?"
            failed_techniques = []  # populated only on partial failure
        }
```

### Technique Design

**HyDE (Hypothetical Document Embeddings)**

Generates a passage that mimics an actual 8-K exhibit in register and vocabulary. When embedded with fastembed, this synthetic passage maps into the same region of embedding space as real document chunks — closing the semantic gap at the source.

Temperature: `0.3` — moderate creativity to generate plausible-sounding passages without hallucinating too wildly.

**Multi-Query**

Generates 3 rephrasings with deliberately varied vocabulary axes:
- Version 1: Formal analyst language (`revenue → net revenue`, `guidance → forward outlook`)
- Version 2: Management commentary style (`what did management say about X`)
- Version 3: Short keyword-style (`AAPL revenue Q4 2024`)

Temperature: `0.7` — higher variance ensures the rephrasings actually differ in vocabulary.

**Step-Back Prompting**

Generates a broader, more abstract question. Purpose: retrieve foundational context chunks that the specific question would miss — segment definitions, methodology notes, management commentary on strategy.

Temperature: `0.1` — near-deterministic, same question should produce same abstraction.

### Graceful Degradation

All three techniques run in a `ThreadPoolExecutor(max_workers=3)`. If any technique fails:
- The failed technique's output falls back to the original query
- `failed_techniques` list is populated (visible in `/query?verbose=true` response and logs)
- Pipeline execution always completes — partial degradation is not a fatal error

### In-Memory LRU Cache

A simple dict-based LRU cache (size: `RAG_QUERY_TRANSFORM_CACHE_SIZE`, default 256) keyed by `sha256(query.strip().lower())`. Eliminates redundant LLM calls for:
- Repeated questions in evaluation harness runs
- CRAG re-generation loops on the same question
- UI demos with repeated queries

---

## Layer 3: Hybrid Retrieval

### Search Strategy

```
TransformedQuery
        │
        ├──▶ Dense: hyde_document        → 1× Qdrant search (10 results)
        ├──▶ Dense: multi_queries[0..3]  → 4× Qdrant searches (10 each)
        ├──▶ Dense: stepback_query       → 1× Qdrant search (10 results)
        ├──▶ BM25:  multi_queries[0..3]  → 4× BM25 searches (10 each)
        └──▶ BM25:  stepback_query       → 1× BM25 search (10 results)
                │
                │ Total raw pool: up to 6×10 dense + 5×10 BM25 = ~110 hits
                │ After deduplication: 30–60 unique chunks
                ▼
        RRF Fusion  (k=60, standard default from Cormack et al. 2009)
                │
                │ score(chunk) = Σ 1/(60 + rank_i) across all result lists
                ▼
        Top 20 candidates (top_k_pre_rerank)  →  FlashRank reranker
                │
                ▼
        Top 5 results (top_k_final)  →  Late parent fetch
                │
                ▼
        list[SearchResult] with parent_text populated
```

### RRF Fusion

Reciprocal Rank Fusion scores each unique chunk across all result lists:

```python
score(chunk_id) = sum(1.0 / (k + rank_i) for rank_i in all_rankings_containing_chunk_id)
```

`k=60` is the standard default. Chunks appearing in multiple result lists (both dense and BM25) accumulate higher scores, naturally promoting robust matches.

### FlashRank Reranking

After RRF, the top `top_k_pre_rerank=20` candidates are passed to FlashRank's `ms-marco-MiniLM-L-12-v2` cross-encoder:

- **Model**: 12-layer MiniLM (~66 MB ONNX), fully local, no API cost
- **Input**: `(query_original, parent_text_or_child_text)` pairs
- **Output**: Relevance scores from 0–1 (higher = more relevant)
- **Latency**: ~8–15 ms for 20 candidates on CPU

The cross-encoder is significantly more accurate than cosine similarity for relevance scoring because it attends to interactions between the query and document tokens, not just their independent embeddings.

When disabled (`RAG_RERANKER_ENABLED=false`), results fall through sorted by RRF score.

### Late Parent Fetch

After reranking determines the final `top_k_final=5` child chunks, a **single batch Qdrant scroll** fetches all corresponding parent chunks:

```python
# One batch call, not N individual lookups
scroll_result, _ = client.scroll(
    collection_name=...,
    scroll_filter=Filter(must=[FieldCondition(key="chunk_id", match=MatchAny(any=parent_ids))]),
    limit=len(parent_ids) + 10,
)
```

This replaces each child's 128-token text with its 512-token parent text. The generation layer receives full context without paying the cost of embedding large parent chunks.

### Metadata Filtering

`MetadataFilter(ticker, year, quarter)` is applied at both Qdrant (server-side filter pushdown) and BM25 (Python-side post-filter) levels. Qdrant payload indices are created during `init_qdrant()` for `ticker` (keyword), `year` (integer), and `quarter` (keyword), ensuring O(log n) filtered queries.

---

## Layer 4: Answer Generation

### Context Window Construction

```python
# generation/context_builder.py

1. Deduplicate by parent_id
   (two children sharing a parent → keep higher rerank_score one)

2. Valley reorder  (lost-in-the-middle mitigation)
   even-indexed ranks → front of context
   odd-indexed ranks  → back of context, reversed
   → rank-1 at position 0, rank-2 at last position

3. Greedy token budget allocation
   add blocks until max_context_tokens (4096) exhausted
   if first chunk alone exceeds budget → hard truncate to fit

4. Format as numbered [1]..[N] blocks
   "--- [1] AAPL | Q4 2024 | Revenue ---\n<parent_text>"
```

**Lost-in-the-Middle Mitigation**: Based on Liu et al. (2023), LLM attention follows a U-shaped pattern — strong at start and end of context, weak in the middle. Valley ordering ensures rank-1 (most relevant) occupies position 0 (highest attention) and rank-2 occupies the last position (second-highest attention).

### Citation Contract

The generation system prompt establishes a strict citation contract with the LLM:

```
Every factual claim MUST be followed immediately by an inline citation: [1], [2], etc.
For claims supported by multiple sources: [1][2]  (no space, no comma)
Do NOT invent a citation number that does not appear in the provided context.
```

Post-generation, `_extract_citations()` uses regex `\[(\d+)\]` to scan the answer and map each cited index to its corresponding `SearchResult`. Out-of-range indices (hallucinated citations) are logged with a warning and skipped — never crash.

### Grounding Check

`_is_grounded()` scans the answer text for 13 phrases that signal insufficient context:

```python
UNGROUNDED_PHRASES = (
    "do not contain sufficient information",
    "cannot determine",
    "not mentioned in",
    ...
)
```

`GenerationResult.grounded = False` signals downstream layers (CRAG, API) that a web-search fallback may be appropriate.

### Retry Strategy

`_call_llm()` uses `tenacity` with exponential backoff:
- Retries on: `RateLimitError`, `APITimeoutError`
- Propagates immediately on: 4xx `APIError` (unrecoverable — retrying wastes money)
- Max retries: 3 (configurable via `RAG_GENERATION_MAX_RETRIES`)

---

## Layer 5: Corrective RAG

### Decision Tree

```
                    ┌─────────────────────────────┐
                    │  GenerationResult received   │
                    └─────────────┬───────────────┘
                                  │
                   ┌──────────────▼──────────────┐
                   │ crag.enabled = False?        │
                   │ YES → return CORRECT         │
                   └──────────────┬──────────────┘
                                  │ NO
                   ┌──────────────▼──────────────┐
                   │ grounded=True AND            │
                   │ grade_even_if_grounded=False? │
                   │ YES → return CORRECT (fast)  │
                   └──────────────┬──────────────┘
                                  │ NO
                   ┌──────────────▼──────────────┐
                   │  RelevanceGrader             │
                   │  grade all retrieved chunks  │
                   │  (concurrent ThreadPool)     │
                   └──────────────┬──────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
        ratio≥0.6           0.2<ratio<0.6         ratio≤0.2
              │                   │                   │
          CORRECT             AMBIGUOUS           INCORRECT
              │                   │                   │
      return original        web search          web search
        gen_result          local+web chunks     web only
                                   │                   │
                             Generator.generate()      │
                                   └────────┬──────────┘
                                            │
                                     CRAGResult
```

### Relevance Grader

`RelevanceGrader` sends one LLM call per chunk concurrently via `ThreadPoolExecutor`. Each call asks the LLM to evaluate whether the chunk **directly** helps answer the question (not just tangentially related). JSON response: `{"relevant": bool, "score": float, "reasoning": str}`.

**Fail-open design**: Any grader error (API timeout, parse failure) defaults to `relevant=True, score=0.5`. This ensures the pipeline degrades gracefully — a grader failure never silently discards potentially useful context.

### Web Search Abstraction

Provider auto-detection at init time:
1. **Tavily** — if `TAVILY_API_KEY` is set (`poetry install --extras crag-tavily`)
2. **DuckDuckGo** — free fallback, no API key, rate-limited (`poetry install --extras crag-ddg`)

Web results are converted to `SearchResult` objects via `_web_to_search_result()` and flow through the existing `Generator.generate()` without modification. The `chunk_id` is `web:<md5_hash_of_url>` and `doc_type="web"`.

---

## API Layer

### Middleware Stack

Registration order (inner → outer on request):

```
PrometheusMiddleware      ← records metrics for every route
RequestIDMiddleware       ← stamps X-Request-Id (from header or generates UUID4)
TimingMiddleware          ← measures latency, adds X-Response-Time-Ms
CORSMiddleware            ← handles preflight OPTIONS
```

**Why pure ASGI (not BaseHTTPMiddleware)?**

`BaseHTTPMiddleware` uses `anyio.create_task_group()` internally. When a route raises and the exception handler sends a 500 response, the inner task group re-raises via `ExceptionGroup` / `collapse_excgroups()` — crashing `TestClient` instead of returning the 500. Pure ASGI middleware wraps `send()` directly and never participates in exception propagation.

### Threading Model

The FastAPI event loop is async; the RAG pipeline is entirely synchronous (blocking). Every pipeline call is offloaded to a module-level `ThreadPoolExecutor(max_workers=4)`:

```python
loop = asyncio.get_running_loop()
result = await loop.run_in_executor(_THREAD_POOL, pipeline.ask, ...)
```

`asyncio.get_running_loop()` is used (not `get_event_loop()`) — the latter is deprecated in Python 3.10+ when called inside a running event loop.

### SSE Streaming Architecture

```
Producer thread (_produce)              Consumer coroutine (_consume)
─────────────────────────────           ─────────────────────────────
pipeline.ask_streaming()                asyncio.Queue(maxsize=128)
  for token in generator:                 while True:
    queue.put(json_payload)                 item = await queue.get(timeout=60)
                                            if item is None: break
  queue.put(None)  # sentinel              yield f"data: {item}\n\n"
                                         yield "data: [DONE]\n\n"
```

`asyncio.Queue(maxsize=128)` provides backpressure — if the client reads slowly, the producer thread blocks until the queue drains, preventing memory accumulation.

**Fix**: The producer future is stored and cancelled on consumer exit — preventing thread leaks if the client disconnects mid-stream.

### Health Check Hierarchy

| Endpoint | Depth | Kubernetes Use | Latency |
|----------|-------|---------------|---------|
| `/health/live` | Process alive check | Liveness probe | <1 ms |
| `/health/ready` | Pipeline singleton check | Readiness probe | <1 ms |
| `/health/` | Full dependency probe | Dashboard / alerting | ~50–200 ms |

The readiness probe accurately reflects model loading state — during the 10–20s startup window, `/health/ready` returns 503, causing Kubernetes to hold traffic until models are loaded.

### Exception Handler Mapping

| Exception | HTTP Status | Notes |
|-----------|------------|-------|
| `RequestValidationError` | 422 | Pydantic field validation |
| `ValueError` | 400 | Domain errors (unknown ticker, empty question) |
| `AuthenticationError` | 401 | Invalid OpenAI API key |
| `RateLimitError` | 429 | `Retry-After: 10` header added |
| `FileNotFoundError` | 503 | BM25 index or Qdrant collection missing |
| `APITimeoutError` | 504 | OpenAI timeout after all retries |
| `APIConnectionError` | 502 | OpenAI network error |
| `Exception` (catch-all) | 500 | Unexpected errors — full traceback in logs |

All error responses share the same JSON shape:
```json
{"error": "<category>", "detail": "<message>", "request_id": "<uuid>"}
```

### Prometheus Metrics

The system uses a **dedicated `RAG_REGISTRY`** (not `prometheus_client.REGISTRY`). This prevents "Duplicated timeseries" errors when pytest creates multiple `TestClient` instances per session, each triggering `create_app()`. The `/metrics` endpoint serves only this custom registry.

---

## Configuration System

All configuration lives in `config/settings.py` as frozen `@dataclass` classes. Every field defaults to a sensible production value but is overridable via environment variable.

```
Settings
  ├── QueryTransformConfig    (RAG_QUERY_TRANSFORM_*)
  ├── GenerationConfig        (RAG_GENERATION_*)
  ├── EmbeddingConfig         (RAG_EMBEDDING_*)
  ├── RetrievalConfig         (RAG_RETRIEVAL_*)
  ├── RerankerConfig          (RAG_RERANKER_*)
  ├── InfraConfig             (QDRANT_URL, OPENAI_API_KEY, SEC_USER_AGENT)
  ├── CRAGConfig              (RAG_CRAG_*)
  └── EvaluationConfig        (RAG_EVAL_*)
```

`settings = Settings()` is a module-level singleton — imported by every other module with `from config import settings`. `settings.validate()` is called at startup and raises `OSError` on missing required values, causing a fast fail before accepting any traffic.

---

## Data Contracts

### Key Dataclass Hierarchy

```
TransformedQuery           (query/models.py)
  └── all_retrieval_queries: list[str]   (property, deduped)

SearchResult               (retrieval/models.py)
  ├── chunk_id, parent_id
  ├── text (child, 128 tokens)
  ├── parent_text (512 tokens, populated after parent fetch)
  ├── rrf_score (pre-rerank)
  └── rerank_score (post-rerank, float("-inf") before)

RetrievalResult            (retrieval/models.py)
  ├── results: list[SearchResult]
  ├── reranked: bool
  └── is_empty: bool  (property)

Citation                   (generation/models.py)
  ├── index (1-based, matches [N] in answer text)
  └── excerpt (first 250 chars of parent_text)

GenerationResult           (generation/models.py)
  ├── answer (with inline [N] citations)
  ├── citations: list[Citation]
  ├── grounded: bool
  ├── unique_sources: list[str]  (property)
  └── format_answer_with_citations(): str

CRAGResult                 (crag/models.py)
  ├── action: CRAGAction (CORRECT/AMBIGUOUS/INCORRECT)
  ├── final_result: GenerationResult
  ├── was_corrected: bool  (property)
  └── relevance_ratio: float  (property)
```

---

## Concurrency Model

| Component | Mechanism | Notes |
|-----------|-----------|-------|
| Query transformation | `ThreadPoolExecutor(max_workers=3)` | HyDE + Multi-Query + Step-Back concurrent |
| API request handling | `asyncio` + `ThreadPoolExecutor(max_workers=4)` | Blocks event loop never |
| SSE producer | Background thread via `run_in_executor` | Bounded queue with backpressure |
| Relevance grading | `ThreadPoolExecutor(max_workers=5)` | One LLM call per chunk, concurrent |
| Evaluation harness | `ThreadPoolExecutor(max_workers=2)` | Parallel pipeline calls |
| BM25 search | Python GIL-protected (single thread) | No concurrency needed |
| Qdrant search | Thread-safe (qdrant-client is thread-safe) | |
| fastembed | Thread-safe after first init | Module-level singleton |
| FlashRank | Thread-safe after first init | Module-level singleton |

---

## Design Decisions & Trade-offs

### Why fastembed + BAAI/bge-large-en-v1.5 (not OpenAI text-embedding-3)?

| Concern | fastembed/bge | OpenAI embeddings |
|---------|--------------|------------------|
| Cost | Free (local ONNX) | ~$0.13/1M tokens |
| Latency | CPU: ~50 ms/batch | Network: ~100–500 ms |
| Privacy | No data leaves machine | Data sent to OpenAI |
| Offline | Works without internet | Requires connectivity |
| Quality | 1024-dim, MTEB competitive | Excellent |

For a project ingesting thousands of chunks and running frequent re-indexing, free local embeddings eliminate cost uncertainty while maintaining retrieval quality.

### Why BM25 + Dense (not dense-only)?

Dense embeddings capture semantic meaning but miss **exact keyword matches** — critical for financial queries containing specific metrics, ticker symbols, and fiscal period identifiers. BM25 excels at these. RRF fusion gives each signal appropriate weight without requiring tuned combination coefficients.

### Why Parent/Child (not fixed-size chunks)?

Fixed-size chunking splits financial tables and section context arbitrarily. Parent/child allows:
- **Precision retrieval** with small children (128 tokens ≈ 1–2 financial sentences)
- **Rich generation context** with large parents (512 tokens ≈ full paragraph + header)
- **Table atomicity** — tables are never split

### Why gpt-4.1-nano for all LLM calls?

At `$0.10/1M input, $0.40/1M output` tokens, nano-tier models are sufficient for all three LLM use cases in this pipeline:
- Query transformation (short instruction-following tasks, ~120–350 input tokens)
- Answer generation (precise financial Q&A, 2000–5000 input tokens)
- Evaluation metrics (structured JSON scoring, ~200 input tokens)

Higher-tier models would provide marginal quality gains at 5–10× cost.

### Why not Ragas for evaluation?

Ragas is an excellent framework but has frequent API changes between versions. The custom evaluation harness provides:
- Identical metric definitions (faithfulness, answer_relevancy, context_precision, context_recall)
- Full control over prompt design tuned for financial domain
- No external dependency that could break CI
- Direct integration with the existing OpenAI client singleton

Ragas integration is on the roadmap as an optional alternative once the API stabilises.

### Why CRAG (not naive RAG)?

Naive RAG silently returns hallucinated answers when retrieval quality is poor. CRAG provides:
- **Explicit quality signal**: the grounding flag triggers corrective action
- **Graceful degradation**: web search fallback ensures useful responses even for out-of-scope companies
- **Diagnostic visibility**: `action`, `relevance_ratio`, `web_search_triggered` are all observable

The cost is ~N additional LLM calls for grading (N = number of retrieved chunks, typically 5). At gpt-4.1-nano pricing, this adds ~$0.0005 per query — acceptable for production use.
