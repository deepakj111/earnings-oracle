# API Reference

> Complete reference for all Financial RAG System REST endpoints.

Base URL: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs` (Swagger UI)
Alternative docs: `http://localhost:8000/redoc` (ReDoc)

---

## Authentication

No authentication is required for local/development deployments. For production, add API key validation middleware before deploying publicly.

All responses include:
- `X-Request-Id` — correlation ID (UUID4, or echoed from incoming `X-Request-ID` header)
- `X-Response-Time-Ms` — end-to-end request latency in milliseconds

---

## Endpoints

### Query

#### `POST /query`

Run the full four-layer RAG pipeline and return a structured JSON response with inline citations, token usage, and grounding diagnostics. Accepts requests directly without trailing slash redirects.

**Pipeline layers executed**: L2 Query Transform → L3 Hybrid Retrieval → L4 Answer Generation

**Request body**

```json
{
  "question": "What was NVIDIA's Data Center revenue in fiscal year 2025?",
  "filter": {
    "ticker": "NVDA",
    "year": 2025
  },
  "session_id": "sess_a1b2c3d4",
  "verbose": false
}
```

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `question` | string | ✅ | 3–2000 chars | Natural language financial question |
| `filter` | object | ❌ | — | Optional scope filter (all fields optional) |
| `filter.ticker` | string | ❌ | Any configured ticker (e.g. NVDA, WMT, NFLX, UNH, AAPL, MSFT) | Company ticker (case-insensitive, dynamic via CompanyRegistry) |
| `filter.year` | integer | ❌ | 2020–2030 | Fiscal year |
| `filter.quarter` | string | ❌ | Q1, Q2, Q3, Q4 | Fiscal quarter (case-insensitive) |
| `session_id` | string | ❌ | Max 128 chars | Chat session ID for conversational memory and follow-up turns |
| `verbose` | boolean | ❌ | default: `false` | Include query transform + retrieval diagnostics in response |

**Response `200 OK`**

```json
{
  "question": "What was NVIDIA's Data Center revenue in fiscal year 2025?",
  "answer": "NVIDIA reported fiscal year 2025 Data Center revenue of $115.2 billion [1], an increase of 142% year-over-year [1][2].",
  "citations": [
    {
      "index": 1,
      "ticker": "NVDA",
      "company": "NVIDIA",
      "date": "2025-01-26",
      "fiscal_period": "FY2025",
      "section_title": "Segment Results",
      "doc_type": "10-K",
      "source": "both",
      "rerank_score": 0.9821,
      "excerpt": "Data Center revenue for fiscal year 2025 was $115.2 billion, up 142% from a year ago..."
    },
    {
      "index": 2,
      "ticker": "NVDA",
      "company": "NVIDIA",
      "date": "2025-01-26",
      "fiscal_period": "FY2025",
      "section_title": "Financial Highlights",
      "doc_type": "10-K",
      "source": "dense",
      "rerank_score": 0.9412,
      "excerpt": "Total revenue for fiscal year 2025 was $130.5 billion, driven primarily by hyper-scale demand for compute..."
    }
  ],
  "grounded": true,
  "confidence_score": 0.962,
  "retrieval_failed": false,
  "model": "gemini-2.5-flash",
  "usage": {
    "prompt_tokens": 2456,
    "completion_tokens": 87,
    "total_tokens": 2543
  },
  "context": {
    "chunks_used": 5,
    "tokens_used": 2048
  },
  "latency_seconds": 2.142,
  "unique_tickers": ["NVDA"],
  "unique_sources": ["NVDA FY2025"],
  "calculations": [
    {
      "expr": "growth(47.5, 115.2)",
      "res": 142.53,
      "fmt": "142.53",
      "ok": true
    }
  ],
  "numerical_hallucination_warnings": [],
  "citation_integrity_warnings": [],
  "reflexion_attempts": 0,
  "query_summary": null,
  "retrieval_summary": null
}
```

**Response fields**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | LLM-synthesised answer with inline `[N]` citations |
| `citations` | array | Structured metadata for each citation number used in answer |
| `citations[].index` | integer | 1-based citation number matching `[N]` in answer text |
| `citations[].source` | string | `"dense"` \| `"bm25"` \| `"both"` \| `"facts"` \| `"knowledge_graph"` — retrieval system that surfaced this chunk |
| `citations[].rerank_score` | float | FlashRank cross-encoder relevance score (higher = more relevant) |
| `citations[].excerpt` | string | First 250 chars of source passage for compact UI cards |
| `citations[].full_text` | string | Full text of retrieved context chunk for evaluation and deep inspection |
| `grounded` | boolean | `false` if model signalled insufficient context or verification failed — triggers calibrated abstention |
| `confidence_score` | float | Calibrated composite confidence score ($0.0–1.0$, clamped to $0.0$ on abstention) |
| `retrieval_failed` | boolean | `true` if zero documents were retrieved from the index |
| `unique_tickers` | array | Deduplicated tickers cited in answer, in citation order |
| `unique_sources` | array | `"TICKER fiscal_period"` labels, e.g. `["NVDA FY2025"]` |
| `calculations` | array | Executed deterministic PAL math formulas and verified outputs |
| `numerical_hallucination_warnings` | array | Quantitative figures flagged by `NumericalHallucinationFence` |
| `citation_integrity_warnings` | array | Entity mismatches flagged by `CitationIntegrityValidator` |
| `reflexion_attempts` | integer | Number of Agentic Reflexion self-correction cycles executed |
| `query_summary` | string\|null | Query transform diagnostics (only when `verbose=true`) |
| `retrieval_summary` | string\|null | Retrieval diagnostics with scores (only when `verbose=true`) |

**Verbose response** (`verbose: true`)

`query_summary` contains:
```
Original      : What was Apple's total revenue in Q4 2024?
HyDE doc      : Apple Inc. reported total net sales of $94.9 billion for Q4 fiscal...
Multi-queries : 4 total
  [1] What was Apple's total revenue in Q4 2024?
  [2] What were Apple's net revenues for the fourth quarter of fiscal year 2024?
  [3] What did management report about Apple's total net sales in Q4 2024?
  [4] AAPL Q4 2024 total revenue results
Step-Back     : What is Apple's revenue breakdown by segment and growth trends?
```

**Error responses**

| Status | Condition |
|--------|-----------|
| `400 Bad Request` | Empty question, unknown ticker in filter |
| `422 Unprocessable Entity` | Pydantic validation failure (question too short/long, invalid quarter) |
| `429 Too Many Requests` | OpenAI rate limit exceeded — retry after `Retry-After: 10` seconds |
| `503 Service Unavailable` | BM25 index or Qdrant collection not found — run ingestion pipeline |
| `504 Gateway Timeout` | OpenAI API timed out after all retries |
| `502 Bad Gateway` | Cannot connect to OpenAI API |
| `500 Internal Server Error` | Unexpected error — check server logs |

All error responses follow this shape:
```json
{
  "error": "Rate limit exceeded",
  "detail": "The LLM provider is rate-limiting this service. Retry in a moment.",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

---

#### `POST /query/stream`

Streaming variant. Runs L2 + L3 synchronously, then streams L4 answer tokens as Server-Sent Events.

**Request body**: Same as `POST /query/` (the `verbose` field is ignored in streaming mode).

**Response**: `text/event-stream`

SSE message format uses typed JSON frames:

```
data: {"log": "Transforming query using HyDE and multi-query..."}

data: {"token": "Apple"}

data: {"token": " reported"}

data: {"token": " $94.9B"}

data: {"type": "done", "grounded": true, "citations": [{"index": 1, "ticker": "AAPL", "fiscal_period": "Q4 2024", "section_title": "Revenue", "excerpt": "..."}], "trace_id": "a1b2c3d4-..."}

data: [DONE]
```

Error events (stream still terminates with `[DONE]`):

```
data: {"error": "Rate limit exceeded. Please retry in a moment."}

data: [DONE]
```

Response headers:
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
X-Request-ID: <uuid>
```

**Note**: Streaming clients receive real-time token events and progress logs, followed by a final structured `{"type": "done"}` frame containing citation cards and grounding status. Use `POST /query` when immediate full structured JSON is required.

##### Streaming Limitations & Compliance Guidance

> [!WARNING]
> **Streaming Mode Architectural Trade-offs**:
> While `POST /query/stream` achieves minimal Time-To-First-Token (TTFT < 250ms), clients should note two intentional architectural degradations relative to the non-streaming `POST /query` endpoint:
> 1. **No Agentic Reflexion Self-Correction**: Tokens stream directly to the client as they are sampled from the LLM. If the post-generation Grounding Verifier or Numerical Hallucination Fence flags an ungrounded claim or arithmetic inconsistency, the pipeline flags the terminal `{"type": "done"}` payload (`grounded: false`, `numerical_hallucination_warnings: [...]`), but cannot recall or rewrite already-emitted tokens.
> 2. **Single-Entity Retrieval Scope**: Streaming mode bypasses multi-subquery decomposition (ADR-014) and comparative candidate interleaving (ADR-012) in favor of linear hybrid search latency.
>
> **Recommendation**: Use `POST /query/stream` for interactive conversational dashboards. Use `POST /query` for programmatic trading systems, quantitative analysis, regulatory compliance audits, and multi-company comparative evaluations.


**Example consumption (JavaScript)**:

```javascript
const response = await fetch('/query/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: "What was Apple's revenue?" })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const text = decoder.decode(value);
  for (const line of text.split('\n')) {
    if (line.startsWith('data: ')) {
      const payload = line.slice(6);
      if (payload === '[DONE]') return;
      const event = JSON.parse(payload);
      if (event.token) process.stdout.write(event.token);
    }
  }
}
```

**Example consumption (Python)**:

```python
import requests, json

with requests.post(
    "http://localhost:8000/query/stream",
    json={"question": "What was NVDA data center revenue?"},
    stream=True,
) as resp:
    for line in resp.iter_lines():
        if not line:
            continue
        data = line.decode("utf-8")
        if data.startswith("data: "):
            payload = data[6:]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if "token" in event:
                print(event["token"], end="", flush=True)
```

---

#### `POST /query/cache/invalidate`

Invalidate cached semantic query responses stored in Qdrant. Allows targeted invalidation by ticker (e.g. after newly ingested quarterly or annual reports) or full flush of the semantic cache.

**Query Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|:---:|---------|-------------|
| `ticker` | string | ❌ | `null` | Optional company ticker (e.g. `NVDA`). If provided, invalidates all cached responses referencing that ticker. If omitted, flushes the entire cache. |

**Response `200 OK` (Targeted Ticker Invalidation)**

```json
{
  "status": "success",
  "ticker": "NVDA",
  "deleted": 1
}
```

**Response `200 OK` (Full Cache Flush)**

```json
{
  "status": "success",
  "message": "Semantic cache flushed completely"
}
```

---

### Companies

#### `GET /companies/` (or `GET /companies`)

List all configured public companies from the dynamic `CompanyRegistry` (`config/companies.json`). Returns metadata including ticker, corporate name, SEC CIK, industry sector, fiscal year-end month, and brand/subsidiary aliases.

Used by the Web UI (`loadCompanies()`) to dynamically populate company selection dropdowns without hardcoding tickers in frontend code.

**Response `200 OK`**

```json
[
  {
    "ticker": "AAPL",
    "name": "Apple",
    "cik": "0000320193",
    "sector": "Technology / Consumer Electronics",
    "fiscal_year_end_month": 9,
    "download_start_date": "2024-01-01",
    "default_portfolio": false,
    "aliases": ["apple", "iphone", "ipad", "macbook"]
  },
  {
    "ticker": "NVDA",
    "name": "NVIDIA",
    "cik": "0001045810",
    "sector": "Technology / Semiconductors",
    "fiscal_year_end_month": 1,
    "download_start_date": "2024-01-01",
    "default_portfolio": true,
    "aliases": ["nvidia", "geforce", "mellanox", "nvd"]
  }
]
```

---

#### `GET /companies/{ticker}`

Retrieve the detailed profile and fiscal metadata for a specific company ticker.

**Path parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ticker` | string | ✅ | Company ticker symbol (case-insensitive, e.g. `NVDA`, `aapl`) |

**Response `200 OK`**

```json
{
  "ticker": "NVDA",
  "name": "NVIDIA",
  "cik": "0001045810",
  "sector": "Technology / Semiconductors",
  "fiscal_year_end_month": 1,
  "download_start_date": "2024-01-01",
  "default_portfolio": true,
  "aliases": ["nvidia", "geforce", "mellanox", "nvd"]
}
```

**Response `404 Not Found`**

```json
{
  "detail": "Company 'XYZ' is not configured in the registry."
}
```

---

### Sessions

Multi-user conversational chat history, session persistence, and thread management for enterprise AI chatbot workflows.

All session endpoints support user and tenant scoping via headers:
- `X-User-ID`: Scopes operations to the specified user (defaults to `default_user`).
- `X-Tenant-ID`: Scopes operations to the specified tenant organization (defaults to `default_tenant`).

#### `GET /sessions`

Retrieve all active chat sessions for the current authenticated or guest user, ordered by most recently updated.

**Query Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|:---:|---------|-------------|
| `limit` | integer | ❌ | `50` | Maximum number of sessions to return (1–100) |

**Response `200 OK`**

```json
[
  {
    "id": "sess_8f3d1e2a",
    "user_id": "default_user",
    "title": "NVIDIA FY2025 Data Center Analysis",
    "message_count": 4,
    "created_at": "2026-09-25T03:12:00Z",
    "updated_at": "2026-09-25T03:15:22Z",
    "metadata": {
      "ticker": "NVDA",
      "year": 2025
    }
  }
]
```

---

#### `POST /sessions`

Initialize a new conversational session for multi-turn financial analysis.

**Request body**

```json
{
  "title": "Walmart vs Netflix Comparative Study",
  "metadata": {
    "primary_ticker": "WMT"
  }
}
```

| Field | Type | Required | Constraints | Description |
|-------|------|:---:|-------------|-------------|
| `title` | string | ❌ | 1–255 chars | Optional display title (auto-generated if omitted) |
| `metadata` | object | ❌ | JSON map | Optional user-defined session attributes |

**Response `201 Created`**

```json
{
  "id": "sess_9c4b2a11",
  "user_id": "default_user",
  "title": "Walmart vs Netflix Comparative Study",
  "messages": [],
  "created_at": "2026-09-25T04:00:00Z",
  "updated_at": "2026-09-25T04:00:00Z",
  "metadata": {
    "primary_ticker": "WMT"
  }
}
```

---

#### `GET /sessions/{session_id}`

Retrieve the full conversational message history, verified calculations, and citation cards for a specific session ID.

**Path parameters**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `session_id` | string | ✅ | Session identifier |

**Response `200 OK`**

```json
{
  "id": "sess_9c4b2a11",
  "user_id": "default_user",
  "title": "Walmart vs Netflix Comparative Study",
  "messages": [
    {
      "role": "user",
      "content": "What was Walmart's global revenue in FY2026?",
      "timestamp": "2026-09-25T04:01:00Z",
      "citations": [],
      "calculations": [],
      "grounded": null,
      "confidence_score": null
    },
    {
      "role": "assistant",
      "content": "Walmart reported total revenue of $681.0 billion for fiscal year 2026 [1]...",
      "timestamp": "2026-09-25T04:01:03Z",
      "citations": [
        {
          "index": 1,
          "ticker": "WMT",
          "fiscal_period": "FY2026",
          "section_title": "Financial Highlights",
          "excerpt": "Total revenue reached $681.0 billion..."
        }
      ],
      "calculations": [],
      "grounded": true,
      "confidence_score": 0.95
    }
  ],
  "created_at": "2026-09-25T04:00:00Z",
  "updated_at": "2026-09-25T04:01:03Z",
  "metadata": {}
}
```

---

#### `PATCH /sessions/{session_id}`

Update the display title of an active chat session.

**Request body**

```json
{
  "title": "Walmart FY2026 Segment Margins"
}
```

**Response `200 OK`**

```json
{
  "id": "sess_9c4b2a11",
  "user_id": "default_user",
  "title": "Walmart FY2026 Segment Margins",
  "messages": [...],
  "created_at": "2026-09-25T04:00:00Z",
  "updated_at": "2026-09-25T04:05:00Z",
  "metadata": {}
}
```

---

#### `DELETE /sessions/{session_id}`

Permanently delete a chat session and its complete conversational history.

**Response `204 No Content`**

---

#### `POST /sessions/{session_id}/clear`

Clear all message history in a session while preserving the session ID, title, and metadata.

**Response `200 OK`**

```json
{
  "status": "cleared",
  "session_id": "sess_9c4b2a11",
  "message_count": 0
}
```

---

### Web Frontend

#### `GET /app` (or `GET /`)

Serves the modern single-page HTML / CSS / JavaScript chat application. Provides interactive conversation history, ticker and fiscal period filters, and citation inspection cards.

**Response `200 OK`**: `text/html; charset=utf-8`

---

### Health

#### `GET /health/live`

Kubernetes liveness probe. Returns 200 immediately if the process is running. Never touches external dependencies.

**Response `200 OK`**
```json
{"status": "alive"}
```

---

#### `GET /health/ready`

Kubernetes readiness probe. Returns 200 only when the `FinancialRAGPipeline` singleton is fully initialised (models loaded, BM25 warm). Returns 503 during the cold-start window.

**Response `200 OK`** (ready)
```json
{"status": "ready"}
```

**Response `503 Service Unavailable`** (model loading in progress)
```json
{"detail": "Pipeline not yet initialised — retry shortly."}
```

---

#### `GET /health/`

Full dependency health check. Actively probes Qdrant connectivity, collection existence, pipeline availability, and BM25 index presence.

**Response `200 OK`**
```json
{
  "status": "healthy",
  "version": "0.7.0",
  "uptime_seconds": 3627.4,
  "components": {
    "qdrant": {
      "status": "ok",
      "detail": "collection 'earnings_transcripts' present (18432 points)"
    },
    "pipeline": {
      "status": "ok",
      "detail": "generation=gemini-2.5-flash | transform=gemini-2.5-flash"
    },
    "bm25_index": {
      "status": "ok",
      "detail": "data/bm25_index.pkl (8.3 MB)"
    }
  }
}
```

**Status values**:
- `"healthy"` — all components OK
- `"degraded"` — some components unavailable (partial service)
- `"unhealthy"` — pipeline unavailable (no queries can be served)

---

### Observability

#### `GET /metrics`

Prometheus text-format metrics. Scraped by Prometheus every 15 seconds (configured in `prometheus.yml`).

**Response**: `text/plain; version=0.0.4`

```
# HELP rag_http_requests_total Total HTTP requests handled by the RAG API
# TYPE rag_http_requests_total counter
rag_http_requests_total{endpoint="/query",method="POST",status_code="200"} 142.0
rag_http_requests_total{endpoint="/health/live",method="GET",status_code="200"} 1847.0

# HELP rag_grounded_responses_total Responses classified as grounded vs ungrounded
# TYPE rag_grounded_responses_total counter
rag_grounded_responses_total{grounded="true"} 138.0
rag_grounded_responses_total{grounded="false"} 4.0

# HELP rag_pipeline_latency_seconds Per-layer pipeline latency in seconds
# TYPE rag_pipeline_latency_seconds histogram
rag_pipeline_latency_seconds_bucket{layer="L2",le="0.25"} 0.0
rag_pipeline_latency_seconds_bucket{layer="L2",le="0.5"} 3.0
rag_pipeline_latency_seconds_bucket{layer="L2",le="1.0"} 89.0
...
```

Not included in OpenAPI schema (`include_in_schema=False`) to keep the Swagger UI focused on application endpoints.

---

## Common Request Patterns

### Basic question (no filter)

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was NVIDIA revenue in Q3 2024?"}' | python3 -m json.tool
```

### Scoped to a specific company and quarter

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was the gross margin?",
    "filter": {"ticker": "AAPL", "year": 2024, "quarter": "Q4"}
  }' | python3 -m json.tool
```

### With verbose diagnostics (debugging)

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What guidance did Apple give for Q1 2025?", "verbose": true}' \
  | python3 -m json.tool
```

### Streaming with correlation ID

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: my-trace-id-001" \
  -d '{"question": "What was Meta Q3 2024 advertising revenue?"}'
```

### Full health check

```bash
curl -s http://localhost:8000/health/ | python -m json.tool
```

---

## Rate Limits & Timeouts

| Limit | Value | Source |
|-------|-------|--------|
| OpenAI rate limit | Varies by tier | OpenAI API tier |
| Request timeout (streaming) | 90 seconds | Client-side (`ui/utils.py`) |
| Producer timeout (SSE) | 60 seconds per token | Server-side queue wait |
| `Retry-After` on 429 | 10 seconds | Response header |

The system uses tenacity exponential backoff (base delay 1s, max 30s) for OpenAI calls. A 429 response from the API means all retries were exhausted.

---

## SDK Usage

The `FinancialRAGPipeline` Python class is the recommended SDK for programmatic access:

```python
from rag_pipeline import FinancialRAGPipeline
from qdrant_client import QdrantClient
from retrieval.models import MetadataFilter

# Initialise (pre-loads all models — do this once, reuse across calls)
pipeline = FinancialRAGPipeline(
    qdrant_client=QdrantClient(url="http://localhost:6333"),
    enable_query_cache=True,
)

# Structured response
result = pipeline.ask(
    question="What was Apple's revenue in Q4 2024?",
    metadata_filter=MetadataFilter(ticker="AAPL", year=2024),
)

print(result.answer)                                  # String with [N] citations
print(result.citations)                               # List of Citation objects
print(result.grounded)                                # Boolean
print(result.format_answer_with_citations())          # Formatted string
print(result.to_json())                               # JSON serialisation

# Streaming
for token in pipeline.ask_streaming("What was NVDA Q3 revenue?"):
    print(token, end="", flush=True)

# Verbose (includes query transform + retrieval summaries)
result, query_summary, retrieval_summary = pipeline.ask_verbose("...")
print(query_summary)
print(retrieval_summary)

# With Agentic Reflexion & Calibrated Abstention (built into ask)
result = await pipeline.ask(
    "What was NVDA data center revenue Q3?",
    strict_verification=True,     # enables sentence-level NLI grounding + PAL math
)
print(result.grounded)            # True if verified, False if abstained
print(result.confidence_score)    # Composite confidence score (0.0-1.0)
print(result.citations)           # Exact SEC 10-K/10-Q citations
print(result.answer)              # Verified financial answer
```
