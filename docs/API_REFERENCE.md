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

#### `POST /query/`

Run the full four-layer RAG pipeline and return a structured JSON response with inline citations, token usage, and grounding diagnostics.

**Pipeline layers executed**: L2 Query Transform → L3 Hybrid Retrieval → L4 Answer Generation

**Request body**

```json
{
  "question": "What was Apple's total revenue in Q4 2024?",
  "filter": {
    "ticker": "AAPL",
    "year": 2024,
    "quarter": "Q4"
  },
  "verbose": false
}
```

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `question` | string | ✅ | 3–2000 chars | Natural language financial question |
| `filter` | object | ❌ | — | Optional scope filter (all fields optional) |
| `filter.ticker` | string | ❌ | One of: AAPL, NVDA, MSFT, AMZN, META, JPM, XOM, UNH, TSLA, WMT | Company ticker (case-insensitive) |
| `filter.year` | integer | ❌ | 2020–2030 | Fiscal year |
| `filter.quarter` | string | ❌ | Q1, Q2, Q3, Q4 | Fiscal quarter (case-insensitive) |
| `verbose` | boolean | ❌ | default: `false` | Include query transform + retrieval diagnostics in response |

**Response `200 OK`**

```json
{
  "question": "What was Apple's total revenue in Q4 2024?",
  "answer": "Apple reported total net sales of $94.9 billion in Q4 fiscal year 2024 [1], representing a 6% increase year-over-year [1][2].",
  "citations": [
    {
      "index": 1,
      "ticker": "AAPL",
      "company": "Apple",
      "date": "2024-10-31",
      "fiscal_period": "Q4 2024",
      "section_title": "Financial Highlights",
      "doc_type": "earnings_release",
      "source": "both",
      "rerank_score": 0.9821,
      "excerpt": "Apple Inc. today announced financial results for its fiscal 2024 fourth quarter ended September 28, 2024. The Company posted quarterly revenue of $94.9 billion..."
    },
    {
      "index": 2,
      "ticker": "AAPL",
      "company": "Apple",
      "date": "2024-10-31",
      "fiscal_period": "Q4 2024",
      "section_title": "Revenue",
      "doc_type": "earnings_release",
      "source": "dense",
      "rerank_score": 0.9412,
      "excerpt": "Total net sales for Q4 2024 were $94,930 million compared to $89,498 million in Q4 2023..."
    }
  ],
  "grounded": true,
  "retrieval_failed": false,
  "model": "gpt-4.1-nano",
  "usage": {
    "prompt_tokens": 2456,
    "completion_tokens": 87,
    "total_tokens": 2543
  },
  "context": {
    "chunks_used": 5,
    "tokens_used": 2048
  },
  "latency_seconds": 3.142,
  "unique_tickers": ["AAPL"],
  "unique_sources": ["AAPL Q4 2024"],
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
| `citations[].source` | string | `"dense"` \| `"bm25"` \| `"both"` — retrieval system that surfaced this chunk |
| `citations[].rerank_score` | float | FlashRank cross-encoder relevance score (higher = more relevant) |
| `citations[].excerpt` | string | First 250 chars of source passage |
| `grounded` | boolean | `false` if model signalled insufficient context — consider CRAG web fallback |
| `retrieval_failed` | boolean | `true` if zero documents were retrieved from the index |
| `unique_tickers` | array | Deduplicated tickers cited in answer, in citation order |
| `unique_sources` | array | `"TICKER fiscal_period"` labels, e.g. `["AAPL Q4 2024"]` |
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

SSE message format:

```
data: {"token": "Apple"}

data: {"token": " reported"}

data: {"token": " total"}

data: [DONE]
```

Error events (stream terminates after this):

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

**Note**: No citation metadata or token counts are available in streaming mode. Use `POST /query/` for structured output with citation details.

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
  "version": "0.1.0",
  "uptime_seconds": 3627.4,
  "components": {
    "qdrant": {
      "status": "ok",
      "detail": "collection 'earnings_transcripts' present (18432 points)"
    },
    "pipeline": {
      "status": "ok",
      "detail": "generation=gpt-4.1-nano | transform=gpt-4.1-nano"
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
curl -s -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What was NVIDIA revenue in Q3 2024?"}'
```

### Scoped to a specific company and quarter

```bash
curl -s -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was the gross margin?",
    "filter": {"ticker": "AAPL", "year": 2024, "quarter": "Q4"}
  }'
```

### With verbose diagnostics (debugging)

```bash
curl -s -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What guidance did Apple give for Q1 2025?", "verbose": true}' \
  | python -m json.tool
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

# With CRAG
crag = pipeline.ask_with_crag("What was Tesla Q3 deliveries?")
print(crag.action.value)          # "correct" | "ambiguous" | "incorrect"
print(crag.was_corrected)         # bool
print(crag.relevance_ratio)       # 0.0–1.0
print(crag.final_result.answer)   # Corrected answer
```
