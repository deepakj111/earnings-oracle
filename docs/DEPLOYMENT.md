# Deployment Guide

> Docker, CI/CD pipeline, production configuration, and operational runbooks for the Financial RAG System.

---

## Table of Contents

1. [Docker](#docker)
2. [docker-compose (Full Stack)](#docker-compose-full-stack)
3. [CI/CD Pipeline](#cicd-pipeline)
4. [Production Configuration](#production-configuration)
5. [Health Probes](#health-probes)
6. [Scaling Considerations](#scaling-considerations)
7. [Kubernetes (Roadmap)](#kubernetes-roadmap)
8. [Operational Runbooks](#operational-runbooks)

---

## Docker

### Multi-stage Dockerfile

The Dockerfile uses a two-stage build to minimise the runtime image size:

```
Stage 1: builder
  └── python:3.11-slim
  └── Install Poetry
  └── poetry install --no-root --without dev
  └── Creates .venv in project root

Stage 2: runtime
  └── python:3.11-slim  (fresh base — no build tools)
  └── Copy .venv from builder
  └── Copy application source
  └── Non-root user: appuser
  └── HEALTHCHECK: curl -sf http://localhost:8000/health/live
  └── CMD: uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Security posture**:
- Non-root `appuser` at runtime
- No build tools in final image
- Secrets injected via `--env-file .env` at runtime (never baked into layers)
- `detect-private-key` pre-commit hook prevents accidental key commits
- TruffleHog verified secret scanning on all staged commits

### Build

```bash
# Build runtime image
docker build -t financial-rag:latest .

# Build specific target
docker build --target runtime -t financial-rag:latest .

# With build args for image labels
docker build \
  --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --build-arg GIT_SHA=$(git rev-parse HEAD) \
  -t financial-rag:latest .
```

### Run individual containers

```bash
# API server
docker run -d \
  --name rag_api \
  -p 8000:8000 \
  --env-file .env \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  financial-rag:latest

# Streamlit UI
docker run -d \
  --name rag_ui \
  -p 8501:8501 \
  -e RAG_API_URL=http://host.docker.internal:8000 \
  financial-rag:latest \
  streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
```

### Image tags

Published to GitHub Container Registry (`ghcr.io`):

| Tag | Trigger | Description |
|-----|---------|-------------|
| `latest` | Push to `main` | Latest main branch build |
| `main` | Push to `main` | Branch name tag |
| `sha-<short>` | Every push | Immutable commit reference |
| `1.2.3` | Tag `v1.2.3` | Semantic version release |
| `1.2` | Tag `v1.2.3` | Major.minor floating tag |

---

## docker-compose (Full Stack)

### Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `qdrant` | `qdrant/qdrant:v1.9.2` | 6333, 6334 | Vector database |
| `api` | `financial-rag:latest` (built) | 8000 | FastAPI backend |
| `ui` | `financial-rag:latest` (built) | 8501 | Streamlit frontend |
| `prometheus` | `prom/prometheus:v2.51.2` | 9090 | Metrics scraping |
| `grafana` | `grafana/grafana:10.4.2` | 3000 | Metrics dashboards |

### Startup

```bash
# 1. Copy and fill environment
cp .env.example .env
# Edit .env: OPENAI_API_KEY, SEC_USER_AGENT, GRAFANA_ADMIN_PASSWORD

# 2. Start all services
docker compose up -d

# 3. Wait for API to be ready (model loading ~20s warm, 2-5min cold)
docker compose logs -f api

# 4. Run ingestion (one-time)
docker compose exec api poetry run python -m ingestion.download_filings
docker compose exec api poetry run python -m ingestion.pipeline
```

### Data persistence

| Volume | Mounted at | Contents |
|--------|-----------|---------|
| `qdrant_data` | `/qdrant/storage` | Qdrant vectors + payloads |
| `app_data` | `/app/data` | BM25 index, transcripts, checkpoint, eval reports |
| `prometheus_data` | `/prometheus` | 15-day metrics retention |
| `grafana_data` | `/var/lib/grafana` | Dashboards + datasource config |

### Service dependencies

```
grafana
  └── depends_on: prometheus
prometheus
  └── depends_on: api
ui
  └── depends_on: api (service_healthy)
api
  └── depends_on: qdrant (service_healthy)
qdrant
  └── (no dependencies)
```

Health-checked startup ensures the API doesn't receive traffic before Qdrant is ready, and the UI doesn't start before the API is serving.

### Prometheus hot-reload

```bash
# Reload prometheus config without restart
curl -X POST http://localhost:9090/-/reload
```

### Useful compose commands

```bash
# View logs
docker compose logs -f api
docker compose logs --tail=50 api

# Restart single service
docker compose restart api

# Scale API workers (not recommended with shared BM25 singleton — see Scaling)
docker compose up -d --scale api=2

# Teardown (preserves volumes)
docker compose down

# Teardown and delete volumes
docker compose down -v
```

---

## CI/CD Pipeline

### CI Workflow (`.github/workflows/ci.yml`)

Six jobs run **in parallel** on every push to `main`/`develop` and on all pull requests:

```
lint ──────────────────────────────────────────┐
typecheck ─────────────────────────────────────┤
security ──────────────────────────────────────┤──► ci-gate (required for merge)
test (matrix: 3.11, 3.12) ─────────────────────┤
validate-configs ──────────────────────────────┤
docker-build ──────────────────────────────────┘
```

**Branch protection**: Only the `ci-gate` job is required. Adding a new job to `ci-gate.needs` automatically enforces it without touching branch protection settings.

**Concurrency control**: `cancel-in-progress: true` — new pushes cancel in-flight runs for the same branch, saving CI minutes.

**Qdrant in CI**: The `test` job spins up a real `qdrant/qdrant:v1.9.2` container as a GitHub Actions service. The test suite waits up to 60s for readiness via a polling curl loop (qdrant image has no curl, so polling runs on the runner host).

**Coverage**: Uploaded to Codecov only from the Python 3.11 matrix leg (`if: matrix.python-version == '3.11'`) to avoid duplicate reports.

**Security SARIF**: Bandit findings are uploaded to the GitHub Security tab (`continue-on-error: true`). Security is a review signal, not a build blocker.

**Config validation**:
- `docker compose config --quiet` — resolves all env interpolations and validates service dependencies
- `python -c "import yaml; yaml.safe_load(open('prometheus.yml'))"` — validates Prometheus config
- `python -c "import tomllib; tomllib.load(...)"` — validates pyproject.toml (stdlib in Python 3.11)

### CD Workflow (`.github/workflows/cd.yml`)

Triggers:
- After CI passes on `main` (via `workflow_run` event — not parallel)
- On semver tags `v*.*.*` (release builds)

**Jobs**:

1. **build-and-push**
   - Docker Buildx with GitHub Actions layer cache (`type=gha`)
   - `docker/metadata-action` generates all tag variants
   - Pushes to `ghcr.io/<owner>/financial-rag`
   - `permissions: packages: write` for GHCR push

2. **smoke-test** (after build)
   - Spins up a real Qdrant service
   - Pulls and runs the published image
   - Polls `/health/live` for up to 150s
   - Asserts `/health` status is `ok` or `degraded` (collection missing is acceptable — ingestion hasn't run)

### CI environment variables

Only the default `GITHUB_TOKEN` is required to push to GHCR:

| Variable | Required In | Description |
|----------|-------------|-------------|
| `GITHUB_TOKEN` | CD | Auto-provided by GitHub Actions for GHCR push |

*Note: No `OPENAI_API_KEY` secret is required! The pipeline injects a hardcoded placeholder (`sk-test-placeholder...`) directly in the workflow files. CI tests never call real OpenAI — all LLM calls are mocked via `unittest.mock.patch`.*

---

## Production Configuration

### Recommended `.env` for production

```dotenv
# Required
OPENAI_API_KEY=sk-...
SEC_USER_AGENT="Company Name ops@company.com"
QDRANT_URL=http://qdrant:6333              # Docker service name

# LLM
RAG_GENERATION_MODEL=gpt-4.1-nano
RAG_QUERY_TRANSFORM_MODEL=gpt-4.1-nano
RAG_EVAL_MODEL=gpt-4.1-nano

# Retrieval tuning
RAG_RETRIEVAL_TOP_K_DENSE=10
RAG_RETRIEVAL_TOP_K_BM25=10
RAG_RETRIEVAL_TOP_K_FINAL=5
RAG_RERANKER_ENABLED=true
RAG_RERANKER_TOP_K_PRE=20

# CRAG
RAG_CRAG_ENABLED=true
RAG_CRAG_HIGH_THRESHOLD=0.6
RAG_CRAG_LOW_THRESHOLD=0.2
TAVILY_API_KEY=tvly-...                    # Recommended for web search quality

# Observability
GRAFANA_ADMIN_PASSWORD=<strong-password>

# Generation safety
RAG_GENERATION_MAX_CONTEXT_TOKENS=4096
RAG_GENERATION_MAX_TOKENS=4096
```

### uvicorn production command

The docker-compose `api` service uses:

```bash
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 2 \
  --loop uvloop \
  --http httptools
```

**Worker count**: Keep `--workers 2` or match CPU cores. Do not exceed 4 without verifying BM25 index memory behaviour — each worker process loads its own BM25 index copy.

**uvloop + httptools**: Both ship with `uvicorn[standard]`. They replace asyncio's default event loop and HTTP protocol with faster C implementations.

---

## Health Probes

### Kubernetes liveness probe

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
  timeoutSeconds: 3
  failureThreshold: 3
```

### Kubernetes readiness probe

```yaml
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 30       # Allow time for model loading
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

`/health/ready` returns 503 until the `FinancialRAGPipeline` singleton is initialised. On first run (model download), startup can take 2–5 minutes. On subsequent runs (warm cache), ~10–20s.

### Full health check response

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

Status values:
- `healthy` — all components OK
- `degraded` — Qdrant unreachable or collection/BM25 missing (some queries may fail)
- `unhealthy` — pipeline singleton unavailable (no queries can be served)

---

## Scaling Considerations

### Current architecture constraints

| Constraint | Cause | Impact |
|-----------|-------|--------|
| BM25 is in-process | `_bm25_index` module-level singleton | Each worker process loads its own copy (~60–100 MB each) |
| fastembed is CPU-bound | ONNX inference on CPU | Embedding latency ~50 ms per batch; scales with CPU cores |
| FlashRank is CPU-bound | Cross-encoder ONNX inference | ~8–15 ms per 20 candidates; independent per request |
| OpenAI rate limits | API tier | Enforced by tenacity backoff; consider `max_workers` in ThreadPool |

### Horizontal scaling recommendations

**Short-term (current architecture)**:
- Run 2–4 API containers behind a load balancer
- Use a shared external Qdrant instance (not in-container)
- BM25 index is read-only — safe to share via mounted volume

**Medium-term**:
- Move BM25 to a shared Redis instance or serve via a dedicated BM25 microservice
- Use Qdrant Cloud for managed vector storage
- Consider quantized embeddings for lower memory footprint

**Long-term**:
- Async ingestion pipeline for concurrent embedding + upsert
- GPU acceleration for fastembed (fastembed supports CUDA via ONNX)
- Streaming token counting for cost attribution per request

---

## Kubernetes (Roadmap)

> Not yet implemented. Reference architecture for future deployment.

```yaml
# Planned resource structure
apiVersion: apps/v1
kind: Deployment
metadata:
  name: financial-rag-api
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: api
        image: ghcr.io/<owner>/financial-rag:latest
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: rag-secrets
        volumeMounts:
        - name: app-data
          mountPath: /app/data
        livenessProbe:
          httpGet: { path: /health/live, port: 8000 }
          initialDelaySeconds: 5
        readinessProbe:
          httpGet: { path: /health/ready, port: 8000 }
          initialDelaySeconds: 30
        resources:
          requests: { cpu: "500m", memory: "1Gi" }
          limits: { cpu: "2000m", memory: "4Gi" }
```

---

## Operational Runbooks

### Runbook: Full re-index

**When**: Qdrant storage was wiped, or checkpoint is corrupt.

```bash
# 1. Stop the API (optional but prevents partial reads during re-index)
docker compose stop api

# 2. Delete checkpoint so all files are re-processed
rm data/pipeline_checkpoint.txt

# 3. Optionally delete existing Qdrant collection (forces fresh start)
# curl -X DELETE http://localhost:6333/collections/earnings_transcripts

# 4. Run pipeline
poetry run python -m ingestion.pipeline

# 5. Restart API
docker compose start api
```

### Runbook: Add new company

**When**: Expanding beyond the current 10 tickers.

```python
# 1. Add to ingestion/download_filings.py
COMPANIES = {
    ...
    "GOOGL": "0001652044",   # Alphabet Inc.
}

# 2. Add to ingestion/metadata_extractor.py
COMPANY_MAP = {
    ...
    "GOOGL": "Alphabet",
}

# 3. Add to api/models.py
_VALID_TICKERS = frozenset({..., "GOOGL"})

# 4. Add to ui/app.py
_TICKERS = ["(all)", ..., "GOOGL"]

# 5. Download and ingest new company filings
poetry run python -m ingestion.download_filings
poetry run python -m ingestion.pipeline   # Checkpoint ensures existing files are skipped
```

### Runbook: Diagnose high latency

```bash
# 1. Check Grafana dashboard for per-layer latency
# rag_pipeline_latency_seconds{layer="L2"} — query transform
# rag_pipeline_latency_seconds{layer="L3"} — retrieval
# rag_pipeline_latency_seconds{layer="L4"} — generation

# 2. Check OpenAI API status
# https://status.openai.com/

# 3. Check Qdrant health
curl http://localhost:6333/healthz

# 4. Disable reranker if FlashRank is slow on CPU
RAG_RERANKER_ENABLED=false docker compose restart api

# 5. Check BM25 index size (large index = slow search)
poetry run python scripts/inspect_index.py
```

### Runbook: Diagnose poor answer quality

```bash
# 1. Enable verbose mode in a test query
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "your question", "verbose": true}'

# 2. Check retrieval_summary — are relevant chunks being retrieved?
# Check query_summary — are all three techniques producing good variants?

# 3. Run evaluation harness
poetry run python -m evaluation.harness --n 5 --metrics faithfulness context_precision

# 4. Common root causes:
# - grounded=false → CRAG not enabled or web search not configured
# - context_precision=low → reranker not enabled or BM25 not returning good results
# - faithfulness=low → context window too small (increase RAG_GENERATION_MAX_CONTEXT_TOKENS)

# 5. Check if ingestion is fresh
poetry run python scripts/inspect_index.py
# Look for: "OK: BM25 and Qdrant counts are consistent"
```

### Runbook: Prometheus metrics not appearing

```bash
# 1. Verify /metrics endpoint responds
curl http://localhost:8000/metrics | head -20

# 2. Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# 3. Hot-reload Prometheus config
curl -X POST http://localhost:9090/-/reload

# 4. Check prometheus.yml scrape config
# targets should be "api:8000" (docker-compose) or "localhost:8000" (local dev)
```
