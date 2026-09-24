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
| `qdrant` | `qdrant/qdrant:v1.11.0` | 6333, 6334 | Vector database (Dense HNSW + Semantic Cache collection) |
| `redis` | `redis:7-alpine` | 6379 | Optional cache layer (available for future use) |
| `jaeger` | `jaegertracing/all-in-one:latest` | 16686, 4317, 4318 | APM distributed tracing & flamegraphs |
| `api` | `financial-rag:latest` (built) | 8000 | FastAPI backend & web frontend |
| `ui` | `financial-rag:latest` (built) | 8501 | Streamlit frontend |
| `prometheus` | `prom/prometheus:v2.51.2` | 9090 | Metrics scraping |
| `grafana` | `grafana/grafana:10.4.2` | 3000 | Real-time metrics dashboards |

### Startup

```bash
# 1. Copy and fill environment
cp .env.example .env
# Edit .env: RAG_LLM_PROVIDER, GOOGLE_CLOUD_PROJECT (or OPENAI_API_KEY), SEC_USER_AGENT, GRAFANA_ADMIN_PASSWORD
# If using Google Cloud ADC, authenticate via: gcloud auth application-default login

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

For a deep technical dive into our automated pipelines, see [CI/CD & Automation](CI_CD.md).

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
# Option A: Google Cloud ADC (Zero API Keys) [Recommended]
RAG_LLM_PROVIDER=gemini
GOOGLE_CLOUD_PROJECT=gleaming-vision-509507-j6
GOOGLE_CLOUD_LOCATION=us-central1

# Option B: OpenAI API Key (Alternative)
# RAG_LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...

SEC_USER_AGENT="Company Name ops@company.com"
QDRANT_URL=http://qdrant:6333              # Docker service name

# Models (Defaults to Gemini 2.5 Flash / text-embedding-004)
RAG_GENERATION_MODEL=gemini-2.5-flash
RAG_QUERY_TRANSFORM_MODEL=gemini-2.5-flash
RAG_EMBEDDING_MODEL=text-embedding-004
RAG_EMBEDDING_VECTOR_DIM=768
RAG_EVAL_MODEL=gemini-2.5-flash

# Retrieval tuning
RAG_RETRIEVAL_TOP_K_DENSE=10
RAG_RETRIEVAL_TOP_K_BM25=10
RAG_RETRIEVAL_TOP_K_FINAL=5
RAG_RERANKER_ENABLED=true
RAG_RERANKER_TOP_K_PRE=20

# Knowledge Graph & Retrieval
RAG_KG_RETRIEVAL_ENABLED=true
RAG_CONTEXT_COMPRESSION_ENABLED=true
RAG_AUDIT_ENABLED=true

# Observability
GRAFANA_ADMIN_PASSWORD=<strong-password>

# Generation safety
RAG_GENERATION_MAX_CONTEXT_TOKENS=4096
RAG_GENERATION_MAX_TOKENS=4096
```

### uvicorn production command

The `poetry run serve-prod` entrypoint and Docker container use:

```bash
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

**Worker count**: Set `--workers 4` or match CPU cores. Do not exceed 4 without verifying BM25 index memory behaviour — each worker process loads its own BM25 index copy.

**Multi-worker compatibility**: `--workers >1` uses multiprocessing. Custom event loop flags (`--loop`/`--http`) are omitted as Uvicorn automatically selects the optimal event loop per worker process based on installed `uvicorn[standard]` extras.

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
      "detail": "generation=gemini-2.5-flash | transform=gemini-2.5-flash"
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

> 📖 **Master Operations Runbook**: For the exhaustive, step-by-step operational guide covering all commands, CLI flags, multi-stage ingestion modes, and granular ablation studies in exact chronological execution order, see **[RUNBOOK.md](RUNBOOK.md)**.

### Runbook: Full re-index

**When**: Qdrant storage was wiped, or store index files are corrupt.

```bash
# 1. Stop the API (optional but prevents partial reads during re-index)
docker compose stop api

# 2. Reset vector DB collection and local index files
poetry run python scripts/reset_index.py

# 3. Run pipeline
poetry run python -m ingestion.pipeline

# 4. Restart API
docker compose start api
```

### Runbook: Add new company

**When**: Registering an additional company into the system.

```python
# 1. Register company profile in config/companies.py
# CompanyRegistry holds CIK, fiscal year end, and sector info

# 2. Add to ingestion/download_filings.py if scraping new SEC filings

# 3. Ingest new company filings
poetry run python -m ingestion.download_filings
poetry run python -m ingestion.pipeline   # Idempotent state manager skips existing filings
```

### Runbook: Diagnose high latency

```bash
# 1. Check Jaeger UI for distributed trace flamegraphs (L2/L3/L4 breakdown)
# http://localhost:16686

# 2. Check Prometheus metrics & Grafana dashboards
# http://localhost:9090 and http://localhost:3000

# 3. Check Qdrant health
curl http://localhost:6333/readyz

# 4. Check Redis cache connectivity
docker compose exec redis redis-cli ping

# 5. Disable reranker if FlashRank is slow on CPU
RAG_RERANKER_ENABLED=false docker compose restart api
```

### Runbook: Diagnose poor answer quality

```bash
# 1. Enable verbose mode in a test query
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "your question", "verbose": true}' | python3 -m json.tool

# 2. Check retrieval_summary — are relevant chunks being retrieved?
# Check query_summary — are all three techniques producing good variants?

# 3. Run evaluation harness
poetry run python evaluation/harness.py

# 4. Check if ingestion is fresh and consistent
poetry run inspect-data
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
