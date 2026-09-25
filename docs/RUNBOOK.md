# 📋 Financial Earnings Oracle: Master Command & Operations Runbook

> **The definitive, single source of truth for all operational procedures, pipeline executions, evaluations, and maintenance workflows.**

This runbook documents every command in the exact order required to set up, ingest, serve, evaluate, and maintain the Financial RAG system.

---

## 📑 Table of Contents

1. [Phase 0: Prerequisites & System Requirements](#phase-0-prerequisites--system-requirements)
2. [Phase 1: Environment & API Key Setup](#phase-1-environment--api-key-setup)
3. [Phase 2: Infrastructure Provisioning (Docker)](#phase-2-infrastructure-provisioning-docker)
4. [Phase 3: Data Ingestion & Indexing Pipeline](#phase-3-data-ingestion--indexing-pipeline)
5. [Phase 4: Serving API & User Interfaces](#phase-4-serving-api--user-interfaces)
6. [Phase 5: Observability, Tracing & Monitoring](#phase-5-observability-tracing--monitoring)
7. [Phase 6: Golden Dataset & Statistical Evaluation](#phase-6-golden-dataset--statistical-evaluation)
8. [Phase 7: Granular Ablation Studies & A/B Testing](#phase-7-granular-ablation-studies--ab-testing)
9. [Phase 8: Code Quality, Testing & Security Gates](#phase-8-code-quality-testing--security-gates)
10. [Phase 9: Maintenance, Recovery & Reset Procedures](#phase-9-maintenance-recovery--reset-procedures)
11. [⚡ Quick Reference Cheat Sheet](#-quick-reference-cheat-sheet)

---

## Phase 0: Prerequisites & System Requirements

Ensure the host machine satisfies the following prerequisites before running commands:

| Tool | Minimum Version | Verification Command |
| :--- | :--- | :--- |
| **Python** | 3.11+ | `python3 --version` |
| **Poetry** | 1.8.0+ | `poetry --version` |
| **Docker & Docker Compose** | 24.0+ (Compose v2) | `docker compose version` |
| **Git** | 2.40+ | `git --version` |

### 1. Clone the Repository
```bash
git clone https://github.com/deepakj111/earnings-oracle.git
cd rag-project
```

### 2. Install Dependencies via Poetry
Installs production and development dependencies into an isolated virtual environment:
```bash
poetry install
```

---

## Phase 1: Environment & API Key Setup

The system relies on a central `.env` file for API keys, infrastructure URLs, and feature toggles.

### 1. Initialize Environment File
```bash
cp .env.example .env
```

### 2. Populate Required Environment Variables
Open `.env` and configure the following parameters:

```dotenv
# ── Authentication: Google Cloud ADC (Recommended) or API Key ──────────────
# Option A: Zero keys needed — authenticate once via Google Cloud CLI:
#   gcloud auth application-default login
RAG_LLM_PROVIDER="gemini"
GOOGLE_CLOUD_PROJECT="gleaming-vision-509507-j6"
GOOGLE_CLOUD_LOCATION="us-central1"

# Option B: OpenAI API Key (Alternative)
# RAG_LLM_PROVIDER="openai"
# OPENAI_API_KEY=sk-your-actual-openai-api-key

# ── SEC EDGAR Identity (Required by SEC fair access policy) ──────────────────
SEC_USER_AGENT="YourName yourname@example.com"

# ── Local Infrastructure Endpoints ──────────────────────────────────────────
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# ── Feature Configurations & Toggles ─────────────────────────────────────────
RAG_RERANKER_ENABLED=true
RAG_CONTEXT_COMPRESSION_ENABLED=true
RAG_SEMANTIC_CACHE_ENABLED=true
```

> [!IMPORTANT]
> **SEC User-Agent Policy**: The SEC requires a custom `User-Agent` header in the format `Name email@domain.com`. Failing to set this in `.env` will cause filing downloads to fail with HTTP 403 Forbidden.

---

## Phase 2: Infrastructure Provisioning (Docker)

The project includes a multi-container Docker Compose stack comprising **Qdrant** (vector search), **Redis** (semantic caching), **Jaeger** (distributed tracing), **Prometheus** (metrics), and **Grafana** (dashboards).

### Recommended Workflow: Hybrid Local Development
In standard development and evaluation, run backing services in Docker while executing Python code natively on the host for rapid feedback and debugging:

```bash
# Start backing data stores and observability tools
docker compose up -d qdrant redis jaeger prometheus grafana
```

Verify that all services are healthy:
```bash
docker compose ps
```

| Service | Host Port | Purpose | Health URL |
| :--- | :--- | :--- | :--- |
| **Qdrant** | `6333` | Vector Database (Dense Vectors & Semantic Cache) | `curl http://localhost:6333/readyz` |
| **Redis** | `6379` | Optional cache layer (available for future use) | `docker compose exec redis redis-cli ping` |
| **Jaeger** | `16686` (UI), `4317` (OTLP) | Distributed APM Tracing | `http://localhost:16686` |
| **Prometheus** | `9090` | Time-series Metrics Scraper | `http://localhost:9090` |
| **Grafana** | `3000` | Real-time Observability Dashboard | `http://localhost:3000` (admin/admin) |

> [!WARNING]
> **Port 8000 Conflict Notice**: Docker Compose also includes an `api` container. If you start the full stack (`docker compose up -d`) and then attempt to run `poetry run serve-prod` locally, both will compete for host port `8000`. Stop the Docker API container when running local Python servers:
> ```bash
> docker stop rag_api
> ```

---

## Phase 3: Data Ingestion & Indexing Pipeline

The ingestion pipeline must be executed sequentially: first downloading raw SEC filings, then parsing, chunking, embedding, and indexing into Qdrant, BM25, and Knowledge Graph stores.

```text
SEC EDGAR ──▶ download_filings ──▶ parse_html ──▶ parent_child_chunks ──▶ contextual_enrichment
                                                                                   │
                 ┌─────────────────────────────────────────────────────────────────┴─────┐
                 ▼                                 ▼                                     ▼
           Qdrant Dense                    BM25 Sparse Corpus                     Knowledge Graph
    (text-embedding-004 768d /          (Tuned financial BM25)              (Entity & Relation Graph)
     text-embedding-3-small 1536d)
```

### Step 3.1: Download SEC Filings
Downloads Form 10-K and 10-Q HTML filings for NVDA, WMT, NFLX, and UNH from SEC EDGAR into `data/company_filings/`:
```bash
poetry run python -m ingestion.download_filings
```

### Step 3.2: Execute Ingestion & Indexing Pipeline
Runs full structure-aware chunking, Anthropic-style contextual enrichment, Google Gemini / OpenAI embedding generation with vector dimension auto-healing, BM25 indexing, and LLM Knowledge Graph extraction:
```bash
poetry run python -m ingestion.pipeline
```

### CLI Execution Modes & Flags
The pipeline provides granular CLI arguments to control execution scope:

```bash
# 1. Fast Ingestion (Skip LLM Knowledge Graph to minimize API cost/time)
poetry run python -m ingestion.pipeline --fast

# 2. Knowledge Graph Only (Extract KG on already indexed files without re-embedding)
poetry run python -m ingestion.pipeline --kg-only

# 3. Custom Concurrency & Threading
poetry run python -m ingestion.pipeline --concurrency 4 --threads 8
```

| Flag | Purpose | Recommended When |
| :--- | :--- | :--- |
| *(default)* | Full end-to-end ingestion across all stores | First full indexing run |
| `--fast` / `--no-kg` | Bypasses LLM Knowledge Graph extraction | Rapid smoke tests and evaluation iterations |
| `--kg-only` | Re-runs only Knowledge Graph extraction | Recovering or enriching graph entities |
| `--concurrency N` | Sets max concurrent documents processed | Tuning for API rate/concurrency limits |
| `--threads N` | Overrides worker thread pool count | Running CPU-bound tokenization / workers |

### Step 3.3: Verify Ingestion State & Idempotency
Verify that all filings were successfully processed into Qdrant and BM25:

```bash
# 1. Inspect collection statistics and chunk distribution
poetry run inspect-data

# 2. Export sample payload records from Qdrant for manual inspection
poetry run export-qdrant

# 3. Review ingestion latency profiling per step
cat data/ingestion_metrics.json | python3 -m json.tool | head -n 30
```

> [!NOTE]
> **Idempotent Ingestion Guarantee**: The pipeline computes SHA-256 chunk hashes recorded in `data/ingestion_state.db` (SQLite) and performs direct in-memory store inspection (`IngestionStoreState`). Re-running `pipeline.py` skips already-indexed chunks automatically, guaranteeing zero duplicate vectors and zero wasted API credits.

---

## Phase 4: Serving API & User Interfaces

Once ingestion is complete, launch the backend application servers and interactive frontends.

### Option A: Production Multi-Worker Server (Recommended)
Launches the high-throughput Uvicorn ASGI server with 4 worker processes:
```bash
poetry run serve-prod
```
*Equivalent command: `poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4`*

### Option B: Development Auto-Reload Server
Launches a single worker instance that automatically reloads when source code changes:
```bash
poetry run serve
```
*Equivalent command: `poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`*

### Access Interfaces & Endpoints
With the server running, open the following URLs in your browser:

- **Web Frontend Application**: [http://localhost:8000/app](http://localhost:8000/app) (or root [http://localhost:8000/](http://localhost:8000/))
- **Interactive Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc API Reference**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Streamlit Alternative UI** (Optional):
  ```bash
  poetry run ui
  # Accessible at http://localhost:8501
  ```

### Verify Endpoints via CLI

```bash
# 1. Check API health
curl -s http://localhost:8000/health | python3 -m json.tool

# 2. Create a persistent chat session
curl -s -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "Walmart FY2026 Analysis"}' | python3 -m json.tool

# 3. Execute a test query (with session tracking)
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was Walmart global revenue and US net sales for fiscal year 2026?",
    "session_id": "sess_your_id"
  }' | python3 -m json.tool
```

---

## Phase 5: Observability, Tracing & Monitoring

The system comes pre-configured with full APM observability:

### 1. Distributed Tracing (Jaeger)
- **Web UI**: [http://localhost:16686](http://localhost:16686)
- **What to look for**: Select service `financial-rag-api` and click **Find Traces**. Inspect the flamegraphs to trace query latency across HyDE query expansion, hybrid retrieval (Dense + BM25), FlashRank reranking, and Gemini / OpenAI generation.

### 2. Prometheus Metrics
- **Web UI**: [http://localhost:9090](http://localhost:9090)
- **Raw Metrics Endpoint**: `http://localhost:8000/metrics`
- **Key Metrics to Query**:
  - `rag_query_latency_seconds_bucket` (P50/P95/P99 latency)
  - `rag_retrieval_chunks_total` (chunks retrieved per query)
  - `rag_cache_hits_total` / `rag_cache_misses_total` (Qdrant semantic cache efficiency)
  - `rag_faithfulness_score` (online verification scores)

### 3. Grafana Dashboard
- **Web UI**: [http://localhost:3000](http://localhost:3000)
- **Credentials**: Username `admin`, Password `admin`
- Contains pre-provisioned dashboards visualizing live traffic, response times, and LLM token expenditures.

### 4. Per-Query Audit Log Trail
Every request is recorded with complete query variants, HyDE generated text, retrieved chunks, LLM parameters, and response text in daily directories:
```bash
# Inspect the append-only JSONL audit log
tail -n 5 data/audit_logs/audit.jsonl | python3 -m json.tool

# View the latest detailed per-query execution trace
ls -t data/audit_logs/*/*.json | head -n 1 | xargs cat | python3 -m json.tool
```

---

## Phase 6: Golden Dataset & Statistical Evaluation

To evaluate the system scientifically without data leakage, the repository uses an audited 129-QA financial evaluation dataset with grounded reference answers across 43 Form 10-K/10-Q filings.

### Step 6.1: (Optional) Re-generate Golden Dataset
If you wish to synthesize a fresh evaluation dataset directly from the downloaded SEC filings using `gemini-2.5-pro`:
```bash
poetry run python scripts/generate_golden_dataset.py
```
*Output is validated and written to `data/golden_dataset.json`.*

### Step 6.2: Run Automated Evaluation Harness (`evaluation/harness.py`)

The evaluation harness benchmarks the **current production RAG pipeline end-to-end** against the curated golden QA dataset (`data/golden_dataset.json`). It evaluates the holistic system performance across retrieval, reranking, synthesis, and mathematical computation.

#### What It Does:
1. **Sample Scoping & Filtering**:
   Loads golden question-answer pairs and applies metadata filtering (ticker, year, quarter) specified in each sample to mirror real-world query scoping.
2. **End-to-End Pipeline Execution**:
   Executes `pipeline.ask_verbose()` through the entire active stack:
   - Query Transformation (HyDE / Multi-Query expansion)
   - Hybrid Search (Qdrant Dense 768-dim + BM25 Sparse with RRF fusion)
   - Cross-Encoder Reranking
   - GraphRAG Multi-hop Context Expansion
   - PAL Program-Aided Deterministic Financial Math
   - SEC Citation Grounding & LLM Answer Generation
3. **LLM-as-a-Judge Evaluation (RAG Triad + Context Metrics)**:
   Scores the retrieved context and generated answer using provider-agnostic LLM calls with structured JSON output:
   - **`faithfulness` (Groundedness)**: Asserts that every factual claim in the generated answer is directly substantiated by the retrieved SEC context chunks ($\text{Score} = \frac{\text{supported claims}}{\text{total claims}}$, detecting hallucinations).
   - **`answer_relevancy`**: Evaluates whether the generated answer directly addresses the user's prompt without drift or omission.
   - **`context_precision`**: Measures the signal-to-noise ratio in retrieval by determining what fraction of retrieved chunks are genuinely relevant to the query.
   - **`context_recall`**: Checks whether the retrieved context covers all factual statements present in the reference ground truth answer.
4. **Statistical Rigor (Bootstrap Confidence Intervals)**:
   Computes **95% Bootstrap Confidence Intervals** (1 000 resamples) across all metric distributions to prove that observed performance is statistically significant rather than stochastic variance.
5. **Multi-Threaded Execution**:
   Runs samples concurrently via a configurable thread pool (`max_workers`) for high evaluation throughput.

#### Execution Commands:
```bash
# Full evaluation on all golden dataset samples
poetry run python evaluation/harness.py

# Quick smoke test on 5 samples (recommended for rapid verification)
poetry run python evaluation/harness.py -n 5

# Evaluate only a specific company ticker
poetry run python evaluation/harness.py -t NVDA -n 10

# Dry-run validation (checks dataset samples and pipeline initialization without making LLM calls)
poetry run python evaluation/harness.py --dry-run

# Custom metrics selection
poetry run python evaluation/harness.py --metrics faithfulness answer_relevancy -n 10
```

#### Output Artifacts:
Reports are saved directly to `data/eval_reports/`:
- **JSON Report (`full_stack_production_eval_<timestamp>.json`)**: Complete record of per-sample inputs, generated outputs, retrieved chunks, citations, latencies, individual metric scores, and judge reasoning.
- **CSV Report (`full_stack_production_eval_<timestamp>.csv`)**: Flattened tabular export ideal for Pandas analysis, spreadsheets, and regression diffs.
- **Console Summary**: Formatted terminal table displaying aggregate metric means, standard deviations, and 95% Bootstrap CIs.
- **Formal Evaluation Report**: Complete empirical findings, statistical computation, and in-depth interpretations are documented in [`docs/BENCHMARKS.md`](file:///home/deepak/rag-project/docs/BENCHMARKS.md).

---

### Workflow Intuition: Evaluation Harness (Phase 6) vs. Ablation Studies (Phase 7)

Here is how the two phases interact in the engineering lifecycle:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               The RAG Optimization Lifecycle                           │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
   [Phase 5: Ingestion] ────────► 43 Filings Indexed (Qdrant Dense, BM25, Knowledge Graph)
                                            │
                                            ▼
   [Phase 6: Initial Harness] ──► Full Stack "Ceiling Benchmark" (All features enabled)
                                  Answers: "What is the maximum quality our system can achieve?"
                                  Identifies: Latency bottlenecks (e.g. 6.5s) & token costs
                                            │
                                            ▼
   [Phase 7: Ablation Study] ───► Dissection into Isolated Arms (Dense-only, +BM25, +HyDE, etc.)
                                  Measures: True marginal ΔFaithfulness and ΔLatency per module
                                  Synthesizes: Dynamic Pareto Tiers (Tier 1 Fast vs Tier 2 SOTA)
                                            │
                                            ▼
   [Production Configuration] ──► Lock winning tier into config/settings.py or .env
                                            │
                                            ▼
   [Ongoing Harness CI/CD] ─────► Automated Release Gate: 1x cost test on every PR / release
```

1. **Phase 6 (Harness as Ceiling Benchmark)**:
   By default, `evaluation/harness.py` initializes the pipeline with your active configuration (all techniques enabled: Dense + BM25 + Query Transforms + Reranker + GraphRAG + PAL Math). This establishes the **quality ceiling**. However, you don't yet know *which* of those components actually contribute to accuracy versus which are dead weight adding latency.
2. **Phase 7 (Ablations as Discovery Engine)**:
   Ablation strips the system down to pure dense retrieval and tests each module in isolation. It discovers whether GraphRAG or HyDE actually earned their compute overhead, and automatically bundles the winners into **Tier 1 (Fast: <1.0s overhead)** and **Tier 2 (SOTA: all positive lift)**.
3. **Execution Ordering Flexibility**:
   - **Top-Down (Runbook order)**: Run Phase 6 to see the full stack baseline, then Phase 7 to optimize and prune.
   - **Bottom-Up (Greenfield design)**: Run Phase 7 immediately after Ingestion to discover the optimal Pareto configuration first, configure `config/settings.py`, and then run Phase 6 to establish the production release gate.

---

## Phase 7: Granular Ablation Studies & Dynamic Pareto Tiers

Rather than hardcoding an arbitrary production configuration, the system uses a **dynamic Pareto frontier**. Each architectural component is first tested in strict isolation against a pure dense baseline to empirically measure its marginal $\Delta\text{Faithfulness}$ lift and latency overhead:
- **Tier 1 (Fast)**: Dynamically bundles all components with positive Faithfulness lift ($\Delta \ge 0$) AND latency overhead $< 1.0\text{s}$.
- **Tier 2 (SOTA)**: Dynamically stacks all net-positive components ($\Delta \ge 0$).
- Regressive components ($\Delta < 0$) are pruned.

Because Pareto tiers depend on empirical measurements, isolated single-component arms are evaluated first.

### Step 7.1: Run Isolated Single-Component Arms
Evaluates the dense-only baseline and all 5 isolated component arms (`bm25`, `querytransform`, `reranker`, `graphrag`, `pal_math`). When all isolated arms complete, it automatically synthesizes and benchmarks the dynamic Pareto production tiers:

```bash
# Smoke test on 5 samples (fast verification of baseline, 5 isolated arms, and Pareto tiers)
poetry run python scripts/run_portfolio_ablations.py --isolated -n 5

# Full evaluation across all samples in the golden dataset
poetry run python scripts/run_portfolio_ablations.py --isolated --all
```

*Alternative options:*
- Run isolated arms only without evaluating Pareto tiers:
  ```bash
  poetry run python scripts/run_portfolio_ablations.py --isolated --no-pareto -n 5
  ```
- Benchmark specific isolated arms:
  ```bash
  poetry run python scripts/run_portfolio_ablations.py --isolated --iso-arms bm25 reranker -n 5
  ```

### Step 7.2: Verify Component Isolation Invariants
Mathematically assert that each isolated arm has 100% component isolation with zero feature leakage (e.g., verifying that the pure dense arm contains no sparse or graph chunks, and that isolated arms only execute their designated component):

```bash
# Verify existing arm checkpoints in data/ablation_results/
poetry run python scripts/verify_ablation_isolation.py

# Or execute a fresh 5-sample benchmark and verify in one shot
poetry run python scripts/verify_ablation_isolation.py --run -n 5
```

### Step 7.3: Generate / Re-evaluate Dynamic Pareto Production Tiers
Once isolated arms exist in cache, you can re-run or evaluate the dynamic Pareto tiers alone:
```bash
poetry run python scripts/run_portfolio_ablations.py
```
This loads cached isolated arms, constructs Tier 1 (Fast) and Tier 2 (SOTA), and prints Table 1.

### Step 7.4: Regenerate Report from Cache (Zero LLM API Calls)
Regenerates the Markdown report (`data/ablation_results/ablation_report.md`) and summary JSON from existing checkpoints without making network calls:
```bash
poetry run python scripts/run_portfolio_ablations.py --report-only
```

### Step 7.5: Run Custom Pairwise A/B Hypothesis Testing
Executes head-to-head A/B experiments with paired Wilcoxon signed-rank tests and Student's t-tests:
```bash
poetry run python -m experiments.retrieval_experiment \
  --baseline '{"top_k_final": 5, "reranker_enabled": false}' \
  --variant  '{"top_k_final": 5, "reranker_enabled": true}' \
  --n 10 \
  --name "reranker_hypothesis_test" \
  --save
```

---

## Phase 8: Code Quality, Testing & Security Gates

Run the local test suite and quality gates to mirror the GitHub Actions CI/CD matrix.

### 1. Run Complete Test Suite
```bash
# Run all 958 unit and integration tests
poetry run pytest tests/

# Run tests with exact CI flags and coverage gate (≥80% required)
poetry run pytest tests/ -m "not integration" \
  --cov=ingestion --cov=query --cov=retrieval --cov=generation \
  --cov=observability --cov=knowledge_graph --cov=evaluation \
  --cov=api --cov=config --cov-fail-under=80
```

### 2. Code Formatting & Linting
```bash
# Check code style and formatting
poetry run ruff check .

# Automatically apply safe formatting fixes
poetry run ruff format .
poetry run ruff check --fix .
```

### 3. Static Type Checking
```bash
poetry run mypy ingestion/ query/ retrieval/ generation/ evaluation/ observability/ api/ config/ knowledge_graph/ --ignore-missing-imports --disallow-untyped-defs --pretty
```

### 4. Security Scanning & Dependency Audit
```bash
# Scan for Python security vulnerabilities
poetry run bandit -r api config generation ingestion observability query retrieval scripts -c pyproject.toml

# Audit installed packages for known CVEs
poetry run pip-audit
```

---

## Phase 9: Maintenance, Recovery & Reset Procedures

Procedures to clean, rebuild, or recover pipeline components from scratch.

### 1. Rebuild BM25 Sparse Index Only
If you need to re-score or update BM25 parameters without modifying Qdrant vectors:
```bash
poetry run python scripts/rebuild_bm25.py
```

### 2. Re-extract Knowledge Graph Only
If the Knowledge Graph needs to be regenerated or updated without re-embedding vectors:
```bash
poetry run python -m ingestion.pipeline --kg-only
```

### 3. Complete Pipeline & Index Reset
Completely purges the local Qdrant collection, BM25 indices, ingestion state DB, and cached knowledge graph:
```bash
poetry run python scripts/reset_index.py
```

### 4. Docker Teardown & Volume Reset
```bash
# Stop all containers
docker compose down

# Stop containers and permanently delete all persistent volumes
docker compose down -v
```

### 5. Semantic Cache Invalidation & Management
The semantic cache indexes query vectors in Qdrant collection `semantic_cache` tagged with company tickers and timestamps. When newly filed 10-K or 10-Q reports are ingested, cached responses for those specific tickers must be invalidated to prevent stale GAAP retrieval:

```python
# Programmatic cache invalidation per ticker (e.g. after ingesting MSFT FY2024)
import asyncio
from retrieval.semantic_cache import SemanticCache

async def purge():
    cache = SemanticCache()
    # Invalidate cached queries touching MSFT
    count = await cache.invalidate_ticker("MSFT")
    print(f"Purged cached entries for MSFT")

asyncio.run(purge())
```

Via REST API:
```bash
# Invalidate cache for a specific ticker via API
curl -X POST "http://localhost:8000/query/cache/invalidate?ticker=NVDA"

# Flush entire semantic cache
curl -X POST "http://localhost:8000/query/cache/invalidate"
```

To view semantic cache hit rate, eviction count, and total lookups:
```python
cache = SemanticCache()
print(cache.stats())
# {'hits': 42, 'misses': 108, 'evictions': 0, 'total_lookups': 150, 'hit_rate': 0.28}
```

---

## ⚡ Quick Reference Cheat Sheet

| Step | Goal | Exact Command |
| :---: | :--- | :--- |
| **1** | Install dependencies | `poetry install` |
| **2** | Configure secrets | `cp .env.example .env` *(or run `gcloud auth application-default login` for ADC)* |
| **3** | Start backing infrastructure | `docker compose up -d qdrant redis jaeger prometheus grafana` |
| **4** | Download SEC filings | `poetry run python -m ingestion.download_filings` |
| **5** | Run ingestion & indexing | `poetry run python -m ingestion.pipeline` |
| **6** | Launch production server | `poetry run serve-prod` |
| **7** | Open web chat interface | Navigate to `http://localhost:8000/app` |
| **8** | Run 129-QA evaluation | `poetry run python evaluation/harness.py` |
| **9** | Run isolated ablations | `poetry run python scripts/run_portfolio_ablations.py --isolated --all` |
| **10** | Run test suite | `poetry run pytest tests/` |
| **11** | Full system reset | `poetry run python scripts/reset_index.py` |
