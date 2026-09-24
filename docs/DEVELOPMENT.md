# Development Guide

> Local setup, coding standards, testing workflow, and contribution process for the Financial RAG System.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Setup](#local-setup)
3. [Project Layout Conventions](#project-layout-conventions)
4. [Running the Stack Locally](#running-the-stack-locally)
5. [Testing](#testing)
6. [Code Quality Gates](#code-quality-gates)
7. [Adding a New Feature](#adding-a-new-feature)
8. [Configuration Reference](#configuration-reference)
9. [Debugging & Diagnostics](#debugging--diagnostics)
10. [Common Issues](#common-issues)

---

## Prerequisites

| Tool | Minimum Version | Install |
|------|----------------|---------|
| Python | 3.11 | [python.org](https://www.python.org/downloads/) |
| Poetry | 1.8.0 | `curl -sSL https://install.python-poetry.org \| python3 -` |
| Docker | 24.0 | [docker.com](https://docs.docker.com/get-docker/) |
| Git | 2.40 | system package manager |

Verify:
```bash
python --version     # Python 3.11.x or 3.12.x
poetry --version     # Poetry 1.8.x
docker --version     # Docker 24.x
```

---

## Local Setup

### 1. Clone and bootstrap

```bash
git clone https://github.com/deepakj111/earnings-oracle.git
cd earnings-oracle      # or cd rag-project
poetry install          # installs all deps including dev group
```

### 2. Configure environment

```bash
cp .env.example .env

# Option A (Recommended): Google Cloud Application Default Credentials (Zero Keys)
gcloud auth application-default login

# Option B: OpenAI API Key
# Set RAG_LLM_PROVIDER=openai and OPENAI_API_KEY in .env
```

Edit `.env` (minimum required keys):

```dotenv
# Provider selection: "gemini" (default with ADC) or "openai"
RAG_LLM_PROVIDER="gemini"
GOOGLE_CLOUD_PROJECT="gleaming-vision-509507-j6"
GOOGLE_CLOUD_LOCATION="us-central1"

# Required by SEC EDGAR fair-access policy
SEC_USER_AGENT="Firstname Lastname firstname@example.com"
QDRANT_URL=http://localhost:6333
```

### 3. Install pre-commit hooks

```bash
poetry run pre-commit install
# Hooks now run automatically on every git commit
```

### 4. Start backing infrastructure

Start the local backing services (Qdrant vector DB, Redis cache, Jaeger tracing, Prometheus, Grafana) via Docker Compose:

```bash
docker compose up -d qdrant redis jaeger prometheus grafana
```

### 5. Ingest data

```bash
# Download SEC 10-K/10-Q filings (default portfolio: NVDA, WMT, NFLX, UNH)
poetry run python -m ingestion.download_filings

# Or download custom company filings (e.g. AAPL, MSFT)
# See docs/ADDING_COMPANIES.md for onboarding any ticker via config/companies.json
poetry run python -m ingestion.download_filings --tickers AAPL,MSFT

# Build Qdrant + BM25 + Knowledge Graph index (idempotent, resumable)
poetry run python -m ingestion.pipeline

# Ingestion options:
#   --fast                    Instant regex entity extraction (skip LLM network calls)
#   --contextual-retrieval    Enable Anthropic-style LLM contextual chunk enrichment
#   --kg-only                 Re-extract Knowledge Graph on existing indexed files
#   --concurrency N           Set worker concurrency limit
```

### 6. Start services

```bash
# Production server (multi-worker)
poetry run serve-prod

# Or development server (auto-reload)
poetry run serve

# Streamlit UI (optional, separate terminal)
poetry run ui
```

Access:
- **Web App**: http://localhost:8000/app (or http://localhost:8000/)
- **API Swagger Docs**: http://localhost:8000/docs
- **Jaeger APM Tracing**: http://localhost:16686
- **Prometheus Metrics**: http://localhost:9090
- **Grafana Dashboard**: http://localhost:3000 (admin / admin)
- **Master Operations Runbook**: See [RUNBOOK.md](RUNBOOK.md) for the complete command reference.

---

## Project Layout Conventions

### Module structure

Each pipeline layer follows this pattern:

```
layer_name/
  __init__.py       # Public API — only export what callers need
  models.py         # Pydantic / dataclass data contracts
  <core>.py         # Business logic
  prompts.py        # LLM prompt templates (if applicable)
```

### Naming conventions

| Item | Convention | Example |
|------|-----------|---------|
| Modules | `snake_case` | `context_builder.py` |
| Classes | `PascalCase` | `QueryTransformer` |
| Functions | `snake_case` | `_build_qdrant_filter` |
| Private functions | `_snake_case` (single underscore) | `_run_hyde` |
| Constants | `UPPER_SNAKE_CASE` | `PARENT_TOKEN_TARGET = 512` |
| Type hints | Always present on all function signatures | Required by mypy |

### Import ordering

Ruff `isort` configuration sorts imports into groups:

```python
# Standard library
import re
from pathlib import Path

# Third-party
from loguru import logger
from qdrant_client import QdrantClient

# First-party (alphabetical within group)
from config import settings
from retrieval.models import SearchResult
```

First-party packages: `ingestion`, `query`, `retrieval`, `generation`, `evaluation`, `knowledge_graph`, `observability`, `api`, `config`, `ui`.

### Type annotations

All functions in `ingestion/` and all public APIs require complete type annotations (enforced by mypy). Private functions (`_`-prefixed) should also be annotated but are not strictly enforced.

```python
# Good
def _build_parents(
    sections: list[tuple[str, str]],
    ticker: str,
    date: str,
    doc_type: str,
) -> list[Chunk]:

# Bad — missing return type
def _build_parents(sections, ticker, date, doc_type):
```

---

## Running the Stack Locally

### Development mode (recommended)

```bash
# Terminal 1: Qdrant
docker run -p 6333:6333 qdrant/qdrant:v1.9.2

# Terminal 2: API (auto-reload on code changes)
poetry run serve

# Terminal 3: UI
poetry run ui
```

### Production-like mode

```bash
# All services via docker-compose
cp .env.example .env   # fill in values
docker compose up -d

# Run ingestion
docker compose exec api poetry run python -m ingestion.download_filings
docker compose exec api poetry run python -m ingestion.pipeline
```

### Inspect the index

```bash
# Comprehensive diagnostic tool
poetry run inspect-data
```

Output includes filesystem stats, BM25 corpus summary (per-ticker/quarter distribution, avg token length), and Qdrant collection stats (point counts per ticker, sample payloads, consistency check).

---

## Testing

### Running tests

```bash
# Full suite (885 tests with verbose output and timing)
poetry run pytest tests/ -v --durations=10

# Quiet (just counts)
poetry run pytest tests/ -q

# With coverage report
poetry run pytest tests/ --cov=. --cov-report=term-missing

# Single module
poetry run pytest tests/test_chunker.py -v

# Single test class
poetry run pytest tests/test_api_query.py::TestAskEndpoint -v

# Single test
poetry run pytest tests/test_generator.py::TestGeneratorGenerate::test_empty_retrieval_returns_no_context_answer -v
```

### Ablation Benchmarking & Component Isolation Testing

The RAG portfolio ablation framework (`scripts/run_portfolio_ablations.py`) enforces strict, unconfounded component isolation between architecture variants. Each isolated arm enables **exactly one** feature over the dense-only baseline:

| Isolated Arm | Feature Tested | All Others |
|:---|:---|:---|
| `bm25` | BM25 keyword matching + hybrid RRF (k=60) | Disabled |
| `querytransform` | HyDE + 3× Multi-Query + Step-Back prompting | Disabled |
| `reranker` | FlashRank cross-encoder (`ms-marco-MiniLM-L-12-v2`) | Disabled |
| `graphrag` | Knowledge Graph entity matching & multi-hop context | Disabled |
| `pal_math` | Sentence NLI Grounding Verification & PAL Math Evaluator | Disabled |

```bash
# Run all isolated arms on the dataset
poetry run python scripts/run_portfolio_ablations.py --isolated --all

# Run automated zero-leakage invariant assertions across all arms (5 samples)
poetry run python scripts/verify_ablation_isolation.py --run -n 5

# Run isolation unit tests
poetry run pytest tests/test_verify_ablation_isolation.py tests/test_retrieval_experiment.py -v
```

### Test architecture

**Fixture strategy** (`tests/conftest.py`):

The FastAPI app's lifespan downloads ML models and validates API keys — both unacceptable in unit tests. The test suite patches `api.main.lifespan` with a lightweight `_test_lifespan` context manager that sets `app.state` from mock fixtures instead of running the real startup.

```python
# conftest.py pattern
@asynccontextmanager
async def _test_lifespan(app):
    app.state.pipeline = mock_pipeline   # MagicMock
    app.state.qdrant = mock_qdrant       # MagicMock
    app.state.startup_time = time.time() - 60.0
    yield

with patch("api.main.lifespan", _test_lifespan):
    test_app = api.main.create_app()

with TestClient(test_app, raise_server_exceptions=False) as c:
    yield c
```

`raise_server_exceptions=False` allows exception handler tests to assert HTTP status codes rather than having exceptions propagate through `TestClient`.

**Mock pipeline fixture**: Pre-wired with:
- `mock_pipeline.ask()` → `sample_generation_result` (grounded, 1 citation)
- `mock_pipeline.ask_verbose()` → `(result, query_summary, retrieval_summary)`
- `mock_pipeline.ask_streaming()` → iterator of 5 token strings

**BM25 tests**: Write a minimal in-memory BM25 index + corpus to `tmp_path` (pytest fixture), then patch the module-level path constants. No disk dependency, no fixture files to maintain.

### Writing new tests

Follow these conventions:

```python
# tests/test_<module>.py

class TestMyClass:
    """Tests for MyClass — one class per logical unit under test."""

    def test_<behaviour>_when_<condition>(self):
        """Docstring optional but test name must be self-documenting."""
        # Arrange
        ...
        # Act
        result = my_function(...)
        # Assert
        assert result.field == expected_value

    @pytest.mark.parametrize("input,expected", [
        ("case1", "output1"),
        ("case2", "output2"),
    ])
    def test_handles_multiple_cases(self, input, expected):
        assert transform(input) == expected
```

**Patching LLM calls**: Always patch at the call site, not the import site:

```python
# Correct — patches the function where it's called from
with patch("generation.generator._call_llm") as mock:
    mock.return_value = ("Answer [1].", 200, 60)
    result = generator.generate(...)

# Wrong — patches the definition, not the usage
with patch("openai.OpenAI") as mock:
    ...
```

**Coverage requirements**: New modules must achieve ≥80% coverage. The CI gate enforces `--cov-fail-under=80` across all covered modules.

---

## Code Quality Gates

### Pre-commit hooks (run on every `git commit`)

```bash
# Run manually against all files
poetry run pre-commit run --all-files

# Run a single hook
poetry run pre-commit run ruff --all-files
poetry run pre-commit run mypy --all-files
```

| Hook | What it checks | Auto-fix? |
|------|---------------|-----------|
| `trailing-whitespace` | Trailing spaces | ✅ Yes |
| `end-of-file-fixer` | Files end with newline | ✅ Yes |
| `check-yaml` / `check-toml` / `check-json` | File validity | ❌ No |
| `check-added-large-files` | Files >500 KB | ❌ No |
| `detect-private-key` | Accidental API key commits | ❌ No |
| `debug-statements` | Forgotten `breakpoint()` / `pdb` | ❌ No |
| `ruff` | PEP8 + bugbear + isort | ✅ Yes (`--fix`) |
| `ruff-format` | Code formatting | ✅ Yes |
| `mypy` | Type annotations | ❌ No |
| `bandit` | Security patterns | ❌ No |
| `trufflehog` | Verified secret scanning (staged changes) | ❌ No |

For a deep dive into our automation pipelines, see [CI/CD & Automation](CI_CD.md).

### Ruff configuration

```toml
# pyproject.toml
[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP", "N"]
ignore = [
    "E501",  # line too long — handled by ruff-format
    "B008",  # no function calls in defaults
]
```

Key rules:
- `F` (pyflakes): undefined names, unused imports — hard errors
- `I` (isort): import ordering — auto-fixed
- `B` (bugbear): common bugs (mutable defaults, comparison issues)
- `UP` (pyupgrade): modernise syntax for Python 3.11 (`Optional[X]` → `X | None`)

### Mypy configuration

```toml
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true     # Third-party stubs may be missing
disallow_untyped_defs = true      # All functions must have annotations
no_strict_optional = true         # Optional[X] treated as X (pragmatic)
```

Scope: `ingestion/` only (strictest gate). Other modules are checked but without `--disallow-untyped-defs`.

### Bandit configuration

```toml
[tool.bandit]
exclude_dirs = ["tests", ".venv"]
skips = ["B101"]  # assert statements (fine in tests)
```

Findings are uploaded as SARIF to the GitHub Security tab (`continue-on-error: true` — security is a review signal, not a hard build blocker).

---

## Adding a New Feature

### Example: Adding a new retrieval technique

**1. Write the core logic** (`retrieval/my_technique.py`):

```python
from retrieval.models import SearchResult

def my_new_search(query: str, top_k: int) -> list[SearchResult]:
    """Docstring with Args/Returns."""
    ...
```

**2. Add configuration** (`config/settings.py`):

```python
@dataclass(frozen=True)
class RetrievalConfig:
    ...
    my_technique_enabled: bool = field(
        default_factory=lambda: _env_bool("RAG_MY_TECHNIQUE_ENABLED", True)
    )
```

**3. Wire into the retrieval pipeline** (`retrieval/searcher.py`):

```python
if settings.retrieval.my_technique_enabled:
    hits = my_new_search(query_text, top_k=cfg.top_k_dense)
    rrf_input.append((ids, "my_technique"))
```

**4. Write tests** (`tests/test_retrieval_my_technique.py`):

```python
class TestMyTechnique:
    def test_returns_relevant_results(self, tmp_path): ...
    def test_respects_top_k(self): ...
    def test_metadata_filter_applied(self): ...
    def test_failure_is_logged_and_continues(self): ...
```

**5. Update the golden dataset if new data patterns are expected** (`evaluation/dataset.py`).

**6. Run the full quality gate**:

```bash
poetry run pre-commit run --all-files
poetry run pytest tests/ --cov-fail-under=80
```

### Example: Adding a new API endpoint

**1. Create route** (`api/routes/my_route.py`):

```python
from fastapi import APIRouter
router = APIRouter()

@router.post("/my-endpoint", response_model=MyResponse, tags=["MyTag"])
async def my_endpoint(body: MyRequest, ...) -> MyResponse:
    ...
```

**2. Add Pydantic models** (`api/models.py`).

**3. Mount the router** (`api/main.py`):

```python
from api.routes import my_route
app.include_router(my_route.router, prefix="/my-prefix", tags=["MyTag"])
```

**4. Write tests** (`tests/test_api_my_route.py`) using the `client` fixture from `conftest.py`.

---

## Configuration Reference

All environment variables follow the pattern `RAG_<SECTION>_<KEY>`. See `config/settings.py` for the complete reference. Key variables for local development:

```dotenv
# Fast development cycle — disable expensive components
RAG_RERANKER_ENABLED=false       # Skip FlashRank (faster iteration)
RAG_CONTEXT_COMPRESSION_ENABLED=false # Skip contextual compression
RAG_KG_RETRIEVAL_ENABLED=false   # Skip knowledge graph retrieval
RAG_RETRIEVAL_TOP_K_DENSE=5      # Fewer Qdrant candidates
RAG_RETRIEVAL_TOP_K_FINAL=3      # Fewer final chunks

# Model selection (default: gemini-2.5-flash via ADC; or gpt-5-mini via OpenAI)
RAG_GENERATION_MODEL=gemini-2.5-flash
RAG_QUERY_TRANSFORM_MODEL=gemini-2.5-flash

# Evaluation
RAG_EVAL_OUTPUT_DIR=data/eval_reports
RAG_EVAL_MAX_WORKERS=2           # Conservative for rate limiting
```

---

## Debugging & Diagnostics

### Inspect the index

```bash
poetry run python scripts/inspect_index.py
```

Checks: filesystem (.htm files per ticker), BM25 corpus (chunk distribution, vocabulary size, avg token length), Qdrant (point counts per ticker/quarter, sample payloads, consistency cross-check).

### Pipeline verbose mode

```python
result, query_summary, retrieval_summary = pipeline.ask_verbose(
    "What was Apple's Q4 2024 revenue?"
)
print(query_summary)      # HyDE doc, all multi-queries, step-back query
print(retrieval_summary)  # Candidate counts, RRF scores, rerank scores
```

### Loguru structured logs

The pipeline logs structured events at each layer:

```
INFO  | QueryTransformer | Transforming query | 'What was Apple's revenue?'
INFO  | QueryTransformer | Transformation complete | 4 multi-queries | 0.95s
INFO  | retrieval.searcher | RRF fusion: 47 unique chunks → top 20 passed to reranker
INFO  | retrieval.reranker | Reranking: 20 candidates → 5 results (top score: 0.9412)
INFO  | generation.generator | Context built | chunks=5 | tokens=2048
INFO  | generation.generator | Generation complete | citations=3 | grounded=True | 1.23s
INFO  | FinancialRAGPipeline | Pipeline complete | total=3.18s (L2=0.95s L3=0.62s L4=1.23s)
```

### Check Prometheus metrics

```bash
# Raw metrics
curl -s http://localhost:8000/metrics | grep rag_

# Query specific metric
curl -s http://localhost:8000/metrics | grep rag_grounded_responses_total
```

### BM25 & Store consistency check

If Qdrant and BM25 counts diverge (inspect_data.py shows a mismatch):

```bash
# Full re-index (reset index files and Qdrant collection)
poetry run python scripts/reset_index.py
poetry run python -m ingestion.pipeline
```

---

## Common Issues

### `BM25 index not found at data/bm25_index.pkl`

The API starts but retrieval fails with 503. Run the ingestion pipeline:

```bash
poetry run python -m ingestion.pipeline
```

### `Neither GEMINI_API_KEY nor valid ADC found` / `OPENAI_API_KEY is not set`

Settings validation fails at startup if the credentials for the active `RAG_LLM_PROVIDER` are missing:
- When `RAG_LLM_PROVIDER="gemini"` (default): Ensure you have run `gcloud auth application-default login` or provided `GEMINI_API_KEY` in `.env`.
- When `RAG_LLM_PROVIDER="openai"`: Ensure `OPENAI_API_KEY` is present in `.env`.

### `Qdrant collection 'company_filings' NOT found`

The collection hasn't been created yet. Run:

```bash
poetry run python -m ingestion.pipeline
```

Or if Qdrant storage was wiped:

```bash
poetry run python scripts/reset_index.py
poetry run python -m ingestion.pipeline
```

### Pre-commit `mypy` fails on new file

Ensure all function signatures have complete type annotations:

```bash
poetry run mypy ingestion/ query/ retrieval/ generation/ evaluation/ observability/ api/ config/ knowledge_graph/ --ignore-missing-imports --disallow-untyped-defs --pretty
```

### Tests fail with `DuplicateTimeseries` Prometheus error

This happens if tests create multiple apps without using the test `client` fixture. Always use the `client` fixture from `conftest.py` which patches `api.main.lifespan` — it prevents the real lifespan (which creates Prometheus metrics) from running.

### `collect_items_mock_error` in pytest

If `ask_streaming` mock is exhausted between tests, reset it in the test:

```python
mock_pipeline.ask_streaming.return_value = iter(["token1", "token2"])
```

Iterators are one-shot — recreate them per test that consumes the stream.
