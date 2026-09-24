# Contributing to Financial RAG System (Earnings Oracle)

Thank you for your interest in contributing to the Financial RAG System! We welcome bug fixes, documentation improvements, evaluation dataset extensions, and architectural enhancements.

This document outlines our development process, standards, and guidelines.

---

## Code of Conduct

We are committed to providing a welcoming, inclusive, and professional environment. Contributors are expected to uphold respectful and constructive collaboration at all times.

---

## Development Setup

### 1. Prerequisites
- **Python 3.11** or **3.12**
- **Poetry** (package manager): `curl -sSL https://install.python-poetry.org | python3 -`
- **Docker & Docker Compose** (for vector database & observability services)
- **Google Cloud Application Default Credentials (ADC)** (`gcloud auth application-default login`, zero keys required) or **OpenAI API Key**

### 2. Clone and Install
```bash
git clone https://github.com/deepakj111/earnings-oracle.git
cd earnings-oracle

# Install all project and development dependencies
poetry install

# Activate virtual environment
poetry shell
```

### 3. Install Pre-Commit Hooks
Our repository uses `pre-commit` to guarantee clean styling, strict typing, and secret protection:
```bash
poetry run pre-commit install
```

### 4. Start Local Infrastructure
```bash
# Launch Qdrant vector database and Prometheus monitoring
docker compose up -d qdrant prometheus grafana
```

---

## Development Workflow

1. **Branch Naming**:
   Create a descriptive feature branch from `main`:
   - `feat/feature-name`
   - `fix/bug-description`
   - `docs/documentation-update`
   - `perf/latency-optimization`

2. **Semantic Commit Messages**:
   Follow Conventional Commits:
   - `feat(retrieval): implement dynamic candidate scaling for comparative queries`
   - `fix(middleware): resolve rate limiting sliding window reset`
   - `docs(llmops): update prometheus alert thresholds`
   - `test(guardrails): add prompt injection adversarial suite`

3. **Using the Makefile**:
   Convenience targets are provided for standard development workflows:
   ```bash
   make help       # List all available targets
   make test       # Run test suite
   make lint       # Ruff lint check
   make format     # Auto-format and fix
   make typecheck  # Mypy type validation
   make audit      # Security audit (Bandit + pip-audit)
   make eval-fast  # 5-sample evaluation run
   ```

---

## Extending the Pipeline

### 1. Adding a New Retrieval Technique
1. Implement the search logic as an isolated module in `retrieval/` or method in `retrieval/searcher.py`.
2. Ensure candidates return normalized `SearchResult` objects containing `chunk_id`, `score`, `text`, and metadata (`ticker`, `filing_type`, `fiscal_period`).
3. Integrate candidate rankings into Reciprocal Rank Fusion (RRF) in `retrieval/searcher.py`:
   $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
4. Feature-gate the technique in `config/settings.py` via an environment variable (`RAG_<TECHNIQUE>_ENABLED`).
5. Write a corresponding Architectural Decision Record in `docs/DESIGN_DECISIONS.md`.

### 2. Contributing to the Golden Evaluation Dataset
The golden evaluation benchmark (`data/golden_dataset.json`) powers regression testing:
1. Each entry must be sourced from official SEC 10-K or 10-Q filings for the portfolio companies (NVDA, WMT, NFLX, UNH, or custom configured tickers).
2. Provide ground-truth numbers verified against the original filing tables.
3. Include the exact SEC accession number, item section (e.g. `Item 7 MD&A`, `Item 8 Financial Statements`), and fiscal period.
4. Verify your proposed entry using:
   ```bash
   poetry run python scripts/generate_golden_dataset.py --verify-only
   ```

### 3. Benchmarking Ablations & Regression Testing
When contributing an algorithmic or prompting change:
1. Run the ablation test isolation runner to verify your component does not leak state:
   ```bash
   poetry run python scripts/verify_ablation_isolation.py
   ```
2. Run the evaluation harness to compare metric averages against baseline:
   ```bash
   poetry run python evaluation/harness.py --sample-size 10
   ```

---

## Quality & Testing Gates

Before opening a Pull Request, verify that all local checks pass:

### 1. Linting & Formatting
```bash
make lint
make format
```

### 2. Static Type Checking
```bash
make typecheck
```

### 3. Security Auditing
```bash
make audit
```

### 4. Automated Tests & Coverage
All changes must maintain or increase overall test coverage ($\ge 80\%$ required by CI):
```bash
poetry run pytest tests/ -v --cov-fail-under=80
```

---

## Submitting a Pull Request

1. Push your branch to GitHub.
2. Open a Pull Request against `main`.
3. Fill out the PR template with:
   - Context and motivation for the change.
   - Summary of changes made across modules.
   - Verification steps and test command outputs.
4. Ensure all 7 GitHub Actions CI jobs pass (including the `ci-gate`).
