# Dockerfile
# Multi-stage build for the Financial RAG System.
#
# Stages:
#   1. builder  — installs Poetry and all Python dependencies into a venv
#   2. runtime  — copies only the venv + source, no build tooling
#
# Usage:
#   docker build -t financial-rag:latest .
#
#   # API server
#   docker run -p 8000:8000 --env-file .env \
#       financial-rag:latest uvicorn api.main:app --host 0.0.0.0 --port 8000
#
#   # Streamlit UI
#   docker run -p 8501:8501 --env-file .env \
#       -e RAG_API_URL=http://api:8000 \
#       financial-rag:latest streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System packages required only during build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (pinned version for reproducibility)
ENV POETRY_VERSION=1.8.3
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /build

# Copy dependency manifests first — layer cache is valid until these change
COPY pyproject.toml poetry.lock* ./

# Install into a venv inside /build/.venv — no system-wide pollution
RUN poetry config virtualenvs.in-project true \
    && poetry install --no-root --no-interaction --no-ansi --without dev

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system packages only (lxml needs libxml2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy the pre-built venv from builder — no pip/poetry needed at runtime
COPY --from=builder /build/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy application source (excludes .env, data/, tests/ via .dockerignore)
COPY --chown=appuser:appuser . .

USER appuser

# Create data directory for runtime artefacts (BM25 index, checkpoints)
RUN mkdir -p /app/data/transcripts /app/data/eval_reports

# Healthcheck for the API container
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health/live || exit 1

# Default command — override in docker-compose or CLI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
