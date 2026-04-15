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

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.3
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /build

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.in-project true \
    && poetry install --no-root --no-interaction --no-ansi --without dev


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy the pre-built venv from builder
COPY --from=builder /build/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy application source
COPY --chown=appuser:appuser . .

# ── FIX: create dirs and set ownership BEFORE switching to non-root user ──────
# Running as root here, so mkdir succeeds. Then chown hands the whole
# /app/data tree to appuser so the app can write BM25 index, checkpoints, etc.
RUN mkdir -p /app/data/transcripts /app/data/eval_reports \
    && chown -R appuser:appuser /app/data

# Switch to non-root for all runtime execution
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health/live || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
