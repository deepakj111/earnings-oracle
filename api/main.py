"""
Financial RAG API — application entry point.

Run with uvicorn:
    # Development (auto-reload on code changes)
    poetry run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

    # Production (4 worker processes, production-grade server)
    poetry run uvicorn api.main:app --workers 4 --host 0.0.0.0 --port 8000

    # Or via the CLI shortcut defined in pyproject.toml:
    poetry run serve

OpenAPI docs (auto-generated):
    http://localhost:8000/docs   — Swagger UI
    http://localhost:8000/redoc  — ReDoc

Architecture:
  This module wires together:
    - Lifespan context manager  (startup / shutdown hooks)
    - CORS + middleware stack    (RequestID → Timing → CORS)
    - Exception handler registry (typed error → HTTP status mapping)
    - Router mounting            (/query, /health)

Design decisions:
  1. Application factory pattern (create_app()) so the app can be
     instantiated in tests with different settings without running startup.

  2. Heavy models (BAAI/bge, BM25, FlashRank) are pre-loaded in the lifespan
     startup rather than lazily on first request.  This gives:
       - Predictable cold-start latency (pays it at process start, not first user)
       - Kubernetes readiness probe accuracy (/health/ready returns 200 only
         after models are loaded)

  3. CORS is configured permissively (allow_origins=["*"]) for development.
     Tighten this in production by setting RAG_CORS_ORIGINS env var.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from qdrant_client import QdrantClient

from api.errors import register_exception_handlers
from api.middleware import RequestIDMiddleware, TimingMiddleware
from api.routes import health, query
from config import settings
from rag_pipeline import FinancialRAGPipeline

# UNIX timestamp at import time — used by /health to compute uptime
_PROCESS_START: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: pre-load all models at startup, clean up on shutdown.

    FastAPI guarantees this completes before the first request is served,
    which means the Kubernetes readiness probe correctly returns 503 during
    the model-loading phase.

    Startup sequence:
      1. Validate required environment variables (fail fast if misconfigured)
      2. Open Qdrant TCP connection
      3. Instantiate FinancialRAGPipeline — this triggers model downloads
         and loads all ONNX models into memory:
           BAAI/bge-large-en-v1.5    ~340 MB (embedding)
           BM25 index                 ~30-100 MB (keyword search)
           ms-marco-MiniLM-L-12-v2   ~66 MB (reranker, if enabled)

    Expected startup time:
      First run: 2–5 min (model download from HuggingFace Hub)
      Subsequent runs: 10–20 s (load from local cache)
    """
    logger.info("=" * 60)
    logger.info("Financial RAG API — startup initiated")
    logger.info("=" * 60)

    # ── Step 1: Config validation ──────────────────────────────────────────────
    try:
        settings.validate()
        logger.info("Config validation passed.")
    except OSError as exc:
        logger.critical(f"Config validation FAILED — cannot start: {exc}")
        raise  # Process exits; Kubernetes restarts the pod

    # ── Step 2: Qdrant connection ──────────────────────────────────────────────
    logger.info(f"Connecting to Qdrant at {settings.infra.qdrant_url} ...")
    qdrant = QdrantClient(url=settings.infra.qdrant_url)
    logger.info("Qdrant connection established.")

    # ── Step 3: Pipeline init (pre-loads all models) ───────────────────────────
    logger.info("Loading RAG pipeline and all models into memory ...")
    pipeline = FinancialRAGPipeline(qdrant_client=qdrant)

    # ── Store singletons on app.state for dependency injection ─────────────────
    app.state.pipeline = pipeline
    app.state.qdrant = qdrant
    app.state.startup_time = _PROCESS_START

    logger.info("=" * 60)
    logger.info("Financial RAG API — ready to serve requests")
    logger.info("=" * 60)

    yield  # ── Serve requests ────────────────────────────────────────────────

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("Financial RAG API — shutdown initiated")
    qdrant.close()
    logger.info("Qdrant connection closed.")


def create_app() -> FastAPI:
    """
    Application factory.

    Called once at module load time to produce the ASGI app.  Isolating
    construction here allows tests to call create_app() without triggering
    the lifespan startup (use TestClient with lifespan='off' in unit tests
    or mock app.state in integration tests).
    """
    app = FastAPI(
        title="Financial RAG API",
        description=(
            "Production-grade Retrieval-Augmented Generation system for querying "
            "SEC 8-K earnings filings from 10 major public companies.\n\n"
            "Uses a hybrid retrieval approach — dense vector search (BAAI/bge-large-en-v1.5 + "
            "Qdrant) combined with sparse keyword search (BM25) — fused via Reciprocal Rank "
            "Fusion and reranked with a FlashRank cross-encoder.  Query transformation uses "
            "HyDE, multi-query expansion, and step-back prompting for maximum recall."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {
                "name": "Query",
                "description": (
                    "Ask financial questions over SEC 8-K earnings filings. "
                    "Supports structured JSON responses and streaming SSE."
                ),
            },
            {
                "name": "Health",
                "description": (
                    "Liveness, readiness, and full dependency health checks. "
                    "Designed for Kubernetes probes and monitoring dashboards."
                ),
            },
        ],
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    # Registration order is REVERSED — Starlette wraps from inside out.
    # Innermost (registered last) executes first on request ingress.

    # 1. CORS — outermost so preflight OPTIONS requests are handled before
    #    any application logic runs.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # ⚠  Tighten in production: ["https://your-app.com"]
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
    )

    # 2. Timing — reads request_id set by RequestIDMiddleware below
    app.add_middleware(TimingMiddleware)

    # 3. Request ID — innermost, sets request.state.request_id first
    app.add_middleware(RequestIDMiddleware)

    # ── Exception handlers ────────────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(query.router, prefix="/query", tags=["Query"])
    app.include_router(health.router, prefix="/health", tags=["Health"])

    return app


# Module-level app instance — this is what uvicorn imports.
app = create_app()
