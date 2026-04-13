# api/routes/health.py
"""
Health check routes — /health, /health/live, /health/ready.

Three endpoints at different depths:

  GET /health/live   — liveness probe (Kubernetes)
    Is the process running?  Returns 200 immediately without touching any
    external dependency.  Should never return 5xx after startup.

  GET /health/ready  — readiness probe (Kubernetes)
    Is the app ready to serve traffic?  Returns 503 during cold start or if
    the pipeline singleton was not successfully initialised.  Load balancers
    use this to decide whether to send traffic to this pod.

  GET /health        — full health check (dashboards / ops)
    Actively probes all downstream dependencies — Qdrant reachability,
    collection existence, pipeline availability — and aggregates them into
    a single structured status response.  More expensive than the probes;
    intended for dashboards and monitoring systems, not tight Kubernetes loops.

Status vocabulary:
  healthy   — all components report 'ok'
  degraded  — some components are unavailable but the service can still answer
              some queries (e.g. Qdrant collection missing → retrieval fails,
              but the process itself is running)
  unhealthy — the pipeline singleton is unavailable; no queries can be served

FIX 1 (Bug): qdrant-client 1.7+ changed CollectionInfo.points_count to
  Optional[int].  Accessing it in an f-string without a guard crashes when
  the collection exists but is still being indexed (points_count=None).
  Fixed with `(points_count or 0)`.

FIX 2 (Bug): get_collection() was called inside the same try block as
  get_collections().  If get_collections() succeeded but the target
  collection's get_collection() raised (e.g., a transient gRPC error),
  we would fall into the `except` branch and mark qdrant as errored even
  though the real issue was isolated to a single get_collection() call.
  Fixed by wrapping get_collection() in its own inner try/except.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from qdrant_client import QdrantClient

from api.dependencies import get_pipeline, get_qdrant, get_uptime_seconds
from api.models import ComponentStatus, HealthResponse
from config import settings

router = APIRouter()

_API_VERSION = "0.1.0"


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Full dependency health check",
    description=(
        "Actively probes Qdrant and the RAG pipeline. "
        "Use /health/live and /health/ready for lightweight Kubernetes probes."
    ),
)
async def health(request: Request) -> HealthResponse:
    uptime = get_uptime_seconds(request)
    components: dict[str, ComponentStatus] = {}
    overall = "healthy"

    # ── 1. Qdrant reachability + collection existence ──────────────────────────
    try:
        qdrant: QdrantClient = get_qdrant(request)
        collections = qdrant.get_collections().collections
        collection_names = [c.name for c in collections]
        target = settings.embedding.collection_name
        found = target in collection_names

        if found:
            # FIX 2: isolate get_collection() from get_collections() so a
            # transient per-collection RPC error does not mask a healthy
            # Qdrant connection.
            try:
                info = qdrant.get_collection(target)
                # FIX 1: points_count is Optional[int] in qdrant-client 1.7+.
                # Guard against None to prevent f-string crash on empty or
                # still-indexing collections.
                points_count: int = info.points_count or 0
                detail = f"collection '{target}' present ({points_count} points)"
            except Exception as coll_exc:
                logger.warning(f"Health: get_collection('{target}') failed: {coll_exc}")
                detail = f"collection '{target}' present but info unavailable: {coll_exc}"

            components["qdrant"] = ComponentStatus(status="ok", detail=detail)
        else:
            components["qdrant"] = ComponentStatus(
                status="error",
                detail=(
                    f"collection '{target}' NOT found — "
                    "run `poetry run python -m ingestion.pipeline`"
                ),
            )
            overall = "degraded"

    except Exception as exc:
        logger.warning(f"Health: Qdrant probe failed: {exc}")
        components["qdrant"] = ComponentStatus(status="error", detail=str(exc))
        overall = "degraded"

    # ── 2. Pipeline / model availability ──────────────────────────────────────
    try:
        _ = get_pipeline(request)
        gen_model = settings.generation.model
        transform_model = settings.query_transform.model
        components["pipeline"] = ComponentStatus(
            status="ok",
            detail=f"generation={gen_model} | transform={transform_model}",
        )
    except Exception as exc:
        logger.warning(f"Health: pipeline probe failed: {exc}")
        components["pipeline"] = ComponentStatus(status="error", detail=str(exc))
        overall = "unhealthy"

    # ── 3. BM25 index on disk ─────────────────────────────────────────────────
    bm25_path = Path("data/bm25_index.pkl")
    if bm25_path.exists():
        size_mb = round(bm25_path.stat().st_size / 1_048_576, 1)
        components["bm25_index"] = ComponentStatus(
            status="ok", detail=f"{bm25_path} ({size_mb} MB)"
        )
    else:
        components["bm25_index"] = ComponentStatus(
            status="error",
            detail=f"{bm25_path} not found — run ingestion pipeline",
        )
        if overall == "healthy":
            overall = "degraded"

    return HealthResponse(
        status=overall,
        version=_API_VERSION,
        uptime_seconds=uptime,
        components=components,
    )


@router.get(
    "/live",
    summary="Liveness probe",
    description="Kubernetes liveness probe. Returns 200 if the process is alive.",
)
async def liveness() -> dict:
    """Never touches external dependencies — always returns 200 after startup."""
    return {"status": "alive"}


@router.get(
    "/ready",
    summary="Readiness probe",
    description=(
        "Kubernetes readiness probe. Returns 200 only when the pipeline singleton "
        "is initialised and ready to serve. Returns 503 during startup cold-start "
        "(model loading takes ~10–20s on first run)."
    ),
)
async def readiness(request: Request) -> dict:
    """
    Lightweight check that the pipeline was successfully initialised.
    Load balancers should hold traffic until this returns 200.
    """
    try:
        pipeline = get_pipeline(request)
        if pipeline is None:
            raise AttributeError("pipeline is None")
    except AttributeError:
        raise HTTPException(
            status_code=503, detail="Pipeline not yet initialised — retry shortly."
        ) from None
    return {"status": "ready"}
