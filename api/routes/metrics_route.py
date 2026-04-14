# api/routes/metrics_route.py
"""
GET /metrics — Prometheus scrape endpoint.

Serves metrics from RAG_REGISTRY (the application-level custom registry).
Prometheus polls this every 15 seconds as configured in prometheus.yml.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.metrics import RAG_REGISTRY

router = APIRouter(tags=["observability"])


@router.get("/metrics", include_in_schema=False)
def get_metrics() -> Response:
    """Prometheus scrape target. Returns metrics in Prometheus text format."""
    return Response(
        content=generate_latest(RAG_REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )
