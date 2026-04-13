# tests/test_api_health.py
"""
Tests for api/routes/health.py

Coverage:
  GET /health/live   — always-200 liveness probe
  GET /health/ready  — 200 when pipeline ready, 503 during cold-start
  GET /health        — full dependency health: healthy / degraded / unhealthy
                       states for Qdrant, pipeline, and BM25 index
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# ─────────────────────────────────────────────────────────────────────────────
# Liveness
# ─────────────────────────────────────────────────────────────────────────────


class TestLiveness:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health/live")
        assert resp.status_code == 200

    def test_body_status_alive(self, client: TestClient) -> None:
        resp = client.get("/health/live")
        assert resp.json() == {"status": "alive"}

    def test_no_external_calls(self, client: TestClient, mock_qdrant: MagicMock) -> None:
        """Liveness must never touch Qdrant or the pipeline."""
        client.get("/health/live")
        mock_qdrant.get_collections.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Readiness
# ─────────────────────────────────────────────────────────────────────────────


class TestReadiness:
    def test_ready_returns_200_when_pipeline_initialised(self, client: TestClient) -> None:
        resp = client.get("/health/ready")
        assert resp.status_code == 200

    def test_ready_body_contains_status(self, client: TestClient) -> None:
        resp = client.get("/health/ready")
        assert resp.json()["status"] == "ready"

    def test_ready_returns_503_when_pipeline_missing(self, client_no_pipeline: TestClient) -> None:
        resp = client_no_pipeline.get("/health/ready")
        assert resp.status_code == 503

    def test_ready_503_detail_message(self, client_no_pipeline: TestClient) -> None:
        resp = client_no_pipeline.get("/health/ready")
        assert "not yet initialised" in resp.json()["detail"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# Full health check — healthy path
# ─────────────────────────────────────────────────────────────────────────────


class TestFullHealthHealthy:
    def test_returns_200(self, client: TestClient) -> None:
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat.return_value.st_size = 5 * 1_048_576  # 5 MB
            resp = client.get("/health/")
        assert resp.status_code == 200

    def test_status_healthy(self, client: TestClient) -> None:
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat.return_value.st_size = 5 * 1_048_576
            data = client.get("/health/").json()
        assert data["status"] == "healthy"

    def test_response_has_version(self, client: TestClient) -> None:
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat.return_value.st_size = 1_048_576
            data = client.get("/health/").json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    def test_response_has_uptime_seconds(self, client: TestClient) -> None:
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat.return_value.st_size = 1_048_576
            data = client.get("/health/").json()
        assert "uptime_seconds" in data
        # uptime was set to 60 s ago in the fixture
        assert data["uptime_seconds"] >= 0

    def test_all_components_present(self, client: TestClient) -> None:
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat.return_value.st_size = 1_048_576
            data = client.get("/health/").json()
        components = data["components"]
        assert "qdrant" in components
        assert "pipeline" in components
        assert "bm25_index" in components

    def test_qdrant_component_ok(self, client: TestClient) -> None:
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat.return_value.st_size = 1_048_576
            data = client.get("/health/").json()
        assert data["components"]["qdrant"]["status"] == "ok"
        assert "5000" in data["components"]["qdrant"]["detail"]

    def test_pipeline_component_ok(self, client: TestClient) -> None:
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat.return_value.st_size = 1_048_576
            data = client.get("/health/").json()
        assert data["components"]["pipeline"]["status"] == "ok"

    def test_bm25_component_ok_when_file_exists(self, client: TestClient) -> None:
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat.return_value.st_size = 10 * 1_048_576  # 10 MB
            data = client.get("/health/").json()
        assert data["components"]["bm25_index"]["status"] == "ok"
        assert "10.0 MB" in data["components"]["bm25_index"]["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# Full health check — degraded / unhealthy paths
# ─────────────────────────────────────────────────────────────────────────────


class TestFullHealthDegraded:
    def test_degraded_when_qdrant_unreachable(
        self, client: TestClient, mock_qdrant: MagicMock
    ) -> None:
        mock_qdrant.get_collections.side_effect = ConnectionError("Connection refused")
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat.return_value.st_size = 1_048_576
            data = client.get("/health/").json()
        assert data["status"] == "degraded"
        assert data["components"]["qdrant"]["status"] == "error"

    def test_degraded_when_collection_missing(
        self, client: TestClient, mock_qdrant: MagicMock
    ) -> None:
        # Return an empty collections list
        empty_resp = MagicMock()
        empty_resp.collections = []
        mock_qdrant.get_collections.return_value = empty_resp

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat.return_value.st_size = 1_048_576
            data = client.get("/health/").json()
        assert data["status"] == "degraded"
        assert "NOT found" in data["components"]["qdrant"]["detail"]

    def test_degraded_when_bm25_missing(self, client: TestClient) -> None:
        with patch("pathlib.Path.exists", return_value=False):
            data = client.get("/health/").json()
        assert data["status"] == "degraded"
        assert data["components"]["bm25_index"]["status"] == "error"
        assert "not found" in data["components"]["bm25_index"]["detail"]

    def test_unhealthy_when_pipeline_error(
        self, client: TestClient, mock_qdrant: MagicMock
    ) -> None:
        """If the pipeline raises during health check the service is unhealthy."""
        import time

        import api.main

        @asynccontextmanager
        async def _bad_pipeline_lifespan(app):  # type: ignore[no-untyped-def]
            # Store qdrant but no pipeline → get_pipeline raises AttributeError
            app.state.qdrant = mock_qdrant
            app.state.startup_time = time.time()
            yield

        with patch("api.main.lifespan", _bad_pipeline_lifespan):
            test_app = api.main.create_app()

        with TestClient(test_app, raise_server_exceptions=False) as c:
            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.stat") as mock_stat,
            ):
                mock_stat.return_value.st_size = 1_048_576
                data = c.get("/health/").json()

        assert data["status"] == "unhealthy"
        assert data["components"]["pipeline"]["status"] == "error"

    def test_points_count_none_does_not_crash(
        self, client: TestClient, mock_qdrant: MagicMock
    ) -> None:
        """
        Regression test for FIX 1: points_count=None must not crash the
        health endpoint (qdrant-client 1.7+ returns Optional[int]).
        """
        mock_coll_info = MagicMock()
        mock_coll_info.points_count = None  # <-- the bug condition
        mock_qdrant.get_collection.return_value = mock_coll_info

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat.return_value.st_size = 1_048_576
            resp = client.get("/health/")
        assert resp.status_code == 200
        # Should show 0 points, not crash
        assert "0 points" in resp.json()["components"]["qdrant"]["detail"]

    def test_get_collection_failure_isolated(
        self, client: TestClient, mock_qdrant: MagicMock
    ) -> None:
        """
        Regression test for FIX 2: get_collection() failure must not mark
        the whole qdrant component as fully unreachable.
        get_collections() succeeds → collection found.
        get_collection() raises → detail note, but qdrant status still 'ok'.
        """
        mock_qdrant.get_collection.side_effect = RuntimeError("gRPC timeout")

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat.return_value.st_size = 1_048_576
            data = client.get("/health/").json()

        # qdrant itself was reachable — collections listing worked
        assert data["components"]["qdrant"]["status"] == "ok"
        assert "unavailable" in data["components"]["qdrant"]["detail"]
