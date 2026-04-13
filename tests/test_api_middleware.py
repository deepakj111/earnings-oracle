# tests/test_api_middleware.py
"""
Tests for api/middleware.py

Coverage:
  RequestIDMiddleware
    · Sets X-Request-ID response header when no header provided in request
    · Generated ID is non-empty (8-char UUID prefix)
    · Passes through caller-supplied X-Request-ID unchanged
    · Every request gets a unique ID when none is provided

  TimingMiddleware
    · Sets X-Response-Time-Ms response header on every response
    · Value is a non-negative integer string
    · Header is present for both 2xx and 4xx responses
"""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestRequestIDMiddleware:
    def test_response_contains_x_request_id_header(self, client: TestClient) -> None:
        resp = client.get("/health/live")
        assert "x-request-id" in resp.headers

    def test_generated_id_is_non_empty(self, client: TestClient) -> None:
        resp = client.get("/health/live")
        assert resp.headers["x-request-id"].strip() != ""

    def test_caller_supplied_id_is_passed_through(self, client: TestClient) -> None:
        custom_id = "my-trace-id-42"
        resp = client.get("/health/live", headers={"X-Request-ID": custom_id})
        assert resp.headers["x-request-id"] == custom_id

    def test_caller_id_takes_precedence_over_generated(self, client: TestClient) -> None:
        """If the caller sets X-Request-ID we must echo it, not generate a new one."""
        caller_id = "upstream-gateway-abc"
        resp = client.get("/health/live", headers={"X-Request-ID": caller_id})
        assert resp.headers["x-request-id"] == caller_id

    def test_unique_ids_generated_for_different_requests(self, client: TestClient) -> None:
        id1 = client.get("/health/live").headers["x-request-id"]
        id2 = client.get("/health/live").headers["x-request-id"]
        # UUIDs are random — two consecutive requests should not collide
        assert id1 != id2

    def test_id_present_on_4xx_responses(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "x"})  # too short → 422
        assert "x-request-id" in resp.headers

    def test_id_present_on_post_endpoints(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        assert "x-request-id" in resp.headers


class TestTimingMiddleware:
    def test_response_time_header_present(self, client: TestClient) -> None:
        resp = client.get("/health/live")
        assert "x-response-time-ms" in resp.headers

    def test_response_time_is_non_negative_integer(self, client: TestClient) -> None:
        resp = client.get("/health/live")
        value = int(resp.headers["x-response-time-ms"])
        assert value >= 0

    def test_response_time_present_on_2xx(self, client: TestClient) -> None:
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert "x-response-time-ms" in resp.headers

    def test_response_time_present_on_4xx(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "x"})  # 422
        assert "x-response-time-ms" in resp.headers

    def test_response_time_present_on_post_query(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "Apple revenue?"})
        assert "x-response-time-ms" in resp.headers
        assert int(resp.headers["x-response-time-ms"]) >= 0

    def test_timing_and_request_id_coexist(self, client: TestClient) -> None:
        """Both middleware headers should appear on the same response."""
        resp = client.get("/health/live")
        assert "x-request-id" in resp.headers
        assert "x-response-time-ms" in resp.headers
