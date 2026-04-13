# tests/test_api_errors.py
"""
Tests for api/errors.py

Each exception handler is tested by configuring the mock pipeline to raise
the target exception inside POST /query, then asserting:
  1. The HTTP status code matches the documented mapping
  2. The JSON body contains `error`, `detail`, and `request_id`
  3. Any extra headers (e.g. Retry-After for 429) are present

Exception → HTTP status mapping under test:
  ValueError              → 400
  RequestValidationError  → 422  (Pydantic validation failure — tested via bad input)
  AuthenticationError     → 401
  RateLimitError          → 429  (+ Retry-After: 10 header)
  FileNotFoundError       → 503
  APITimeoutError         → 504
  APIConnectionError      → 502
  Exception               → 500  (catch-all)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
from fastapi.testclient import TestClient

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — constructing openai exceptions that require httpx primitives
# ─────────────────────────────────────────────────────────────────────────────


def _make_rate_limit_error() -> Exception:
    from openai import RateLimitError

    response = httpx.Response(
        429,
        content=b'{"error":{"message":"rate limit","type":"tokens","code":"rate_limit_exceeded"}}',
        headers={"content-type": "application/json"},
    )
    return RateLimitError("Rate limit exceeded", response=response, body=None)


def _make_auth_error() -> Exception:
    from openai import AuthenticationError

    response = httpx.Response(
        401,
        content=b'{"error":{"message":"invalid api key","type":"invalid_request_error"}}',
        headers={"content-type": "application/json"},
    )
    return AuthenticationError("Invalid API key", response=response, body=None)


def _make_timeout_error() -> Exception:
    from openai import APITimeoutError

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return APITimeoutError(request=request)


def _make_connection_error() -> Exception:
    from openai import APIConnectionError

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return APIConnectionError(message="Connection refused", request=request)


# ─────────────────────────────────────────────────────────────────────────────
# Shared assertion helpers
# ─────────────────────────────────────────────────────────────────────────────


def _assert_error_shape(data: dict) -> None:
    """Every error response must carry these three fields."""
    assert "error" in data, f"Missing 'error' in: {data}"
    assert "detail" in data, f"Missing 'detail' in: {data}"
    assert "request_id" in data, f"Missing 'request_id' in: {data}"


# ─────────────────────────────────────────────────────────────────────────────
# 422 — Pydantic / Request validation error
# ─────────────────────────────────────────────────────────────────────────────


class TestValidationError:
    def test_status_422(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "x"})  # too short
        assert resp.status_code == 422

    def test_body_shape(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "x"})
        _assert_error_shape(resp.json())

    def test_error_category(self, client: TestClient) -> None:
        resp = client.post("/query/", json={"question": "x"})
        assert "validation" in resp.json()["error"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 400 — ValueError
# ─────────────────────────────────────────────────────────────────────────────


class TestValueError:
    def test_status_400(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = ValueError("Unknown ticker XYZ in metadata filter")
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        assert resp.status_code == 400

    def test_body_shape(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = ValueError("Bad input")
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        _assert_error_shape(resp.json())

    def test_error_category(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = ValueError("Bad input")
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        assert "bad request" in resp.json()["error"].lower()

    def test_detail_contains_original_message(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        mock_pipeline.ask.side_effect = ValueError("specific error text")
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        assert "specific error text" in resp.json()["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# 401 — AuthenticationError
# ─────────────────────────────────────────────────────────────────────────────


class TestAuthError:
    def test_status_401(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = _make_auth_error()
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        assert resp.status_code == 401

    def test_body_shape(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = _make_auth_error()
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        _assert_error_shape(resp.json())

    def test_error_category(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = _make_auth_error()
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        assert "auth" in resp.json()["error"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 429 — RateLimitError (+ Retry-After header)
# ─────────────────────────────────────────────────────────────────────────────


class TestRateLimitError:
    def test_status_429(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = _make_rate_limit_error()
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        assert resp.status_code == 429

    def test_body_shape(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = _make_rate_limit_error()
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        _assert_error_shape(resp.json())

    def test_retry_after_header_present(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = _make_rate_limit_error()
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        assert "retry-after" in resp.headers
        assert resp.headers["retry-after"] == "10"

    def test_error_category(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = _make_rate_limit_error()
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        assert "rate limit" in resp.json()["error"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 503 — FileNotFoundError (BM25 index / Qdrant collection missing)
# ─────────────────────────────────────────────────────────────────────────────


class TestFileNotFoundError:
    def test_status_503(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = FileNotFoundError(
            "BM25 index not found at data/bm25_index.pkl"
        )
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        assert resp.status_code == 503

    def test_body_shape(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = FileNotFoundError("BM25 index not found")
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        _assert_error_shape(resp.json())

    def test_error_category(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = FileNotFoundError("BM25 index not found")
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        assert "unavailable" in resp.json()["error"].lower()

    def test_detail_includes_ingestion_hint(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        mock_pipeline.ask.side_effect = FileNotFoundError("BM25 index not found")
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        assert (
            "ingestion" in resp.json()["detail"].lower()
            or "pipeline" in resp.json()["detail"].lower()
        )


# ─────────────────────────────────────────────────────────────────────────────
# 504 — APITimeoutError
# ─────────────────────────────────────────────────────────────────────────────


class TestAPITimeoutError:
    def test_status_504(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = _make_timeout_error()
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        assert resp.status_code == 504

    def test_body_shape(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = _make_timeout_error()
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        _assert_error_shape(resp.json())

    def test_error_category(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = _make_timeout_error()
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        assert "timeout" in resp.json()["error"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 502 — APIConnectionError
# ─────────────────────────────────────────────────────────────────────────────


class TestAPIConnectionError:
    def test_status_502(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = _make_connection_error()
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        assert resp.status_code == 502

    def test_body_shape(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = _make_connection_error()
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        _assert_error_shape(resp.json())

    def test_error_category(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = _make_connection_error()
        resp = client.post("/query/", json={"question": "Apple revenue Q4 2024?"})
        assert "gateway" in resp.json()["error"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 500 — catch-all (unexpected exception)
# ─────────────────────────────────────────────────────────────────────────────


class TestGenericException:
    def test_status_500(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = RuntimeError("Something completely unexpected")
        resp = client.post(
            "/query/",
            json={"question": "Apple revenue Q4 2024?"},
        )
        assert resp.status_code == 500

    def test_body_shape(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = RuntimeError("Unexpected")
        resp = client.post(
            "/query/",
            json={"question": "Apple revenue Q4 2024?"},
        )
        _assert_error_shape(resp.json())

    def test_error_category(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        mock_pipeline.ask.side_effect = RuntimeError("Unexpected")
        resp = client.post(
            "/query/",
            json={"question": "Apple revenue Q4 2024?"},
        )
        assert "internal" in resp.json()["error"].lower()

    def test_request_id_present_even_on_500(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        mock_pipeline.ask.side_effect = RuntimeError("Unexpected")
        resp = client.post(
            "/query/",
            json={"question": "Apple revenue Q4 2024?"},
        )
        assert resp.json()["request_id"] != "-"
