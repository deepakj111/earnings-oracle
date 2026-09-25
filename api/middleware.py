"""
Custom ASGI middleware for the Financial RAG API.

RequestIDMiddleware  — stamps every request with a unique correlation ID
                       (X-Request-Id header, echoed back in the response)
TimingMiddleware     — measures end-to-end latency, adds X-Response-Time-Ms

WHY PURE ASGI (not BaseHTTPMiddleware):
  BaseHTTPMiddleware uses anyio.create_task_group() internally to run
  call_next(). When a route raises an unhandled exception, the exception
  handler sends a 500 response, BUT the inner task group still sees the
  unhandled exception and re-raises it via ExceptionGroup → collapse_excgroups()
  → starlette/base.py:168. This propagates through both dispatch() methods and
  crashes the TestClient instead of returning the 500 response.

  Pure ASGI middleware wraps the send() callable directly. It intercepts the
  http.response.start message for every response — success, 4xx, or 5xx —
  without ever participating in exception propagation. The exception handlers
  work exactly as intended.

Reference: https://github.com/encode/starlette/issues/1176
"""

from __future__ import annotations

import time
import uuid

from loguru import logger
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_MAX_REQUEST_ID_LEN = 64


class RequestIDMiddleware:
    """
    Stamp every HTTP request with a unique correlation ID.

    Sources the ID from the incoming X-Request-Id header (so callers can
    inject their own trace ID for distributed tracing), or generates a UUID4
    if the header is absent.

    The request_id is:
      - Stored on request.state.request_id for use in route handlers/logs
      - Written to scope["state"] so all downstream middleware can read it
      - Echoed back as X-Request-Id on EVERY response, including 4xx and 5xx
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        request_id = (request.headers.get("X-Request-Id") or str(uuid.uuid4()))[
            :_MAX_REQUEST_ID_LEN
        ]

        request.state.request_id = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Request-Id"] = request_id
            await send(message)

        await self.app(scope, receive, send_with_request_id)


class TimingMiddleware:
    """
    Measure end-to-end request latency and write a structured access log line.

    Adds X-Response-Time-Ms to every response.  Log line format:
      [<rid>] METHOD /path -> STATUS | <N>ms

    Timing is measured from the first byte of the request scope until the
    http.response.start message is sent — this is identical to the wall-clock
    latency a client experiences (minus TCP overhead).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        t0 = time.perf_counter()

        async def send_with_timing(message: Message) -> None:
            if message["type"] == "http.response.start":
                elapsed_ms = round((time.perf_counter() - t0) * 1000)

                headers = MutableHeaders(scope=message)
                headers["X-Response-Time-Ms"] = str(elapsed_ms)

                rid = getattr(request.state, "request_id", "-")
                logger.info(
                    f"[{rid}] {request.method} {request.url.path} "
                    f"-> {message['status']} | {elapsed_ms}ms"
                )
            await send(message)

        await self.app(scope, receive, send_with_timing)


class UserContextMiddleware:
    """
    Extracts multi-tenant user and organization/workspace identity from incoming requests.

    Inspects:
      - X-User-ID: Authenticated user identity (e.g. analyst@fund.com or user_42)
      - X-Tenant-ID: Multi-tenant organization / workspace identifier
      - Authorization: Bearer token (parses token prefix or subject if present)

    Stores:
      - request.state.user_id
      - request.state.tenant_id
      - request.state.is_authenticated

    Echoes:
      - X-User-ID on the response for tracing and audit confirmation.
    """

    def __init__(
        self,
        app: ASGIApp,
        default_user: str = "default_user",
        default_tenant: str = "default_tenant",
    ) -> None:
        self.app = app
        self.default_user = default_user
        self.default_tenant = default_tenant

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        user_id = request.headers.get("x-user-id", "").strip()
        tenant_id = request.headers.get("x-tenant-id", "").strip()
        auth_header = request.headers.get("authorization", "").strip()

        # If Authorization header provided: Bearer <token>
        is_auth = False
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if token:
                is_auth = True
                if not user_id:
                    # Use token prefix as user identification if no explicit header
                    user_id = f"token_{token[:12]}"

        final_user = user_id or self.default_user
        final_tenant = tenant_id or self.default_tenant

        scope.setdefault("state", {})
        scope["state"]["user_id"] = final_user
        scope["state"]["tenant_id"] = final_tenant
        scope["state"]["is_authenticated"] = is_auth

        async def send_with_user_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                h = MutableHeaders(scope=message)
                h["X-User-ID"] = final_user
                h["X-Tenant-ID"] = final_tenant
            await send(message)

        await self.app(scope, receive, send_with_user_headers)


class RateLimitMiddleware:
    """
    Sliding-window rate limiter with User + IP multi-tier throttling.

    Features:
    - User-aware: Throttles by X-User-ID or API Token when present, avoiding NAT IP bottlenecking.
    - IP fallback: Falls back to client IP for anonymous requests.
    - Emits standard RFC rate limit headers:
        X-RateLimit-Limit: <rpm>
        X-RateLimit-Remaining: <remaining requests in current window>
        X-RateLimit-Reset: <seconds until quota resets>
    - Returns 429 Too Many Requests if rate limit is exceeded.
    - Skips rate limiting for health checks, metrics, and documentation endpoints.
    """

    def __init__(
        self,
        app: ASGIApp,
        rpm: int = 120,
        exempt_paths: set[str] | None = None,
    ) -> None:
        self.app = app
        self.rpm = rpm
        self.exempt_paths = exempt_paths or {
            "/health",
            "/health/live",
            "/health/ready",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        }
        self._requests: dict[str, list[float]] = {}
        self._last_cleanup = time.time()

    def _clean_stale(self, now: float) -> None:
        if now - self._last_cleanup > 60.0:
            window_start = now - 60.0
            cleaned = {}
            for key, ts_list in self._requests.items():
                valid = [t for t in ts_list if t > window_start]
                if valid:
                    cleaned[key] = valid
            self._requests = cleaned
            self._last_cleanup = now

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.rpm <= 0:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
            request.client.host if request.client else "unknown"
        )
        user_id = (
            getattr(request.state, "user_id", None) or request.headers.get("x-user-id", "").strip()
        )

        # Prioritise user-based throttling key if authenticated, else client IP
        throttle_key = (
            f"user:{user_id}" if user_id and user_id != "default_user" else f"ip:{client_ip}"
        )

        now = time.time()
        self._clean_stale(now)

        window_start = now - 60.0
        timestamps = [t for t in self._requests.get(throttle_key, []) if t > window_start]
        current_count = len(timestamps)

        remaining = max(0, self.rpm - current_count)
        reset_seconds = int(max(1.0, 60.0 - (now - timestamps[0]))) if timestamps else 60

        if current_count >= self.rpm:
            headers = [
                (b"content-type", b"application/json"),
                (b"retry-after", str(reset_seconds).encode()),
                (b"x-ratelimit-limit", str(self.rpm).encode()),
                (b"x-ratelimit-remaining", b"0"),
                (b"x-ratelimit-reset", str(reset_seconds).encode()),
            ]
            rid = getattr(request.state, "request_id", "-")
            logger.warning(
                f"[{rid}] Rate limit exceeded for {throttle_key} on {path} ({self.rpm} RPM)"
            )
            body = (
                f'{{"error":"Too Many Requests","detail":"Rate limit of {self.rpm} req/min exceeded.",'
                f'"retry_after":{reset_seconds}}}'
            ).encode()
            await send({"type": "http.response.start", "status": 429, "headers": headers})
            await send({"type": "http.response.body", "body": body})
            return

        timestamps.append(now)
        self._requests[throttle_key] = timestamps
        remaining = max(0, self.rpm - len(timestamps))

        async def send_with_rate_limit_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                h = MutableHeaders(scope=message)
                h["X-RateLimit-Limit"] = str(self.rpm)
                h["X-RateLimit-Remaining"] = str(remaining)
                h["X-RateLimit-Reset"] = str(reset_seconds)
            await send(message)

        await self.app(scope, receive, send_with_rate_limit_headers)


__all__ = [
    "RequestIDMiddleware",
    "TimingMiddleware",
    "UserContextMiddleware",
    "RateLimitMiddleware",
]
