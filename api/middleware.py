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
