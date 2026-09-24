"""
Unit tests for observability/otel.py — OpenTelemetry initialization and instrumentation.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI

from observability.otel import init_otel


def test_init_otel_success() -> None:
    app = FastAPI()
    with (
        patch("observability.otel.OTLPSpanExporter") as mock_exporter,
        patch("observability.otel.BatchSpanProcessor") as mock_processor,
        patch("observability.otel.trace.set_tracer_provider") as mock_set_provider,
        patch("observability.otel.FastAPIInstrumentor.instrument_app") as mock_instrument,
    ):
        init_otel(app, service_name="test_service")
        mock_exporter.assert_called_once()
        mock_processor.assert_called_once()
        mock_set_provider.assert_called_once()
        mock_instrument.assert_called_once_with(app)


def test_init_otel_exception_logged() -> None:
    app = FastAPI()
    with (
        patch("observability.otel.OTLPSpanExporter", side_effect=RuntimeError("OTLP failure")),
        patch("observability.otel.logger.warning") as mock_log,
    ):
        init_otel(app, service_name="test_service")
        mock_log.assert_called_once()
