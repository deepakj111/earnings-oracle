import os

from fastapi import FastAPI
from loguru import logger
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def init_otel(app: FastAPI, service_name: str = "rag_api") -> None:
    """
    Initialize OpenTelemetry and instrument the FastAPI app.
    Exports traces to the OTLP endpoint (Jaeger) if configured.
    """
    otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://jaeger:4317")

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # Configure OTLP Exporter
    try:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        trace.set_tracer_provider(provider)

        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)

        logger.info(f"OpenTelemetry initialized. Exporting traces to {otlp_endpoint}")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenTelemetry: {e}")


# Global tracer instance for manual instrumentation in pipeline
otel_tracer = trace.get_tracer(__name__)
