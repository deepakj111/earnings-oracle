# config/logging.py
"""
Centralized logging configuration for Financial RAG.

Supports human-readable colorized text format for local development
and structured JSON format for Kubernetes, Docker, Datadog, and ELK.
"""

from __future__ import annotations

import sys

from loguru import logger

from config.settings import settings


def configure_logging() -> None:
    """
    Configure loguru logger according to the LOG_FORMAT environment setting.

    - "text" (default): standard loguru colored console output
    - "json": serialized JSON logs per line for cloud-native ingestion
    """
    log_format = getattr(settings.infra, "log_format", "text").strip().lower()
    if log_format == "json":
        logger.remove()
        logger.add(
            sys.stderr,
            serialize=True,
            level="INFO",
            backtrace=True,
            diagnose=False,
        )
