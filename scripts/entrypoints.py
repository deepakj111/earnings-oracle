# scripts/entrypoints.py
"""
CLI entry-point wrappers for poetry run scripts.
Each function is a zero-argument callable that poetry scripts can invoke.
"""

import subprocess
import sys


def serve_dev() -> None:
    """Development server with auto-reload."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
        ],
        check=True,
    )


def serve_prod() -> None:
    """Production server with multiple workers."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--workers",
            "4",
            "--loop",
            "uvloop",
            "--http",
            "httptools",
        ],
        check=True,
    )


def run_ui() -> None:
    """Start Streamlit UI."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "ui/app.py",
            "--server.port",
            "8501",
            "--server.address",
            "0.0.0.0",
            "--server.headless",
            "true",
        ],
        check=True,
    )
