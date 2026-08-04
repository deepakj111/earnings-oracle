# scripts/entrypoints.py
"""
CLI entry-point wrappers for poetry run scripts.
Each function is a zero-argument callable that poetry scripts can invoke.
"""

import socket
import subprocess
import sys


def _check_port(port: int) -> bool:
    """Return True if *port* is free; print an error and return False if bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            print(
                f"\n[ERROR] Port {port} is already in use.\n"
                "  If the Docker stack is running, stop the API container first:\n"
                "    docker stop rag_api\n"
                "  Or, to stop all Docker services:\n"
                "    docker compose down\n",
                file=sys.stderr,
            )
            return False


def serve_dev() -> None:
    """Development server with auto-reload."""
    if not _check_port(8000):
        sys.exit(1)
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
    if not _check_port(8000):
        sys.exit(1)
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


def inspect_data() -> None:
    """Run data store inspection."""
    from scripts.inspect_data import main

    main()


def export_qdrant() -> None:
    """Export Qdrant chunks, payloads, and embeddings to JSON."""
    from scripts.export_qdrant_data import main

    main()
