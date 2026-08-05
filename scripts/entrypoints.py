# scripts/entrypoints.py
"""
CLI entry-point wrappers for poetry run scripts.
Each function is a zero-argument callable that poetry scripts can invoke.
"""

import socket
import subprocess
import sys


def _check_port(port: int, *, service_hint: str = "") -> bool:
    """
    Return True if *port* is free.

    Prints a clear, actionable error if the port is occupied — including
    hints about both local processes AND Docker containers that might hold
    the port.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            hint = f" ({service_hint})" if service_hint else ""
            print(
                f"\n[ERROR] Port {port}{hint} is already in use.\n"
                "\nPossible causes and fixes:\n"
                "  1. A local server is running on this port:\n"
                f"       lsof -ti:{port} | xargs kill -9\n"
                "  2. A Docker container is occupying this port:\n"
                f"       docker ps | grep {port}     # find the container\n"
                "       docker compose down           # stop all RAG services\n"
                "  3. To stop only the API container:\n"
                "       docker stop rag_api\n",
                file=sys.stderr,
            )
            return False


def serve_dev() -> None:
    """Development server with auto-reload (single worker)."""
    if not _check_port(8000, service_hint="FastAPI dev server"):
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
            "--log-level",
            "info",
        ],
        check=True,
    )


def serve_prod() -> None:
    """
    Production server — 4 Uvicorn worker processes managed by a master process.

    NOTE: --workers >1 uses multiprocessing (gunicorn-style). The --loop and
    --http flags are NOT compatible with multi-worker mode and must NOT be
    passed here. Uvicorn handles the event loop selection per-worker
    automatically based on installed extras (uvicorn[standard]).
    """
    if not _check_port(8000, service_hint="FastAPI prod server"):
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
            "--log-level",
            "info",
        ],
        check=True,
    )


def run_ui() -> None:
    """Start Streamlit UI on port 8501."""
    if not _check_port(8501, service_hint="Streamlit UI"):
        sys.exit(1)
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
