#!/usr/bin/env python3
"""
scripts/export_qdrant_data.py
=============================
Export all points, payloads, and embeddings from Qdrant vector DB
to `data/qdrant_data_export.json` for manual inspection.

Usage:
    poetry run python scripts/export_qdrant_data.py
    poetry run python scripts/export_qdrant_data.py --no-vectors   # lightweight payload-only export
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# ── Ensure repo root is on sys.path ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from config import settings  # noqa: E402

OUTPUT_PATH = ROOT / "data" / "qdrant_data_export.json"
QDRANT_URL = settings.infra.qdrant_url
COLLECTION = settings.embedding.collection_name


def export_qdrant(output_path: Path = OUTPUT_PATH, include_vectors: bool = True) -> None:
    from qdrant_client import QdrantClient

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(QDRANT_URL, timeout=30, check_compatibility=False)

    try:
        col_info = client.get_collection(COLLECTION)
        print(f"Collection '{COLLECTION}' found | {col_info.points_count:,} total points.")
    except Exception as e:
        print(f"[ERROR] Could not access collection '{COLLECTION}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Scrolling all points (with_payload=True, with_vectors={include_vectors})...")
    offset = None
    all_points = []
    fetched_count = 0

    while True:
        points, offset = client.scroll(
            COLLECTION,
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=include_vectors,
        )
        for pt in points:
            point_dict: dict[str, Any] = {
                "id": str(pt.id),
                "payload": pt.payload or {},
            }
            if include_vectors and pt.vector is not None:
                # pt.vector can be a list[float] or a dict (named vectors)
                if isinstance(pt.vector, dict):
                    point_dict["vector"] = {k: list(v) for k, v in pt.vector.items()}
                    point_dict["vector_dim"] = None
                elif isinstance(pt.vector, list):
                    point_dict["vector"] = list(pt.vector)
                    point_dict["vector_dim"] = len(pt.vector)
                else:
                    point_dict["vector"] = list(pt.vector)  # type: ignore[arg-type]
                    point_dict["vector_dim"] = None

            all_points.append(point_dict)
            fetched_count += 1

        print(f"  Fetched {fetched_count}/{col_info.points_count:,} points...")
        if offset is None:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting export file → {output_path.resolve()}...")

    export_payload = {
        "collection": COLLECTION,
        "total_points": len(all_points),
        "include_vectors": include_vectors,
        "points": all_points,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"✓ Export complete! Exported {len(all_points):,} points to {output_path.name} ({file_size_mb:.2f} MB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all Qdrant chunks, payloads, and embeddings to JSON."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Target output file path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--no-vectors",
        action="store_true",
        help="Exclude vector embeddings to create a smaller payload-only JSON export.",
    )
    args = parser.parse_args()

    export_qdrant(output_path=args.output, include_vectors=not args.no_vectors)


if __name__ == "__main__":
    main()
