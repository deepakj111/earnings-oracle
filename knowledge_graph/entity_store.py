# knowledge_graph/entity_store.py
"""
JSON-backed persistence for the Financial Knowledge Graph.

The graph is stored as a single JSON file at `data/knowledge_graph.json`.
This is intentionally simple — the graph is small (hundreds of entities,
not millions), so a full graph database would be over-engineering.

Thread-safety: load() and save() are not thread-safe. The ingestion
pipeline runs single-threaded, so this is acceptable. The retrieval
layer only reads (via load), never writes.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from knowledge_graph.models import KnowledgeGraph

DEFAULT_GRAPH_PATH = Path("data/knowledge_graph.json")


class EntityStore:
    """
    Persistent storage layer for the KnowledgeGraph.

    Usage:
        store = EntityStore()
        graph = store.load()           # load from disk (or empty graph)
        graph.add_entity(entity)       # mutate in memory
        store.save(graph)              # persist back to disk
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_GRAPH_PATH

    def load(self) -> KnowledgeGraph:
        """Load the knowledge graph from disk, or return an empty graph."""
        if not self.path.exists():
            logger.info(f"No knowledge graph found at {self.path} — starting fresh.")
            return KnowledgeGraph()

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            graph = KnowledgeGraph.from_dict(data)
            logger.info(f"Knowledge graph loaded from {self.path} | {graph.summary()}")
            return graph
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(f"Knowledge graph at {self.path} is corrupt ({exc}). Starting fresh.")
            return KnowledgeGraph()

    def save(self, graph: KnowledgeGraph) -> None:
        """Persist the knowledge graph to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(graph.to_json(indent=2), encoding="utf-8")
        logger.info(f"Knowledge graph saved to {self.path} | {graph.summary()}")

    def exists(self) -> bool:
        """Check if the knowledge graph file exists on disk."""
        return self.path.exists()
