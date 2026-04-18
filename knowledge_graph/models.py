# knowledge_graph/models.py
"""
Core data models for the Financial Knowledge Graph.

The knowledge graph captures entities and relationships extracted from
SEC 8-K earnings filings. Entities represent real-world objects mentioned
in the filings (people, products, metrics, competitors, risk factors),
and relationships capture the connections between them.

Design decisions:
  - Dataclasses for clarity and JSON serialization
  - Entity names are normalized to lowercase for deduplication
  - Each entity tracks which chunks it was extracted from (provenance)
  - Relationships are directional (source → target) with typed edges
  - The KnowledgeGraph aggregate enables cross-document entity resolution
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum


class EntityType(str, Enum):
    """Canonical entity types extracted from financial filings."""

    PERSON = "PERSON"  # CEO, CFO, board members
    PRODUCT = "PRODUCT"  # iPhone, Azure, AWS
    SEGMENT = "SEGMENT"  # Services, Cloud, Advertising
    METRIC = "METRIC"  # Revenue, EPS, Gross Margin
    COMPETITOR = "COMPETITOR"  # named competitor companies
    RISK = "RISK"  # risk factors, headwinds, challenges
    INITIATIVE = "INITIATIVE"  # AI, sustainability, restructuring


class RelationType(str, Enum):
    """Canonical relationship types between financial entities."""

    LEADS = "LEADS"  # person → company/segment
    REPORTS = "REPORTS"  # metric → segment/company
    DRIVES_REVENUE = "DRIVES_REVENUE"  # product/initiative → company
    COMPETES_WITH = "COMPETES_WITH"  # company → company
    RISK_TO = "RISK_TO"  # risk → company/segment
    PART_OF = "PART_OF"  # product/segment → company
    MENTIONED_WITH = "MENTIONED_WITH"  # co-occurrence in same chunk


@dataclass
class Entity:
    """
    A named entity extracted from financial filings.

    Attributes:
        name: Canonical normalized name (lowercased, trimmed)
        entity_type: One of the EntityType enum values
        ticker: Source company ticker (e.g., "AAPL")
        fiscal_period: Filing period (e.g., "Q4 2024")
        chunk_ids: List of chunk IDs where this entity was mentioned
        aliases: Alternative names for deduplication (e.g., ["tim cook", "mr. cook"])
        properties: Arbitrary metadata (e.g., {"role": "CEO", "value": "$94.9B"})
    """

    name: str
    entity_type: str
    ticker: str = ""
    fiscal_period: str = ""
    chunk_ids: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = self.name.strip().lower()
        self.aliases = [a.strip().lower() for a in self.aliases]
        # Normalize enum → string value for consistent canonical keys
        if hasattr(self.entity_type, "value"):
            self.entity_type = self.entity_type.value

    @property
    def canonical_key(self) -> str:
        """Unique key for deduplication: type::ticker::name."""
        return f"{self.entity_type}::{self.ticker}::{self.name}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "ticker": self.ticker,
            "fiscal_period": self.fiscal_period,
            "chunk_ids": self.chunk_ids,
            "aliases": self.aliases,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Entity:
        return cls(**data)


@dataclass
class Relationship:
    """
    A directional edge between two entities.

    Attributes:
        source: Source entity name (normalized)
        target: Target entity name (normalized)
        relation: One of the RelationType enum values
        ticker: Source company ticker
        fiscal_period: Filing period
        chunk_id: Chunk where this relationship was extracted
        properties: Arbitrary edge metadata (e.g., {"value": "$94.9B"})
    """

    source: str
    target: str
    relation: str
    ticker: str = ""
    fiscal_period: str = ""
    chunk_id: str = ""
    properties: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source = self.source.strip().lower()
        self.target = self.target.strip().lower()
        # Normalize enum → string value for consistent edge keys
        if hasattr(self.relation, "value"):
            self.relation = self.relation.value

    @property
    def edge_key(self) -> str:
        """Unique key for deduplication."""
        return f"{self.source}--{self.relation}-->{self.target}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Relationship:
        return cls(**data)


@dataclass
class KnowledgeGraph:
    """
    In-memory knowledge graph aggregating entities and relationships
    across all ingested documents.

    Provides query methods for entity resolution and relationship traversal
    used by the graph-fused retrieval layer.
    """

    entities: dict[str, Entity] = field(default_factory=dict)  # canonical_key → Entity
    relationships: list[Relationship] = field(default_factory=list)

    # ── Mutation ───────────────────────────────────────────────────────────

    def add_entity(self, entity: Entity) -> Entity:
        """Add or merge an entity. Returns the canonical entity."""
        key = entity.canonical_key
        if key in self.entities:
            existing = self.entities[key]
            # Merge chunk IDs (dedup)
            existing.chunk_ids = list(set(existing.chunk_ids + entity.chunk_ids))
            # Merge aliases
            existing.aliases = list(set(existing.aliases + entity.aliases))
            return existing
        self.entities[key] = entity
        return entity

    def add_relationship(self, rel: Relationship) -> None:
        """Add a relationship, deduplicating by edge_key."""
        existing_keys = {r.edge_key for r in self.relationships}
        if rel.edge_key not in existing_keys:
            self.relationships.append(rel)

    # ── Query ──────────────────────────────────────────────────────────────

    def find_entity(self, name: str, entity_type: str | None = None) -> Entity | None:
        """Find an entity by exact name match, optionally filtered by type."""
        normalized = name.strip().lower()
        for entity in self.entities.values():
            if entity.name == normalized:
                if entity_type is None or entity.entity_type == entity_type:
                    return entity
            # Also check aliases
            if normalized in entity.aliases:
                if entity_type is None or entity.entity_type == entity_type:
                    return entity
        return None

    def find_by_type(self, entity_type: str, ticker: str | None = None) -> list[Entity]:
        """Find all entities of a given type, optionally filtered by ticker."""
        results = []
        for entity in self.entities.values():
            if entity.entity_type != entity_type:
                continue
            if ticker and entity.ticker != ticker:
                continue
            results.append(entity)
        return results

    def find_related(self, entity_name: str) -> list[tuple[Relationship, Entity | None]]:
        """
        Find all entities connected to `entity_name` via relationships.

        Returns list of (relationship, target_entity) tuples.
        Target entity may be None if it's not in the graph (external reference).
        """
        normalized = entity_name.strip().lower()
        results: list[tuple[Relationship, Entity | None]] = []
        for rel in self.relationships:
            if rel.source == normalized:
                target = self.find_entity(rel.target)
                results.append((rel, target))
            elif rel.target == normalized:
                source = self.find_entity(rel.source)
                results.append((rel, source))
        return results

    def find_cross_company_entities(self, entity_type: str) -> dict[str, list[str]]:
        """
        Find entities of a type that appear across multiple tickers.

        Returns dict[entity_name → list[tickers]].
        Useful for answering "Which companies mentioned X?"
        """
        entity_tickers: dict[str, set[str]] = {}
        for entity in self.entities.values():
            if entity.entity_type != entity_type:
                continue
            if entity.name not in entity_tickers:
                entity_tickers[entity.name] = set()
            entity_tickers[entity.name].add(entity.ticker)
        return {
            name: sorted(tickers) for name, tickers in entity_tickers.items() if len(tickers) > 1
        }

    def get_entity_chunk_ids(self, entity_name: str) -> list[str]:
        """Get all chunk IDs associated with an entity (across all types)."""
        normalized = entity_name.strip().lower()
        chunk_ids: list[str] = []
        for entity in self.entities.values():
            if entity.name == normalized or normalized in entity.aliases:
                chunk_ids.extend(entity.chunk_ids)
        return list(set(chunk_ids))

    # ── Statistics ─────────────────────────────────────────────────────────

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relationship_count(self) -> int:
        return len(self.relationships)

    def summary(self) -> str:
        type_counts: dict[str, int] = {}
        for e in self.entities.values():
            type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1
        type_str = ", ".join(f"{t}={c}" for t, c in sorted(type_counts.items()))
        return (
            f"KnowledgeGraph: {self.entity_count} entities ({type_str}), "
            f"{self.relationship_count} relationships"
        )

    # ── Serialization ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relationships": [r.to_dict() for r in self.relationships],
        }

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeGraph:
        graph = cls()
        for e_data in data.get("entities", []):
            entity = Entity.from_dict(e_data)
            graph.entities[entity.canonical_key] = entity
        for r_data in data.get("relationships", []):
            graph.relationships.append(Relationship.from_dict(r_data))
        return graph

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> KnowledgeGraph:
        return cls.from_dict(json.loads(json_str))
