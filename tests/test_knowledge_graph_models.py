# tests/test_knowledge_graph_models.py
"""
Tests for knowledge_graph/models.py — Entity, Relationship, KnowledgeGraph.

Tests cover:
  - Entity normalization and canonical key generation
  - Relationship edge key deduplication
  - KnowledgeGraph CRUD operations
  - Cross-company entity queries
  - JSON round-trip serialization
  - Graph traversal and entity resolution
"""

from knowledge_graph.models import (
    Entity,
    EntityType,
    KnowledgeGraph,
    Relationship,
    RelationType,
)

# ── Entity ─────────────────────────────────────────────────────────────────────


class TestEntity:
    """Verify Entity normalization and serialization."""

    def test_name_normalized_to_lowercase(self) -> None:
        e = Entity(name="Tim Cook", entity_type=EntityType.PERSON)
        assert e.name == "tim cook"

    def test_name_stripped(self) -> None:
        e = Entity(name="  iPhone 16  ", entity_type=EntityType.PRODUCT)
        assert e.name == "iphone 16"

    def test_canonical_key(self) -> None:
        e = Entity(name="Revenue", entity_type=EntityType.METRIC, ticker="AAPL")
        assert e.canonical_key == "METRIC::AAPL::revenue"

    def test_aliases_normalized(self) -> None:
        e = Entity(
            name="Tim Cook",
            entity_type=EntityType.PERSON,
            aliases=["Mr. Cook", "Timothy Cook"],
        )
        assert "mr. cook" in e.aliases
        assert "timothy cook" in e.aliases

    def test_to_dict_round_trip(self) -> None:
        e = Entity(
            name="Apple Services",
            entity_type=EntityType.SEGMENT,
            ticker="AAPL",
            fiscal_period="Q4 2024",
            chunk_ids=["chunk_1", "chunk_2"],
            properties={"revenue": "$94.9B"},
        )
        d = e.to_dict()
        restored = Entity.from_dict(d)
        assert restored.name == e.name
        assert restored.entity_type == e.entity_type
        assert restored.chunk_ids == e.chunk_ids
        assert restored.properties == e.properties


# ── Relationship ───────────────────────────────────────────────────────────────


class TestRelationship:
    """Verify Relationship normalization and edge keys."""

    def test_names_normalized(self) -> None:
        r = Relationship(
            source="Tim Cook",
            target="Apple",
            relation=RelationType.LEADS,
        )
        assert r.source == "tim cook"
        assert r.target == "apple"

    def test_edge_key(self) -> None:
        r = Relationship(
            source="iPhone",
            target="Apple",
            relation=RelationType.DRIVES_REVENUE,
        )
        assert r.edge_key == "iphone--DRIVES_REVENUE-->apple"

    def test_to_dict_round_trip(self) -> None:
        r = Relationship(
            source="AI",
            target="AAPL",
            relation=RelationType.RISK_TO,
            ticker="AAPL",
            chunk_id="chunk_1",
            properties={"severity": "high"},
        )
        d = r.to_dict()
        restored = Relationship.from_dict(d)
        assert restored.source == r.source
        assert restored.relation == r.relation
        assert restored.properties == r.properties


# ── KnowledgeGraph ─────────────────────────────────────────────────────────────


class TestKnowledgeGraph:
    """Verify KnowledgeGraph CRUD, queries, and serialization."""

    def _build_graph(self) -> KnowledgeGraph:
        """Helper to build a small test graph."""
        graph = KnowledgeGraph()
        graph.add_entity(
            Entity(
                name="Tim Cook",
                entity_type=EntityType.PERSON,
                ticker="AAPL",
                fiscal_period="Q4 2024",
                chunk_ids=["aapl_1"],
                aliases=["Mr. Cook"],
            )
        )
        graph.add_entity(
            Entity(
                name="iPhone",
                entity_type=EntityType.PRODUCT,
                ticker="AAPL",
                fiscal_period="Q4 2024",
                chunk_ids=["aapl_2"],
            )
        )
        graph.add_entity(
            Entity(
                name="AI",
                entity_type=EntityType.INITIATIVE,
                ticker="AAPL",
                fiscal_period="Q4 2024",
                chunk_ids=["aapl_3"],
            )
        )
        graph.add_entity(
            Entity(
                name="AI",
                entity_type=EntityType.INITIATIVE,
                ticker="NVDA",
                fiscal_period="Q4 2024",
                chunk_ids=["nvda_1"],
            )
        )
        graph.add_relationship(
            Relationship(
                source="Tim Cook",
                target="Apple",
                relation=RelationType.LEADS,
                ticker="AAPL",
                chunk_id="aapl_1",
            )
        )
        graph.add_relationship(
            Relationship(
                source="iPhone",
                target="Apple",
                relation=RelationType.DRIVES_REVENUE,
                ticker="AAPL",
                chunk_id="aapl_2",
            )
        )
        return graph

    def test_add_entity(self) -> None:
        graph = KnowledgeGraph()
        e = Entity(name="Revenue", entity_type=EntityType.METRIC)
        graph.add_entity(e)
        assert graph.entity_count == 1

    def test_merge_entity_deduplicates_chunk_ids(self) -> None:
        graph = KnowledgeGraph()
        e1 = Entity(name="Revenue", entity_type=EntityType.METRIC, ticker="AAPL", chunk_ids=["c1"])
        e2 = Entity(name="Revenue", entity_type=EntityType.METRIC, ticker="AAPL", chunk_ids=["c2"])
        graph.add_entity(e1)
        graph.add_entity(e2)
        assert graph.entity_count == 1
        assert set(graph.entities["METRIC::AAPL::revenue"].chunk_ids) == {"c1", "c2"}

    def test_add_relationship_deduplicates(self) -> None:
        graph = KnowledgeGraph()
        r = Relationship(source="A", target="B", relation=RelationType.LEADS)
        graph.add_relationship(r)
        graph.add_relationship(r)  # duplicate
        assert graph.relationship_count == 1

    def test_find_entity_by_name(self) -> None:
        graph = self._build_graph()
        found = graph.find_entity("tim cook")
        assert found is not None
        assert found.entity_type == EntityType.PERSON

    def test_find_entity_by_alias(self) -> None:
        graph = self._build_graph()
        found = graph.find_entity("Mr. Cook")
        assert found is not None
        assert found.name == "tim cook"

    def test_find_entity_not_found(self) -> None:
        graph = self._build_graph()
        assert graph.find_entity("Satya Nadella") is None

    def test_find_by_type(self) -> None:
        graph = self._build_graph()
        products = graph.find_by_type(EntityType.PRODUCT)
        assert len(products) == 1
        assert products[0].name == "iphone"

    def test_find_by_type_with_ticker_filter(self) -> None:
        graph = self._build_graph()
        aapl_initiatives = graph.find_by_type(EntityType.INITIATIVE, ticker="AAPL")
        assert len(aapl_initiatives) == 1

    def test_find_related(self) -> None:
        graph = self._build_graph()
        related = graph.find_related("tim cook")
        assert len(related) >= 1
        relations = [r.relation for r, _ in related]
        assert RelationType.LEADS in relations

    def test_find_cross_company_entities(self) -> None:
        graph = self._build_graph()
        cross = graph.find_cross_company_entities(EntityType.INITIATIVE)
        assert "ai" in cross
        assert len(cross["ai"]) == 2
        assert "AAPL" in cross["ai"]
        assert "NVDA" in cross["ai"]

    def test_get_entity_chunk_ids(self) -> None:
        graph = self._build_graph()
        chunk_ids = graph.get_entity_chunk_ids("tim cook")
        assert "aapl_1" in chunk_ids

    def test_summary(self) -> None:
        graph = self._build_graph()
        s = graph.summary()
        assert "KnowledgeGraph:" in s
        assert "entities" in s
        assert "relationships" in s

    def test_json_round_trip(self) -> None:
        graph = self._build_graph()
        json_str = graph.to_json()
        restored = KnowledgeGraph.from_json(json_str)
        assert restored.entity_count == graph.entity_count
        assert restored.relationship_count == graph.relationship_count
        # Verify entity data survival
        found = restored.find_entity("tim cook")
        assert found is not None
        assert found.ticker == "AAPL"

    def test_empty_graph(self) -> None:
        graph = KnowledgeGraph()
        assert graph.entity_count == 0
        assert graph.relationship_count == 0
        assert graph.find_entity("nobody") is None
        assert graph.find_related("nobody") == []
        assert graph.get_entity_chunk_ids("nobody") == []
