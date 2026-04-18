# tests/test_graph_retriever.py
"""
Tests for knowledge_graph/graph_retriever.py — Graph-Fused Retrieval.

Tests cover:
  - Entity matching from question text
  - Relationship traversal for related chunk IDs
  - Chunk deduplication against existing results
  - Disabled graph retrieval
  - Error handling (fail-open)
  - GraphRetrievalSpan observability
"""

from unittest.mock import MagicMock, patch

import pytest

from knowledge_graph.graph_retriever import (
    _collect_related_chunk_ids,
    _match_entities_from_question,
    graph_retrieve,
)
from knowledge_graph.models import (
    Entity,
    EntityType,
    KnowledgeGraph,
    Relationship,
    RelationType,
)
from observability.trace_models import SpanStatus
from retrieval.models import SearchResult

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_graph() -> KnowledgeGraph:
    """Build a small test knowledge graph."""
    graph = KnowledgeGraph()
    graph.add_entity(
        Entity(
            name="iPhone",
            entity_type=EntityType.PRODUCT,
            ticker="AAPL",
            chunk_ids=["aapl_c1", "aapl_c2"],
        )
    )
    graph.add_entity(
        Entity(
            name="Tim Cook",
            entity_type=EntityType.PERSON,
            ticker="AAPL",
            chunk_ids=["aapl_c3"],
            aliases=["Mr. Cook"],
        )
    )
    graph.add_entity(
        Entity(
            name="services",
            entity_type=EntityType.SEGMENT,
            ticker="AAPL",
            chunk_ids=["aapl_c4"],
        )
    )
    graph.add_relationship(
        Relationship(
            source="iPhone",
            target="services",
            relation=RelationType.PART_OF,
            chunk_id="aapl_c1",
        )
    )
    graph.add_relationship(
        Relationship(
            source="Tim Cook",
            target="services",
            relation=RelationType.LEADS,
            chunk_id="aapl_c3",
        )
    )
    return graph


@pytest.fixture()
def existing_result() -> SearchResult:
    """A SearchResult that's already in the retrieval results."""
    return SearchResult(
        chunk_id="aapl_c1",
        parent_id=None,
        text="iPhone revenue...",
        parent_text="iPhone revenue...",
        rrf_score=0.5,
        rerank_score=0.8,
        ticker="AAPL",
        company="Apple",
        date="2024-10-31",
        year=2024,
        quarter="Q4",
        fiscal_period="Q4 2024",
        section_title="Revenue",
        doc_type="earnings_release",
        source="dense",
    )


# ── Entity matching ───────────────────────────────────────────────────────────


class TestEntityMatching:
    """Verify entity matching from question text."""

    def test_matches_entity_by_name(self, sample_graph: KnowledgeGraph) -> None:
        matched = _match_entities_from_question("What was iPhone revenue?", sample_graph)
        assert "iphone" in matched

    def test_matches_entity_by_alias(self, sample_graph: KnowledgeGraph) -> None:
        matched = _match_entities_from_question(
            "What did Mr. Cook say about revenue?", sample_graph
        )
        assert "tim cook" in matched

    def test_no_match_for_unknown_entity(self, sample_graph: KnowledgeGraph) -> None:
        matched = _match_entities_from_question("What was Tesla's revenue?", sample_graph)
        assert len(matched) == 0

    def test_case_insensitive(self, sample_graph: KnowledgeGraph) -> None:
        matched = _match_entities_from_question("IPHONE sales grew", sample_graph)
        assert "iphone" in matched


# ── Chunk ID collection ───────────────────────────────────────────────────────


class TestChunkCollection:
    """Verify relationship traversal and chunk ID collection."""

    def test_collects_direct_chunk_ids(self, sample_graph: KnowledgeGraph) -> None:
        chunk_ids = _collect_related_chunk_ids(["iphone"], sample_graph, existing_chunk_ids=set())
        assert "aapl_c1" in chunk_ids
        assert "aapl_c2" in chunk_ids

    def test_collects_related_chunk_ids(self, sample_graph: KnowledgeGraph) -> None:
        chunk_ids = _collect_related_chunk_ids(["iphone"], sample_graph, existing_chunk_ids=set())
        # iPhone is PART_OF services → should include services chunks
        assert "aapl_c4" in chunk_ids

    def test_deduplicates_against_existing(self, sample_graph: KnowledgeGraph) -> None:
        chunk_ids = _collect_related_chunk_ids(
            ["iphone"], sample_graph, existing_chunk_ids={"aapl_c1"}
        )
        # aapl_c1 already exists, should not be in result
        assert "aapl_c1" not in chunk_ids
        # But aapl_c2 should still be present
        assert "aapl_c2" in chunk_ids


# ── Full graph_retrieve ───────────────────────────────────────────────────────


class TestGraphRetrieve:
    """Verify end-to-end graph retrieval with mocked Qdrant."""

    @patch("knowledge_graph.graph_retriever._load_graph")
    def test_returns_graph_chunks(
        self,
        mock_load: MagicMock,
        sample_graph: KnowledgeGraph,
        existing_result: SearchResult,
    ) -> None:
        mock_load.return_value = sample_graph

        # Mock Qdrant scroll to return a result
        mock_qdrant = MagicMock()
        mock_point = MagicMock()
        mock_point.payload = {
            "chunk_id": "aapl_c2",
            "parent_id": None,
            "text": "iPhone 16 features...",
            "ticker": "AAPL",
            "company": "Apple",
            "date": "2024-10-31",
            "year": 2024,
            "quarter": "Q4",
            "fiscal_period": "Q4 2024",
            "section_title": "Products",
            "doc_type": "earnings_release",
        }
        mock_qdrant.scroll.return_value = ([mock_point], None)

        with patch("knowledge_graph.graph_retriever.settings") as mock_settings:
            mock_settings.knowledge_graph.retrieval_enabled = True
            mock_settings.knowledge_graph.max_graph_chunks = 3
            mock_settings.embedding.collection_name = "test_collection"

            results, span = graph_retrieve(
                "What was iPhone revenue?",
                [existing_result],
                mock_qdrant,
            )

        assert len(results) >= 1
        assert results[0].source == "graph"
        assert span.entities_matched >= 1
        assert span.status == SpanStatus.OK

    @patch("knowledge_graph.graph_retriever._load_graph")
    def test_disabled_returns_empty(
        self,
        mock_load: MagicMock,
        existing_result: SearchResult,
    ) -> None:
        mock_qdrant = MagicMock()

        with patch("knowledge_graph.graph_retriever.settings") as mock_settings:
            mock_settings.knowledge_graph.retrieval_enabled = False

            results, span = graph_retrieve(
                "What was iPhone revenue?",
                [existing_result],
                mock_qdrant,
            )

        assert results == []
        mock_load.assert_not_called()

    @patch("knowledge_graph.graph_retriever._load_graph")
    def test_empty_graph_returns_empty(
        self,
        mock_load: MagicMock,
        existing_result: SearchResult,
    ) -> None:
        mock_load.return_value = KnowledgeGraph()
        mock_qdrant = MagicMock()

        with patch("knowledge_graph.graph_retriever.settings") as mock_settings:
            mock_settings.knowledge_graph.retrieval_enabled = True

            results, span = graph_retrieve(
                "What was iPhone revenue?",
                [existing_result],
                mock_qdrant,
            )

        assert results == []
        assert span.entities_matched == 0

    @patch("knowledge_graph.graph_retriever._load_graph")
    def test_error_returns_empty_with_error_span(
        self,
        mock_load: MagicMock,
        existing_result: SearchResult,
    ) -> None:
        mock_load.side_effect = RuntimeError("disk error")
        mock_qdrant = MagicMock()

        with patch("knowledge_graph.graph_retriever.settings") as mock_settings:
            mock_settings.knowledge_graph.retrieval_enabled = True

            results, span = graph_retrieve(
                "What was iPhone revenue?",
                [existing_result],
                mock_qdrant,
            )

        assert results == []
        assert span.status == SpanStatus.ERROR
