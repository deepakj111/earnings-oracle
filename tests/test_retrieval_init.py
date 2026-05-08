from __future__ import annotations

from unittest.mock import Mock, patch

from query.models import TransformedQuery
from retrieval import retrieve
from retrieval.models import MetadataFilter, SearchResult


def test_retrieve_success() -> None:
    query = TransformedQuery(
        original="test",
        hyde_document="hyde",
        multi_queries=["q1"],
        stepback_query="step",
    )
    client = Mock()

    with (
        patch("retrieval.search") as mock_search,
        patch("retrieval.rerank") as mock_rerank,
        patch("knowledge_graph.graph_retriever.graph_retrieve") as mock_graph,
        patch("retrieval._fetch_parent_texts") as mock_fetch,
    ):
        r1 = SearchResult.from_payload({"chunk_id": "c1", "text": "t1"}, rrf_score=1.0, source="dense")
        g1 = SearchResult.from_payload({"chunk_id": "g1", "text": "g1"}, rrf_score=1.0, source="graph")
        mock_search.return_value = [r1]
        mock_rerank.return_value = [r1]
        mock_graph.return_value = ([g1], None)
        mock_fetch.side_effect = lambda c, res: res

        result = retrieve(query, client, MetadataFilter(ticker="AAPL"))

        assert len(result.results) == 2
        assert result.results[0].chunk_id == "c1"
        assert result.results[1].chunk_id == "g1"
        assert result.reranked is True
        assert result.metadata_filter is not None
        assert result.metadata_filter.ticker == "AAPL"


def test_retrieve_graph_fail_open() -> None:
    query = TransformedQuery(
        original="test",
        hyde_document="hyde",
        multi_queries=["q1"],
        stepback_query="step",
    )
    client = Mock()

    with (
        patch("retrieval.search") as mock_search,
        patch("retrieval.rerank") as mock_rerank,
        patch("knowledge_graph.graph_retriever.graph_retrieve") as mock_graph,
        patch("retrieval._fetch_parent_texts") as mock_fetch,
    ):
        r1 = SearchResult.from_payload({"chunk_id": "c1", "text": "t1"}, rrf_score=1.0, source="dense")
        mock_search.return_value = [r1]
        mock_rerank.return_value = [r1]
        mock_graph.side_effect = Exception("Graph failed")
        mock_fetch.side_effect = lambda c, res: res

        result = retrieve(query, client, None)

        assert len(result.results) == 1
        assert result.results[0].chunk_id == "c1"
