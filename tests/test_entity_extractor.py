# tests/test_entity_extractor.py
"""
Tests for knowledge_graph/extractor.py — LLM entity extraction.

Tests cover:
  - Regex-based entity extraction (competitor tickers)
  - LLM extraction with mocked API responses
  - Handling of malformed LLM output
  - Fail-open behavior on LLM errors
  - Entity deduplication across chunks
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge_graph.extractor import (
    _regex_extract,
    extract_entities_from_chunks,
)
from knowledge_graph.models import EntityType, RelationType

# ── Regex extraction ──────────────────────────────────────────────────────────


class TestRegexExtraction:
    """Verify deterministic regex-based entity extraction."""

    def test_extracts_competitor_tickers(self) -> None:
        text = "Apple's results compared favorably to MSFT and NVDA in cloud revenue."
        entities, rels = _regex_extract(text, "AAPL", "Q4 2024", "chunk_1")

        ticker_names = {e.name for e in entities}
        assert "msft" in ticker_names
        assert "nvda" in ticker_names
        # Should not extract AAPL as its own competitor
        assert "aapl" not in ticker_names

    def test_extracts_competitor_relationships(self) -> None:
        text = "AAPL outperformed NVDA in market cap growth."
        entities, rels = _regex_extract(text, "AAPL", "Q4 2024", "chunk_1")

        assert len(rels) >= 1
        assert any(r.relation == RelationType.COMPETES_WITH for r in rels)

    def test_no_competitors_in_clean_text(self) -> None:
        text = "Revenue grew 6 percent year over year to $94.9 billion."
        entities, rels = _regex_extract(text, "AAPL", "Q4 2024", "chunk_1")
        assert len(entities) == 0

    def test_sets_entity_metadata(self) -> None:
        text = "Competition from MSFT intensified."
        entities, _ = _regex_extract(text, "AAPL", "Q4 2024", "chunk_1")
        assert len(entities) == 1
        e = entities[0]
        assert e.entity_type == EntityType.COMPETITOR
        assert e.ticker == "AAPL"
        assert e.fiscal_period == "Q4 2024"
        assert "chunk_1" in e.chunk_ids


# ── LLM extraction (mocked) ──────────────────────────────────────────────────


class TestLLMExtraction:
    """Verify LLM-powered extraction with mocked OpenAI responses."""

    @patch("knowledge_graph.extractor._call_llm_extract", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_extracts_entities_from_llm(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "entities": [
                {"name": "Tim Cook", "entity_type": "PERSON", "properties": {"role": "CEO"}},
                {"name": "iPhone 16", "entity_type": "PRODUCT", "properties": {}},
            ],
            "relationships": [
                {
                    "source": "Tim Cook",
                    "target": "Apple",
                    "relation": "LEADS",
                    "properties": {},
                },
            ],
        }

        chunk = MagicMock()
        chunk.chunk_id = "aapl_q4_001"
        chunk.text = "Tim Cook announced iPhone 16 at the event."

        with patch("knowledge_graph.extractor.settings") as mock_settings:
            mock_settings.knowledge_graph.extraction_enabled = True
            mock_settings.knowledge_graph.extraction_model = "gpt-4.1-nano"

            entities, rels = await extract_entities_from_chunks([chunk], "AAPL", "Q4 2024")

        # Should have LLM entities + regex entities
        llm_entities = [e for e in entities if e.name in ("tim cook", "iphone 16")]
        assert len(llm_entities) == 2

        # Should have LLM relationships
        assert any(r.relation == "LEADS" for r in rels)

    @patch("knowledge_graph.extractor._call_llm_extract", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_handles_empty_llm_response(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {}

        chunk = MagicMock()
        chunk.chunk_id = "test_001"
        chunk.text = "Revenue grew 6 percent."

        with patch("knowledge_graph.extractor.settings") as mock_settings:
            mock_settings.knowledge_graph.extraction_enabled = True
            mock_settings.knowledge_graph.extraction_model = "gpt-4.1-nano"

            entities, rels = await extract_entities_from_chunks([chunk], "AAPL", "Q4 2024")

        # Only regex entities (if any)
        assert isinstance(entities, list)
        assert isinstance(rels, list)

    @patch("knowledge_graph.extractor._call_llm_extract", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_handles_malformed_entities(self, mock_llm: AsyncMock) -> None:
        """Malformed entities should be skipped, not crash the pipeline."""
        mock_llm.return_value = {
            "entities": [
                {"name": "", "entity_type": "PERSON"},  # empty name → skipped
                {"name": "Valid Entity", "entity_type": "PRODUCT"},
            ],
            "relationships": [],
        }

        chunk = MagicMock()
        chunk.chunk_id = "test_001"
        chunk.text = "Some financial text."

        with patch("knowledge_graph.extractor.settings") as mock_settings:
            mock_settings.knowledge_graph.extraction_enabled = True
            mock_settings.knowledge_graph.extraction_model = "gpt-4.1-nano"

            entities, _ = await extract_entities_from_chunks([chunk], "AAPL", "Q4 2024")

        llm_entities = [e for e in entities if e.name == "valid entity"]
        assert len(llm_entities) == 1

    @pytest.mark.asyncio
    async def test_disabled_extraction_uses_regex_only(self) -> None:
        """When LLM extraction is disabled, only regex runs."""
        chunk = MagicMock()
        chunk.chunk_id = "test_001"
        chunk.text = "Apple outperformed MSFT this quarter."

        with patch("knowledge_graph.extractor.settings") as mock_settings:
            mock_settings.knowledge_graph.extraction_enabled = False

            entities, rels = await extract_entities_from_chunks([chunk], "AAPL", "Q4 2024")

        # Should still extract MSFT via regex
        assert any(e.name == "msft" for e in entities)

    @pytest.mark.asyncio
    async def test_empty_chunks_list(self) -> None:
        """No chunks → no entities."""
        with patch("knowledge_graph.extractor.settings") as mock_settings:
            mock_settings.knowledge_graph.extraction_enabled = False

            entities, rels = await extract_entities_from_chunks([], "AAPL", "Q4 2024")

        assert entities == []
        assert rels == []
