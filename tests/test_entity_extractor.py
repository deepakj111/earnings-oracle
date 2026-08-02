# tests/test_entity_extractor.py
"""
Tests for knowledge_graph/extractor.py — Per-chunk LLM entity extraction.

Tests cover:
  - Per-chunk LLM entity and relationship extraction
  - Structured JSON parsing and fallback repair
  - Fail-open behavior on LLM errors
  - Proper binding of chunk_ids to extracted entities
  - Disabled extraction returns empty results
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge_graph.extractor import (
    extract_entities_from_chunks,
)


class TestLLMExtraction:
    """Verify per-chunk LLM-powered extraction with mocked OpenAI responses."""

    @patch("knowledge_graph.extractor._call_llm_extract", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_extracts_entities_from_llm_per_chunk(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "entities": [
                {"name": "Tim Cook", "entity_type": "PERSON", "properties": {"role": "CEO"}},
                {"name": "Revenue", "entity_type": "METRIC", "properties": {"value": "$94.9B"}},
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

        chunk1 = MagicMock()
        chunk1.chunk_id = "aapl_q4_001"
        chunk1.text = "Tim Cook announced Revenue of $94.9B."

        chunk2 = MagicMock()
        chunk2.chunk_id = "aapl_q4_002"
        chunk2.text = "Second chunk text."

        with patch("knowledge_graph.extractor.settings") as mock_settings:
            mock_settings.knowledge_graph.extraction_enabled = True
            mock_settings.knowledge_graph.extraction_model = "gpt-4.1-nano"

            entities, rels = await extract_entities_from_chunks([chunk1, chunk2], "AAPL", "Q4 2024")

        # Verify _call_llm_extract was called twice (once per chunk)
        assert mock_llm.call_count == 2

        # Verify entities extracted with proper chunk provenance
        extracted_names = {e.name for e in entities}
        assert "tim cook" in extracted_names
        assert "revenue" in extracted_names

        tim_cook_entity = next(e for e in entities if e.name == "tim cook")
        assert "aapl_q4_001" in tim_cook_entity.chunk_ids
        assert tim_cook_entity.ticker == "AAPL"

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

        assert entities == []
        assert rels == []

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
    async def test_disabled_extraction_returns_empty(self) -> None:
        """When LLM extraction is disabled, returns empty lists."""
        chunk = MagicMock()
        chunk.chunk_id = "test_001"
        chunk.text = "Apple outperformed MSFT this quarter."

        with patch("knowledge_graph.extractor.settings") as mock_settings:
            mock_settings.knowledge_graph.extraction_enabled = False

            entities, rels = await extract_entities_from_chunks([chunk], "AAPL", "Q4 2024")

        assert entities == []
        assert rels == []

    @pytest.mark.asyncio
    async def test_empty_chunks_list(self) -> None:
        """No chunks → no entities."""
        with patch("knowledge_graph.extractor.settings") as mock_settings:
            mock_settings.knowledge_graph.extraction_enabled = True

            entities, rels = await extract_entities_from_chunks([], "AAPL", "Q4 2024")

        assert entities == []
        assert rels == []


class TestJSONRepair:
    """Verify JSON repair fallback parsing functionality."""

    def test_valid_json(self) -> None:
        from knowledge_graph.extractor import _parse_json_response

        res = _parse_json_response('{"entities": [{"name": "AWS"}]}')
        assert res == {"entities": [{"name": "AWS"}]}

    def test_empty_input(self) -> None:
        from knowledge_graph.extractor import _parse_json_response

        assert _parse_json_response("") == {}
        assert _parse_json_response("  ") == {}

    def test_repairs_trailing_comma(self) -> None:
        from knowledge_graph.extractor import _parse_json_response

        res = _parse_json_response('{"entities": [{"name": "AWS"},]}')
        assert res == {"entities": [{"name": "AWS"}]}

    def test_repairs_unclosed_json(self) -> None:
        from knowledge_graph.extractor import _parse_json_response

        res = _parse_json_response('{"entities": [{"name": "AWS"')
        assert res == {"entities": [{"name": "AWS"}]}
