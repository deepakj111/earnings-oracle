"""
Unit tests for retrieval/contextual_compression.py (Layer 3f — Contextual Compression).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval.contextual_compression import ContextualCompressor
from retrieval.models import SearchResult


def _make_sample_result(
    chunk_id: str = "c1",
    source: str = "dense",
    text: str = "Short child text.",
    parent_text: str = (
        "Item 7. Management Discussion and Analysis. Apple reported Services revenue of $24.2B in Q3 2024, "
        "up 12% YoY. Total net sales reached $85.8B compared to $81.8B in the year-ago quarter. "
        "Forward-looking statements involve risks and uncertainties. Safe harbor provisions apply."
    ),
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        parent_id="p1",
        text=text,
        parent_text=parent_text,
        rrf_score=0.1,
        rerank_score=0.95,
        ticker="AAPL",
        company="Apple Inc.",
        date="2024-06-30",
        year=2024,
        quarter="Q3",
        fiscal_period="Q3 2024",
        section_title="MD&A",
        doc_type="10-Q",
        source=source,
    )


class TestContextualCompressor:
    @pytest.mark.asyncio
    async def test_disabled_by_default_returns_original(self) -> None:
        compressor = ContextualCompressor()
        r = _make_sample_result()
        with patch.object(compressor, "_enabled", False):
            compressed = await compressor.compress_result("What was Services revenue?", r)
            assert compressed.parent_text == r.parent_text

    @pytest.mark.asyncio
    async def test_short_text_skipped(self) -> None:
        compressor = ContextualCompressor()
        r = _make_sample_result(parent_text="Short text under 250 characters.")
        with patch.object(compressor, "_enabled", True):
            compressed = await compressor.compress_result("What was Services revenue?", r)
            assert compressed.parent_text == "Short text under 250 characters."

    @pytest.mark.asyncio
    async def test_facts_store_source_preserved_without_compression(self) -> None:
        compressor = ContextualCompressor()
        r = _make_sample_result(source="facts")
        with patch.object(compressor, "_enabled", True):
            compressed = await compressor.compress_result("What was Services revenue?", r)
            assert compressed.parent_text == r.parent_text

    @pytest.mark.asyncio
    async def test_successful_compression_updates_parent_text(self) -> None:
        compressor = ContextualCompressor()
        r = _make_sample_result()
        mock_extracted = "Apple reported Services revenue of $24.2B in Q3 2024, up 12% YoY."
        with (
            patch.object(compressor, "_enabled", True),
            patch.object(
                compressor, "_compress_text", new_callable=AsyncMock, return_value=mock_extracted
            ),
        ):
            compressed = await compressor.compress_result("What was Services revenue?", r)
            assert compressed.parent_text == mock_extracted
            assert "Services revenue of $24.2B" in compressed.text

    @pytest.mark.asyncio
    async def test_compress_all_list(self) -> None:
        compressor = ContextualCompressor()
        r1 = _make_sample_result(chunk_id="c1")
        r2 = _make_sample_result(chunk_id="c2")
        mock_extracted = "Apple reported Services revenue of $24.2B in Q3 2024, up 12% YoY."
        with (
            patch.object(compressor, "_enabled", True),
            patch.object(
                compressor, "_compress_text", new_callable=AsyncMock, return_value=mock_extracted
            ),
        ):
            compressed_list = await compressor.compress_all("What was Services revenue?", [r1, r2])
            assert len(compressed_list) == 2
            assert compressed_list[0].parent_text == mock_extracted
            assert compressed_list[1].parent_text == mock_extracted

    def test_is_enabled_property(self) -> None:
        compressor = ContextualCompressor()
        assert isinstance(compressor.is_enabled, bool)

    @pytest.mark.asyncio
    async def test_compress_all_empty(self) -> None:
        compressor = ContextualCompressor()
        with patch.object(compressor, "_enabled", True):
            assert await compressor.compress_all("query", []) == []

    @pytest.mark.asyncio
    async def test_compress_text_acomplete_execution(self) -> None:
        compressor = ContextualCompressor()
        mock_resp = MagicMock(content="Apple Services revenue was $24.2B.")
        with (
            patch(
                "retrieval.contextual_compression.get_async_openai_client", side_effect=RuntimeError
            ),
            patch("config.llm_client.acomplete", new_callable=AsyncMock, return_value=mock_resp),
        ):
            extracted = await compressor._compress_text("Services revenue?", "Long text passage...")
            assert extracted == "Apple Services revenue was $24.2B."

    @pytest.mark.asyncio
    async def test_compress_result_exception_handled_gracefully(self) -> None:
        compressor = ContextualCompressor()
        r = _make_sample_result()
        with (
            patch.object(compressor, "_enabled", True),
            patch.object(compressor, "_compress_text", side_effect=RuntimeError("LLM error")),
        ):
            compressed = await compressor.compress_result("Services revenue?", r)
            assert compressed.parent_text == r.parent_text
