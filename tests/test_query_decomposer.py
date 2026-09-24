"""
Unit tests for retrieval/query_decomposer.py (Layer 2b — Sub-Query Decomposition).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from retrieval.query_decomposer import DecomposedQueryPlan, QueryDecomposer


class TestQueryDecomposer:
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self) -> None:
        decomposer = QueryDecomposer()
        is_decomp, queries, reason = await decomposer.decompose("   ")
        assert is_decomp is False
        assert queries == []
        assert "Empty" in reason

    @pytest.mark.asyncio
    async def test_disabled_by_default_returns_original(self) -> None:
        decomposer = QueryDecomposer()
        with patch.object(decomposer, "_enabled", False):
            is_decomp, queries, reason = await decomposer.decompose(
                "Compare Apple and Microsoft revenues in 2024"
            )
            assert is_decomp is False
            assert queries == ["Compare Apple and Microsoft revenues in 2024"]
            assert "disabled" in reason.lower()

    @pytest.mark.asyncio
    async def test_atomic_query_classification(self) -> None:
        decomposer = QueryDecomposer()
        plan = DecomposedQueryPlan(
            is_complex=False,
            sub_queries=[],
            reasoning="Single company single metric query",
        )
        with (
            patch.object(decomposer, "_enabled", True),
            patch.object(decomposer, "_call_llm", new_callable=AsyncMock, return_value=plan),
        ):
            is_decomp, queries, reason = await decomposer.decompose(
                "What was Apple's revenue in Q4 2024?"
            )
            assert is_decomp is False
            assert queries == ["What was Apple's revenue in Q4 2024?"]
            assert reason == "Single company single metric query"

    @pytest.mark.asyncio
    async def test_complex_query_decomposed_into_subqueries(self) -> None:
        decomposer = QueryDecomposer()
        plan = DecomposedQueryPlan(
            is_complex=True,
            sub_queries=[
                "What was Apple's Services gross margin in Q3 2024?",
                "What was Microsoft's Intelligent Cloud gross margin in Q4 2024?",
            ],
            reasoning="Multi-entity cross-company comparative analysis",
        )
        with (
            patch.object(decomposer, "_enabled", True),
            patch.object(decomposer, "_call_llm", new_callable=AsyncMock, return_value=plan),
        ):
            is_decomp, queries, reason = await decomposer.decompose(
                "Compare Apple's Services gross margin in Q3 2024 to Microsoft's Intelligent Cloud in Q4 2024"
            )
            assert is_decomp is True
            assert len(queries) == 2
            assert "Apple's Services" in queries[0]
            assert "Microsoft's Intelligent Cloud" in queries[1]
            assert "comparative" in reason.lower()

    @pytest.mark.asyncio
    async def test_fallback_on_llm_exception(self) -> None:
        decomposer = QueryDecomposer()
        with (
            patch.object(decomposer, "_enabled", True),
            patch.object(
                decomposer,
                "_call_llm",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API timeout"),
            ),
        ):
            is_decomp, queries, reason = await decomposer.decompose(
                "Compare Nvidia and AMD AI revenue in 2024"
            )
            assert is_decomp is False
            assert queries == ["Compare Nvidia and AMD AI revenue in 2024"]
            assert "Fallback due to error" in reason
