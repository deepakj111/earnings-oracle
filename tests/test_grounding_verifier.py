from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from generation.grounding_verifier import ClaimGroundingVerifier, GroundingReportModel
from retrieval.models import SearchResult


def _make_search_result(text: str) -> SearchResult:
    return SearchResult(
        chunk_id="chunk_1",
        parent_id="parent_1",
        text=text,
        parent_text=text,
        rrf_score=0.9,
        rerank_score=0.9,
        ticker="NVDA",
        company="NVIDIA",
        date="2025-01-26",
        year=2025,
        quarter="Q4",
        fiscal_period="Q4 2025",
        section_title="Financial Highlights",
        doc_type="10-K",
        source="dense",
    )


def _mock_parse_response(report_model: GroundingReportModel):
    choice = MagicMock()
    choice.message.parsed = report_model
    response = MagicMock()
    response.choices = [choice]
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=response)
    return mock_client


@pytest.mark.asyncio
async def test_grounded_factual_claim():
    context = [
        _make_search_result(
            "Data Center revenue for fiscal 2025 was $47,544 million, up 162% compared to fiscal 2024."
        )
    ]
    answer = "NVIDIA's Data Center revenue was $47,544 million in fiscal 2025 [1]."
    mock_model = GroundingReportModel(
        is_grounded=True,
        grounding_score=1.0,
        verified_claims=["Data Center revenue for fiscal 2025 was $47,544 million"],
        ungrounded_claims=[],
        hallucinated_citations=[],
        reasoning="Grounded in context",
    )
    with patch(
        "generation.grounding_verifier.get_async_openai_client",
        return_value=_mock_parse_response(mock_model),
    ):
        report = await ClaimGroundingVerifier.verify(answer, context)
        assert report.is_grounded is True
        assert report.grounding_score == 1.0
        assert len(report.verified_claims) == 1
        assert len(report.ungrounded_claims) == 0


@pytest.mark.asyncio
async def test_numerical_hallucination_detected():
    context = [_make_search_result("Data Center revenue was $47,544 million.")]
    answer = "NVIDIA's Data Center revenue was $88,000 million [1]."
    mock_model = GroundingReportModel(
        is_grounded=False,
        grounding_score=0.0,
        verified_claims=[],
        ungrounded_claims=["Numerical discrepancy: $88,000 million not found in context"],
        hallucinated_citations=[],
        reasoning="Numerical discrepancy",
    )
    with patch(
        "generation.grounding_verifier.get_async_openai_client",
        return_value=_mock_parse_response(mock_model),
    ):
        report = await ClaimGroundingVerifier.verify(answer, context)
        assert report.is_grounded is False
        assert len(report.ungrounded_claims) == 1
        assert "Numerical discrepancy" in report.ungrounded_claims[0]


@pytest.mark.asyncio
async def test_missing_citation_detected():
    context = [_make_search_result("Data Center revenue was $47,544 million.")]
    answer = "NVIDIA's Data Center revenue reached $47,544 million."
    mock_model = GroundingReportModel(
        is_grounded=False,
        grounding_score=0.0,
        verified_claims=[],
        ungrounded_claims=["Missing citation: assertion has no citation link"],
        hallucinated_citations=[],
        reasoning="Missing citation",
    )
    with patch(
        "generation.grounding_verifier.get_async_openai_client",
        return_value=_mock_parse_response(mock_model),
    ):
        report = await ClaimGroundingVerifier.verify(answer, context)
        assert len(report.ungrounded_claims) == 1
        assert "Missing citation" in report.ungrounded_claims[0]


@pytest.mark.asyncio
async def test_verified_calculation_accepted():
    context = [
        _make_search_result("Revenue in 2024 was $100 million and in 2025 was $150 million.")
    ]
    answer = "Revenue grew by 50% YoY [1]."
    mock_model = GroundingReportModel(
        is_grounded=True,
        grounding_score=1.0,
        verified_claims=["Revenue grew by 50% YoY [1]"],
        ungrounded_claims=[],
        hallucinated_citations=[],
        reasoning="Calculation verified",
    )
    with patch(
        "generation.grounding_verifier.get_async_openai_client",
        return_value=_mock_parse_response(mock_model),
    ):
        report = await ClaimGroundingVerifier.verify(answer, context, verified_calculations=[50.0])
        assert report.is_grounded is True
        assert len(report.verified_claims) == 1


@pytest.mark.asyncio
async def test_empty_answer_and_context_handling():
    report_empty = await ClaimGroundingVerifier.verify("", [_make_search_result("data")])
    assert report_empty.is_grounded is False
    assert report_empty.grounding_score == 0.0

    report_no_ctx = await ClaimGroundingVerifier.verify("Answer", [])
    assert report_no_ctx.is_grounded is False
    assert report_no_ctx.grounding_score == 0.0
