"""
Citation Integrity Validator for Multi-Entity Financial RAG.

Detects cross-entity citation contamination in comparative or multi-company queries
where a metric for one company is attributed with a citation pointing to a different company's filing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

from config.companies import CompanyRegistry
from generation.models import Citation


@dataclass
class CitationValidationReport:
    """Report of citation integrity and cross-entity consistency."""

    total_citations: int = 0
    valid_citations: int = 0
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = True


class CitationIntegrityValidator:
    """
    Validates that inline citations in answers map to the correct entity mentioned in context.
    """

    _CITATION_TAG_RE = re.compile(r"\[(\d+)\]")

    @classmethod
    def _build_alias_map(cls) -> dict[str, str]:
        """Build mapping of lowercase entity names, aliases, and tickers to standard ticker."""
        return CompanyRegistry.get_alias_map()

    @classmethod
    def validate(
        cls,
        answer: str,
        citations: list[Citation],
    ) -> CitationValidationReport:
        """
        Validate citation integrity across sentences.

        Args:
            answer: Synthesised answer containing inline [N] citations.
            citations: List of Citation objects corresponding to [1], [2], ...

        Returns:
            CitationValidationReport with any cross-ticker citation warnings.
        """
        if not answer.strip() or not citations:
            return CitationValidationReport()

        alias_map = cls._build_alias_map()
        citation_by_index: dict[int, Citation] = {c.index: c for c in citations}

        # Split answer into sentence-like clauses
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        warnings: list[str] = []
        total_citations_checked = 0

        for sent in sentences:
            citation_matches = [int(m) for m in cls._CITATION_TAG_RE.findall(sent)]
            if not citation_matches:
                continue

            # Identify which companies are mentioned in this specific sentence
            sent_lower = sent.lower()
            detected_tickers: set[str] = set()
            for alias, ticker in alias_map.items():
                pattern = rf"\b{re.escape(alias)}\b"
                if re.search(pattern, sent_lower):
                    detected_tickers.add(ticker)

            # If the sentence uniquely mentions a single company, check all citations in it
            if len(detected_tickers) == 1:
                target_ticker = next(iter(detected_tickers))
                for c_idx in citation_matches:
                    total_citations_checked += 1
                    cit = citation_by_index.get(c_idx)
                    if cit and cit.ticker:
                        cit_ticker = cit.ticker.upper()
                        if cit_ticker != target_ticker:
                            warn_msg = (
                                f"Citation [{c_idx}] cites {cit_ticker} ({cit.fiscal_period}) "
                                f"in a statement specifically discussing {target_ticker}: "
                                f"'{sent.strip()[:100]}...'"
                            )
                            warnings.append(warn_msg)
                            logger.warning(f"[CitationIntegrityValidator] {warn_msg}")
            else:
                total_citations_checked += len(citation_matches)

        is_valid = len(warnings) == 0
        valid_citations = max(0, total_citations_checked - len(warnings))

        return CitationValidationReport(
            total_citations=total_citations_checked,
            valid_citations=valid_citations,
            warnings=warnings,
            is_valid=is_valid,
        )
