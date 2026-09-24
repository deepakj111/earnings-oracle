"""
Numerical Hallucination Fence for Financial RAG.

Extracts all quantitative/financial figures from generated answers (currencies,
percentages, multiples, scale abbreviations like B/M/K) and verifies their
grounding against retrieved source context chunks and verified PAL calculations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from retrieval.models import SearchResult


@dataclass
class NumericalHallucinationReport:
    """Audit report of quantitative grounding for a generated answer."""

    total_extracted: int = 0
    sources_verified: int = 0
    flagged_numbers: list[str] = field(default_factory=list)
    verified_numbers: list[str] = field(default_factory=list)
    has_hallucinations: bool = False
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        """Fraction of extracted numbers that were successfully grounded."""
        if self.total_extracted == 0:
            return 1.0
        return self.sources_verified / self.total_extracted


class NumericalHallucinationFence:
    """
    Regex and tolerance-based numerical grounding verifier.

    Catches fabricated numbers, swapped statistics, and unit hallucination
    prior to returning final answers to end users.
    """

    # Matches numbers with optional currency prefix, minus/en-dash, parentheses, comma grouping, decimals, and scale/percentage suffixes
    _NUMERIC_PATTERN = re.compile(
        r"(?:[-–—]\s*)?(?:\$|€|£)?\b\d+(?:,\d{3})*(?:\.\d+)?(?:%|\s*(?:billion|million|trillion|bps|basis points|[BMKbmk])\b)?"
        r"|(?:\$|€|£)?\s*\(\s*(?:\$|€|£)?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|\s*(?:billion|million|trillion|bps|basis points|[BMKbmk])\b)?\s*\)"
        r"(?:%|\s*(?:billion|million|trillion|bps|basis points|[BMKbmk])\b)?",
        re.IGNORECASE,
    )

    # Years like 2020..2035 that are typically temporal labels rather than metric values
    _YEAR_PATTERN = re.compile(r"^(?:19|20)\d{2}$")

    @classmethod
    def extract_numbers(cls, text: str) -> list[str]:
        """Extract all candidate numeric expressions from text, ignoring bare calendar years."""
        raw_matches = cls._NUMERIC_PATTERN.findall(text)
        candidates: list[str] = []
        for m in raw_matches:
            cleaned = m.strip()
            # Skip bare year mentions like '2024' (unless they have currency or suffix)
            if cls._YEAR_PATTERN.match(cleaned) and not cleaned.startswith(("$", "€", "£")):
                continue
            # Skip single-digit citation indices if matched alone (handled by context)
            if cleaned.isdigit() and len(cleaned) == 1:
                continue
            if cleaned:
                candidates.append(cleaned)
        return candidates

    @classmethod
    def _normalize_number(cls, num_str: str) -> tuple[float | None, str]:
        """
        Normalize a numeric string into (float_value, suffix).

        Applies the actual scale multiplier so cross-unit comparisons work:
          '$14.2B'   → (14_200_000_000.0, 'B')
          '$14,200M' → (14_200_000_000.0, 'M')
          '12%'      → (0.12, '%')  — stored as fraction for comparison
          '($500M)'  → (-500_000_000.0, 'M') — SEC accounting parentheses negative
          '-3.2%'    → (-3.2, '%')

        This ensures the fence does NOT raise false-positive hallucination
        warnings when the answer correctly states '$14.2B' and the source
        text contains the equivalent '$14,200 million', or when negative
        figures appear in financial statements.
        """
        s = num_str.strip()
        is_negative = False

        # Detect accounting parentheses: ($1,234) or (1,234) or (5.2)%
        if s.startswith("(") and ")" in s:
            is_negative = True
            s = s.replace("(", "").replace(")", "").strip()
        elif s.startswith(("-", "–", "—")):
            is_negative = True
            s = s.lstrip("-–— ").strip()

        s = s.replace(",", "")
        if s and s[0] in ("$", "€", "£"):
            s = s[1:].strip()
            if s.startswith("(") and s.endswith(")"):
                is_negative = True
                s = s[1:-1].strip()

        scale = 1.0
        suffix = ""
        lower = s.lower()

        if lower.endswith("trillion") or (lower.endswith("t") and not lower.endswith("%")):
            scale = 1e12
            suffix = "T"
            s = re.sub(r"[a-zA-Z]+", "", s).strip()
        elif lower.endswith("billion") or (lower.endswith("b") and not lower.endswith("%")):
            scale = 1e9
            suffix = "B"
            s = re.sub(r"[a-zA-Z]+", "", s).strip()
        elif lower.endswith("million") or (lower.endswith("m") and not lower.endswith("%")):
            scale = 1e6
            suffix = "M"
            s = re.sub(r"[a-zA-Z]+", "", s).strip()
        elif lower.endswith("thousand") or (lower.endswith("k") and not lower.endswith("%")):
            scale = 1e3
            suffix = "K"
            s = re.sub(r"[a-zA-Z]+", "", s).strip()
        elif lower.endswith(("%", "percent")):
            suffix = "%"
            s = s.replace("%", "").replace("percent", "").strip()
            try:
                val = float(s)
                return (-val if is_negative else val), suffix
            except ValueError:
                return None, ""
        elif lower.endswith(("bps", "basis points")):
            suffix = "bps"
            s = re.sub(r"(?i)\s*(?:basis\s+points|bps)", "", s).strip()
            try:
                val = float(s)
                return (-val if is_negative else val), suffix
            except ValueError:
                return None, ""

        try:
            val = float(s) * scale
            return (-val if is_negative else val), suffix
        except ValueError:
            return None, ""

    @classmethod
    def verify(
        cls,
        answer: str,
        retrieved_chunks: list[SearchResult] | list[str],
        verified_calculations: list[float] | None = None,
        tolerance: float = 0.005,
    ) -> NumericalHallucinationReport:
        """
        Verify that numbers in the answer appear in the source chunks or verified calculations.

        Args:
            answer: Generated answer text.
            retrieved_chunks: List of SearchResult objects or raw text chunks.
            verified_calculations: List of calculation results from PAL calculator.
            tolerance: Relative float tolerance for numeric equality (default 0.5%).
        """
        if not answer.strip():
            return NumericalHallucinationReport()

        extracted = cls.extract_numbers(answer)
        if not extracted:
            return NumericalHallucinationReport()

        # Combine all source text into a single search space
        source_texts: list[str] = []
        for c in retrieved_chunks:
            if isinstance(c, SearchResult):
                source_texts.append(c.parent_text or c.text)
            elif isinstance(c, str):
                source_texts.append(c)

        combined_source = " ".join(source_texts)
        combined_source_normalized = combined_source.replace(",", "")

        # Pre-extract and normalize numbers from source for cross-unit comparison
        source_numbers = cls.extract_numbers(combined_source)
        source_scaled_values: list[float] = []
        for sn in source_numbers:
            s_val, _ = cls._normalize_number(sn)
            if s_val is not None:
                source_scaled_values.append(s_val)

        verified: list[str] = []
        flagged: list[str] = []
        details: list[dict[str, Any]] = []

        calc_floats = verified_calculations or []

        for raw_num in extracted:
            val, suffix = cls._normalize_number(raw_num)
            is_grounded = False
            reason = "unverified"

            # 1. Exact string match in source (fast path)
            clean_num = raw_num.replace(",", "")
            if raw_num in combined_source or clean_num in combined_source_normalized:
                is_grounded = True
                reason = "exact_source_match"

            # 2. Check if produced by verified PAL math
            if not is_grounded and val is not None:
                for c_val in calc_floats:
                    diff = abs(c_val - val)
                    denom = max(abs(c_val), 1e-9)
                    if (diff / denom) <= tolerance:
                        is_grounded = True
                        reason = f"verified_pal_calculation ({c_val})"
                        break
                    # If percent, also check fraction equivalence (e.g. c_val=0.20 vs val=20.0)
                    if suffix == "%":
                        diff_pct = abs(c_val * 100 - val)
                        if (diff_pct / max(abs(val), 1e-9)) <= tolerance:
                            is_grounded = True
                            reason = f"verified_pal_calculation ({c_val})"
                            break
                    elif suffix == "bps":
                        # 1% = 100 bps: check if PAL calculated a percentage (e.g. 1.5% -> 150 bps)
                        diff_bps = abs(c_val * 100 - val)
                        if (diff_bps / max(abs(val), 1e-9)) <= tolerance:
                            is_grounded = True
                            reason = f"verified_pal_calculation ({c_val})"
                            break

            # 3. Scaled cross-unit match in source (e.g. $14.2B matches $14,200M, or 150 bps matches 1.5%)
            if not is_grounded and val is not None:
                for s_val in source_scaled_values:
                    diff = abs(s_val - val)
                    denom = max(abs(s_val), 1e-9)
                    if (diff / denom) <= tolerance:
                        is_grounded = True
                        reason = f"scaled_source_match ({s_val})"
                        break
                    if suffix == "bps":
                        # Match 150 bps in answer against 1.5% in source
                        diff_bps = abs(s_val * 100 - val)
                        if (diff_bps / max(abs(val), 1e-9)) <= tolerance:
                            is_grounded = True
                            reason = f"scaled_source_match ({s_val}%)"
                            break
                    elif suffix == "%":
                        # Match 1.5% in answer against 150 bps in source
                        diff_pct = abs(val * 100 - s_val)
                        if (diff_pct / max(abs(s_val), 1e-9)) <= tolerance:
                            is_grounded = True
                            reason = f"scaled_source_match ({s_val} bps)"
                            break

            # 4. Numeric regex match in source (e.g. 14.2 in '14.2 billion')
            if not is_grounded and val is not None:
                # Search for float string representation in normalized source
                pattern = rf"\b{re.escape(str(round(val, 2)))}\b"
                if re.search(pattern, combined_source_normalized):
                    is_grounded = True
                    reason = "numeric_source_match"

            if is_grounded:
                verified.append(raw_num)
            else:
                flagged.append(raw_num)

            details.append(
                {
                    "number": raw_num,
                    "parsed_value": val,
                    "grounded": is_grounded,
                    "reason": reason,
                }
            )

        has_hallucinations = len(flagged) > 0
        if has_hallucinations:
            logger.warning(
                f"[NumericalHallucinationFence] Flagged {len(flagged)} ungrounded numbers: {flagged}"
            )

        return NumericalHallucinationReport(
            total_extracted=len(extracted),
            sources_verified=len(verified),
            flagged_numbers=flagged,
            verified_numbers=verified,
            has_hallucinations=has_hallucinations,
            details=details,
        )
