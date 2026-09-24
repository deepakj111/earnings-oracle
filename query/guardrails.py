# query/guardrails.py
"""
Input validation, safety guardrails, and security filtering for Financial RAG.

Provides multi-layer protection before queries reach the LLM or vector search:
  1. Prompt injection & jailbreak detection (heuristic & regex pattern matching)
  2. PII detection & masking (SSN, credit cards with Luhn validation)
  3. Token budget enforcement (prevents context overflow & token denial-of-service)
  4. Unicode normalization & invisible character stripping (prevents token smuggling)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import tiktoken
from loguru import logger

try:
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


class ViolationSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class GuardrailViolation:
    violation_type: str
    message: str
    severity: ViolationSeverity
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailResult:
    passed: bool
    sanitized_query: str
    violations: list[GuardrailViolation] = field(default_factory=list)
    token_count: int = 0
    risk_score: float = 0.0

    @property
    def has_critical_violation(self) -> bool:
        return any(
            v.severity in (ViolationSeverity.HIGH, ViolationSeverity.CRITICAL)
            for v in self.violations
        )


# ── Prompt Injection Patterns ──────────────────────────────────────────────────
_PROMPT_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str, ViolationSeverity]] = [
    (
        re.compile(
            r"(?i)\b(?:ignore|disregard|forget|override|bypass|cancel)\s+(?:all\s+)?(?:(?:previous|prior|above|system)\s+)*(?:instructions|rules|prompts|directives|constraints)\b",
        ),
        "Attempt to override or disregard prior system instructions.",
        ViolationSeverity.CRITICAL,
    ),
    (
        re.compile(
            r"(?i)\b(?:you\s+are\s+now|act\s+as|roleplay\s+as)\s+(?:(?:an?|in)\s+)?(?:unfiltered|dan|jailbreak|unrestricted|godmode)\b",
        ),
        "Attempted persona adoption or jailbreak mode.",
        ViolationSeverity.CRITICAL,
    ),
    (
        re.compile(
            r"(?i)\b(?:reveal|print|show|dump|repeat|output)\s+(?:all\s+)?(?:your\s+)?(?:system\s+prompt|developer\s+message|system\s+instructions|pre-prompt)\b",
        ),
        "Attempt to extract confidential system prompt.",
        ViolationSeverity.HIGH,
    ),
    (
        re.compile(
            r"(?i)(?:<\|im_start\|>|<\|im_end\|>|<<sys>>|\[inst\]|\[\/inst\]|<system>|<\/system>)",
        ),
        "Attempt to inject raw instruction/chat formatting tokens.",
        ViolationSeverity.CRITICAL,
    ),
]

# ── PII Patterns ──────────────────────────────────────────────────────────────
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b|\b\d{15,16}\b")


def _luhn_checksum_valid(num_str: str) -> bool:
    """Validate credit card number using the Luhn mod-10 algorithm."""
    digits = [int(c) for c in re.sub(r"\D", "", num_str)]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for idx, d in enumerate(reverse_digits):
        if idx % 2 == 1:
            d = d * 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def sanitize_text(text: str) -> str:
    """
    Normalize unicode, strip non-printable characters and zero-width smuggling tokens.
    """
    if not text:
        return ""
    # Strip zero-width and bidirectional formatting characters
    cleaned = re.sub(r"[\u200B-\u200D\uFEFF\u202A-\u202E]", "", text)
    # Normalize unicode to NFKC
    cleaned = unicodedata.normalize("NFKC", cleaned)
    # Strip unprintable control characters (keep newline and tab)
    cleaned = "".join(ch for ch in cleaned if ch.isprintable() or ch in "\n\t")
    return cleaned.strip()


class QueryGuardrails:
    """
    Multi-tier input guardrails for financial question-answering.

    Enforces prompt injection defense, PII redaction, and token budget limits
    before queries enter the retrieval pipeline.
    """

    def __init__(self, max_tokens: int = 500, enforce_pnd: bool = True) -> None:
        self.max_tokens = max_tokens
        self.enforce_pnd = enforce_pnd

    def validate(self, query: str) -> GuardrailResult:
        """
        Validate and sanitize user query against all active safety rules.
        """
        sanitized = sanitize_text(query)
        violations: list[GuardrailViolation] = []

        # 1. Token budget check
        token_count = 0
        if _ENC:
            token_count = len(_ENC.encode(sanitized))
            if token_count > self.max_tokens:
                violations.append(
                    GuardrailViolation(
                        violation_type="token_limit_exceeded",
                        message=f"Query exceeds maximum token limit of {self.max_tokens} (found {token_count}).",
                        severity=ViolationSeverity.HIGH,
                        details={"token_count": token_count, "max_tokens": self.max_tokens},
                    )
                )
        else:
            token_count = len(sanitized.split())

        # 2. Prompt injection detection
        for pattern, message, severity in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(sanitized):
                violations.append(
                    GuardrailViolation(
                        violation_type="prompt_injection",
                        message=message,
                        severity=severity,
                        details={"matched_pattern": pattern.pattern},
                    )
                )

        # 3. PII Detection (SSN)
        ssn_matches = _SSN_PATTERN.findall(sanitized)
        if ssn_matches:
            violations.append(
                GuardrailViolation(
                    violation_type="pii_ssn_detected",
                    message="Social Security Number (SSN) pattern detected in query.",
                    severity=ViolationSeverity.HIGH,
                    details={"count": len(ssn_matches)},
                )
            )
            # Redact SSN
            sanitized = _SSN_PATTERN.sub("[REDACTED_SSN]", sanitized)

        # 4. PII Detection (Credit Card)
        card_matches = _CREDIT_CARD_PATTERN.findall(sanitized)
        valid_cards = [card for card in card_matches if _luhn_checksum_valid(card)]
        if valid_cards:
            violations.append(
                GuardrailViolation(
                    violation_type="pii_credit_card_detected",
                    message="Valid credit card number detected in query via Luhn check.",
                    severity=ViolationSeverity.HIGH,
                    details={"count": len(valid_cards)},
                )
            )
            for card in valid_cards:
                sanitized = sanitized.replace(card, "[REDACTED_CREDIT_CARD]")

        # Compute risk score
        risk_score = 0.0
        for v in violations:
            if v.severity == ViolationSeverity.CRITICAL:
                risk_score = max(risk_score, 1.0)
            elif v.severity == ViolationSeverity.HIGH:
                risk_score = max(risk_score, 0.75)
            elif v.severity == ViolationSeverity.MEDIUM:
                risk_score = max(risk_score, 0.40)
            elif v.severity == ViolationSeverity.LOW:
                risk_score = max(risk_score, 0.15)

        passed = not any(
            v.severity in (ViolationSeverity.HIGH, ViolationSeverity.CRITICAL) for v in violations
        )

        if not passed:
            logger.warning(
                f"Query failed guardrails (risk_score={risk_score}): "
                f"{[v.violation_type for v in violations]}"
            )

        return GuardrailResult(
            passed=passed,
            sanitized_query=sanitized,
            violations=violations,
            token_count=token_count,
            risk_score=risk_score,
        )


__all__ = [
    "QueryGuardrails",
    "GuardrailResult",
    "GuardrailViolation",
    "ViolationSeverity",
    "sanitize_text",
]
