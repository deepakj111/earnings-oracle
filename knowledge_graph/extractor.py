# knowledge_graph/extractor.py
"""
LLM-powered financial entity and relationship extraction for SEC filings.

Runs during ingestion to populate the knowledge graph using pure LLM extraction
(e.g., gpt-4.1-nano) evaluated per parent chunk for exhaustive metric, segment,
executive, and relationship discovery with precise chunk-level provenance.

Design decisions:
  - Evaluated per parent chunk (512 tokens) to capture all financial data without truncation
  - Pure LLM extraction (no regex fallback) for high accuracy and clean structured output
  - Fail-open: extraction errors skip the chunk, never crash the ingestion pipeline
  - Concurrent per-chunk LLM calls bounded by asyncio Semaphore
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import tiktoken
from loguru import logger

from config import settings
from knowledge_graph.models import Entity, EntityType, Relationship, RelationType

_ENC = tiktoken.get_encoding("cl100k_base")
_KG_MAX_TOKENS = 3500  # safe budget for extraction LLM context


def _truncate_to_tokens(text: str, max_tokens: int = _KG_MAX_TOKENS) -> str:
    """Truncate text to max_tokens using tiktoken, preserving whole tokens."""
    tokens = _ENC.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return _ENC.decode(tokens[:max_tokens])


# ── LLM extraction prompt ─────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """\
You are an expert financial document analyst. Extract all key financial entities, metrics, numbers, segment performance, guidance, products, executives, risks, and relationships from the provided SEC filing text section.

Return ONLY a JSON object with this exact structure:
{
  "entities": [
    {
      "name": "entity or metric name",
      "entity_type": "PERSON|PRODUCT|SEGMENT|METRIC|COMPETITOR|RISK|INITIATIVE",
      "properties": {"value": "optional exact numerical value or description"}
    }
  ],
  "relationships": [
    {
      "source": "entity name",
      "target": "entity name",
      "relation": "LEADS|REPORTS|DRIVES_REVENUE|COMPETES_WITH|RISK_TO|PART_OF|MENTIONED_WITH",
      "properties": {"value": "optional relationship details or metric value"}
    }
  ]
}

Rules:
- Extract METRICS exhaustively (Revenue, Net Income, Operating Margin, EPS, CapEx, Free Cash Flow, Growth Rates) with exact numbers in properties.
- Extract SEGMENTS (e.g., Services, Cloud, Hardware) and link them to reported METRICS.
- Extract PERSONS (CEOs, CFOs, executives) and their roles.
- Extract PRODUCTS, COMPETITORS, strategic INITIATIVES (AI, cloud), and RISKS.
- Keep entity names canonical, normalized, and concise (e.g., "revenue", "cloud segment", "tim cook").
- Only extract entities explicitly mentioned in the text.
- Return empty lists if no entities or relationships are present.

---
EXAMPLES

Example 1 — Table section:
Text: "| Segment | Q4 2024 Revenue | YoY Growth | \n| Services | $26.3B | +12% | \n| Products | $70.4B | +5% |"
Output:
{
  "entities": [
    {"name": "services", "entity_type": "SEGMENT", "properties": {"value": "$26.3B"}},
    {"name": "products", "entity_type": "SEGMENT", "properties": {"value": "$70.4B"}},
    {"name": "services revenue", "entity_type": "METRIC", "properties": {"value": "$26.3B", "growth": "+12%"}},
    {"name": "products revenue", "entity_type": "METRIC", "properties": {"value": "$70.4B", "growth": "+5%"}}
  ],
  "relationships": [
    {"source": "services revenue", "target": "services", "relation": "REPORTS", "properties": {}},
    {"source": "products revenue", "target": "products", "relation": "REPORTS", "properties": {}}
  ]
}

Example 2 — Executive commentary:
Text: "CEO Tim Cook said Services revenue reached a new all-time high of $26.3 billion, driven by the App Store and Apple Music. CFO Luca Maestri noted that diluted EPS grew 12% year over year to $2.18."
Output:
{
  "entities": [
    {"name": "tim cook", "entity_type": "PERSON", "properties": {"role": "CEO"}},
    {"name": "luca maestri", "entity_type": "PERSON", "properties": {"role": "CFO"}},
    {"name": "services", "entity_type": "SEGMENT", "properties": {"value": "$26.3B"}},
    {"name": "services revenue", "entity_type": "METRIC", "properties": {"value": "$26.3 billion"}},
    {"name": "diluted eps", "entity_type": "METRIC", "properties": {"value": "$2.18", "growth": "12%"}},
    {"name": "app store", "entity_type": "PRODUCT", "properties": {}},
    {"name": "apple music", "entity_type": "PRODUCT", "properties": {}}
  ],
  "relationships": [
    {"source": "tim cook", "target": "services", "relation": "LEADS", "properties": {}},
    {"source": "app store", "target": "services", "relation": "PART_OF", "properties": {}},
    {"source": "apple music", "target": "services", "relation": "PART_OF", "properties": {}},
    {"source": "services revenue", "target": "services", "relation": "REPORTS", "properties": {}}
  ]
}

Example 3 — Risk factor:
Text: "The macroeconomic environment, including foreign exchange headwinds, poses risks to our international revenue. Increased competition from Google and Samsung in the smartphone market may pressure iPhone margins."
Output:
{
  "entities": [
    {"name": "foreign exchange headwinds", "entity_type": "RISK", "properties": {}},
    {"name": "international revenue", "entity_type": "METRIC", "properties": {}},
    {"name": "google", "entity_type": "COMPETITOR", "properties": {}},
    {"name": "samsung", "entity_type": "COMPETITOR", "properties": {}},
    {"name": "iphone", "entity_type": "PRODUCT", "properties": {}},
    {"name": "iphone margins", "entity_type": "METRIC", "properties": {}}
  ],
  "relationships": [
    {"source": "foreign exchange headwinds", "target": "international revenue", "relation": "RISK_TO", "properties": {}},
    {"source": "google", "target": "iphone", "relation": "COMPETES_WITH", "properties": {}},
    {"source": "samsung", "target": "iphone", "relation": "COMPETES_WITH", "properties": {}}
  ]
}
"""

EXTRACTION_USER_TEMPLATE = """\
Company: {ticker} | Period: {fiscal_period}

Text:
{text}
"""


def _parse_json_response(content: str) -> dict:
    """Parse JSON string with fallback repair for truncated or trailing-comma output."""
    if not content or not content.strip():
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        cleaned = content.strip()
        # Remove trailing comma before closing braces/brackets
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # If inside unclosed string, append quote
        quote_count = len(re.findall(r'(?<!\\)"', cleaned))
        if quote_count % 2 != 0:
            cleaned += '"'

        # Track nesting stack of '{' and '[' outside strings
        stack: list[str] = []
        in_string = False
        escaped = False
        for char in cleaned:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char in ("{", "["):
                    stack.append(char)
                elif char == "}" and stack and stack[-1] == "{":
                    stack.pop()
                elif char == "]" and stack and stack[-1] == "[":
                    stack.pop()

        # Close open elements in reverse order
        for opener in reversed(stack):
            if opener == "{":
                cleaned += "}"
            elif opener == "[":
                cleaned += "]"

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}


async def _call_llm_extract(
    text: str,
    ticker: str,
    fiscal_period: str,
) -> dict:
    """
    Call the LLM to extract entities and relationships from text asynchronously.

    Returns parsed JSON dict or empty dict on failure.
    """
    try:
        from config.openai_client import get_async_openai_client

        client = get_async_openai_client()
        # Token-level truncation prevents mid-sentence cuts that character
        # slicing (text[:4000]) could produce on multibyte or BPE-split content.
        truncated_text = _truncate_to_tokens(text)
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": EXTRACTION_USER_TEMPLATE.format(
                    ticker=ticker,
                    fiscal_period=fiscal_period,
                    text=truncated_text,
                ),
            },
        ]
        kwargs: dict[str, Any] = {
            "model": settings.knowledge_graph.extraction_model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        # Attempt 1: Try max_completion_tokens; only add temperature if model supports it
        call_kwargs = dict(kwargs)
        call_kwargs["max_completion_tokens"] = settings.knowledge_graph.extraction_max_tokens
        if settings.knowledge_graph.extraction_temperature != 1.0:
            call_kwargs["temperature"] = settings.knowledge_graph.extraction_temperature

        try:
            response = await client.chat.completions.create(**call_kwargs)
        except Exception as first_exc:
            err_msg = str(first_exc)
            logger.debug(
                f"KG LLM Attempt 1 failed ({err_msg[:120]}), retrying with fallback params"
            )
            if "429" in err_msg or "rate limit" in err_msg.lower():
                await asyncio.sleep(2.0)

            retry_kwargs = dict(kwargs)

            # Explicitly skip temperature if it caused the first failure
            if (
                "temperature" not in err_msg
                and settings.knowledge_graph.extraction_temperature != 1.0
            ):
                retry_kwargs["temperature"] = settings.knowledge_graph.extraction_temperature
            # else: temperature is omitted entirely so the model uses its default (1.0)

            # Alternate token limit parameter if max_completion_tokens failed
            if "max_completion_tokens" in err_msg or "unsupported_parameter" in err_msg:
                retry_kwargs["max_tokens"] = settings.knowledge_graph.extraction_max_tokens
            else:
                retry_kwargs["max_completion_tokens"] = (
                    settings.knowledge_graph.extraction_max_tokens
                )

            try:
                response = await client.chat.completions.create(**retry_kwargs)
            except Exception as second_exc:
                second_err = str(second_exc)
                if "429" in second_err or "rate limit" in second_err.lower():
                    await asyncio.sleep(4.0)
                    response = await client.chat.completions.create(**retry_kwargs)
                else:
                    raise

        content = response.choices[0].message.content or "{}"
        return _parse_json_response(content)
    except Exception as exc:
        logger.warning(f"LLM entity extraction failed (fail-open): {exc}")
        return {}


async def _extract_single_chunk(
    chunk: Any,
    ticker: str,
    fiscal_period: str,
    semaphore: asyncio.Semaphore,
    chunk_index: int = 0,
    total_chunks: int = 0,
) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities and relationships from a single chunk using OpenAI LLM."""
    async with semaphore:
        raw = await _call_llm_extract(chunk.text, ticker, fiscal_period)
        entities: list[Entity] = []
        relationships: list[Relationship] = []

        chunk_id = getattr(chunk, "chunk_id", "")

        for e_data in raw.get("entities", []):
            try:
                entity = Entity(
                    name=e_data.get("name", ""),
                    entity_type=e_data.get("entity_type", EntityType.METRIC),
                    ticker=ticker,
                    fiscal_period=fiscal_period,
                    chunk_ids=[chunk_id] if chunk_id else [],
                    properties=e_data.get("properties", {}),
                )
                if entity.name:
                    entities.append(entity)
            except Exception as exc:
                logger.debug(f"Skipping malformed entity: {exc}")

        for r_data in raw.get("relationships", []):
            try:
                rel = Relationship(
                    source=r_data.get("source", ""),
                    target=r_data.get("target", ""),
                    relation=r_data.get("relation", RelationType.MENTIONED_WITH),
                    ticker=ticker,
                    fiscal_period=fiscal_period,
                    chunk_id=chunk_id,
                    properties=r_data.get("properties", {}),
                )
                if rel.source and rel.target:
                    relationships.append(rel)
            except Exception as exc:
                logger.debug(f"Skipping malformed relationship: {exc}")

        logger.info(
            f"[KG chunk {chunk_index}/{total_chunks}] {ticker} | "
            f"{len(entities)}e {len(relationships)}r | id={chunk_id[:20] if chunk_id else 'n/a'}"
        )
        return entities, relationships


async def extract_entities_from_chunks(
    parent_chunks: list,
    ticker: str,
    fiscal_period: str,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Extract financial entities and relationships from parent chunks asynchronously.

    Runs OpenAI LLM extraction per parent chunk in parallel with controlled concurrency,
    ensuring exhaustive financial data extraction and accurate chunk provenance.

    Args:
        parent_chunks: List of Chunk objects (parent type only)
        ticker: Company ticker symbol
        fiscal_period: e.g., "Q4 2024"

    Returns:
        Tuple of (entities, relationships) ready for knowledge graph insertion.
    """
    llm_enabled = settings.knowledge_graph.extraction_enabled

    if not llm_enabled or not parent_chunks:
        return [], []

    total = len(parent_chunks)

    logger.info(f"[KG Extract] {ticker} {fiscal_period} | starting {total} chunks (concurrency=5)")

    semaphore = asyncio.Semaphore(5)
    tasks = [
        _extract_single_chunk(chunk, ticker, fiscal_period, semaphore, idx + 1, total)
        for idx, chunk in enumerate(parent_chunks)
    ]

    results = await asyncio.gather(*tasks)

    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []
    for chunk_entities, chunk_rels in results:
        all_entities.extend(chunk_entities)
        all_relationships.extend(chunk_rels)

    logger.info(
        f"[KG Extract] {ticker} {fiscal_period} | DONE | "
        f"{len(all_entities)} entities, {len(all_relationships)} relationships "
        f"from {total} parent chunks"
    )
    return all_entities, all_relationships
