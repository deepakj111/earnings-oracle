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

from loguru import logger

from config import settings
from knowledge_graph.models import Entity, EntityType, Relationship, RelationType

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
        response = await client.chat.completions.create(
            model=settings.knowledge_graph.extraction_model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": EXTRACTION_USER_TEMPLATE.format(
                        ticker=ticker,
                        fiscal_period=fiscal_period,
                        text=text[:4000],
                    ),
                },
            ],
            temperature=settings.knowledge_graph.extraction_temperature,
            max_tokens=settings.knowledge_graph.extraction_max_tokens,
            response_format={"type": "json_object"},
        )
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

    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []

    semaphore = asyncio.Semaphore(5)
    tasks = [
        _extract_single_chunk(chunk, ticker, fiscal_period, semaphore) for chunk in parent_chunks
    ]
    results = await asyncio.gather(*tasks)

    for chunk_entities, chunk_rels in results:
        all_entities.extend(chunk_entities)
        all_relationships.extend(chunk_rels)

    logger.info(
        f"[KG Extract] {ticker} {fiscal_period} | "
        f"{len(all_entities)} entities, {len(all_relationships)} relationships "
        f"extracted from {len(parent_chunks)} parent chunks (pure LLM)"
    )
    return all_entities, all_relationships
