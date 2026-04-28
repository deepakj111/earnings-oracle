# knowledge_graph/extractor.py
"""
LLM-powered entity and relationship extraction for SEC 8-K filings.

Runs during ingestion to populate the knowledge graph. Uses gpt-4.1-nano
for cost-efficient extraction with structured JSON output.

Design decisions:
  - Extracts from parent chunks (512 tokens), not children, to minimize LLM calls
  - Includes regex-based fallback for common financial entities (tickers, dollar amounts)
  - Uses structured JSON output format for reliable parsing
  - Fail-open: extraction errors skip the chunk, never crash the pipeline
  - Batch-friendly: processes all parent chunks from one document at once
"""

from __future__ import annotations

import asyncio
import json
import re

from loguru import logger

from config import settings
from knowledge_graph.models import Entity, EntityType, Relationship, RelationType

# ── Regex fallback patterns for deterministic extraction ──────────────────────

_DOLLAR_RE = re.compile(
    r"\$[\d,]+\.?\d*\s*(?:billion|million|B|M|bn|mn|trillion|T)",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    r"[\d.]+\s*(?:percent|%)",
    re.IGNORECASE,
)
_TICKER_RE = re.compile(
    r"\b(?:AAPL|NVDA|MSFT|AMZN|META|JPM|XOM|UNH|TSLA|WMT|GOOG|GOOGL)\b",
)

# Known executive patterns
_EXEC_RE = re.compile(
    r"(?:CEO|CFO|COO|CTO|Chief\s+(?:Executive|Financial|Operating|Technology)\s+Officer)",
    re.IGNORECASE,
)

# ── LLM extraction prompt ─────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """\
You are a financial document analyst. Extract entities and relationships \
from the given SEC 8-K earnings filing text.

Return ONLY a JSON object with this exact structure:
{
  "entities": [
    {
      "name": "entity name",
      "entity_type": "PERSON|PRODUCT|SEGMENT|METRIC|COMPETITOR|RISK|INITIATIVE",
      "properties": {}
    }
  ],
  "relationships": [
    {
      "source": "entity name",
      "target": "entity name",
      "relation": "LEADS|REPORTS|DRIVES_REVENUE|COMPETES_WITH|RISK_TO|PART_OF|MENTIONED_WITH",
      "properties": {}
    }
  ]
}

Rules:
- Extract PERSONS (executives, board members) with their roles in properties
- Extract PRODUCTS and SEGMENTS mentioned in the filing
- Extract key METRICS (revenue, EPS, margins) with values in properties
- Extract COMPETITORS by name
- Extract RISK factors and headwinds
- Extract strategic INITIATIVES (AI, cloud, sustainability)
- Only extract entities explicitly mentioned in the text
- Keep entity names short and canonical (e.g., "iPhone" not "the iPhone product line")
- Return empty lists if no entities/relationships are found
"""

EXTRACTION_USER_TEMPLATE = """\
Company: {ticker} | Period: {fiscal_period}

Text:
{text}
"""


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
                        text=text[:3000],  # cap input to control cost
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as exc:
        logger.warning(f"LLM entity extraction failed (fail-open): {exc}")
        return {}


def _regex_extract(
    text: str,
    ticker: str,
    fiscal_period: str,
    chunk_id: str,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Deterministic regex-based extraction as a fallback/supplement.

    Extracts:
      - Dollar amounts as METRIC entities
      - Percentage changes as METRIC entities
      - Competitor ticker mentions as COMPETITOR entities
      - Executive role mentions as PERSON entities
    """
    entities: list[Entity] = []
    relationships: list[Relationship] = []

    # Extract competitor tickers mentioned in text
    for match in _TICKER_RE.finditer(text):
        mentioned_ticker = match.group()
        if mentioned_ticker != ticker:
            entities.append(
                Entity(
                    name=mentioned_ticker,
                    entity_type=EntityType.COMPETITOR,
                    ticker=ticker,
                    fiscal_period=fiscal_period,
                    chunk_ids=[chunk_id],
                )
            )
            relationships.append(
                Relationship(
                    source=ticker.lower(),
                    target=mentioned_ticker.lower(),
                    relation=RelationType.COMPETES_WITH,
                    ticker=ticker,
                    fiscal_period=fiscal_period,
                    chunk_id=chunk_id,
                )
            )

    return entities, relationships


async def extract_entities_from_chunks(
    parent_chunks: list,
    ticker: str,
    fiscal_period: str,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Extract entities and relationships from a list of parent chunks asynchronously.

    Combines concurrent LLM extraction (if enabled) with regex-based fallback.

    Args:
        parent_chunks: List of Chunk objects (parent type only)
        ticker: Company ticker symbol
        fiscal_period: e.g., "Q4 2024"

    Returns:
        Tuple of (entities, relationships) ready for knowledge graph insertion.
    """
    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []
    llm_enabled = settings.knowledge_graph.extraction_enabled

    # ── LLM extraction (Concurrent) ─────────────────────────────────────────────
    if llm_enabled:
        tasks = []
        for chunk in parent_chunks:
            tasks.append(_call_llm_extract(chunk.text, ticker, fiscal_period))

        raw_responses = await asyncio.gather(*tasks) if tasks else []

        for chunk, raw in zip(parent_chunks, raw_responses, strict=False):
            chunk_id = chunk.chunk_id
            for e_data in raw.get("entities", []):
                try:
                    entity = Entity(
                        name=e_data.get("name", ""),
                        entity_type=e_data.get("entity_type", EntityType.METRIC),
                        ticker=ticker,
                        fiscal_period=fiscal_period,
                        chunk_ids=[chunk_id],
                        properties=e_data.get("properties", {}),
                    )
                    if entity.name:
                        all_entities.append(entity)
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
                        all_relationships.append(rel)
                except Exception as exc:
                    logger.debug(f"Skipping malformed relationship: {exc}")

    # ── Regex fallback (always runs sequentially, CPU bounds) ───────────────────
    for chunk in parent_chunks:
        regex_entities, regex_rels = _regex_extract(
            chunk.text, ticker, fiscal_period, chunk.chunk_id
        )
        all_entities.extend(regex_entities)
        all_relationships.extend(regex_rels)

    logger.info(
        f"[KG Extract] {ticker} {fiscal_period} | "
        f"{len(all_entities)} entities, {len(all_relationships)} relationships "
        f"from {len(parent_chunks)} parent chunks "
        f"(llm={'on' if llm_enabled else 'off'})"
    )
    return all_entities, all_relationships
