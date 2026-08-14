# scripts/generate_golden_dataset.py
"""
Automatic Golden QA Dataset Generator using OpenAI Structured Outputs (`gpt-5`).

Parses SEC 10-K and 10-Q HTML filings in `data/company_filings/` and uses `gpt-5`
to extract diverse qualitative and quantitative Question & Answer pairs with ground-truth answers.

Usage:
    poetry run python -m scripts.generate_golden_dataset --max-files 4
    poetry run python -m scripts.generate_golden_dataset --max-files 0  # Process all filings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from config import settings
from config.openai_client import get_openai_client
from ingestion.parser import parse_html

# ── Pydantic Output Schemas ───────────────────────────────────────────────────


class GeneratedSample(BaseModel):
    sample_id: str = Field(
        description="Unique snake_case identifier, e.g. 'NVDA_2025_datacenter_revenue' or 'WMT_2026_mda_strategy'"
    )
    question: str = Field(
        description="Clear, natural language financial question (quantitative or qualitative) about the filing"
    )
    ground_truth: str = Field(
        description="Factual, complete ground-truth answer stating exact monetary values, units, or strategic details from text"
    )
    ticker: str = Field(description="Company ticker, e.g. 'NVDA', 'WMT', 'UNH', 'NFLX'")
    year: int | None = Field(default=None, description="Fiscal year integer e.g. 2025")
    quarter: str | None = Field(
        default=None, description="Fiscal quarter ('Q1', 'Q2', 'Q3', 'Q4') or None for annual"
    )


class GenerationBatch(BaseModel):
    samples: list[GeneratedSample]


# ── System Prompt ─────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """You are a senior financial analyst constructing a golden evaluation benchmark dataset for a RAG system indexing SEC 10-K and 10-Q filings.
Your task is to analyze the provided SEC filing text and generate 3 to 5 diverse, high-quality Question & Answer pairs.

Dataset Requirements:
1. **Diverse Balance**: Generate BOTH quantitative questions (exact revenue, net income, operating margins, segment breakdown, EPS, free cash flow) AND qualitative questions (management MD&A observations, strategic initiatives, key risk factors, market trends).
2. **Grounding & Precision**: Answers must be 100% factual and directly grounded in the provided filing text. Include exact dollar amounts (e.g., "$30.8 billion"), percentage changes, and fiscal dates where applicable.
3. **Natural & Unambiguous Questions**: Questions should sound like real questions asked by portfolio managers or equity research analysts (e.g. "What key drivers contributed to Walmart's international sales growth in Q1 2026?").
4. **Metadata**: Ensure sample_id is descriptive, snake_case, and unique. Fill ticker, year, and quarter accurately.
"""


def generate_qa_for_doc(
    file_path: Path,
    model: str = "gpt-5",
) -> list[dict[str, Any]]:
    """Parse one HTML filing and generate structured QA pairs using OpenAI."""
    parsed = parse_html(file_path)
    if not parsed or not parsed.sections:
        logger.warning(f"Skipping empty or unparseable filing: {file_path.name}")
        return []

    # Extract ticker, doc_type, date from filename stem
    stem = file_path.stem
    parts = stem.split("_")
    ticker = parts[0].upper()
    doc_type = parts[1].upper() if len(parts) > 1 else ""
    date_str = parts[2] if len(parts) > 2 else ""

    year = None
    if date_str and len(date_str) >= 4 and date_str[:4].isdigit():
        year = int(date_str[:4])

    quarter = None
    if "10-Q" in doc_type:
        if date_str and len(date_str) >= 7:
            month = int(date_str[5:7])
            if month in (3, 4, 5):
                quarter = "Q1"
            elif month in (6, 7, 8):
                quarter = "Q2"
            elif month in (9, 10, 11):
                quarter = "Q3"
            else:
                quarter = "Q4"

    # Select financial sections
    selected_text = "\n\n".join(parsed.sections[:12])[:25000]

    user_prompt = (
        f"Filing Metadata: Ticker={ticker}, DocType={doc_type}, Date={date_str}, Year={year}, Quarter={quarter}\n\n"
        f"Filing Content Excerpt:\n{selected_text}\n\n"
        "Respond with a JSON object with a key 'samples' containing an array of 3 to 5 objects with keys:\n"
        "- sample_id (string, snake_case e.g. NVDA_2025_datacenter_revenue)\n"
        "- question (string, clear natural language financial question)\n"
        "- ground_truth (string, factual answer with exact dollar figures or MD&A details)\n"
        "- ticker (string e.g. NVDA)\n"
        "- year (integer e.g. 2025)\n"
        "- quarter (string e.g. Q1, Q2, Q3, Q4, or null)\n"
    )

    client = get_openai_client()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        content = completion.choices[0].message.content or "{}"
        data = json.loads(content)
        raw_samples = data.get("samples") or data.get("qa_pairs") or data.get("data") or []

        results = []
        for s in raw_samples:
            if not isinstance(s, dict):
                continue
            sid = s.get("sample_id") or f"{ticker}_{year or 'doc'}_sample"
            q = s.get("question")
            gt = s.get("ground_truth")
            if not q or not gt:
                continue
            d = {
                "sample_id": sid,
                "question": q,
                "ground_truth": gt,
                "ticker": ticker,
                "year": s.get("year") or year,
                "quarter": s.get("quarter") or quarter,
            }
            results.append(d)
        return results

    except Exception as exc:
        logger.error(f"OpenAI QA generation failed for {file_path.name} with model={model}: {exc}")
        return []


def generate_golden_dataset(
    input_dir: Path = Path("data/company_filings"),
    output_file: Path = Path("data/golden_dataset.json"),
    max_files: int = 0,
    model: str = "gpt-5",
) -> list[dict[str, Any]]:
    """Process files in input_dir and write golden dataset to output_file."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(list(input_dir.glob("*.htm")) + list(input_dir.glob("*.html")))
    if not files:
        logger.warning(f"No HTML files found in {input_dir}")
        return []

    if max_files > 0:
        files = files[:max_files]

    logger.info(
        f"Generating golden dataset from {len(files)} filings using model={model} (parallel workers=5)..."
    )

    all_samples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_file = {
            executor.submit(generate_qa_for_doc, f, model): (idx, f)
            for idx, f in enumerate(files, 1)
        }

        for future in as_completed(future_to_file):
            idx, f = future_to_file[future]
            try:
                qa_pairs = future.result()
                logger.info(
                    f"[{idx}/{len(files)}] Generated {len(qa_pairs)} QA pairs from {f.name}"
                )
                for qa in qa_pairs:
                    sid = qa["sample_id"]
                    if sid in seen_ids:
                        qa["sample_id"] = f"{sid}_{idx}"
                    seen_ids.add(qa["sample_id"])
                    all_samples.append(qa)
            except Exception as exc:
                logger.error(f"[{idx}/{len(files)}] Processing {f.name} failed: {exc}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(all_samples, indent=2), encoding="utf-8")
    logger.info(f"Golden dataset written to {output_file} ({len(all_samples)} total samples).")
    return all_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate golden QA dataset from SEC HTML filings."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/company_filings"),
        help="Input directory containing SEC HTML filings",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/golden_dataset.json"),
        help="Output JSON dataset path",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum files to process (0 = all files)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.generation.model,
        help="OpenAI model for dataset generation (default: gpt-5)",
    )

    args = parser.parse_args()
    generate_golden_dataset(
        input_dir=args.dir,
        output_file=args.out,
        max_files=args.max_files,
        model=args.model,
    )
