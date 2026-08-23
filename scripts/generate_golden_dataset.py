# scripts/generate_golden_dataset.py
"""
Automatic Production-Grade Golden QA Dataset Generator.

Parses SEC 10-K and 10-Q HTML filings in `data/company_filings/` and uses OpenAI
structured completions (default `gpt-5-mini`) to extract balanced, high-value
qualitative and quantitative Question & Answer pairs with authentic ground-truth answers.

Features:
  - 4-Pillar Balance: Strategy/MD&A (~35%), Risks/Regulatory (~25%), Segment/Financials (~30%), Capital Allocation (~10%)
  - Substantive Text Extraction: Prioritizes MD&A, Segment details, Risk Factors, and Footnotes while skipping cover boilerplate
  - 100% Self-Containment: Automatically anchors company names and precise fiscal periods in questions
  - Multi-Worker Parallelism: ThreadPoolExecutor for fast concurrent processing across filings
  - Quota Balancing: Proportional sampling across portfolio tickers (NFLX, UNH, NVDA, WMT)

Usage:
    # Full dataset generation across all filings
    poetry run python -m scripts.generate_golden_dataset

    # Dry-run or test on 2 files with custom model
    poetry run python -m scripts.generate_golden_dataset --max-files 2 --out data/test_dataset.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from loguru import logger

from config.companies import CompanyRegistry
from config.openai_client import get_openai_client
from ingestion.parser import parse_html

# ── System Prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a Principal Financial Analyst and MLOps Evaluation Architect constructing a golden evaluation benchmark dataset for a RAG system indexing SEC Form 10-K and 10-Q filings.

Your task is to analyze the provided SEC filing text and generate distinct, high-quality Question & Answer pairs adhering to the following 4 pillars:

1. **Segment & Core Financials**: Specific segment revenue, operating margins, regional ARPU/ARM, growth metrics, or balance sheet figures.
2. **MD&A & Strategic Initiatives**: Operational strategies, product roadmaps (e.g. Blackwell/liquid-cooling for NVDA, supply chain automation/Walmart Connect for WMT, ad-tier/paid sharing/live content for NFLX, Optum Care/value-based healthcare for UNH).
3. **Risk Factors & Regulatory / Operational Disclosures**: Key risks (e.g. US export controls/BIS licensing, cyber incident impact/Change Healthcare for UNH, retail shrink/tariffs, content amortization/licensing).
4. **Capital Allocation & Cash Flow**: Share repurchases, free cash flow generation, capex priorities, or debt maturity.
5. **Operational Dynamics or Multi-Period Trends**: Cause-and-effect explanations of how headwinds/tailwinds affected performance.

RULES:
- AVOID generic trivia (do NOT ask for website URLs, social media handles, CIK numbers, or verbatim coupon rates of standard senior notes).
- AVOID repeating the same foreign exchange hedge accounting formula across multiple quarters.
- EVERY ground-truth answer must be 100% grounded in the provided text excerpt, factual, specific, and self-contained with exact numbers/percentages/facts where applicable.
- Make questions sound natural, clear, and unambiguous, as asked by equity research analysts.
"""


def extract_meaningful_text(sections: list[str], max_chars: int = 35000) -> str:
    """Combine informative sections while skipping cover page boilerplate."""
    candidate_sections = sections[5:] if len(sections) > 10 else sections

    priority_keywords = [
        "management's discussion",
        "item 7",
        "item 2",
        "item 1",
        "revenue",
        "segment",
        "operating results",
        "risk factors",
        "liquidity",
        "note",
        "balance sheet",
        "cash flow",
        "outlook",
        "guidance",
        "optum",
        "blackwell",
        "walmart connect",
        "subscribers",
        "membership",
        "medical care ratio",
    ]

    scored_sections: list[tuple[int, str]] = []
    for s in candidate_sections:
        s_lower = s.lower()
        score = sum(1 for kw in priority_keywords if kw in s_lower)
        if len(s.strip()) > 200:
            scored_sections.append((score, s))

    scored_sections.sort(key=lambda x: x[0], reverse=True)

    meaningful: list[str] = []
    total_len = 0
    for _, s in scored_sections:
        if total_len + len(s) > max_chars:
            remaining = max_chars - total_len
            if remaining > 500:
                meaningful.append(s[:remaining])
            break
        meaningful.append(s)
        total_len += len(s)

    return "\n\n---\n\n".join(meaningful) if meaningful else "\n\n".join(sections[:12])[:max_chars]


def polish_qa_sample(
    question: str,
    ground_truth: str,
    ticker: str,
    year: int | None,
    quarter: str | None,
) -> tuple[str, str]:
    """Ensure explicit company naming, precise fiscal period context, and clean answers."""
    comp_prof = CompanyRegistry.get_company(ticker)
    comp_name = comp_prof.name if comp_prof else ticker
    period_str = f"{quarter} {year}" if quarter else f"FY{year}"

    q_clean = question.strip()

    replacements = [
        (r"\bthis Form 10‑K\b", f"{comp_name}'s {year} Form 10-K"),
        (r"\bthis Form 10-K\b", f"{comp_name}'s {year} Form 10-K"),
        (r"\bthis 10-K\b", f"{comp_name}'s {year} Form 10-K"),
        (r"\bthis Form 10‑Q\b", f"{comp_name}'s {period_str} Form 10-Q"),
        (r"\bthis Form 10-Q\b", f"{comp_name}'s {period_str} Form 10-Q"),
        (r"\bthis 10-Q\b", f"{comp_name}'s {period_str} Form 10-Q"),
        (r"\bin the filing\b", f"in {comp_name}'s {period_str} SEC filing"),
        (r"\bas reported in this filing\b", f"as reported in {comp_name}'s {period_str} filing"),
        (r"\bas of the filing\b", f"as of {comp_name}'s {period_str} filing"),
        (r"\bAccording to the filing\b", f"According to {comp_name}'s {period_str} filing"),
        (r"\bthe filing states\b", f"{comp_name}'s {period_str} filing states"),
        (r"\bthe Company’s\b", f"{comp_name}'s"),
        (r"\bthe Company\b", f"{comp_name}"),
        (r"\bthe company\b", f"{comp_name}"),
    ]

    for pat, repl in replacements:
        q_clean = re.sub(pat, repl, q_clean, flags=re.IGNORECASE)

    targets = [ticker.lower(), comp_name.lower()]
    if ticker == "UNH":
        targets.append("optum")
    if ticker == "WMT":
        targets.append("walmart")

    has_company = any(t in q_clean.lower() for t in targets)
    if not has_company:
        if q_clean.lower().startswith("for sam's club"):
            q_clean = "For Walmart's Sam's Club segment," + q_clean[len("for sam's club") :]
        elif q_clean.lower().startswith("how much of sam's club"):
            q_clean = (
                "How much of Walmart's Sam's Club segment"
                + q_clean[len("how much of sam's club") :]
            )
        elif q_clean.lower().startswith("how much did sam's club"):
            q_clean = (
                "How much did Walmart's Sam's Club segment"
                + q_clean[len("how much did sam's club") :]
            )
        elif q_clean.lower().startswith("by how much did"):
            q_clean = f"For {comp_name}, " + q_clean[0].lower() + q_clean[1:]
        elif q_clean.lower().startswith("what drove the"):
            q_clean = f"What drove {comp_name}'s" + q_clean[len("What drove the") :]
        elif q_clean.lower().startswith("for the six months"):
            q_clean = f"For {comp_name}'s six months" + q_clean[len("For the six months") :]
        else:
            q_clean = f"For {comp_name} ({ticker}), {q_clean[0].lower() + q_clean[1:]}"

    has_period = (
        (str(year) in q_clean)
        or (quarter is not None and quarter in q_clean)
        or (f"fiscal {year}" in q_clean.lower())
        or (f"fy{year}" in q_clean.lower())
        or (f"fy {year}" in q_clean.lower())
        or ("fiscal 2026" in q_clean.lower())
        or ("fiscal 2025" in q_clean.lower())
    )
    if not has_period and year:
        q_clean = f"{q_clean.rstrip('?')} in {period_str}?"

    q_clean = re.sub(r"\s+", " ", q_clean).strip()

    gt_clean = ground_truth.strip()
    openers = [
        ("The filing states that ", f"In {comp_name}'s filing, "),
        ("The filing states ", f"In {comp_name}'s filing, "),
        ("The filing notes ", f"In {comp_name}'s filing, "),
        ("The filing lists ", f"In {comp_name}'s filing, "),
        ("The filing describes ", f"In {comp_name}'s filing, "),
        ("It discloses that ", f"{comp_name} discloses that "),
        ("It states ", f"{comp_name} states "),
    ]
    for old_op, new_op in openers:
        if gt_clean.startswith(old_op):
            gt_clean = new_op + gt_clean[len(old_op) :]
            break

    gt_clean = re.sub(r"\s+", " ", gt_clean).strip()
    return q_clean, gt_clean


def generate_qa_for_doc(
    file_path: Path,
    target_count: int = 5,
    model: str = "gpt-5-mini",
) -> list[dict[str, Any]]:
    """Parse one HTML filing and generate structured QA pairs using OpenAI."""
    parsed = parse_html(file_path)
    if not parsed or not parsed.sections:
        logger.warning(f"Skipping empty or unparseable filing: {file_path.name}")
        return []

    stem = file_path.stem
    parts = stem.split("_")
    ticker = parts[0].upper()
    doc_type = parts[1].upper() if len(parts) > 1 else ""
    date_str = parts[2] if len(parts) > 2 else ""

    year = int(date_str[:4]) if date_str and len(date_str) >= 4 and date_str[:4].isdigit() else None

    quarter = None
    if "10-Q" in doc_type and date_str and len(date_str) >= 7:
        month = int(date_str[5:7])
        if month in (3, 4, 5):
            quarter = "Q1"
        elif month in (6, 7, 8):
            quarter = "Q2"
        elif month in (9, 10, 11):
            quarter = "Q3"
        else:
            quarter = "Q4"

    comp_prof = CompanyRegistry.get_company(ticker)
    company_name = comp_prof.name if comp_prof else ticker

    context_text = extract_meaningful_text(parsed.sections)
    if len(context_text) < 1000:
        context_text = "\n\n".join(parsed.sections[:15])[:30000]

    user_prompt = f"""Filing Information:
Company: {company_name} ({ticker})
Filing Type: {doc_type}
Date: {date_str}
Fiscal Year: {year}
Quarter: {quarter or "Full Year (Annual)"}

Filing Content Excerpt:
{context_text}

Generate exactly {target_count} diverse, high-impact QA pairs adhering to the 4 pillars.
Return valid JSON formatted as:
{{
  "samples": [
    {{
      "sample_id": "{ticker.lower()}_{year}_{quarter.lower() if quarter else "annual"}_<short_topic_slug>",
      "question": "<Clear, natural language question>",
      "ground_truth": "<Factual, complete answer with exact numbers/facts directly from text>",
      "ticker": "{ticker}",
      "year": {year},
      "quarter": {"null" if quarter is None else f'"{quarter}"'}
    }}
  ]
}}
"""

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

        results: list[dict[str, Any]] = []
        for s in raw_samples:
            if not isinstance(s, dict):
                continue
            sid = s.get("sample_id") or f"{ticker.lower()}_{year or 'doc'}_sample"
            q = s.get("question", "").strip()
            gt = s.get("ground_truth", "").strip()
            if not q or not gt or len(q) < 15 or len(gt) < 20:
                continue

            q_polished, gt_polished = polish_qa_sample(
                question=q,
                ground_truth=gt,
                ticker=ticker,
                year=s.get("year", year),
                quarter=s.get("quarter", quarter),
            )

            results.append(
                {
                    "sample_id": sid,
                    "question": q_polished,
                    "ground_truth": gt_polished,
                    "ticker": ticker,
                    "year": s.get("year", year),
                    "quarter": s.get("quarter", quarter),
                }
            )

        logger.info(f"Generated {len(results)} valid QA pairs from {file_path.name}")
        return results[:target_count]

    except Exception as exc:
        logger.error(f"OpenAI QA generation failed for {file_path.name} with model={model}: {exc}")
        return []


def generate_golden_dataset(
    input_dir: Path = Path("data/company_filings"),
    output_file: Path = Path("data/golden_dataset.json"),
    max_files: int = 0,
    model: str = "gpt-5-mini",
    workers: int = 6,
) -> list[dict[str, Any]]:
    """Process files in input_dir and write golden dataset to output_file."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    all_files = sorted(list(input_dir.glob("*.htm")) + list(input_dir.glob("*.html")))
    files_found = len(all_files)
    if not files_found:
        logger.warning(f"No HTML files found in {input_dir}")
        return []

    logger.info(
        f"Found {files_found} filing files. Generating dataset using model={model} (workers={workers})..."
    )

    by_ticker: dict[str, list[Path]] = {}
    for f in all_files:
        ticker = f.stem.split("_")[0].upper()
        by_ticker.setdefault(ticker, []).append(f)

    quotas = {
        "NFLX": 35,
        "UNH": 35,
        "NVDA": 30,
        "WMT": 30,
    }

    if max_files > 0:
        # Scale quotas down proportionally if max_files is set
        total_files = min(max_files, files_found)
        scale = total_files / files_found
        quotas = {t: max(1, int(q * scale)) for t, q in quotas.items()}

    tasks: list[tuple[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for ticker, target_total in quotas.items():
            files = by_ticker.get(ticker, [])
            if not files:
                continue
            if max_files > 0:
                files = files[: max(1, int(len(files) * (max_files / files_found)))]

            per_file = target_total // len(files)
            remainder = target_total % len(files)

            logger.info(f"Submitting {ticker}: {len(files)} files, target={target_total}...")
            for idx, file_path in enumerate(files):
                count = per_file + (1 if idx < remainder else 0)
                future = executor.submit(
                    generate_qa_for_doc,
                    file_path=file_path,
                    target_count=count,
                    model=model,
                )
                tasks.append((ticker, future))

        ticker_results: dict[str, list[dict[str, Any]]] = {t: [] for t in quotas}
        for ticker, future in tasks:
            try:
                res = future.result()
                ticker_results[ticker].extend(res)
            except Exception as e:
                logger.error(f"Task failed for {ticker}: {e}")

    all_refined_samples: list[dict[str, Any]] = []
    for ticker, target_total in quotas.items():
        ticker_samples = ticker_results.get(ticker, [])
        logger.info(
            f"Collected {len(ticker_samples)} samples for {ticker} (Target: {target_total})"
        )
        all_refined_samples.extend(ticker_samples[:target_total])

    seen_ids: set[str] = set()
    for s in all_refined_samples:
        sid = s["sample_id"]
        if sid in seen_ids:
            s["sample_id"] = f"{sid}_{int(time.time() * 1000) % 10000}"
        seen_ids.add(s["sample_id"])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(all_refined_samples, indent=2), encoding="utf-8")
    logger.success(
        f"Golden dataset written to {output_file} ({len(all_refined_samples)} total samples)."
    )
    return all_refined_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate production golden QA dataset from SEC HTML filings."
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
        default="gpt-5-mini",
        help="OpenAI model for dataset generation (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of concurrent worker threads (default: 6)",
    )

    args = parser.parse_args()
    generate_golden_dataset(
        input_dir=args.dir,
        output_file=args.out,
        max_files=args.max_files,
        model=args.model,
        workers=args.workers,
    )
