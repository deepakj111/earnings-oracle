"""
Production-grade 2026 chunking pipeline for financial documents.

Architecture:
  Stage 1 — Structure-aware splitting:  financial section headers as hard boundaries,
                                         Markdown table BLOCKS detected and kept atomic
  Stage 2 — Parent chunks with overlap: ~512 tokens, parent-level overlap preserved
  Stage 3 — Sentence-aware child chunks: ~128 tokens, sentence boundaries respected,
                                          contextual prefix re-applied to every child
"""

import re
import uuid
from dataclasses import dataclass, field

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def _token_count(text: str) -> int:
        return len(_ENC.encode(text, disallowed_special=()))
except ImportError:
    raise ImportError("tiktoken is required: poetry add tiktoken") from None


PARENT_TOKEN_TARGET = 512
CHILD_TOKEN_TARGET = 128
PARENT_OVERLAP_TOKENS = 64
CHILD_OVERLAP_TOKENS = 32
TABLE_LINE_THRESHOLD = 2

# Real SEC section headers are short: "Revenue", "## Segment Results", "Outlook".
# Prose sentences that happen to start with "Revenue grew 6 percent year over year..."
# must NOT be classified as headers — they have far more than 8 words.
HEADER_MAX_WORDS = 8

SECTION_HEADERS_RE = re.compile(
    "|".join(
        [
            r"^(financial highlights?|financial results?)",
            r"^(revenue|net (revenue|sales))",
            r"^(earnings per share|eps|diluted)",
            r"^(segment (results?|performance|revenue))",
            r"^(products?|services?|iphone|mac|ipad|wearables)",
            r"^(outlook|guidance|forward.looking)",
            r"^(balance sheet|cash flow|liquidity)",
            r"^(operating (income|expenses?|margin))",
            r"^(gross (margin|profit))",
            r"^(quarterly|annual|fiscal (year|quarter))",
            r"^(management|ceo|cfo).{0,30}(comment|remark|statement)",
            r"^(about|conference call|webcast|investor)",
            r"^(cautionary|forward.looking|safe harbor)",
            r"^#{1,4}\s+\S",
        ]
    ),
    re.IGNORECASE | re.MULTILINE,
)

_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str | None
    ticker: str
    date: str
    doc_type: str
    chunk_type: str
    text: str
    section_title: str = ""
    metadata: dict = field(default_factory=dict)


def _is_table_block(lines: list) -> bool:
    """
    Returns True if the line block looks like a Markdown table.
      - Ignores blank lines entirely
      - At least TABLE_LINE_THRESHOLD non-blank lines must be pipe rows
      - Pipe rows must make up >= 40% of non-blank lines
        (prevents a stray pipe in prose from triggering a false positive)
    """
    non_blank = [line for line in lines if line.strip()]
    if not non_blank:
        return False
    pipe_lines = [line for line in non_blank if _TABLE_ROW_RE.match(line.strip())]
    if len(pipe_lines) < TABLE_LINE_THRESHOLD:
        return False
    return (len(pipe_lines) / len(non_blank)) >= 0.40


def _split_text_by_token_budget(text: str, budget: int) -> list[str]:
    tokens = _ENC.encode(text, disallowed_special=())
    pages = []
    for start in range(0, len(tokens), budget):
        chunk_tokens = tokens[start : start + budget]
        pages.append(_ENC.decode(chunk_tokens))
    return pages


_ABBREV_RE = re.compile(
    r"\b(U\.S|No|vs|approx|est|Corp|Inc|Ltd|Dr|Mr|Mrs|e\.g|i\.e|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.$",
    re.IGNORECASE,
)


def _split_into_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(text)
    merged, buf = [], ""
    for part in parts:
        candidate = (buf + " " + part).strip() if buf else part
        if buf and _ABBREV_RE.search(buf.rstrip()):
            buf = candidate
        else:
            if buf:
                merged.append(buf)
            buf = part
    if buf:
        merged.append(buf)
    return [s.strip() for s in merged if s.strip()]


def _make_chunk_id(ticker: str, date: str, index: int, suffix: str = "") -> str:
    ns = uuid.uuid5(uuid.NAMESPACE_DNS, f"{ticker}:{date}")
    base = f"{ticker}_{date}_{str(ns)[:8]}_{index}"
    return f"{base}_{suffix}" if suffix else base


def _contextual_prefix(ticker: str, date: str, doc_type: str, section_title: str) -> str:
    prefix = f"[Context: {ticker} | {doc_type} | {date}"
    if section_title and section_title not in ("__TABLE__", "Financial Table"):
        prefix += f" | Section: {section_title}"
    return prefix + "]\n\n"


def _split_into_semantic_sections(sections: list[str]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    current_title: str = ""
    current_lines: list[str] = []
    table_buffer: list[str] = []

    def _is_section_header(line: str) -> bool:
        """
        A line is a section header only if:
          1. It matches the financial/markdown header patterns, AND
          2. It is short (<= HEADER_MAX_WORDS words)
        This prevents "Revenue grew 6 percent year over year..." from
        being mistakenly classified as a header.
        """
        return (
            bool(line)
            and len(line.split()) <= HEADER_MAX_WORDS
            and bool(SECTION_HEADERS_RE.match(line))
        )

    def flush_text(title: str, lines: list[str]) -> None:
        block = "\n".join(lines).strip()
        if _token_count(block) > 10:
            result.append((title, block))

    def flush_table() -> None:
        if len(table_buffer) >= TABLE_LINE_THRESHOLD:
            result.append(("__TABLE__", "\n".join(table_buffer)))
        elif table_buffer:
            current_lines.extend(table_buffer)
        table_buffer.clear()

    for section in sections:
        for line in section.split("\n"):
            stripped = line.strip()

            if stripped and _TABLE_ROW_RE.match(stripped):
                if not table_buffer:
                    flush_text(current_title, current_lines)
                    current_lines = []
                table_buffer.append(stripped)
                continue

            if table_buffer:
                flush_table()

            if _is_section_header(stripped):
                flush_text(current_title, current_lines)
                current_title = stripped.lstrip("#").strip()
                current_lines = []  # header is captured in current_title; don't duplicate in body
            else:
                current_lines.append(line)

    if table_buffer:
        flush_table()
    flush_text(current_title, current_lines)

    return [(t, s) for t, s in result if len(s.split()) >= 10]


def _build_parents(
    sections: list[tuple[str, str]],
    ticker: str,
    date: str,
    doc_type: str,
) -> list[Chunk]:
    parents: list[Chunk] = []
    current_texts: list[str] = []
    current_tokens: int = 0
    current_title: str = ""
    overlap_text: str = ""

    def emit_parent(texts: list[str], title: str, has_overlap: bool) -> Chunk:
        body = "\n\n".join(texts)
        prefix = _contextual_prefix(ticker, date, doc_type, title)
        full = (prefix + body).strip()
        pid = _make_chunk_id(ticker, date, len(parents))
        return Chunk(
            chunk_id=pid,
            parent_id=None,
            ticker=ticker,
            date=date,
            doc_type=doc_type,
            chunk_type="parent",
            text=full,
            section_title=title,
            metadata={
                "ticker": ticker,
                "date": date,
                "doc_type": doc_type,
                "chunk_index": len(parents),
                "section": title,
                "has_overlap": has_overlap,
                "is_table": False,
            },
        )

    def _flush_and_reset() -> str:
        """Emit the current accumulation as a parent and return the new overlap tail."""
        nonlocal overlap_text
        parents.append(emit_parent(current_texts, current_title, bool(overlap_text)))
        body = "\n\n".join(current_texts)
        tail = " ".join(body.split()[-PARENT_OVERLAP_TOKENS:])
        return tail + "\n\n" if tail else ""

    def _accumulate(text: str, title: str) -> None:
        """Add a text block (≤ PARENT_TOKEN_TARGET) to the current accumulation."""
        nonlocal current_texts, current_tokens, current_title, overlap_text

        tokens = _token_count(text)

        if current_tokens + tokens > PARENT_TOKEN_TARGET and current_texts:
            overlap_text = _flush_and_reset()
            current_texts = []
            current_tokens = 0

        if not current_texts and overlap_text:
            current_texts = [overlap_text.strip()]
            current_tokens = _token_count(overlap_text)

        if title and title != current_title:
            current_title = title

        current_texts.append(text)
        current_tokens += tokens

    for section_title, section_text in sections:
        if section_title == "__TABLE__":
            if current_texts:
                overlap_text = _flush_and_reset()
                current_texts = []
                current_tokens = 0

            prefix = _contextual_prefix(ticker, date, doc_type, "Financial Table")
            tid = _make_chunk_id(ticker, date, len(parents), "tbl")
            parents.append(
                Chunk(
                    chunk_id=tid,
                    parent_id=None,
                    ticker=ticker,
                    date=date,
                    doc_type=doc_type,
                    chunk_type="table",
                    text=(prefix + section_text).strip(),
                    section_title="Financial Table",
                    metadata={
                        "ticker": ticker,
                        "date": date,
                        "doc_type": doc_type,
                        "chunk_index": len(parents),
                        "section": "Financial Table",
                        "has_overlap": False,
                        "is_table": True,
                    },
                )
            )
            overlap_text = ""
            continue

        section_tokens = _token_count(section_text)

        if section_tokens > PARENT_TOKEN_TARGET:
            # Oversized section: split into word-budget pages first, then
            # accumulate each page normally so overlap logic still applies.
            for page in _split_text_by_token_budget(section_text, PARENT_TOKEN_TARGET):
                _accumulate(page, section_title)
        else:
            _accumulate(section_text, section_title)

    if current_texts:
        parents.append(emit_parent(current_texts, current_title, bool(overlap_text)))

    return parents


def _split_parent_into_children(parent: Chunk) -> list[Chunk]:
    if parent.chunk_type == "table":
        return [
            Chunk(
                chunk_id=f"{parent.chunk_id}_c0",
                parent_id=parent.chunk_id,
                ticker=parent.ticker,
                date=parent.date,
                doc_type=parent.doc_type,
                chunk_type="child",
                text=parent.text,
                section_title=parent.section_title,
                metadata={
                    **parent.metadata,
                    "parent_id": parent.chunk_id,
                    "child_index": 0,
                    "is_table": True,
                },
            )
        ]

    sentinel = "]\n\n"
    body_text = parent.text.split(sentinel, 1)[-1] if sentinel in parent.text else parent.text

    sentences: list[str] = _split_into_sentences(body_text)
    children: list[Chunk] = []
    current_sents: list[str] = []
    current_tokens: int = 0
    child_index: int = 0

    def emit_child(sents: list[str]) -> Chunk:
        nonlocal child_index
        child_text = " ".join(sents)
        prefix = _contextual_prefix(
            parent.ticker, parent.date, parent.doc_type, parent.section_title
        )
        c = Chunk(
            chunk_id=f"{parent.chunk_id}_c{child_index}",
            parent_id=parent.chunk_id,
            ticker=parent.ticker,
            date=parent.date,
            doc_type=parent.doc_type,
            chunk_type="child",
            text=(prefix + child_text).strip(),
            section_title=parent.section_title,
            metadata={
                **parent.metadata,
                "parent_id": parent.chunk_id,
                "child_index": child_index,
                "is_table": False,
            },
        )
        child_index += 1
        return c

    for sentence in sentences:
        sent_tokens = _token_count(sentence)

        if current_tokens + sent_tokens > CHILD_TOKEN_TARGET and current_sents:
            children.append(emit_child(current_sents))

            overlap_sents: list[str] = []
            overlap_tokens: int = 0
            for s in reversed(current_sents):
                t = _token_count(s)
                if overlap_tokens + t <= CHILD_OVERLAP_TOKENS:
                    overlap_sents.insert(0, s)
                    overlap_tokens += t
                else:
                    break

            current_sents = overlap_sents
            current_tokens = overlap_tokens

        current_sents.append(sentence)
        current_tokens += sent_tokens

    if current_sents:
        children.append(emit_child(current_sents))

    return children


def create_parent_child_chunks(
    ticker: str,
    date: str,
    sections: list[str],
    doc_type: str = "earnings_release",
) -> list[Chunk]:
    """Create structured parent and sentence-aware child chunks from document sections."""
    semantic_sections = _split_into_semantic_sections(sections)
    if not semantic_sections:
        return []

    parents = _build_parents(semantic_sections, ticker, date, doc_type)
    all_chunks = []
    for parent in parents:
        all_chunks.append(parent)
        all_chunks.extend(_split_parent_into_children(parent))

    return all_chunks
