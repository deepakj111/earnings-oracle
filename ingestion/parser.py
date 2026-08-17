import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import lxml.html
from bs4 import XMLParsedAsHTMLWarning
from lxml import etree

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


@dataclass
class ParsedDocument:
    ticker: str
    date: str
    file_path: str
    raw_text: str
    sections: list[str] = field(default_factory=list)


def _table_to_markdown(table_elem: etree._Element) -> str:
    """Convert an HTML <table> element into a clean, structured Markdown table.

    Preserves 2D row/column alignment, headers, and numeric values so that
    financial statements and schedules remain structurally intact.
    """
    rows: list[list[str]] = []
    for tr in table_elem.xpath(".//tr"):
        cells = tr.xpath("./th | ./td")
        if not cells:
            continue
        row_cells = []
        for c in cells:
            # Strip excessive inner whitespace and newlines
            c_text = " ".join(c.text_content().split()).strip()
            # Escape pipes inside cell content to avoid breaking markdown syntax
            c_text = c_text.replace("|", "\\|")
            row_cells.append(c_text)
        if any(row_cells):
            rows.append(row_cells)

    if not rows or len(rows) < 2:
        return ""

    max_cols = max(len(r) for r in rows)
    if max_cols < 2:
        return ""

    # Normalize row lengths to match max_cols
    norm_rows = [r + [""] * (max_cols - len(r)) for r in rows]

    header = "| " + " | ".join(norm_rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * max_cols) + " |"
    body = ["| " + " | ".join(r) + " |" for r in norm_rows[1:]]

    return "\n\n" + "\n".join([header, separator] + body) + "\n\n"


def parse_html(file_path: Path) -> ParsedDocument | None:
    """Parse pure content and financial sections from an SEC HTML file using fast C-based lxml parser."""
    stem = file_path.stem
    stem_parts = stem.split("_")
    ticker = stem_parts[0]

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", stem)
    if date_match:
        date = date_match.group(0)
    elif len(stem_parts) > 1:
        date = stem_parts[1]
    else:
        date = "unknown"

    raw_bytes = file_path.read_bytes()
    if not raw_bytes or not raw_bytes.strip():
        return None

    try:
        # Fast C-based HTML parsing via lxml (bytes input avoids XML declaration encoding errors)
        parser = lxml.html.HTMLParser(encoding="utf-8", recover=True)
        tree = lxml.html.fromstring(raw_bytes, parser=parser)
        if tree is not None:
            # 1. Strip non-content and layout metadata tags
            etree.strip_elements(
                tree,
                "script",
                "style",
                "nav",
                "header",
                "footer",
                "meta",
                "link",
                "noscript",
                with_tail=False,
            )

            # 2. Purge Inline XBRL (iXBRL) header/metadata nodes that contain raw schema URLs
            for elem in tree.xpath('//*[starts-with(name(), "ix:")]'):
                tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if tag_local.lower() in (
                    "ix:header",
                    "ix:hidden",
                    "ix:references",
                    "ix:relationship",
                    "ix:resources",
                    "header",
                    "hidden",
                    "references",
                    "relationship",
                    "resources",
                ):
                    parent = elem.getparent()
                    if parent is not None:
                        parent.remove(elem)

            # 3. Convert HTML tables into structured Markdown tables
            for table in list(tree.xpath("//table")):
                md_table = _table_to_markdown(table)
                parent = table.getparent()
                if md_table and parent is not None:
                    table_div = etree.Element("div")
                    table_div.text = md_table
                    table.addprevious(table_div)
                    parent.remove(table)

            text = tree.text_content()
        else:
            text = ""
    except Exception:
        # Fallback to BeautifulSoup if lxml fails on edge cases
        from bs4 import BeautifulSoup

        html = raw_bytes.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer", "meta", "link", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)

    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Skip pure cover pages / empty files
    if len(text.split()) < 100:
        return None

    sections = [s.strip() for s in text.split("\n\n") if len(s.strip().split()) >= 15]

    if not sections:
        return None

    return ParsedDocument(
        ticker=ticker,
        date=date,
        file_path=str(file_path),
        raw_text=text,
        sections=sections,
    )
