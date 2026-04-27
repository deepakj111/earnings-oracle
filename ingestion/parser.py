import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


@dataclass
class ParsedDocument:
    ticker: str
    date: str
    file_path: str
    raw_text: str
    sections: list[str] = field(default_factory=list)


def parse_html(file_path: Path) -> ParsedDocument | None:
    """Parse pure content and financial sections from an SEC HTML file."""
    stem_parts = file_path.stem.split("_")
    ticker = stem_parts[0]
    date = stem_parts[1] if len(stem_parts) > 1 else "unknown"

    with open(file_path, encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "meta", "link"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
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
