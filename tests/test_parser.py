"""
Tests for ingestion/parser.py

Strategy: build minimal real HTML strings in fixtures.
No file I/O in individual tests — only the integration test touches disk.
Coverage:
  - ticker + date extracted from filename stem
  - script/style/nav stripped from output text
  - sections split on double newline, filtered by word floor
  - files under 100 words return None
  - ParsedDocument fields populated correctly
"""

from pathlib import Path

import pytest

from ingestion.parser import ParsedDocument, parse_html

# Minimal HTML that produces enough words to pass the 100-word guard
_BODY = " ".join(["word"] * 120)

MINIMAL_HTML = f"""
<html>
<head>
  <script>var x = 1;</script>
  <style>body {{ color: red; }}</style>
</head>
<body>
  <nav>Skip navigation</nav>
  <main>
    <p>{_BODY}</p>
  </main>
  <footer>Footer content</footer>
</body>
</html>
"""

MULTI_SECTION_HTML = """
<html><body>
<p>Revenue grew six percent year over year driven by strong iPhone demand
across all geographies and all major product categories this fiscal quarter.
Total net revenue for the period reached ninety four point nine billion dollars,
ahead of analyst consensus estimates of ninety three point one billion dollars.</p>

<p>Services reached a new all time high of twenty four point nine billion dollars
representing fourteen percent growth versus the prior year comparable period.
The installed base of active devices reached a new all time high across all
geographic segments and all major product categories worldwide this quarter.</p>

<p>The company returned twenty nine billion dollars to shareholders through share
buybacks and dividends during this fiscal quarter of operations and remains on
track to achieve its stated capital return targets for the full fiscal year ahead.</p>
</body></html>
"""

SHORT_HTML = "<html><body><p>Too short.</p></body></html>"


@pytest.fixture
def tmp_htm(tmp_path):
    """Returns a helper that writes HTML to a temp .htm file with a given stem."""

    def _write(stem: str, content: str) -> Path:
        p = tmp_path / f"{stem}.htm"
        p.write_text(content, encoding="utf-8")
        return p

    return _write


class TestParseHtml:
    def test_returns_parsed_document(self, tmp_htm) -> None:
        path = tmp_htm("AAPL_2024-10-31_0001234567", MINIMAL_HTML)
        result = parse_html(path)
        assert isinstance(result, ParsedDocument)

    def test_ticker_extracted_from_filename(self, tmp_htm) -> None:
        path = tmp_htm("NVDA_2024-07-15_0001234567", MINIMAL_HTML)
        result = parse_html(path)
        assert result.ticker == "NVDA"

    def test_date_extracted_from_filename(self, tmp_htm) -> None:
        path = tmp_htm("AAPL_2024-10-31_0001234567", MINIMAL_HTML)
        result = parse_html(path)
        assert result.date == "2024-10-31"

    def test_script_tags_stripped(self, tmp_htm) -> None:
        path = tmp_htm("AAPL_2024-10-31_0001234567", MINIMAL_HTML)
        result = parse_html(path)
        assert "var x = 1" not in result.raw_text

    def test_style_tags_stripped(self, tmp_htm) -> None:
        path = tmp_htm("AAPL_2024-10-31_0001234567", MINIMAL_HTML)
        result = parse_html(path)
        assert "color: red" not in result.raw_text

    def test_nav_stripped(self, tmp_htm) -> None:
        path = tmp_htm("AAPL_2024-10-31_0001234567", MINIMAL_HTML)
        result = parse_html(path)
        assert "Skip navigation" not in result.raw_text

    def test_raw_text_non_empty(self, tmp_htm) -> None:
        path = tmp_htm("AAPL_2024-10-31_0001234567", MINIMAL_HTML)
        result = parse_html(path)
        assert len(result.raw_text.strip()) > 0

    def test_sections_is_list(self, tmp_htm) -> None:
        path = tmp_htm("AAPL_2024-10-31_0001234567", MULTI_SECTION_HTML)
        result = parse_html(path)
        assert isinstance(result.sections, list)

    def test_sections_all_meet_word_floor(self, tmp_htm) -> None:
        path = tmp_htm("AAPL_2024-10-31_0001234567", MULTI_SECTION_HTML)
        result = parse_html(path)
        for section in result.sections:
            assert len(section.split()) >= 15, f"Section below 15-word floor: {section[:60]!r}"

    def test_short_file_returns_none(self, tmp_htm) -> None:
        path = tmp_htm("AAPL_2024-10-31_0001234567", SHORT_HTML)
        result = parse_html(path)
        assert result is None

    def test_file_path_stored(self, tmp_htm) -> None:
        path = tmp_htm("AAPL_2024-10-31_0001234567", MINIMAL_HTML)
        result = parse_html(path)
        assert result.file_path == str(path)

    def test_no_triple_newlines_in_raw_text(self, tmp_htm) -> None:
        path = tmp_htm("AAPL_2024-10-31_0001234567", MULTI_SECTION_HTML)
        result = parse_html(path)
        assert "\n\n\n" not in result.raw_text

    def test_unknown_date_when_stem_has_no_date(self, tmp_htm) -> None:
        path = tmp_htm("AAPL", MINIMAL_HTML)
        result = parse_html(path)
        assert result.date == "unknown"

    def test_encoding_errors_do_not_crash(self, tmp_path) -> None:
        path = tmp_path / "AAPL_2024-10-31_0001234567.htm"
        # Write with latin-1 encoding — parser uses errors='ignore'
        path.write_bytes((MINIMAL_HTML + "\nCaf\xe9 revenue grew strongly.").encode("latin-1"))
        result = parse_html(path)
        assert result is not None
