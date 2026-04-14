# tests/test_ui_utils.py
"""
Tests for the UI utility functions in ui/utils.py.

All functions under test are pure (no Streamlit imports, no side effects)
so they run as standard pytest unit tests without a Streamlit runtime.

Coverage:
  parse_sse_line            — bytes/str, valid JSON, [DONE], empty, comments
  build_metadata_filter     — all combos of set/unset fields
  format_latency            — sub-second, seconds, boundary
  format_token_count        — below 1K, above 1K
  citation_badge_label      — complete, missing fields
  group_citations_by_source — grouping, ordering, duplicates
  health_status_emoji       — known and unknown statuses
  stream_query              — mocked requests.post, token yields, error events
  fetch_structured          — mocked requests.post, success, HTTP error
  fetch_health              — mocked response, connection error
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

# ── parse_sse_line ─────────────────────────────────────────────────────────────


class TestParseSseLine:
    def test_parses_token_event(self) -> None:
        from ui.utils import parse_sse_line

        result = parse_sse_line('data: {"token": "Apple"}')
        assert result == {"token": "Apple"}

    def test_parses_bytes_input(self) -> None:
        from ui.utils import parse_sse_line

        result = parse_sse_line(b'data: {"token": "hello"}')
        assert result == {"token": "hello"}

    def test_returns_none_for_done_sentinel(self) -> None:
        from ui.utils import parse_sse_line

        assert parse_sse_line("data: [DONE]") is None

    def test_returns_none_for_empty_string(self) -> None:
        from ui.utils import parse_sse_line

        assert parse_sse_line("") is None

    def test_returns_none_for_blank_whitespace(self) -> None:
        from ui.utils import parse_sse_line

        assert parse_sse_line("   \n\t  ") is None

    def test_returns_none_for_comment_line(self) -> None:
        from ui.utils import parse_sse_line

        assert parse_sse_line(": this is a keep-alive comment") is None

    def test_returns_none_for_line_without_data_prefix(self) -> None:
        from ui.utils import parse_sse_line

        assert parse_sse_line('event: message\n{"token": "x"}') is None

    def test_returns_none_for_invalid_json(self) -> None:
        from ui.utils import parse_sse_line

        # Invalid JSON after "data: " prefix should return None gracefully
        assert parse_sse_line("data: {not valid json}") is None

    def test_parses_error_event(self) -> None:
        from ui.utils import parse_sse_line

        result = parse_sse_line('data: {"error": "API timeout"}')
        assert result == {"error": "API timeout"}

    def test_parses_multikey_event(self) -> None:
        from ui.utils import parse_sse_line

        result = parse_sse_line('data: {"token": "x", "index": 3}')
        assert result is not None
        assert result["token"] == "x"
        assert result["index"] == 3

    def test_strips_surrounding_whitespace(self) -> None:
        from ui.utils import parse_sse_line

        result = parse_sse_line('  data: {"token": "hi"}  ')
        assert result == {"token": "hi"}


# ── build_metadata_filter ──────────────────────────────────────────────────────


class TestBuildMetadataFilter:
    def test_all_none_returns_none(self) -> None:
        from ui.utils import build_metadata_filter

        assert build_metadata_filter(None, None, None) is None

    def test_ticker_only(self) -> None:
        from ui.utils import build_metadata_filter

        result = build_metadata_filter("aapl", None, None)
        assert result == {"ticker": "AAPL"}  # uppercased

    def test_year_only(self) -> None:
        from ui.utils import build_metadata_filter

        result = build_metadata_filter(None, 2024, None)
        assert result == {"year": 2024}

    def test_quarter_only(self) -> None:
        from ui.utils import build_metadata_filter

        result = build_metadata_filter(None, None, "q4")
        assert result == {"quarter": "Q4"}  # uppercased

    def test_all_fields_set(self) -> None:
        from ui.utils import build_metadata_filter

        result = build_metadata_filter("NVDA", 2024, "Q3")
        assert result == {"ticker": "NVDA", "year": 2024, "quarter": "Q3"}

    def test_ticker_and_year_without_quarter(self) -> None:
        from ui.utils import build_metadata_filter

        result = build_metadata_filter("MSFT", 2023, None)
        assert result == {"ticker": "MSFT", "year": 2023}
        assert "quarter" not in result

    def test_ticker_uppercased(self) -> None:
        from ui.utils import build_metadata_filter

        result = build_metadata_filter("meta", None, None)
        assert result is not None
        assert result["ticker"] == "META"

    def test_empty_string_ticker_returns_none(self) -> None:
        from ui.utils import build_metadata_filter

        result = build_metadata_filter("", None, None)
        assert result is None

    def test_year_converted_to_int(self) -> None:
        from ui.utils import build_metadata_filter

        result = build_metadata_filter(None, "2024", None)  # type: ignore[arg-type]
        assert result == {"year": 2024}
        assert isinstance(result["year"], int)


# ── format_latency ─────────────────────────────────────────────────────────────


class TestFormatLatency:
    def test_sub_second_shows_ms(self) -> None:
        from ui.utils import format_latency

        assert format_latency(0.095) == "95ms"

    def test_exactly_1_second(self) -> None:
        from ui.utils import format_latency

        assert format_latency(1.0) == "1.00s"

    def test_seconds_two_decimal_places(self) -> None:
        from ui.utils import format_latency

        assert format_latency(2.456) == "2.46s"

    def test_zero(self) -> None:
        from ui.utils import format_latency

        assert format_latency(0.0) == "0ms"

    def test_large_value(self) -> None:
        from ui.utils import format_latency

        result = format_latency(125.7)
        assert result == "125.70s"

    def test_just_below_1_second(self) -> None:
        from ui.utils import format_latency

        result = format_latency(0.999)
        assert "ms" in result


# ── format_token_count ─────────────────────────────────────────────────────────


class TestFormatTokenCount:
    def test_below_1k_no_suffix(self) -> None:
        from ui.utils import format_token_count

        assert format_token_count(500) == "500"

    def test_exactly_1000(self) -> None:
        from ui.utils import format_token_count

        assert format_token_count(1000) == "1.0K"

    def test_large_number(self) -> None:
        from ui.utils import format_token_count

        assert format_token_count(12400) == "12.4K"

    def test_zero(self) -> None:
        from ui.utils import format_token_count

        assert format_token_count(0) == "0"

    def test_999(self) -> None:
        from ui.utils import format_token_count

        assert format_token_count(999) == "999"


# ── citation_badge_label ───────────────────────────────────────────────────────


class TestCitationBadgeLabel:
    def test_full_citation(self) -> None:
        from ui.utils import citation_badge_label

        c = {"index": 1, "ticker": "AAPL", "fiscal_period": "Q4 2024"}
        assert citation_badge_label(c) == "[1] AAPL Q4 2024"

    def test_missing_ticker(self) -> None:
        from ui.utils import citation_badge_label

        c = {"index": 2, "fiscal_period": "Q3 2024"}
        result = citation_badge_label(c)
        assert "[2]" in result

    def test_missing_index_uses_question_mark(self) -> None:
        from ui.utils import citation_badge_label

        c = {"ticker": "NVDA", "fiscal_period": "Q1 2024"}
        result = citation_badge_label(c)
        assert "[?]" in result

    def test_empty_dict_does_not_crash(self) -> None:
        from ui.utils import citation_badge_label

        result = citation_badge_label({})
        assert "[?]" in result


# ── group_citations_by_source ──────────────────────────────────────────────────


class TestGroupCitationsBySource:
    def test_groups_same_source_together(self) -> None:
        from ui.utils import group_citations_by_source

        citations = [
            {"ticker": "AAPL", "fiscal_period": "Q4 2024", "index": 1},
            {"ticker": "AAPL", "fiscal_period": "Q4 2024", "index": 2},
            {"ticker": "NVDA", "fiscal_period": "Q3 2024", "index": 3},
        ]
        groups = group_citations_by_source(citations)
        assert len(groups) == 2
        assert len(groups["AAPL Q4 2024"]) == 2
        assert len(groups["NVDA Q3 2024"]) == 1

    def test_preserves_insertion_order(self) -> None:
        from ui.utils import group_citations_by_source

        citations = [
            {"ticker": "MSFT", "fiscal_period": "Q1 2024", "index": 1},
            {"ticker": "AAPL", "fiscal_period": "Q4 2024", "index": 2},
        ]
        groups = group_citations_by_source(citations)
        keys = list(groups.keys())
        assert keys[0] == "MSFT Q1 2024"
        assert keys[1] == "AAPL Q4 2024"

    def test_empty_list_returns_empty_dict(self) -> None:
        from ui.utils import group_citations_by_source

        assert group_citations_by_source([]) == {}

    def test_single_citation(self) -> None:
        from ui.utils import group_citations_by_source

        c = {"ticker": "JPM", "fiscal_period": "Q2 2024", "index": 1}
        groups = group_citations_by_source([c])
        assert list(groups.keys()) == ["JPM Q2 2024"]
        assert groups["JPM Q2 2024"] == [c]


# ── health_status_emoji ────────────────────────────────────────────────────────


class TestHealthStatusEmoji:
    @pytest.mark.parametrize(
        "status, expected",
        [
            ("ok", "✅"),
            ("healthy", "✅"),
            ("error", "❌"),
            ("unhealthy", "❌"),
            ("degraded", "⚠️"),
        ],
    )
    def test_known_statuses(self, status: str, expected: str) -> None:
        from ui.utils import health_status_emoji

        assert health_status_emoji(status) == expected

    def test_unknown_status_returns_question_mark(self) -> None:
        from ui.utils import health_status_emoji

        assert health_status_emoji("starting") == "❓"

    def test_case_insensitive(self) -> None:
        from ui.utils import health_status_emoji

        assert health_status_emoji("OK") == "✅"
        assert health_status_emoji("ERROR") == "❌"


# ── stream_query ───────────────────────────────────────────────────────────────


class TestStreamQuery:
    """Tests for stream_query() using mocked requests.post."""

    def _make_mock_response(self, lines: list[str]) -> MagicMock:
        """Build a mock requests.Response whose iter_lines() yields the given lines."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_lines.return_value = iter([line.encode("utf-8") for line in lines])
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_yields_tokens(self) -> None:
        from ui.utils import stream_query

        lines = [
            'data: {"token": "Apple"}',
            'data: {"token": " reported"}',
            "data: [DONE]",
        ]
        mock_resp = self._make_mock_response(lines)
        with patch("requests.post", return_value=mock_resp):
            tokens = list(stream_query("http://localhost:8000", "What is AAPL revenue?"))

        assert tokens == ["Apple", " reported"]

    def test_skips_blank_lines(self) -> None:
        from ui.utils import stream_query

        lines = [
            "",
            'data: {"token": "hello"}',
            "",
            "data: [DONE]",
        ]
        mock_resp = self._make_mock_response(lines)
        with patch("requests.post", return_value=mock_resp):
            tokens = list(stream_query("http://localhost:8000", "question"))

        assert tokens == ["hello"]

    def test_error_event_yields_error_token(self) -> None:
        from ui.utils import stream_query

        lines = [
            'data: {"error": "Rate limit exceeded"}',
        ]
        mock_resp = self._make_mock_response(lines)
        with patch("requests.post", return_value=mock_resp):
            tokens = list(stream_query("http://localhost:8000", "question"))

        assert len(tokens) == 1
        assert "Rate limit exceeded" in tokens[0]

    def test_passes_metadata_filter_in_payload(self) -> None:
        from ui.utils import stream_query

        lines = ["data: [DONE]"]
        mock_resp = self._make_mock_response(lines)
        with patch("requests.post", return_value=mock_resp) as mock_post:
            list(
                stream_query(
                    "http://localhost:8000",
                    "question",
                    metadata_filter={"ticker": "AAPL", "year": 2024},
                )
            )

        call_kwargs = mock_post.call_args
        sent_json = (
            call_kwargs.kwargs.get("json") or call_kwargs.args[1] if call_kwargs.args else {}
        )
        if not sent_json:
            sent_json = call_kwargs.kwargs.get("json", {})
        assert sent_json.get("filter") == {"ticker": "AAPL", "year": 2024}

    def test_raises_on_connection_error(self) -> None:
        from ui.utils import stream_query

        with patch("requests.post", side_effect=requests.ConnectionError("refused")):
            with pytest.raises(requests.ConnectionError):
                list(stream_query("http://localhost:8000", "question"))

    def test_raises_on_http_error(self) -> None:
        from ui.utils import stream_query

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("503")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                list(stream_query("http://localhost:8000", "question"))

    def test_strips_trailing_slash_from_base_url(self) -> None:
        from ui.utils import stream_query

        lines = ["data: [DONE]"]
        mock_resp = self._make_mock_response(lines)
        with patch("requests.post", return_value=mock_resp) as mock_post:
            list(stream_query("http://localhost:8000/", "question"))

        called_url = mock_post.call_args.args[0]
        assert called_url == "http://localhost:8000/query/stream"


# ── fetch_structured ───────────────────────────────────────────────────────────


class TestFetchStructured:
    def test_returns_parsed_json(self) -> None:
        from ui.utils import fetch_structured

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "answer": "Revenue was $94.9B",
            "citations": [],
            "grounded": True,
        }
        with patch("requests.post", return_value=mock_resp):
            result = fetch_structured("http://localhost:8000", "What was revenue?")

        assert result["answer"] == "Revenue was $94.9B"
        assert result["grounded"] is True

    def test_passes_verbose_flag(self) -> None:
        from ui.utils import fetch_structured

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {}

        with patch("requests.post", return_value=mock_resp) as mock_post:
            fetch_structured("http://localhost:8000", "q", verbose=True)

        sent_json = mock_post.call_args.kwargs.get("json", {})
        assert sent_json.get("verbose") is True

    def test_raises_on_http_error(self) -> None:
        from ui.utils import fetch_structured

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("422")

        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                fetch_structured("http://localhost:8000", "q")


# ── fetch_health ───────────────────────────────────────────────────────────────


class TestFetchHealth:
    def test_returns_health_dict_on_success(self) -> None:
        from ui.utils import fetch_health

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "status": "healthy",
            "components": {"qdrant": {"status": "ok"}},
        }

        with patch("requests.get", return_value=mock_resp):
            result = fetch_health("http://localhost:8000")

        assert result["status"] == "healthy"
        assert "qdrant" in result["components"]

    def test_returns_error_dict_on_connection_failure(self) -> None:
        from ui.utils import fetch_health

        with patch("requests.get", side_effect=requests.ConnectionError("refused")):
            result = fetch_health("http://localhost:8000")

        assert result["status"] == "unhealthy"
        assert "error" in result
        assert result["components"] == {}

    def test_returns_error_dict_on_http_error(self) -> None:
        from ui.utils import fetch_health

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("503")

        with patch("requests.get", return_value=mock_resp):
            result = fetch_health("http://localhost:8000")

        assert result["status"] == "unhealthy"
        assert "error" in result

    def test_never_raises(self) -> None:
        """fetch_health must be exception-safe — the UI sidebar must never crash."""
        from ui.utils import fetch_health

        with patch("requests.get", side_effect=RuntimeError("unexpected")):
            result = fetch_health("http://localhost:8000")

        assert isinstance(result, dict)
        assert result["status"] == "unhealthy"
