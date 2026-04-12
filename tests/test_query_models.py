# tests/test_query_models.py


from query.models import TransformedQuery


def _make(**kwargs) -> TransformedQuery:
    defaults = {
        "original": "What was Apple's revenue in Q4 2024?",
        "hyde_document": "Apple reported total revenue of $94.9 billion for Q4 2024, "
        "representing 6% growth year-over-year.",
        "multi_queries": [
            "What was Apple's revenue in Q4 2024?",
            "What were Apple's net revenues for Q4 fiscal 2024?",
            "How much revenue did Apple generate in the fourth quarter of 2024?",
            "AAPL Q4 2024 total revenue results",
        ],
        "stepback_query": "What is Apple's revenue breakdown and segment financial performance trends?",
    }
    defaults.update(kwargs)
    return TransformedQuery(**defaults)


# ── TransformedQuery dataclass ────────────────────────────────────────────────


class TestTransformedQueryFields:
    def test_original_stored(self):
        tq = _make(original="test query text")
        assert tq.original == "test query text"

    def test_hyde_document_stored(self):
        tq = _make(hyde_document="a passage about earnings")
        assert tq.hyde_document == "a passage about earnings"

    def test_multi_queries_stored(self):
        tq = _make(multi_queries=["q1", "q2"])
        assert tq.multi_queries == ["q1", "q2"]

    def test_stepback_query_stored(self):
        tq = _make(stepback_query="broader financial question")
        assert tq.stepback_query == "broader financial question"

    def test_failed_techniques_defaults_to_empty_list(self):
        tq = _make()
        assert tq.failed_techniques == []

    def test_failed_techniques_can_be_populated(self):
        tq = _make(failed_techniques=["hyde", "stepback"])
        assert "hyde" in tq.failed_techniques
        assert "stepback" in tq.failed_techniques

    def test_failed_techniques_is_mutable_list(self):
        tq = _make()
        tq.failed_techniques.append("multi")
        assert "multi" in tq.failed_techniques


# ── all_retrieval_queries property ───────────────────────────────────────────


class TestAllRetrievalQueries:
    def test_returns_list(self):
        tq = _make()
        assert isinstance(tq.all_retrieval_queries, list)

    def test_all_multi_queries_included(self):
        tq = _make(multi_queries=["q1", "q2", "q3"], stepback_query="sq_unique")
        result = tq.all_retrieval_queries
        assert "q1" in result
        assert "q2" in result
        assert "q3" in result

    def test_stepback_query_included(self):
        tq = _make(multi_queries=["q1"], stepback_query="the broader context question")
        assert "the broader context question" in tq.all_retrieval_queries

    def test_exact_duplicate_removed(self):
        tq = _make(multi_queries=["same query"], stepback_query="same query")
        result = tq.all_retrieval_queries
        assert result.count("same query") == 1

    def test_case_insensitive_deduplication(self):
        tq = _make(multi_queries=["Apple Revenue Q4"], stepback_query="apple revenue q4")
        result = tq.all_retrieval_queries
        lowered = [q.lower() for q in result]
        assert len(lowered) == len(set(lowered))

    def test_multi_queries_appear_before_stepback(self):
        tq = _make(multi_queries=["q1", "q2"], stepback_query="broader_q3")
        result = tq.all_retrieval_queries
        assert result.index("q1") < result.index("broader_q3")
        assert result.index("q2") < result.index("broader_q3")

    def test_whitespace_stripped_from_result(self):
        tq = _make(multi_queries=["  q1  "], stepback_query="  q2  ")
        result = tq.all_retrieval_queries
        assert "q1" in result
        assert "q2" in result

    def test_four_unique_queries_all_present(self):
        tq = _make(multi_queries=["q1", "q2", "q3"], stepback_query="q4")
        assert len(tq.all_retrieval_queries) == 4

    def test_stepback_not_duplicated_when_matches_multi(self):
        tq = _make(
            multi_queries=["What is Apple's revenue?"],
            stepback_query="What is Apple's revenue?",
        )
        assert tq.all_retrieval_queries.count("What is Apple's revenue?") == 1

    def test_no_empty_strings_in_result(self):
        tq = _make(multi_queries=["q1", "q2"], stepback_query="sq")
        for q in tq.all_retrieval_queries:
            assert q.strip() != ""


# ── summary() method ──────────────────────────────────────────────────────────


class TestSummary:
    def test_returns_string(self):
        tq = _make()
        assert isinstance(tq.summary(), str)

    def test_contains_original_query(self):
        tq = _make(original="Apple Q4 revenue results")
        assert "Apple Q4 revenue results" in tq.summary()

    def test_contains_multi_query_count(self):
        tq = _make(multi_queries=["q1", "q2", "q3"])
        assert "3" in tq.summary()

    def test_contains_stepback_query(self):
        tq = _make(stepback_query="What is the broader revenue trend?")
        assert "What is the broader revenue trend?" in tq.summary()

    def test_shows_degraded_label_when_failed(self):
        tq = _make(failed_techniques=["hyde"])
        summary = tq.summary().lower()
        assert "degraded" in summary or "hyde" in summary

    def test_no_degraded_label_when_fully_successful(self):
        tq = _make(failed_techniques=[])
        assert "degraded" not in tq.summary().lower()

    def test_hyde_doc_truncated_in_summary(self):
        long_hyde = "X" * 500
        tq = _make(hyde_document=long_hyde)
        summary = tq.summary()
        assert len(summary) < 500 + 300
