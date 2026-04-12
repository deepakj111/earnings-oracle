# tests/test_query_prompts.py


from query.prompts import (
    HYDE_SYSTEM,
    HYDE_USER,
    MULTI_QUERY_SYSTEM,
    MULTI_QUERY_USER,
    STEPBACK_SYSTEM,
    STEPBACK_USER,
)

# ── HyDE prompts ──────────────────────────────────────────────────────────────


class TestHydePrompts:
    def test_system_is_nonempty_string(self):
        assert isinstance(HYDE_SYSTEM, str)
        assert len(HYDE_SYSTEM) > 50

    def test_user_template_contains_query_placeholder(self):
        assert "{query}" in HYDE_USER

    def test_user_template_formats_without_leftover_placeholder(self):
        result = HYDE_USER.format(query="What was Apple's Q4 revenue?")
        assert "What was Apple's Q4 revenue?" in result
        assert "{query}" not in result

    def test_system_references_financial_domain(self):
        lower = HYDE_SYSTEM.lower()
        assert "earnings" in lower or "8-k" in lower or "financial" in lower

    def test_system_requests_passage_not_explanation(self):
        lower = HYDE_SYSTEM.lower()
        assert "passage" in lower or "press release" in lower or "exhibit" in lower

    def test_system_instructs_no_preamble_in_output(self):
        lower = HYDE_SYSTEM.lower()
        assert "only" in lower or "no preamble" in lower or "no explanation" in lower


# ── Multi-Query prompts ───────────────────────────────────────────────────────


class TestMultiQueryPrompts:
    def test_system_is_nonempty_string(self):
        assert isinstance(MULTI_QUERY_SYSTEM, str)
        assert len(MULTI_QUERY_SYSTEM) > 50

    def test_user_template_contains_query_placeholder(self):
        assert "{query}" in MULTI_QUERY_USER

    def test_user_template_formats_without_leftover_placeholder(self):
        result = MULTI_QUERY_USER.format(query="NVDA earnings Q3 2024")
        assert "NVDA earnings Q3 2024" in result
        assert "{query}" not in result

    def test_system_specifies_three_alternatives(self):
        assert "3" in MULTI_QUERY_SYSTEM or "three" in MULTI_QUERY_SYSTEM.lower()

    def test_system_instructs_no_numbering_or_bullets(self):
        lower = MULTI_QUERY_SYSTEM.lower()
        assert "no" in lower or "without" in lower or "only" in lower

    def test_system_mentions_semantic_intent_preservation(self):
        lower = MULTI_QUERY_SYSTEM.lower()
        assert "intent" in lower or "semantic" in lower or "preserve" in lower


# ── Step-Back prompts ─────────────────────────────────────────────────────────


class TestStepbackPrompts:
    def test_system_is_nonempty_string(self):
        assert isinstance(STEPBACK_SYSTEM, str)
        assert len(STEPBACK_SYSTEM) > 50

    def test_user_template_contains_query_placeholder(self):
        assert "{query}" in STEPBACK_USER

    def test_user_template_formats_without_leftover_placeholder(self):
        result = STEPBACK_USER.format(query="MSFT gross margin Q2")
        assert "MSFT gross margin Q2" in result
        assert "{query}" not in result

    def test_system_instructs_broader_or_abstract_output(self):
        lower = STEPBACK_SYSTEM.lower()
        assert "broader" in lower or "general" in lower or "abstract" in lower

    def test_system_includes_examples(self):
        assert "Specific" in STEPBACK_SYSTEM or "broader" in STEPBACK_SYSTEM.lower()

    def test_system_instructs_no_explanation(self):
        lower = STEPBACK_SYSTEM.lower()
        assert "only" in lower or "no explanation" in lower or "no preamble" in lower
