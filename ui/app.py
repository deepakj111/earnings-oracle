# ui/app.py
"""
Financial RAG System — Streamlit UI.

Run with:
    poetry run streamlit run ui/app.py

    # Or with a custom API URL:
    RAG_API_URL=http://my-api-host:8000 poetry run streamlit run ui/app.py

Features:
  - Streaming chat interface (POST /query/stream SSE)
  - Structured mode with full citation display (POST /query)
  - Company / year / quarter metadata filters in sidebar
  - Per-response diagnostics: tokens, latency, grounding status
  - Citation cards with excerpt, source type, and rerank score
  - System health panel (GET /health)
  - Chat history with session state
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

from ui.utils import (
    build_metadata_filter,
    citation_badge_label,
    fetch_health,
    fetch_structured,
    format_latency,
    format_token_count,
    group_citations_by_source,
    health_status_emoji,
    stream_query,
)

# ── Config ─────────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv("RAG_API_URL", "http://localhost:8000")

_TICKERS = ["(all)", "AAPL", "NVDA", "MSFT", "AMZN", "META", "JPM", "XOM", "UNH", "TSLA", "WMT"]
_YEARS = ["(all)", 2024, 2023]
_QUARTERS = ["(all)", "Q1", "Q2", "Q3", "Q4"]

# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Financial RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (minimal, no external CDN) ─────────────────────────────────────

st.markdown(
    """
    <style>
    .citation-card {
        background: #f8f9fa;
        border-left: 3px solid #1f77b4;
        border-radius: 4px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.85rem;
    }
    .citation-card .meta {
        color: #555;
        font-size: 0.78rem;
        margin-bottom: 4px;
    }
    .citation-card .excerpt {
        color: #333;
        font-style: italic;
    }
    .stat-chip {
        display: inline-block;
        background: #e9ecef;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.80rem;
        margin-right: 6px;
    }
    .grounded-true  { color: #198754; font-weight: 600; }
    .grounded-false { color: #dc3545; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ───────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": ..., "meta": ...}

if "last_response" not in st.session_state:
    st.session_state.last_response = None  # full AskResponse dict from /query


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Financial RAG")
    st.caption("SEC 8-K Earnings Intelligence")
    st.divider()

    # --- Filters ---
    st.subheader("🔍 Filters")
    ticker_sel = st.selectbox("Company", _TICKERS, index=0)
    year_sel = st.selectbox("Year", _YEARS, index=0)
    quarter_sel = st.selectbox("Quarter", _QUARTERS, index=0)

    ticker = None if ticker_sel == "(all)" else ticker_sel
    year = None if year_sel == "(all)" else int(str(year_sel))
    quarter = None if quarter_sel == "(all)" else quarter_sel

    metadata_filter = build_metadata_filter(ticker, year, quarter)

    st.divider()

    # --- Response mode ---
    st.subheader("⚙️ Settings")
    streaming_mode = st.toggle(
        "Streaming mode",
        value=True,
        help="Stream tokens as they arrive. Disable for full citation display.",
    )
    verbose_mode = st.toggle(
        "Verbose diagnostics",
        value=False,
        help="Include query transform + retrieval summaries in response.",
    )
    if streaming_mode and verbose_mode:
        st.caption("ℹ️ Verbose diagnostics only available in non-streaming mode.")

    st.divider()

    # --- Clear history ---
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_response = None
        st.rerun()

    st.divider()

    # --- Health panel ---
    st.subheader("🏥 System Health")
    if st.button("Refresh", use_container_width=True, key="health_refresh"):
        st.session_state._health = fetch_health(API_BASE_URL)

    if "_health" not in st.session_state:
        st.session_state._health = fetch_health(API_BASE_URL)

    health = st.session_state._health
    overall = health.get("status", "unknown")
    st.markdown(f"**Status:** {health_status_emoji(overall)} `{overall.upper()}`")

    for component, info in health.get("components", {}).items():
        s = info.get("status", "unknown")
        detail = info.get("detail", "")
        st.markdown(f"{health_status_emoji(s)} **{component}** — {detail[:60]}")

    if health.get("uptime_seconds"):
        uptime = health["uptime_seconds"]
        st.caption(f"Uptime: {format_latency(uptime)}")

    if health.get("error"):
        st.error(f"API unreachable: {health['error']}")


def _render_response_meta(meta: dict[str, Any]) -> None:
    """
    Render token usage, cost, latency, grounding chips and citation cards
    for a completed assistant response.

    Separated from the main flow so it can be called for both new and
    historical messages from session state.
    """
    if not meta:
        return

    # --- Stats row ---
    usage = meta.get("usage", {})
    context = meta.get("context", {})
    grounded = meta.get("grounded", True)
    latency = meta.get("latency_seconds", 0.0)
    total_tokens = usage.get("total_tokens", 0)
    context_tokens = context.get("tokens_used", 0)

    was_cached = meta.get("was_cached", False)

    grounded_cls = "grounded-true" if grounded else "grounded-false"
    grounded_label = "✓ Grounded" if grounded else "✗ Ungrounded"

    cache_badge = (
        '<span class="stat-chip" style="background:#fff3cd;color:#856404;border:1px solid #ffe69c;">⚡ Cached</span>'
        if was_cached
        else ""
    )

    st.markdown(
        f"""
        <div style="margin: 8px 0 4px 0;">
          {cache_badge}
          <span class="stat-chip">⏱ {format_latency(latency)}</span>
          <span class="stat-chip">🪙 {format_token_count(total_tokens)} tokens</span>
          <span class="stat-chip">📄 {context.get("chunks_used", 0)} chunks
          ({format_token_count(context_tokens)} ctx tokens)</span>
          <span class="stat-chip {grounded_cls}">{grounded_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Citations ---
    citations = meta.get("citations", [])
    if citations:
        with st.expander(
            f"📎 Sources ({len(citations)} citation{'s' if len(citations) != 1 else ''})",
            expanded=False,
        ):
            grouped = group_citations_by_source(citations)
            for source_label, source_cits in grouped.items():
                st.markdown(f"**{source_label}**")
                for c in source_cits:
                    score_pct = f"{c.get('rerank_score', 0) * 100:.1f}%"
                    src_icon = {
                        "dense": "🔷",
                        "bm25": "🔶",
                        "both": "🔷🔶",
                        "knowledge_graph": "🕸️",
                    }.get(c.get("source", ""), "📄")
                    st.markdown(
                        f"""
                        <div class="citation-card">
                          <div class="meta">
                            {citation_badge_label(c)}
                            &nbsp;·&nbsp; {c.get("section_title", "")}
                            &nbsp;·&nbsp; {src_icon} {c.get("source", "")}
                            &nbsp;·&nbsp; relevance {score_pct}
                          </div>
                          <div class="excerpt">{c.get("excerpt", "")[:300]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # --- Verbose diagnostics ---
    if meta.get("query_summary") or meta.get("retrieval_summary"):
        with st.expander("🔬 Pipeline diagnostics", expanded=False):
            if meta.get("query_summary"):
                st.markdown("**Query transformation:**")
                st.code(meta["query_summary"], language=None)
            if meta.get("retrieval_summary"):
                st.markdown("**Retrieval:**")
                st.code(meta["retrieval_summary"], language=None)

    # --- Sources pill row ---
    unique_sources = meta.get("unique_sources", [])
    if unique_sources:
        st.caption("Sources: " + " · ".join(f"`{s}`" for s in unique_sources))


# ── Re-render existing assistant messages (history playback) ──────────────────
# We need to re-call _render_response_meta for history items. Since Streamlit
# re-runs from top, patch the above loop's render call here.
# The loop above only renders content; the meta call references the function
# defined after, so we regenerate for the last rendered assistant message.
# NOTE: Streamlit re-runs the entire script on every interaction — the loop
#       at the top will correctly call _render_response_meta (now defined) on
#       reruns from session state.

# ── Main chat area ─────────────────────────────────────────────────────────────

st.header("💬 Ask a Financial Question", divider="gray")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show metadata for assistant messages
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            _render_response_meta(meta)  # defined below — forward reference via function


# ── Chat input ────────────────────────────────────────────────────────────────

question = st.chat_input(
    "Ask about earnings, revenue, guidance… (e.g. What was Apple's revenue in Q4 2024?)",
    key="chat_input",
)

if question:
    # --- Display user message ---
    st.session_state.messages.append({"role": "user", "content": question, "meta": None})
    with st.chat_message("user"):
        st.markdown(question)

    # --- Run pipeline ---
    with st.chat_message("assistant"):
        if streaming_mode:
            # ── Streaming path ──────────────────────────────────────────────
            answer_placeholder = st.empty()
            full_answer = ""
            error_msg = None

            try:
                with st.spinner("Thinking…"):
                    # Let spinner show while the first token arrives
                    token_gen = stream_query(
                        base_url=API_BASE_URL,
                        question=question,
                        metadata_filter=metadata_filter,
                    )
                    # Consume first token inside spinner to cover L2+L3 latency
                    first_token = next(token_gen, None)

                if first_token is not None:
                    full_answer = first_token
                    answer_placeholder.markdown(full_answer + "▌")
                    for token in token_gen:
                        full_answer += token
                        answer_placeholder.markdown(full_answer + "▌")
                    answer_placeholder.markdown(full_answer)
                else:
                    answer_placeholder.warning("No response received from the API.")

            except requests.ConnectionError:
                error_msg = "❌ Cannot connect to the API. Is the server running?"
                st.error(error_msg)
            except requests.HTTPError as exc:
                error_msg = f"❌ API error: {exc.response.status_code}"
                st.error(error_msg)
            except Exception as exc:
                error_msg = f"❌ Unexpected error: {exc}"
                st.error(error_msg)

            resp_meta: dict[str, Any] = {}  # no structured meta in streaming mode
            content = full_answer or error_msg or ""

        else:
            # ── Structured path ─────────────────────────────────────────────
            content = ""
            resp_meta = {}
            error_msg = None

            with st.spinner("Running RAG pipeline…"):
                try:
                    data = fetch_structured(
                        base_url=API_BASE_URL,
                        question=question,
                        metadata_filter=metadata_filter,
                        verbose=verbose_mode,
                    )
                    content = data.get("answer", "")
                    resp_meta = data
                    st.markdown(content)
                    _render_response_meta(resp_meta)

                except requests.ConnectionError:
                    error_msg = "❌ Cannot connect to the API. Is the server running?"
                    st.error(error_msg)
                    content = error_msg
                except requests.HTTPError as exc:
                    try:
                        detail = exc.response.json().get("detail", str(exc))
                    except Exception:
                        detail = str(exc)
                    error_msg = f"❌ API error {exc.response.status_code}: {detail}"
                    st.error(error_msg)
                    content = error_msg
                except Exception as exc:
                    error_msg = f"❌ Unexpected error: {exc}"
                    st.error(error_msg)
                    content = error_msg

    # --- Persist to session state ---
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "meta": resp_meta,
        }
    )
    st.session_state.last_response = resp_meta if resp_meta else None

    st.rerun()


# ── Footer ─────────────────────────────────────────────────────────────────────

st.caption(
    f"Financial RAG v0.1.0 · API: `{API_BASE_URL}` · "
    "Model: BAAI/bge-large-en-v1.5 + Qdrant + BM25 + FlashRank"
)
