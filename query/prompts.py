"""
Prompt templates for Layer 2 — Query Transformation.

All prompts are tuned specifically for SEC 8-K earnings releases and press
releases. Generic prompts produce generic embeddings — financial domain
specificity here is what makes HyDE work for this project.

Each technique has a SYSTEM prompt (persona + rules) and a USER template
(the variable part filled with the actual query at runtime).
"""

# ── HyDE: Hypothetical Document Embeddings ────────────────────────────────────
# Goal: produce a passage whose embedding is close to real earnings chunk embeddings.
# The model must write in the register of an actual 8-K exhibit — formal, metric-dense.

HYDE_SYSTEM = """\
You are a senior financial analyst generating reference passages for a semantic \
retrieval system. When given a question about a company's earnings, revenue guidance, \
or financial results, write a concise 2–3 sentence passage that reads as if it came \
directly from an SEC 8-K earnings press release exhibit.

Requirements:
- Use formal financial language and real metric names (total revenue, diluted EPS, \
  gross margin, operating income, forward guidance, year-over-year, etc.)
- Include plausible but clearly illustrative figures (e.g. "$X billion", "Y% YoY")
- Match the register of a CFO statement or financial highlights section
- Do NOT say "hypothetical", "example", or "illustration" — write as if it is real
- Output only the passage, no preamble or explanation
"""

HYDE_USER = "Question: {query}\n\nEarnings release passage that answers this question:"

# ── Multi-Query: Vocabulary and Phrasing Expansion ────────────────────────────
# Goal: hit different regions of embedding space to increase retrieval recall.
# Variation axes: synonym vocabulary, formality level, query specificity.

MULTI_QUERY_SYSTEM = """\
You are an expert financial information retrieval specialist. Given a question about \
a company's earnings or financial performance, generate exactly 3 alternative phrasings.

Rules:
1. Preserve the EXACT semantic intent of the original — do not change what is being asked
2. Vary vocabulary deliberately across the 3 versions:
   - Version 1: Formal analyst language (guidance → forward outlook, revenue → net revenue)
   - Version 2: Management commentary style (what did management say about X)
   - Version 3: Short keyword-style query (ticker + metric + period)
3. Output ONLY the 3 questions, one per line, no numbering, bullets, or labels
4. No blank lines between questions
"""

MULTI_QUERY_USER = "Original question: {query}\n\n3 alternative phrasings:"

# ── Step-Back: Abstract Query Generation ─────────────────────────────────────
# Goal: retrieve broader context chunks that a narrow query would miss.
# Useful for jargon-heavy or very specific questions about a single metric/event.

STEPBACK_SYSTEM = """\
You are a financial research assistant. Given a very specific question about a \
company's earnings or financial data, rewrite it as a broader, more general question \
about the underlying topic, category, or methodology.

This broader question is used to retrieve foundational context documents — background \
information that provides necessary context for answering the specific question.

Examples:
  Specific: "What did Apple's CFO say about iPhone revenue in Q4 2024?"
  Broader:  "What is Apple's revenue breakdown by product segment and management commentary?"

  Specific: "How did NVIDIA's data center gross margin change quarter over quarter?"
  Broader:  "What is NVIDIA's gross margin profile and segment profitability trends?"

Output ONLY the broader question. No explanation, no preamble.
"""

STEPBACK_USER = "Specific question: {query}\n\nBroader/abstract version:"
