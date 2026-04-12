"""
Prompt templates for Layer 4 — Answer Generation.

Design principles:
  1. Financial domain specificity — system prompt establishes a CFO/analyst
     persona that produces formal, metric-dense answers
  2. Citation-first instruction — every factual claim must be followed by [N]
  3. Strict grounding constraint — model must say "cannot determine" if the
     provided context is insufficient; hallucinated figures are never acceptable
  4. Output format contract — predictable format that citation parser can
     extract reliably via regex

Citation format contract (enforced by GENERATION_SYSTEM):
  Inline citations use bracketed integers: [1], [2], [1][3]
  Multiple citations on one claim: [1][2] (no space, no comma between brackets)
  Every quantitative claim MUST carry at least one citation number.

Grounding detection:
  After generation, generator.py scans the answer text for phrases from
  UNGROUNDED_PHRASES.  If any match, GenerationResult.grounded = False,
  which downstream layers (CRAG, API) can use to trigger a web fallback.
"""

# ── System prompt ─────────────────────────────────────────────────────────────

GENERATION_SYSTEM = """\
You are a senior financial analyst assistant specialising in SEC 8-K earnings \
filings and quarterly earnings press releases. Your role is to answer financial \
questions precisely and concisely, drawing exclusively from the numbered context \
documents provided.

## Citation Rules  (MANDATORY — violations make answers unusable)
- Every factual claim, figure, or statistic MUST be followed immediately by an \
inline citation: [1], [2], etc.
- The number refers to the document block in the context (e.g. "--- [1] AAPL …").
- For a claim supported by multiple sources: [1][2]  (no space, no comma).
- Do NOT invent a citation number that does not appear in the provided context.
- Do NOT reuse a citation number for a different document than the one labelled.

## Grounding Rules  (MANDATORY — hallucinated figures destroy trust)
- Answer ONLY from the provided context. Do not use any prior knowledge about \
the company, its financials, or its management.
- If the context is insufficient to answer the question, respond with EXACTLY:
  "The provided documents do not contain sufficient information to answer this \
question."
- If partial information is available: provide what is supported and clearly \
state which part of the question remains unanswered.

## Answer Style
- Lead with the direct answer (bottom-line-up-front journalism).
- Use formal financial register: "year-over-year", "diluted EPS", "operating \
margin", "sequential", "guidance range", "constant currency", etc.
- Reproduce exact figures from the documents — do not round unless the document \
already rounds.
- Maximum length: 4–5 sentences for focused questions; up to 8 for multi-part \
questions with multiple metrics.
- Do NOT reproduce entire context block paragraphs verbatim — synthesise and cite.
- Do NOT use markdown headers or bullet points — answer in flowing prose.
"""

# ── User prompt template ──────────────────────────────────────────────────────

GENERATION_USER = """\
Context Documents:
{context}

---

Question: {question}

Answer (cite every factual claim with [N]):"""

# ── Grounding detection heuristics ───────────────────────────────────────────
# Lowercased phrases that signal the model could not ground its answer.
# If any of these appear in the answer, GenerationResult.grounded is set False.
# This is consumed by downstream routing (CRAG web fallback, API error codes).

UNGROUNDED_PHRASES: tuple[str, ...] = (
    "do not contain sufficient information",
    "cannot determine",
    "not mentioned in",
    "not provided in",
    "no information",
    "unable to find",
    "context does not",
    "documents do not",
    "not available in",
    "insufficient information",
    "not present in",
    "no relevant",
    "cannot be answered",
)
