# LLMOps Guide

> Evaluation, observability, cost management, and continuous quality improvement for the Financial RAG System.

---

## Table of Contents

1. [LLMOps Philosophy](#llmops-philosophy)
2. [Evaluation Framework](#evaluation-framework)
3. [Observability & Metrics](#observability--metrics)
4. [Cost Management](#cost-management)
5. [Quality Monitoring](#quality-monitoring)
6. [Prompt Management](#prompt-management)
7. [Retrieval Ablations](#retrieval-ablations)
8. [Data Quality](#data-quality)
9. [Incident Response](#incident-response)

---

## LLMOps Philosophy

The Financial RAG System treats LLM-powered components as **measurable, improvable software systems** — not black boxes. Every LLM call produces:

1. **Observable outputs** — structured data that can be validated and scored
2. **Cost attribution** — token counts tracked per model and call type
3. **Quality signals** — grounding flags, citation counts, CRAG actions
4. **Latency measurements** — per-layer timing for bottleneck identification

The evaluation harness closes the loop: offline evaluation against a golden dataset detects quality regressions before they reach production.

---

## Evaluation Framework

### Golden Dataset

`evaluation/dataset.py` loads a **130-question curated QA dataset** drawn from real SEC 10-K Annual Reports and 10-Q Quarterly Filings across four Fortune 50 companies:

| Company | Ticker | Sector | # Samples |
|---------|--------|--------|-----------|
| NVIDIA | NVDA | Technology / Semiconductors | 30 |
| Walmart | WMT | Consumer Staples / Retail | 30 |
| Netflix | NFLX | Communication Services / Streaming | 35 |
| UnitedHealth Group | UNH | Healthcare / Managed Care | 35 |

Questions span both annual (10-K) and quarterly (10-Q) filings and adhere to a structured **4-pillar distribution**:

- **MD&A & Strategic Initiatives (~35%)**: Operational roadmaps, technological modernization (e.g. Blackwell GPU rollout for NVDA, supply chain automation for WMT, live events & ad-tier for NFLX, Optum Care value-based care for UNH).
- **Risk Factors & Regulatory Disclosures (~25%)**: US export control restrictions, cyber incident recovery, healthcare/PBM regulation, retail shrink, and content amortization.
- **Segment & Core Financials (~30%)**: Segment revenue breakdowns, operating margin shifts, regional membership/ARM metrics.
- **Capital Allocation & Cash Flow (~10%)**: Share buybacks, capex allocation, and liquidity management.

All questions are **100% self-contained**, explicitly naming the target company and precise fiscal period (*FY2025*, *Q1 2026 Form 10-Q*, etc.).

#### Reproducing / Regenerating the Dataset
The dataset can be regenerated or scaled at any time via the unified multi-threaded generator:
```bash
poetry run python -m scripts.generate_golden_dataset
```

The adversarial evaluation harness also tests out-of-corpus resilience: CRAG-enabled queries for companies not in the knowledge base should correctly signal `grounded=False` and invoke web search fallback rather than hallucinating an answer.

### Metrics

All four metrics use the same pattern: structured prompt → LLM call → JSON parsing → score 0–1.

#### Faithfulness
*Are all claims in the answer supported by the retrieved context?*

```
Score = supported_claims / total_claims

1.0 — every claim has a citation-verified source in context
0.0 — all claims appear hallucinated or unverifiable
```

Common failure: LLM uses prior knowledge about Apple/NVIDIA instead of the specific filing.

#### Answer Relevancy
*Does the answer directly address the question?*

```
1.0 — fully and precisely answers what was asked
0.7 — mostly answers but misses some aspect (e.g., asks for Q4 but answers Q3)
0.4 — partially relevant (tangential or incomplete)
0.0 — completely off-topic or non-responsive
```

#### Context Precision
*What fraction of retrieved chunks are relevant to the question?*

```
Score = relevant_retrieved_chunks / total_retrieved_chunks

High precision = retrieval is targeted and efficient
Low precision = lots of noise (wrong company/period, boilerplate) in context
```

#### Context Recall
*Does retrieved context cover the key facts in the ground truth?*

```
Score = covered_ground_truth_statements / total_statements

High recall = all key financial figures are present in context
Low recall = important facts missing (likely due to chunking or retrieval gaps)
```

#### Context Retention & LLM-as-a-Judge Prompt Windows
Unlike basic evaluation setups that pass short UI excerpts (e.g. 250 characters), the evaluation harness passes the **complete retrieved context** (`GenerationResult.retrieved_chunks` and `Citation.full_text`) to the LLM-as-a-Judge. The evaluation prompts allow up to 3,000 characters per chunk, ensuring complex multi-column financial tables and detailed management commentary are fully visible during scoring.

### Running Evaluations

```bash
# Quick smoke test (5 samples, 2 metrics)
poetry run python -m evaluation.harness \
  --n 5 \
  --metrics faithfulness answer_relevancy \
  --name smoke_test

# Full evaluation
poetry run python -m evaluation.harness \
  --metrics faithfulness answer_relevancy context_precision context_recall \
  --name full_eval_v1

# Programmatic — fine-grained control
from evaluation import EvaluationHarness
from evaluation.dataset import get_dataset_by_ticker, get_dataset_subset

harness = EvaluationHarness(pipeline)

# Test only AAPL samples
report = harness.run(
    dataset=get_dataset_by_ticker("AAPL"),
    metrics=["faithfulness", "context_precision"],
    dataset_name="aapl_focused",
)

print(report.summary())
json_path, csv_path = harness.save_report(report)
```

### Interpreting Reports

```
=== EvalReport: full_eval_v1 ===
Timestamp  : 2024-11-15T14:22:31+00:00
Samples    : 16 total, 0 failed (pass rate 100%)
Latency    : 47.3s total
Metrics    :
  faithfulness              ████████████████░░░░  0.81
  answer_relevancy          ██████████████████░░  0.90
  context_precision         ██████████████░░░░░░  0.72
  context_recall            ████████████████░░░░  0.79
```

**Target thresholds** (suggested baselines):

| Metric | Minimum | Target |
|--------|---------|--------|
| faithfulness | 0.75 | ≥0.85 |
| answer_relevancy | 0.80 | ≥0.90 |
| context_precision | 0.65 | ≥0.75 |
| context_recall | 0.70 | ≥0.80 |

**context_precision < 0.65** typically indicates:
- Reranker is disabled or ineffective
- Query transformation producing irrelevant variants
- Metadata filtering not scoped correctly

**faithfulness < 0.75** typically indicates:
- Context window too small (increase `RAG_GENERATION_MAX_CONTEXT_TOKENS`)
- Generation model using prior knowledge instead of retrieved context
- Ungrounded phrases being generated (check `grounded` flag distribution in Prometheus)

### CSV analysis with pandas

```python
import pandas as pd

df = pd.read_csv("data/eval_reports/full_eval_v1_2024-11-15.csv")

# Average score per metric
print(df.groupby("metric")["score"].mean())

# Worst-performing samples
worst = df[df["metric"] == "faithfulness"].nsmallest(5, "score")
print(worst[["sample_id", "score", "reasoning"]])

# Failed samples
failed = df[df["pipeline_failed"] == True]
print(f"Pipeline failures: {len(failed)}")
```

---

## Observability & Metrics

### Prometheus Metrics Reference

All metrics are in the `RAG_REGISTRY` (not the default global registry). Access at `GET /metrics`.

#### HTTP Layer

```promql
# Request rate (requests per minute)
rate(rag_http_requests_total[5m]) * 60

# Error rate (5xx responses)
rate(rag_http_requests_total{status_code=~"5.."}[5m])

# p99 latency
histogram_quantile(0.99, rate(rag_http_request_duration_seconds_bucket[5m]))

# p50/p95/p99 for query endpoint only
histogram_quantile(0.95,
  rate(rag_http_request_duration_seconds_bucket{endpoint="/query"}[5m])
)
```

#### LLM Cost & Tokens

```promql
# Total token consumption rate
rate(rag_llm_tokens_total[1h])

# Tokens by type (prompt vs completion)
rate(rag_llm_tokens_total{token_type="prompt"}[1h])
rate(rag_llm_tokens_total{token_type="completion"}[1h])

# Daily token burn (for cost estimation)
increase(rag_llm_tokens_total[24h])
```

#### Retrieval Quality

```promql
# Average candidates entering reranker
histogram_quantile(0.5, rate(rag_retrieval_candidates_bucket[1h]))

# Average final results returned
histogram_quantile(0.5, rate(rag_retrieval_results_returned_bucket[1h]))

# Context window utilisation
histogram_quantile(0.9, rate(rag_context_tokens_used_bucket[1h]))
```

#### Answer Quality

```promql
# Grounding rate (fraction of answers that are grounded)
rate(rag_grounded_responses_total{grounded="true"}[1h])
/ rate(rag_grounded_responses_total[1h])

# Retrieval failure rate
rate(rag_retrieval_failed_total[1h])

# CRAG action distribution
rate(rag_crag_actions_total[1h])
```

#### Pipeline Latency

```promql
# P95 per-layer latency
histogram_quantile(0.95, rate(rag_pipeline_latency_seconds_bucket{layer="L2"}[5m]))
histogram_quantile(0.95, rate(rag_pipeline_latency_seconds_bucket{layer="L3"}[5m]))
histogram_quantile(0.95, rate(rag_pipeline_latency_seconds_bucket{layer="L4"}[5m]))
```

### Grafana Dashboard Setup

1. Log in at http://localhost:3000 (admin / `GRAFANA_ADMIN_PASSWORD`)
2. Add Prometheus datasource: `http://prometheus:9090`
3. Import dashboard panels using the queries above

Recommended panels:
- **Request Rate** — `rate(rag_http_requests_total[5m])`
- **Error Rate** — `rate(rag_http_requests_total{status_code=~"5.."}[5m])`
- **P95 End-to-End Latency** — histogram_quantile on `/query` endpoint
- **Grounding Rate** — grounded true vs total (area chart)
- **CRAG Action Distribution** — pie chart of correct/ambiguous/incorrect
- **Token Burn Rate** — rate of rag_llm_tokens_total by token_type
- **Per-Layer Latency** — stacked bar of L2/L3/L4 p95

### 🔍 Per-Query Audit Trail (`data/audit_logs/`)

For full transparency, debugging, and offline auditability, every query executed through `/query` or `/query/stream` is automatically recorded into structured log files on disk (controlled by `RAG_AUDIT_ENABLED=true` and `RAG_AUDIT_LOG_DIR=data/audit_logs`).

#### Output Directory Structure
```
data/audit_logs/
├── audit.jsonl                       # Global append-only log (one summary line per request)
└── YYYY-MM-DD/                       # Daily rotating subdirectories
    ├── trace_134501_e74f5de1-9012-4abc-8def-1234567890ab.json  # Full per-query trace JSON
    └── trace_134512_f85a6b7c-1234-5678-9abc-def012345678.json
```

#### What Each Format Contains

1. **Global `audit.jsonl` (Append-Only Summary)**:
   - One line per query execution.
   - Contains request context (`trace_id`, `request_id`, `endpoint`, `received_at`, `question`, `filter`), total latency & layer timing breakdown, token counts (`prompt_tokens`, `completion_tokens`, `total_tokens`), cost estimate (`total_cost_usd`), L2 techniques status, L3 candidate/results summary, and L4 model metadata.
   - Ideal for instant command-line analysis using `grep`, `jq`, or `pandas.read_json("data/audit_logs/audit.jsonl", lines=True)`.

2. **Per-Trace JSON (`data/audit_logs/YYYY-MM-DD/trace_<time>_<id>.json`)**:
   - **Full Detail Audit Record** containing:
     - `schema_version`: `"1.0"`
     - `request`: Full question, API endpoint, timestamp, `request_id`, user-supplied filter.
     - `query_transform`: Latency, techniques used/failed, `original_question`, full **`hyde_document`** text, all **`multi_queries`** variants, and the **`stepback_query`**.
     - `retrieval`: Reranker model, latency, total candidates, and **per-chunk audit records (`chunks`)** containing:
       - `rank`: 1-based final rank
       - `chunk_id` & `parent_id`
       - Filing metadata (`ticker`, `company`, `date`, `fiscal_period`, `section_title`, `doc_type`)
       - Search source (`dense`, `bm25`, `both`)
       - `rrf_score` & `rerank_score`
       - `text_excerpt` (first 300 chars of chunk) and `parent_text_excerpt`
     - `generation`: Latency, model used, prompt/completion/total tokens, `context_chunks_used`, `grounded` status, `citation_count`, and the full **`answer`** text.
     - `llm_calls`: Granular breakdown of every single LLM call (HyDE, Multi-Query, Step-Back, Generation) with model, token counts, latency, and cost in USD.

#### Helpful Audit Inspection Commands

```bash
# View recent summary entries in JSONL
tail -n 5 data/audit_logs/audit.jsonl | jq .

# Search for queries containing 'Walmart' in audit log
grep -i "Walmart" data/audit_logs/audit.jsonl | jq '{question, total_latency_seconds, total_cost_usd}'

# Find all queries where reranking score was low
jq 'select(.retrieval.top_rerank_score < 0.5) | {question, top_rerank: .retrieval.top_rerank_score}' data/audit_logs/audit.jsonl

# Load and analyze in Pandas
python -c "import pandas as pd; df = pd.read_json('data/audit_logs/audit.jsonl', lines=True); print(df[['received_at', 'question', 'total_latency_seconds', 'total_cost_usd']])"
```

### Structured logging

loguru outputs structured log lines that can be forwarded to log aggregators (Loki, Elasticsearch, CloudWatch):

```
2024-11-15 14:22:31.412 | INFO | rag_pipeline | Pipeline complete | grounded=True | citations=3 | tokens=1280 | total=3.18s (L2=0.95s L3=0.62s L4=1.23s)
```

For JSON log format (production):

```python
# main.py or entrypoint
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, serialize=True)  # JSON output
```

---

## Cost Management

### Token cost model

At `gpt-5-mini` pricing:

| Operation | Avg Input Tokens | Avg Output Tokens | Cost per Call |
|-----------|-----------------|-------------------|--------------|
| HyDE | ~150 | ~120 | $0.000063 |
| Multi-Query | ~120 | ~80 | $0.000044 |
| Step-Back | ~120 | ~30 | $0.000024 |
| Generation | ~2500 | ~200 | $0.000330 |
| CRAG grading (×5 chunks) | ~700 | ~30 | $0.000082 |
| **Total per query (with CRAG)** | | | **~$0.000543** |

~1,800 queries per dollar. A typical development session of 100 queries costs ~$0.05.

### Cost reduction strategies

**Disable expensive components for development**:
```dotenv
RAG_CRAG_ENABLED=false          # Saves 5 LLM grading calls
RAG_QUERY_TRANSFORM_MODEL=gpt-5-mini  # Default model
```

**Cache aggressively**:
- Query transform cache (`RAG_QUERY_TRANSFORM_CACHE_SIZE=256`) — eliminates duplicate LLM calls for repeated questions
- Evaluation harness: use `get_dataset_subset(5)` for iteration, full dataset only for release evaluation

**Monitor with Prometheus**:
```promql
# Daily cost estimate (nano pricing)
(
  increase(rag_llm_tokens_total{token_type="prompt"}[24h]) * 0.0000001
  + increase(rag_llm_tokens_total{token_type="completion"}[24h]) * 0.0000004
)
```

---

## Quality Monitoring

### Grounding rate alert

A declining grounding rate indicates retrieval quality degradation (ingestion staleness, Qdrant storage issues, or BM25 corruption).

```promql
# Alert: grounding rate drops below 70% in the last hour
(
  rate(rag_grounded_responses_total{grounded="true"}[1h])
  / rate(rag_grounded_responses_total[1h])
) < 0.70
```

### CRAG action drift

A rising `incorrect` action rate indicates the knowledge base is becoming stale relative to user queries.

```promql
# Alert: CRAG "incorrect" actions exceed 30% of all CRAG calls
rate(rag_crag_actions_total{action="incorrect"}[1h])
/ rate(rag_crag_actions_total[1h])
> 0.30
```

### Retrieval failure spike

Zero-result queries indicate BM25 or Qdrant is unavailable.

```promql
# Alert: retrieval failure rate > 5%
rate(rag_retrieval_failed_total[5m])
/ rate(rag_http_requests_total{endpoint="/query"}[5m])
> 0.05
```

### Scheduled re-evaluation

Run the evaluation harness weekly to catch quality regressions:

```bash
# cron: 0 2 * * 1 (Monday 02:00)
poetry run python -m evaluation.harness \
  --metrics faithfulness answer_relevancy context_precision context_recall \
  --name weekly_$(date +%Y%m%d)

# Compare with previous week
python - << 'EOF'
import json, glob, sys
reports = sorted(glob.glob("data/eval_reports/weekly_*.json"))
if len(reports) < 2: sys.exit(0)
curr = json.load(open(reports[-1]))
prev = json.load(open(reports[-2]))
for m in curr["metric_averages"]:
    delta = curr["metric_averages"][m] - prev["metric_averages"].get(m, 0)
    flag = "⚠️ REGRESSION" if delta < -0.05 else "✅"
    print(f"{flag} {m}: {prev['metric_averages'].get(m, 0):.3f} → {curr['metric_averages'][m]:.3f} (Δ {delta:+.3f})")
EOF
```

---

## Prompt Management

### Prompt versioning

Prompts live in `*/prompts.py` files, versioned with the codebase. Changes to prompts should:

1. Be proposed in a PR with a description of the intended effect
2. Include evaluation results showing improvement (run harness before and after)
3. Document the failure mode being addressed in the commit message

### Prompt testing pattern

```python
# tests/test_generation_prompts.py — structural validation
from generation.prompts import GENERATION_SYSTEM, GENERATION_USER, UNGROUNDED_PHRASES

def test_generation_user_template_formats():
    result = GENERATION_USER.format(context="[1] test chunk", question="test?")
    assert "{context}" not in result
    assert "{question}" not in result

def test_ungrounded_phrases_all_lowercase():
    for phrase in UNGROUNDED_PHRASES:
        assert phrase == phrase.lower(), f"Phrase must be lowercase: {phrase!r}"
```

### A/B testing prompts

To compare two prompt variants:

```python
from evaluation import EvaluationHarness
from evaluation.dataset import GOLDEN_DATASET
from generation import prompts as gen_prompts

# Variant A (baseline)
report_a = harness.run(GOLDEN_DATASET, dataset_name="prompt_A")

# Temporarily swap system prompt
original = gen_prompts.GENERATION_SYSTEM
gen_prompts.GENERATION_SYSTEM = NEW_SYSTEM_PROMPT
report_b = harness.run(GOLDEN_DATASET, dataset_name="prompt_B")
gen_prompts.GENERATION_SYSTEM = original  # restore

# Compare
for m in report_a.metric_averages:
    delta = report_b.metric_averages[m] - report_a.metric_averages[m]
    print(f"{m}: A={report_a.metric_averages[m]:.3f} B={report_b.metric_averages[m]:.3f} Δ={delta:+.3f}")
```

---

## Retrieval Ablations & Architecture Benchmarks

The system provides a comprehensive 6-arm incremental ablation framework (`scripts/run_portfolio_ablations.py`) to quantify the exact marginal gain of each RAG architecture layer:

```
[Arm 1] Dense Vector Search Only (top_k_bm25=0, no transforms, no reranking, no KG, no CRAG)
   ↓ (+ BM25 Keyword Matching + Reciprocal Rank Fusion)
[Arm 2] Hybrid Dense + BM25 Search
   ↓ (+ HyDE Passage Synthesis + Multi-Query Expansion + Step-Back Abstraction)
[Arm 3] Hybrid Search + Query Transformations
   ↓ (+ FlashRank Cross-Encoder ms-marco-MiniLM-L-12-v2)
[Arm 4] Hybrid Search + Transforms + FlashRank Reranker
   ↓ (+ GraphRAG Entity Matching & Multi-Hop Context Injection)
[Arm 5] Hybrid Search + Transforms + Reranker + Knowledge Graph
   ↓ (+ Relevance Grader + Dynamic Web-Search Fallback)
[Arm 6] Full Stack Pipeline (+ CRAG Corrective Loop)
```

### 6-Arm Architecture Reference

| Arm # | Architecture Arm | Active Subsystems & Configurations |
|:---:|:---|:---|
| **1** | `1. Base Naive RAG (Dense Only)` | Dense vector retrieval only (`top_k_bm25=0`, reranker disabled, transforms disabled, KG disabled, CRAG disabled) |
| **2** | `2. + BM25 Sparse (Hybrid RRF)` | Dense vector + BM25 keyword matching fused via Reciprocal Rank Fusion (k=60) |
| **3** | `3. + Query Transform` | Hybrid search + HyDE synthesis, 3× Multi-Query expansion, and Step-Back prompting |
| **4** | `4. + FlashRank Reranker` | Hybrid search + Transforms + FlashRank cross-encoder (`ms-marco-MiniLM-L-12-v2`) |
| **5** | `5. + Knowledge Graph (GraphRAG)` | Hybrid + Transforms + Reranker + GraphRAG entity context injection |
| **6** | `6. Full Stack (+ CRAG Fallback)` | Complete pipeline with Corrective RAG (CRAG) web-search fallback |

### Running Ablation Studies

```bash
# Run all 6 arms across the ENTIRE golden dataset (all questions)
poetry run python scripts/run_portfolio_ablations.py --all

# Run all 6 arms on a custom sample size (e.g. 5 samples for rapid validation)
poetry run python scripts/run_portfolio_ablations.py -n 5

# Force fresh re-evaluation bypassing cached sample checkpoints
poetry run python scripts/run_portfolio_ablations.py --all --no-cache

# Run a single arm in isolation (e.g. Arm 1 only)
poetry run python scripts/run_portfolio_ablations.py --arm 1

# Run specific arms (e.g. Arm 1, Arm 2, and Arm 4)
poetry run python scripts/run_portfolio_ablations.py --arms 1 2 4
```

### Automated Invariant Verification (Zero Leakage Check)

To guarantee that components are strictly isolated during ablation testing (e.g. ensuring Arm 1 does not execute FlashRank reranking or query expansion), use `scripts/verify_ablation_isolation.py`. It inspects per-sample execution telemetry:

```bash
# Run a fresh 5-sample verification test across all 6 arms and assert structural invariants
poetry run python scripts/verify_ablation_isolation.py --run -n 5

# Audit existing saved results in data/ablation_results/
poetry run python scripts/verify_ablation_isolation.py
```

### Custom Pairwise A/B Experiments

To run head-to-head A/B experiments between arbitrary pipeline configurations:

```bash
poetry run python -m experiments.retrieval_experiment \
  --baseline '{"top_k_final": 5, "reranker_enabled": false}' \
  --variant  '{"top_k_final": 5, "reranker_enabled": true}' \
  --n 10 \
  --name "reranker_ablation" \
  --save
```

---

## Data Quality

### Filing freshness

The ingestion pipeline uses automatic store-state inspection — it never re-processes chunks or files already indexed in Qdrant, BM25, and Knowledge Graph. To ingest new filings (e.g., after a new earnings cycle):

```bash
# Run download (only fetches new filings not on disk)
poetry run python -m ingestion.download_filings

# Run pipeline (auto store-state inspection skips already-indexed chunks; only processes new ones)
poetry run python -m ingestion.pipeline
```

### Coverage validation

```bash
poetry run python scripts/inspect_index.py
```

Check the output for:
- All 10 tickers have expected filing counts (typically 4–8 per year)
- BM25 corpus and Qdrant point counts match (`OK: BM25 and Qdrant counts are consistent`)
- No gaps in fiscal periods (missing Q3 filings, etc.)

### Chunk quality signals

Warning signs in the inspect output:
- `Avg chunk tokens < 50` — chunks too small, likely parsing errors or empty sections
- `Max chunk tokens > 500` — token budget enforcement may be broken
- Single ticker with 0 chunks — download or parsing failure for that company

### Re-index with quality filters

If parsing quality is poor for a specific company, debug with:

```python
from ingestion.parser import parse_html
from pathlib import Path

doc = parse_html(Path("data/transcripts/AAPL_2024-10-31_0001234567.htm"))
print(f"Word count: {len(doc.raw_text.split())}")
print(f"Section count: {len(doc.sections)}")
print(f"First section: {doc.sections[0][:200]}")
```

---

## Incident Response

### High error rate (>5% 5xx responses)

1. Check `/health` endpoint — identify which component is degraded
2. Check Qdrant reachability: `curl http://localhost:6333/healthz`
3. Check BM25 file exists: `ls -la data/bm25_index.pkl`
4. Check OpenAI status: https://status.openai.com/
5. Review recent deployments — roll back if issue coincides with a deploy

### Hallucination reports from users

1. Collect the question and answer
2. Check `grounded` flag in API response — was it `false`?
3. Run in verbose mode to inspect retrieved context
4. Check if filing for the relevant company/period is in the index
5. If filing is missing → run `download_filings` + `pipeline`
6. If filing is present but wrong context retrieved → likely retrieval quality issue → run evaluation harness

### Cost spike

1. Check Prometheus: `increase(rag_llm_tokens_total[1h])`
2. Identify if CRAG grading is being triggered excessively: `rag_crag_actions_total{action="incorrect"}`
3. Check if query transform cache is working: high cache miss rate → many duplicate queries
4. Verify no infinite retry loops in tenacity (check logs for repeated retry warnings)
