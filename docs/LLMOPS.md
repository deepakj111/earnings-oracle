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
3. **Quality signals** — grounding flags, citation counts, Agentic Reflexion self-corrections, and calibrated abstention rates
4. **Latency measurements** — per-layer timing for bottleneck identification

The evaluation harness closes the loop: offline evaluation against a golden dataset detects quality regressions before they reach production.

---

## Evaluation Framework

### Golden Dataset

`evaluation/dataset.py` loads the **129-question audited golden dataset** drawn from real SEC 10-K Annual Reports and 10-Q Quarterly Filings across four Fortune 50 companies (exactly 3 questions per filing across all 43 filings):

| Company | Ticker | Sector | # Samples | Downloaded Filings |
|---------|--------|--------|:---:|:---:|
| NVIDIA | NVDA | Technology / Semiconductors | **33** | 11 filings (3 × 10-K, 8 × 10-Q) |
| Walmart | WMT | Consumer Staples / Retail | **33** | 11 filings (3 × 10-K, 8 × 10-Q) |
| Netflix | NFLX | Communication Services / Streaming | **33** | 11 filings (3 × 10-K, 8 × 10-Q) |
| UnitedHealth Group | UNH | Healthcare / Managed Care | **30** | 10 filings (2 × 10-K, 8 × 10-Q) |
| **Total Benchmark** | — | **4 Cross-Sector Leaders** | **129** | **43 Filings (FY 2023–2026)** |

Questions span both annual (10-K) and quarterly (10-Q) filings and adhere to a structured **4-pillar distribution**:

- **MD&A & Strategic Initiatives (~35%)**: Operational roadmaps, technological modernization (e.g. Blackwell GPU rollout for NVDA, supply chain automation for WMT, live events & ad-tier for NFLX, Optum Care value-based care for UNH).
- **Risk Factors & Regulatory Disclosures (~25%)**: US export control restrictions, cyber incident recovery, healthcare/PBM regulation, retail shrink, and content amortization.
- **Segment & Core Financials (~30%)**: Segment revenue breakdowns, operating margin shifts, regional membership/ARM metrics.
- **Capital Allocation & Cash Flow (~10%)**: Share buybacks, capex allocation, and liquidity management.

All questions are **100% self-contained**, explicitly naming the target company and precise fiscal period (*FY2025*, *Q1 2026 Form 10-Q*, etc.).

The dataset size scales linearly with the number of ingested filings. Run `--max-files N` to generate a larger dataset for more statistically robust ablation studies.

#### Reproducing / Regenerating the Dataset
The dataset can be regenerated or scaled at any time via the unified multi-threaded generator:
```bash
poetry run python -m scripts.generate_golden_dataset
```

The evaluation harness also tests out-of-corpus resilience: queries for companies or metrics not present in the SEC filing corpus correctly trigger Calibrated Abstention and `grounded=False` rather than hallucinating an answer.

### Structured Output Evaluation Mode (2026 JSON-Schema Standard)

The evaluation harness natively supports **Structured Output Evaluation Mode**, toggled via:
```bash
export RAG_GENERATION_STRUCTURED_OUTPUT=true
```

When enabled, the generator switches from prose synthesis to OpenAI JSON-Schema structured mode (`GENERATION_SYSTEM_STRUCTURED`). The LLM emits a strictly formatted JSON object:
```json
{
  "answer": "NVIDIA's Data Center revenue grew 112% year-over-year to $26,044 million [1]...",
  "citations": [1, 2],
  "calculations": [
    {
      "expression": "(26044 - 12280) / 12280 * 100",
      "result": 112.08,
      "formatted": "+112.08%"
    }
  ],
  "confidence_rationale": "High confidence supported by audited 10-K segment tables.",
  "grounded": true,
  "abstained": false
}
```

**Key Advantages for LLMOps & Continuous Integration:**
1. **Deterministic PAL Auditing**: Math expressions and percentage changes are separated into a dedicated `calculations` array, allowing automated testing of every arithmetic operation without regex extraction brittleness.
2. **Zero Regex Ambiguity**: Citations are emitted as discrete index integers, eliminating markdown format discrepancies across model versions.
3. **Automated Abstention Classification**: Distinguishes between model-detected insufficient context (`abstained: true`) versus ungrounded hallucinations (`grounded: false`), preventing false-positive scoring penalties during out-of-domain evaluation suites.

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

#### Architectural Separation: Evaluation Harness vs. Portfolio Ablation Studies

In our LLMOps framework, **Release Evaluation** and **Component Ablation** are deliberately decoupled:

| Capability | Evaluation Harness (`evaluation/harness.py`) | Portfolio Ablations (`scripts/run_portfolio_ablations.py`) |
| :--- | :--- | :--- |
| **Role** | **Operational Release Gate & CI/CD** | **Scientific Research & Architecture Discovery** |
| **Tested Surface** | **Single pipeline** (active production configuration) | **7–8 modular pipelines** (baseline + isolated components + Pareto tiers) |
| **Objective** | Validate that the current system meets accuracy & safety SLAs | Empirically measure marginal $\Delta\text{Faithfulness}$ & $\Delta\text{Latency}$ per feature |
| **Cost Profile** | $1\times$ API overhead (fast, suitable for automated CI/CD) | $8\times$ API overhead (comprehensive, reserved for design milestones) |
| **Output** | `EvalReport` JSON/CSV in `data/eval_reports/` (Audit report in [`docs/BENCHMARKS.md`](file:///home/deepak/rag-project/docs/BENCHMARKS.md)) | Multi-arm delta tables & `ablation_report.md` in `data/ablation_results/` |

**The Optimization & Guardrail Lifecycle:**
1. **Initial Ceiling Test**: Running `evaluation.harness` on the fully integrated pipeline establishes the system's quality ceiling.
2. **Dissection & Discovery**: Running `run_portfolio_ablations.py --isolated` isolates each component against a pure dense baseline, discovers true marginal lift vs. latency cost, and dynamically synthesizes **Tier 1 (Fast: <1.0s overhead)** and **Tier 2 (SOTA: all net-positive components)**.
3. **Production Guardrail**: The winning tier is locked into `config/settings.py` / `.env`, and `evaluation.harness` serves as the automated daily CI/CD release gate to prevent regressions.

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

#### Answer Quality & Grounding

```promql
# Grounding rate (fraction of answers that are verified and grounded)
rate(rag_grounded_responses_total{grounded="true"}[1h])
/ rate(rag_grounded_responses_total[1h])

# Retrieval failure rate
rate(rag_retrieval_failed_total[1h])
```

#### Pipeline Latency

```promql
# P95 per-layer latency
histogram_quantile(0.95, rate(rag_pipeline_latency_seconds_bucket{layer="L2"}[5m]))
histogram_quantile(0.95, rate(rag_pipeline_latency_seconds_bucket{layer="L3"}[5m]))
histogram_quantile(0.95, rate(rag_pipeline_latency_seconds_bucket{layer="L4"}[5m]))
```

### 🛰️ OpenTelemetry (OTel) Distributed Tracing (Jaeger)

The system is instrumented with native OpenTelemetry tracing (`observability/otel.py`) exporting OTLP spans over gRPC to Jaeger (`localhost:4317`):

- **Jaeger Web UI**: [http://localhost:16686](http://localhost:16686)
- **Spans captured**:
  - `FinancialRAGPipeline.ask` (root span with user query, filter metadata, request ID)
  - `QueryRouter.route` (intent classification, ticker extraction)
  - `SemanticCache.get_cached_response` (vector similarity cache hit/miss)
  - `QueryTransformer.transform` (HyDE, Multi-Query, Step-Back)
  - `HybridSearcher.retrieve` (Qdrant dense, BM25 sparse, SEC FactStore, GraphRAG, and FlashRank cross-encoder reranking)
  - `Generator.generate` (LLM prompt synthesis, token consumption, context chunk injection)
- **Flamegraphs**: Provide microsecond-level visibility into query execution waterfalls to immediately diagnose latency bottlenecks across network, embedding API, or local cross-encoders.

### Grafana Dashboard Setup

1. Log in at http://localhost:3000 (admin / `GRAFANA_ADMIN_PASSWORD`)
2. Add Prometheus datasource: `http://prometheus:9090`
3. Import dashboard panels using the queries above

Recommended panels:
- **Request Rate** — `rate(rag_http_requests_total[5m])`
- **Error Rate** — `rate(rag_http_requests_total{status_code=~"5.."}[5m])`
- **P95 End-to-End Latency** — histogram_quantile on `/query` endpoint
- **Grounding Rate** — grounded true vs total (area chart)
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

#### Model Tier Pricing Reference (per 1M Tokens)

| Model Tier | Role in Architecture | Input / 1M Tokens | Output / 1M Tokens |
|:---|:---|:---:|:---:|
| `gemini-2.5-flash` (Default) | L1 Router, L2 Transforms, L4 Generation, L5 Verification | $0.075 | $0.30 |
| `gemini-2.5-pro` | Deep Financial Synthesis & Multi-Filing Reasoning | $1.25 | $5.00 |
| `text-embedding-004` (Default) | L3 Dense Embedding & Semantic Caching (768-dim) | $0.025 | N/A |
| `gpt-5-mini` | OpenAI Alternative: L2 Transforms, L5 Verification | $0.15 | $0.60 |
| `gpt-5` | OpenAI Alternative: L4 High-Fidelity Generation | $1.25 | $5.00 |
| `text-embedding-3-small` | OpenAI Alternative: L3 Dense Embedding (1536-dim) | $0.02 | N/A |

#### Typical Per-Query Cost Breakdown by Pipeline Stage

##### Google Cloud Vertex AI ADC (Production Default)
| Pipeline Stage | Model Tier | Avg Input Tokens | Avg Output Tokens | Cost per Call |
|:---|:---|:---:|:---:|:---:|
| **L2: HyDE Expansion** | `gemini-2.5-flash` | ~150 | ~120 | $0.000047 |
| **L2: Multi-Query Variants** | `gemini-2.5-flash` | ~120 | ~80 | $0.000033 |
| **L2: Step-Back Abstraction** | `gemini-2.5-flash` | ~120 | ~30 | $0.000018 |
| **L3: Query Embeddings** (original + variants) | `text-embedding-004` | ~180 | 0 | $0.000005 |
| **L4: LLM Generation** | `gemini-2.5-flash` | ~2,500 | ~350 | $0.000293 |
| **L5: Claim-Level NLI Verification** | `gemini-2.5-flash` | ~800 | ~60 | $0.000078 |
| **Total per Query (Standard Uncached)** | | | | **~$0.000474** |
| **Total per Query (Semantic Cache Hit)** | `text-embedding-004` (1 lookup) | ~25 | 0 | **<$0.000001** |

*Note: With Google Cloud Vertex AI ADC defaults, query costs are ~10× lower than frontier alternatives (~2,100 queries per dollar).*

##### OpenAI Alternative Setup
| Pipeline Stage | Model Tier | Avg Input Tokens | Avg Output Tokens | Cost per Call |
|:---|:---|:---:|:---:|:---:|
| **L2: HyDE Expansion** | `gpt-5-mini` | ~150 | ~120 | $0.000095 |
| **L2: Multi-Query Variants** | `gpt-5-mini` | ~120 | ~80 | $0.000066 |
| **L2: Step-Back Abstraction** | `gpt-5-mini` | ~120 | ~30 | $0.000036 |
| **L3: Query Embeddings** (original + variants) | `text-embedding-3-small` | ~180 | 0 | $0.000004 |
| **L4: LLM Generation** | `gpt-5` | ~2,500 | ~350 | $0.004875 |
| **L5: Claim-Level NLI Verification** | `gpt-5-mini` | ~800 | ~60 | $0.000156 |
| **Total per Query (Standard Uncached)** | | | | **~$0.005230** |

#### Monthly Operating Cost Projection at Scale (10,000 Queries / Day)

Assuming production enterprise workload of 300,000 queries/month across analytical users (default Vertex AI ADC stack):

| Scenario | Semantic Cache Hit Rate | Monthly LLM Generation Spend | Monthly Embedding Spend | Total Monthly Cost | Cost per 1k Queries |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Worst-Case (Zero Cache)** | 0% | $142.20 | $1.50 | **$143.70** | $0.48 |
| **Baseline Production** | 25% | $106.65 | $1.44 | **$108.09** | $0.36 |
| **High Cache Re-use (Earnings Season)** | 45% | $78.21 | $1.35 | **$79.56** | $0.27 |

#### Key Cost Optimization Levers

1. **Selective HyDE Gating**: `QueryRouter` bypasses HyDE for specific company/metric queries (e.g. "What was NVDA Q2 FY2026 revenue?"), saving 270 tokens per query on ~60% of traffic.
2. **Contextual Compression**: When enabled (`RAG_CONTEXT_COMPRESSION_ENABLED=true`), extracts only relevant financial rows from parent chunks, reducing L4 prompt tokens by 35–50% (~$0.0012 saved per call).
3. **Semantic Cache Invalidation Granularity**: Precise ticker-level invalidation prevents full-cache flushes when a single company files a 10-Q, maintaining a 30–45% cache hit rate.

### Cost reduction strategies

**Disable expensive components for development**:
```dotenv
# Fast development cycle — disable heavy reranking or late interaction if needed
RAG_CONTEXT_COMPRESSION_ENABLED=false
RAG_QUERY_TRANSFORM_MODEL=gemini-2.5-flash  # Fast path model (or gpt-5-mini)
```

**Cache Aggressively & Manage Invalidation**:
- **Query Transform LRU Cache** (`RAG_QUERY_TRANSFORM_CACHE_SIZE=256`): In-memory caching eliminates duplicate LLM expansions for identical queries during active analytical sessions.
- **Qdrant Semantic Cache** (`retrieval/semantic_cache.py`, ADR-007): Caches query embeddings and validated generation results in Qdrant collection `semantic_cache`. Fast cosine lookup ($\text{score} \ge 0.96$) skips retrieval and generation entirely, yielding sub-50ms responses for recurring user queries.
- **Targeted Ticker Invalidation on Ingestion**: When new quarterly 10-Q or annual 10-K filings are ingested, stale cached outputs must not persist. The pipeline triggers `SemanticCache.invalidate_ticker(ticker)` to atomically delete all points matching `{"tickers": ticker.upper()}`:
  ```python
  # Programmatic ticker invalidation trigger
  from retrieval.semantic_cache import SemanticCache

  async def on_new_filing_ingested(ticker: str):
      cache = SemanticCache()
      await cache.invalidate_ticker(ticker)
  ```
- **TTL Expiration Policy**: Entries older than `ttl_hours` (default 24h) are rejected on read and lazily purged.
- **Evaluation Harness**: Use `get_dataset_subset(5)` for rapid development iteration; run the full 129-QA golden dataset only for formal verification runs.

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

### Ungrounded response and abstention drift

A rising ungrounded response rate indicates the knowledge base is missing context or retrieval relevance has degraded.

```promql
# Alert: ungrounded responses exceed 25% of all queries over 1h
rate(rag_grounded_responses_total{grounded="false"}[1h])
/ rate(rag_grounded_responses_total[1h])
> 0.25
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

### Online Quality Monitoring & Continuous SLI Auditing (2026 SOTA)

While offline evaluation runs on a weekly schedule against the static golden dataset, real-time production quality is audited using `evaluation/online_monitor.py`. This component ingests live trace events from `data/audit_logs/audit.jsonl` to calculate rolling SLIs and detect quality regressions immediately:

```bash
# Run online audit over the most recent 100 production requests
poetry run python -m evaluation.online_monitor \
  --audit-log data/audit_logs/audit.jsonl \
  --sample-size 100 \
  --output data/online_quality.jsonl
```

#### Production SLI Targets & Automated Alerting

| Quality Metric / SLI | Target Threshold | Description | Remediation if Breached |
| :--- | :--- | :--- | :--- |
| **Grounded Rate** | $\ge 90.0\%$ | Fraction of production responses verified as grounded in filing context | Audit chunk retrieval coverage; check if filing ingest is missing recent periods |
| **Citation Coverage Rate** | $\ge 95.0\%$ | Fraction of answers containing verifiable in-line `[n]` citations | Review prompt adherence; verify generator temperature is strictly 0.0 |
| **Faithfulness Proxy Score** | $\ge 0.880$ | Joint score of claim-level entailment and citation grounding | Run offline evaluation harness; inspect Reflexion critique loop |
| **p95 Latency SLA** | $\le 4.0\text{s}$ | 95th percentile end-to-end response time | Check Qdrant cluster latency, HyDE gating rate, and OpenAI API latency |

The monitor automatically logs `[CRITICAL]` warnings and can be scheduled via cron or run as an asynchronous background worker in FastAPI to write metrics for Prometheus/Grafana scrapers.

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

The system provides a rigorous, causal ablation framework (`scripts/run_portfolio_ablations.py`) to quantify the exact marginal gain of each RAG architecture layer without confounding variables:

### 1. Isolated Single-Component Arms (`--isolated`)

To establish true unconfounded causal attribution, each isolated arm enables **exactly one** feature over the dense-only baseline while keeping all other components disabled:

| Isolated Arm | Feature Tested | All Other Components |
|:---|:---|:---|
| `bm25` | BM25 keyword matching + hybrid RRF (k=60) | Disabled |
| `querytransform` | HyDE + 3× Multi-Query + Step-Back prompting | Disabled |
| `reranker` | FlashRank cross-encoder (`ms-marco-MiniLM-L-12-v2`) | Disabled |
| `graphrag` | Knowledge Graph entity extraction & context injection | Disabled |
| `pal_math` | Sentence NLI Grounding Verification & PAL Math Evaluator | Disabled |

### 2. Dynamic Pareto Production Tiers

Using the causal results of isolated arms, the pipeline dynamically filters out any regressions and constructs optimal production tiers:
- **Tier 1 (Fast)**: Stacks all positive components adding <1.0s latency overhead (~2.0s P95).
- **Tier 2 (Full Production SOTA)**: Stacks all positive components including Calibrated Abstention & PAL math (~4.5s P95).

### 3. Three-Table Reporting Architecture

The ablation harness produces `data/ablation_results/ablation_report.md`:
- **Table 1**: Absolute metrics per dynamic Pareto arm + 95% Bootstrap Confidence Intervals (1 000 iterations).
- **Table 2**: (Legacy waterfall incremental progression).
- **Table 3**: Isolated single-component $\Delta$ vs. dense-only baseline for unconfounded causal attribution.

### Running Ablation Studies

```bash
# 1. Run all isolated arms on the dataset to populate causal benchmarks
poetry run python scripts/run_portfolio_ablations.py --isolated --all

# 2. Run on 5 samples for rapid validation
poetry run python scripts/run_portfolio_ablations.py --isolated -n 5

# 3. Dynamically generate and run optimal Pareto tiers
poetry run python scripts/run_portfolio_ablations.py

# 4. Regenerate Markdown report from cache (zero LLM API calls)
poetry run python scripts/run_portfolio_ablations.py --report-only
```

### Automated Invariant Verification (Zero Leakage Check)

To mathematically assert that components are strictly isolated during ablation testing, use `scripts/verify_ablation_isolation.py`. It inspects per-sample execution telemetry:

```bash
# Run a fresh 5-sample verification test and assert structural invariants
poetry run python scripts/verify_ablation_isolation.py --run -n 5

# Audit existing saved results in data/ablation_results/
poetry run python scripts/verify_ablation_isolation.py
```

### Custom Pairwise A/B Experiments

To run head-to-head A/B experiments with paired Wilcoxon signed-rank tests and Student's t-tests:

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
2. Identify if ungrounded responses or Reflexion loops are firing frequently: `rag_grounded_responses_total{grounded="false"}`
3. Check if query transform cache is working: high cache miss rate → many duplicate queries
4. Verify no infinite retry loops in tenacity (check logs for repeated retry warnings)

---

## Production Hallucination Fences & Verification

### 1. Numerical Hallucination Fence (`generation/hallucination_fence.py`)
In financial enterprise applications, quantitative metrics (currencies, growth rates, margins) require strict grounding. The Numerical Hallucination Fence automatically:
- Extracts all numerical values, percentages, and multiples (excluding bare calendar years).
- Cross-references each figure against retrieved context chunks and verified Program-Aided Language (PAL) calculation outputs.
- Flags ungrounded numerical claims as `numerical_hallucination_warnings` in the `GenerationResult`.

### 2. Citation Integrity Validator (`generation/citation_validator.py`)
Comparative financial questions (e.g. comparing Microsoft and Apple revenues) are prone to cross-entity attribution errors. The Citation Integrity Validator:
- Inspects sentence-level company mentions via the dynamic `CompanyRegistry`.
- Flags instances where a sentence discussing Company A cites a filing from Company B (`citation_integrity_warnings`).

### 3. Reflexion Loop Observability & Depth Guards
When an initial answer fails grounding verification:
- The pipeline triggers an Agentic Reflexion loop with bounded depth (`max_reflexion_attempts = 1`).
- Both the initial generation span and the reflexion attempt are explicitly recorded in OpenTelemetry and `PipelineTrace`.
- Prevents unbounded latency or token consumption while retaining full auditability of the model's self-correction trajectory.

---

## Covariate Shift & Semantic Drift Monitoring

### Maximum Mean Discrepancy (MMD) Drift Detection (`evaluation/drift_detector.py`)
Over time, incoming production query distributions diverge from static evaluation benchmarks. The system implements non-parametric distribution shift detection via MMD:

$$\text{MMD}^2(P, Q) = \frac{1}{m^2}\sum_{i,j} k(x_i, x_j) - \frac{2}{mn}\sum_{i,j} k(x_i, y_j) + \frac{1}{n^2}\sum_{i,j} k(y_i, y_j)$$

Run automated drift audits:
```bash
poetry run python -m evaluation.drift_detector \
    --current data/audit_logs/query_embeddings.jsonl \
    --baseline data/golden_dataset.json \
    --threshold 0.05 \
    --kernel rbf
```
An alert is raised when `mmd_score >= threshold`, signalling the ML engineering team to ingest new filings or update golden evaluation benchmarks.
