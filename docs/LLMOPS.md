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

`evaluation/dataset.py` contains 16 curated QA pairs drawn from real SEC 8-K earnings filings:

| Company | # Samples | Topics |
|---------|-----------|--------|
| Apple | 3 | Revenue, Services, EPS |
| NVIDIA | 2 | Data Center revenue, Total revenue |
| Microsoft | 2 | Intelligent Cloud, Total revenue |
| Amazon | 1 | AWS revenue |
| Meta | 2 | Ad revenue, DAP |
| JPMorgan | 1 | NII |
| Tesla | 2 | Revenue, Deliveries |
| Walmart | 1 | Total net sales |
| ExxonMobil | 1 | Earnings |
| Out-of-scope | 1 | Berkshire (should return ungrounded) |

The adversarial out-of-scope sample tests that the system correctly signals `grounded=False` rather than hallucinating an answer for a company not in the knowledge base.

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

At gpt-4.1-nano pricing (`$0.10/1M input, $0.40/1M output`):

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
RAG_QUERY_TRANSFORM_MODEL=gpt-4.1-nano  # Already cheapest tier
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

## Retrieval Ablations

Use environment variable overrides to ablate components and measure impact on evaluation metrics:

| Ablation | Config | Purpose |
|----------|--------|---------|
| Dense-only retrieval | `RAG_RETRIEVAL_TOP_K_BM25=0` | Measure BM25 contribution |
| BM25-only retrieval | `RAG_RETRIEVAL_TOP_K_DENSE=0` | Measure dense search contribution |
| No reranking | `RAG_RERANKER_ENABLED=false` | Measure reranker contribution |
| No HyDE | Patch `_run_hyde` to return original | Measure HyDE contribution |
| No step-back | Patch `_run_stepback` to return original | Measure step-back contribution |
| No parent fetch | `RAG_RETRIEVAL_PARENT_FETCH=false` | Measure parent context contribution |
| No valley ordering | Patch `_valley_reorder` to return input | Measure lost-in-middle mitigation |

Example ablation run:

```bash
# Run with reranker disabled
RAG_RERANKER_ENABLED=false poetry run python -m evaluation.harness \
  --name ablation_no_reranker

# Compare with baseline
python compare_reports.py data/eval_reports/baseline_*.json data/eval_reports/ablation_no_reranker_*.json
```

---

## Data Quality

### Filing freshness

The ingestion pipeline is checkpoint-based — it never re-processes files it has already indexed. To ingest new filings (e.g., after a new earnings cycle):

```bash
# Run download (only fetches new filings not on disk)
poetry run python -m ingestion.download_filings

# Run pipeline (checkpoint skips already-indexed files; only processes new ones)
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
