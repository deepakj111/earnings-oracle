# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.7.x   | :white_check_mark: |
| 0.6.x   | :white_check_mark: |
| 0.5.x   | :white_check_mark: |
| 0.4.x   | :white_check_mark: |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :white_check_mark: |

---

## Security Architecture & Design Principles

The Financial RAG System is engineered with security-in-depth across the application lifecycle:

### 1. AST-Sandboxed Financial Math (No `eval()`)
Financial calculations in `generation/calculator.py` are executed using a restricted Python Abstract Syntax Tree (AST) evaluator (`SafeFinancialCalculator`):
- Explicitly whitelists only arithmetic operators (`+`, `-`, `*`, `/`, `**`).
- Strictly prohibits `eval()`, `exec()`, `__import__`, variable mutations, attribute lookups, and built-in functions.
- Prevents arbitrary code execution even if the LLM generates malicious or malformed code.

### 2. Multi-Tier Input Guardrails (`query/guardrails.py`)
- **Prompt Injection Prevention**: Blocks adversarial prompt override attempts, role-reversal jailbreaks (DAN mode), and raw instruction token injections (`<|im_start|>`, `<<SYS>>`).
- **PII Detection & Redaction**: Automatically identifies and redacts Social Security Numbers (SSN) and credit card numbers verified via the Luhn mod-10 algorithm before passing to LLMs.
- **Token Budget & DoS Shield**: Restricts input token lengths via `tiktoken` to prevent context poisoning and token consumption denial-of-service.

### 3. API Hardening & Rate Limiting (`api/middleware.py`)
- **Per-IP Sliding Window Rate Limiting**: Enforces request caps with RFC-compliant headers (`X-RateLimit-*`) and `429 Too Many Requests` status codes.
- **Request Correlation**: Stamps every request with a cryptographically random UUID4 `X-Request-ID`.
- **CORS Configuration**: Configurable origin controls.

### 4. Multi-User Session Isolation & Tenant Boundaries (`api/chat_store.py`)
- **Tenant & User Scoping**: Chat sessions (`/sessions`) are strictly scoped by `user_id` and `tenant_id` (derived from authenticated JWT claims or `X-User-ID` / `X-Tenant-ID` headers).
- **Cross-Session Leak Prevention**: Direct object references (`GET /sessions/{session_id}`) verify ownership against the active user context. Cross-user message enumeration or thread tampering is rejected with `404 Not Found`.
- **Sliding Window Token Bounds**: Chat message history is bounded and pruned in memory / storage to prevent unbounded context growth or memory exhaustion.

### 5. SEC Fair Access Governance (`config/settings.py`)
- **Mandatory User-Agent Verification**: Rejects blank or default placeholder User-Agents (`Your Name your@email.com`) during startup validation. Requires valid declared identity strings (`FirstName LastName email@domain.com`) conforming to SEC EDGAR Fair Access specifications, protecting infrastructure from IP bans and automated throttling.

### 6. Production Error Masking & Key Sanitization (`api/errors.py`)
- **Sanitized Exception Handlers**: In production mode, raw upstream provider error messages, database credentials, or system tracebacks are intercepted and replaced with opaque correlation IDs (`X-Request-ID`).
- **Credential Redaction**: LLM client logs and OpenTelemetry span attributes filter out authorization tokens, bearer headers, and API keys.

### 7. Container & Process Isolation
- **Non-Root Execution**: Docker container runs under an unprivileged user (`appuser` with UID 10001).
- **Zero Shell Subprocess Invocations**: No usage of `subprocess.Popen(..., shell=True)` anywhere in the codebase.
- **Hermetic Secret Management & Zero-Key ADC**: Production workloads and developers authenticate via Google Cloud Application Default Credentials (`gcloud auth application-default login`), eliminating static API keys entirely. Any alternative external vendor keys are strictly sourced from environment variables, never hardcoded or committed to git.

### 8. Automated CI Security Gates
- **Bandit SAST**: Scans all Python ASTs on every pull request for high/medium security issues (`poetry run bandit -r . -c pyproject.toml -ll`).
- **pip-audit Dependency Scanning**: Continuously audits installed dependencies against the Python Packaging Advisory Database (PyPA) and OSV for known CVEs (`poetry run pip-audit`).
- **TruffleHog**: Deep git commit history secret scanner fails CI builds if unencrypted tokens or credentials are staged.

### 9. BM25 Index Serialization Threat Model & Data Provenance
Inverted indices in `retrieval/searcher.py` and `ingestion/pipeline.py` utilize `rank_bm25` (BM25Okapi), which serializes internal inverted term frequencies via Python `pickle` (marked with `# nosec B301`):
- **Local Hermetic Boundary**: Pickled index artifacts (`data/bm25_index.pkl`) reside strictly on the local ephemeral container filesystem or private volume mount. The application rejects deserialization of any index payload received over network sockets, API endpoints, or unauthenticated user uploads.
- **Data Provenance**: Ingested corpora are downloaded exclusively over TLS from authoritative SEC EDGAR endpoints (`data.sec.gov`), preventing adversarial data injection into the offline tokenizer.
- **Architectural Mitigation Roadmap**: In multi-tenant environments with distributed storage workers, the architecture supports replacing standard pickle serialization with cryptographic SHA-256 signed payloads or sparse scipy CSR matrices / numpy arrays.

---

## Reporting a Vulnerability

If you discover a potential security vulnerability, please report it responsibly:
- **Email**: Reach out to the maintainer directly at `deepakj111@users.noreply.github.com`.
- **Details**: Please include a description of the issue, affected modules, reproduction steps or proof-of-concept payload, and any suggested fixes.
- **Response Target**: We strive to acknowledge reports within 48 hours and provide a patch timeline within 7 business days.

Please do not open public GitHub issues for undisclosed security vulnerabilities.
