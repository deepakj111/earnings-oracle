# CI/CD and Automation

> Documentation for GitHub Actions workflows, pre-commit hooks, and local quality gates.

---

## Overview

The Financial RAG System employs a multi-stage automation pipeline to maintain 2026 production standards. This ensures that every commit is linted, type-checked, and tested before merging.

## 1. Local Persistence: Pre-commit Hooks

We use `pre-commit` to catch errors before they are pushed to the repository.

### Setup
```bash
poetry run pre-commit install
```

### Active Hooks
- **Ruff**: Modern Python linting and formatting (auto-fixes `isort` and simple PEP8 issues).
- **Mypy**: Strict static type analysis.
- **Bandit**: Security scanning for common vulnerabilities.
- **TruffleHog**: Secret scanning to prevent API key leaks.
- **Pytest**: (Optional but recommended) Run unit tests before commit.

To run manually:
```bash
poetry run pre-commit run --all-files
```

---

## 2. CI Pipeline (`ci.yml`)

Triggered on: `push` to `main`, `pull_request` to `main`.

### Jobs
| Job | Purpose | Tool |
|-----|---------|------|
| **Linting** | Enforce coding standards | `ruff check .` |
| **Formatting** | Enforce consistent style | `ruff format --check .` |
| **Type Checking** | Verify type safety | `mypy .` |
| **Security** | Static security audit | `bandit -r .` |
| **Unit Tests** | Verify correctness | `pytest tests/` |

---

## 3. CD Pipeline (`cd.yml`)

Triggered on: Successful `merge` or `push` to `main`.

### Jobs
| Job | Purpose |
|-----|---------|
| **Docker Build** | Builds the production API image |
| **Image Scanning** | Scans container for vulnerabilities |
| **Smoke Test** | Spins up the container and probes `/health/live` |
| **Push** | Pushes to GitHub Container Registry (GHCR) |

---

## Debugging Workflow Failures

1. **Lint/Format errors**: Run `poetry run ruff format .` and `poetry run ruff check . --fix` locally.
2. **Type errors**: Run `poetry run mypy .` and ensure all functions have return types.
3. **Test failures**: Individual tests can be debugged with `poetry run pytest tests/test_name.py -vv`.
4. **Secrets**: If TruffleHog fails, you likely committed a real or placeholder key. Use `git reset` and remove the sensitive data before re-committing.
