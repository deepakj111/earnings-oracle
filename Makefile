.PHONY: help install up down test test-cov lint format typecheck audit serve eval eval-fast monitor clean

PYTHON := poetry run python
PYTEST := poetry run pytest
RUFF := poetry run ruff
MYPY := poetry run mypy

help:  ## Show available commands
	@echo "Financial Earnings Oracle - Developer Commands"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies via Poetry
	poetry install

up:  ## Start background backing infrastructure (Qdrant, Redis, Jaeger, Prometheus, Grafana)
	docker compose up -d qdrant redis jaeger prometheus grafana

down:  ## Stop background backing infrastructure
	docker compose down

test:  ## Run the full pytest test suite
	$(PYTEST) tests/ -q

test-cov:  ## Run pytest with test coverage report
	$(PYTEST) tests/ --cov=. --cov-report=term-missing

lint:  ## Check code style and formatting via Ruff
	$(RUFF) check .
	$(RUFF) format --check .

format:  ## Automatically format code and apply safe fixes via Ruff
	$(RUFF) format .
	$(RUFF) check --fix .

typecheck:  ## Run static type checking via Mypy
	$(MYPY) api config generation ingestion observability query retrieval scripts

audit:  ## Run security audit (Bandit SAST and pip-audit CVE check)
	poetry run bandit -r api config generation ingestion observability query retrieval scripts -c pyproject.toml
	poetry run pip-audit

serve:  ## Launch the FastAPI production server locally on port 8000
	poetry run serve-prod

eval:  ## Run the 50-QA evaluation harness
	$(PYTHON) evaluation/harness.py

eval-fast:  ## Run the evaluation harness on a 5-sample subset for rapid iteration
	$(PYTHON) evaluation/harness.py --sample-size 5

monitor:  ## Run the online production quality SLI monitor
	$(PYTHON) -m evaluation.online_monitor

clean:  ## Clean temporary Python caches and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
