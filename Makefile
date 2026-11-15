.DEFAULT_GOAL := help
PROVIDER ?= fake

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install Python and UI dependencies
	uv sync
	cd ui && npm ci
	uv run pre-commit install || true

lint: ## Ruff lint + format check, mypy --strict
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

test: ## Unit + integration + e2e tests (offline, no API key)
	uv run pytest --cov --cov-report=term-missing

test-live: ## Smoke tests against the real Claude API (needs ANTHROPIC_API_KEY)
	uv run pytest -m live -v

eval: ## Quality evals. PROVIDER=fake (default) or PROVIDER=anthropic
	uv run python -m evals.run_evals --provider $(PROVIDER)

demo: ## Run all sample claims offline
	uv run claim-agent demo --mock

run: ## Run one sample with Claude, e.g. make run FILE=samples/inputs/01_long_delay_eligible.json
	uv run claim-agent run $(FILE)

ui: ## Build the review UI (served by the API at /)
	cd ui && npm run build

serve: ui ## Start API + UI on http://127.0.0.1:8000 (add MOCK=1 for offline)
	uv run claim-agent serve $(if $(MOCK),--mock,)

docker: ## Build and run with docker compose
	docker compose up --build

clean: ## Remove local runs, caches and build output
	rm -rf runs .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov ui/dist

.PHONY: help setup lint test test-live eval demo run ui serve docker clean
