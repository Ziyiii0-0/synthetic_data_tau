.PHONY: help install test test-all lint format clean

help:  ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install the package in editable mode
	pip install -e .

test:  ## Run the offline pipeline tests
	pytest tests/test_synthesis.py

test-all:  ## Run every test (some call an LLM)
	pytest tests

lint:  ## Lint with ruff
	ruff check src tests

format:  ## Format with ruff
	ruff format src/synthesis tests/test_synthesis.py

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
