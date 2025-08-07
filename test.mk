"""Makefile helper for testing infrastructure."""

# Testing commands for LeanVibe Agent Hive

.PHONY: test test-unit test-integration test-e2e test-quick test-coverage

# Quick test without coverage (for development)
test-quick: ## Run tests without coverage requirements
	python -m pytest tests/test_infrastructure.py -v

# Run all unit tests
test-unit: ## Run unit tests only
	python -m pytest tests/unit/ -v -m "not slow"

# Run integration tests (requires services)
test-integration: ## Run integration tests
	python -m pytest tests/integration/ -v

# Run end-to-end tests (requires full system)
test-e2e: ## Run end-to-end tests
	python -m pytest tests/e2e/ -v

# Run tests with coverage (for CI)
test-coverage: ## Run tests with coverage requirements
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# Run specific test file
test-file: ## Run specific test file (use FILE=path/to/test.py)
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make test-file FILE=tests/unit/test_example.py"; \
	else \
		python -m pytest $(FILE) -v; \
	fi

# Run tests by marker
test-marker: ## Run tests by marker (use MARKER=unit)
	@if [ -z "$(MARKER)" ]; then \
		echo "Usage: make test-marker MARKER=unit"; \
		echo "Available markers: unit, integration, e2e, slow, requires_redis, requires_postgres, requires_cli_tools"; \
	else \
		python -m pytest -m "$(MARKER)" -v; \
	fi

# Test setup for CI
test-setup: ## Set up test environment
	@echo "Setting up test environment..."
	@mkdir -p tests/unit tests/integration tests/e2e
	@echo "Test environment ready"

# Clean test artifacts
test-clean: ## Clean test artifacts
	@echo "Cleaning test artifacts..."
	@rm -rf .pytest_cache htmlcov .coverage
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Test artifacts cleaned"

help-test: ## Show test-related help
	@echo "Testing Commands:"
	@echo "  test-quick      - Quick tests without coverage"
	@echo "  test-unit       - Unit tests only"
	@echo "  test-integration- Integration tests"  
	@echo "  test-e2e        - End-to-end tests"
	@echo "  test-coverage   - Tests with coverage"
	@echo "  test-file       - Run specific test file"
	@echo "  test-marker     - Run tests by marker"
	@echo "  test-setup      - Set up test environment"
	@echo "  test-clean      - Clean test artifacts"