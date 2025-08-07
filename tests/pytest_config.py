"""Test configuration for running tests without coverage on placeholder files."""

import pytest


def pytest_configure(config):
    """Configure pytest to handle our current development state."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_redis: mark test as requiring Redis")
    config.addinivalue_line(
        "markers", "requires_postgres: mark test as requiring PostgreSQL"
    )
    config.addinivalue_line(
        "markers", "requires_cli_tools: mark test as requiring CLI tools"
    )
