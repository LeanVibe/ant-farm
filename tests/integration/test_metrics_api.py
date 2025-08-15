"""Integration tests for metrics API endpoints."""

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestMetricsAPI:
    """Test cases for metrics API endpoints."""

    def test_get_custom_metrics_without_auth_fails(self, client):
        """Test that custom metrics endpoint requires authentication."""
        response = client.get("/api/v1/metrics/custom")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_custom_metrics_with_feature_flag_disabled(self, client):
        """Test custom metrics endpoint when feature flag is disabled."""
        # Mock authentication
        with patch("src.api.main.get_current_user"):
            response = client.get("/api/v1/metrics/custom")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert "data" in data
            assert "cli_tool_counters" in data["data"]
            assert "dlq_counters" in data["data"]

    @patch.dict(os.environ, {"HIVE_METRICS_FLUSH": "1"})
    def test_get_custom_metrics_with_feature_flag_enabled(self, client):
        """Test custom metrics endpoint when feature flag is enabled."""
        # Mock authentication
        with patch("src.api.main.get_current_user"):
            response = client.get("/api/v1/metrics/custom")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert "data" in data
            assert "cli_tool_counters" in data["data"]
            assert "dlq_counters" in data["data"]
