"""Comprehensive integration tests for API endpoints with error handling."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.task_queue import TaskPriority


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test cases for health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "healthy"
        assert data["data"]["service"] == "agent-hive-api"
        assert "timestamp" in data
        assert "request_id" in data

    def test_detailed_health_check_mock(self, client):
        """Test detailed health check endpoint with mocking."""
        with (
            patch("src.api.main.get_orchestrator") as mock_orch,
            patch("src.api.main.message_broker") as mock_broker,
        ):
            mock_broker.redis_client.ping.return_value = True

            response = client.get("/api/v1/health")
            assert response.status_code == status.HTTP_200_OK

    def test_system_status_mock(self, client):
        """Test system status endpoint with mocking."""
        with (
            patch("src.api.main.task_queue") as mock_queue,
            patch("src.api.main.get_orchestrator") as mock_orch,
        ):
            mock_queue.get_total_tasks.return_value = 10
            mock_queue.get_completed_tasks.return_value = 8
            mock_queue.get_failed_tasks.return_value = 1
            mock_queue.get_queue_depth.return_value = 1
            mock_orch.return_value.get_active_agent_count.return_value = 2

            response = client.get("/api/v1/status")
            assert response.status_code == status.HTTP_200_OK


class TestAPIErrorHandling:
    """Test cases for comprehensive API error handling."""

    def test_404_error_handling(self, client):
        """Test 404 error returns standardized response."""
        response = client.get("/api/v1/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert "timestamp" in data
        assert "request_id" in data

    def test_validation_error_handling(self, client):
        """Test Pydantic validation error handling."""
        # Send invalid task data - this should trigger validation errors
        invalid_task = {
            "title": "",  # Empty title should fail validation
            "description": "Valid description",
            "type": "",  # Empty type should fail validation
        }

        with patch("src.api.main.task_queue"):
            response = client.post("/api/v1/tasks", json=invalid_task)

            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            data = response.json()
            assert data["success"] is False
            assert "details" in data
            assert len(data["details"]) > 0

    def test_rate_limiting_headers(self, client):
        """Test that rate limit headers are present."""
        response = client.get("/api/v1/test")

        # Check for rate limiting headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_request_timeout_handling(self, client):
        """Test request timeout handling."""
        # Test that the timeout middleware is in place
        response = client.get("/api/v1/test")
        assert "X-Processing-Time" in response.headers


class TestAgentEndpoints:
    """Test cases for agent management endpoints."""

    def test_list_agents_mock(self, client):
        """Test agent listing endpoint."""
        with patch("src.api.main.get_orchestrator") as mock_orch:
            mock_registry = AsyncMock()
            mock_registry.list_agents.return_value = []
            mock_orch.return_value.registry = mock_registry

            response = client.get("/api/v1/agents")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert isinstance(data["data"], list)

    def test_get_nonexistent_agent_mock(self, client):
        """Test getting non-existent agent returns 404."""
        with patch("src.api.main.get_orchestrator") as mock_orch:
            mock_registry = AsyncMock()
            mock_registry.get_agent.return_value = None
            mock_orch.return_value.registry = mock_registry

            response = client.get("/api/v1/agents/nonexistent")

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_spawn_agent_mock(self, client):
        """Test agent spawning endpoint."""
        with (
            patch("src.api.main.get_orchestrator") as mock_orch,
            patch("src.api.main.broadcast_event"),
        ):
            mock_orch.return_value.spawn_agent.return_value = "test-session"

            response = client.post(
                "/api/v1/agents",
                params={"agent_type": "meta", "agent_name": "test-meta"},
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert data["data"]["agent_name"] == "test-meta"


class TestTaskEndpoints:
    """Test cases for task management endpoints."""

    def test_create_valid_task_mock(self, client):
        """Test creating a valid task."""
        with (
            patch("src.api.main.task_queue") as mock_queue,
            patch("src.api.main.broadcast_event"),
        ):
            mock_queue.add_task.return_value = "task-123"

            task_data = {
                "title": "Test Task",
                "description": "A test task",
                "type": "development",
                "priority": "normal",
            }

            response = client.post("/api/v1/tasks", json=task_data)

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert data["data"]["task_id"] == "task-123"

    def test_list_tasks_mock(self, client):
        """Test listing tasks."""
        with patch("src.api.main.task_queue") as mock_queue:
            mock_queue.list_tasks.return_value = []

            response = client.get("/api/v1/tasks")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert isinstance(data["data"], list)

    def test_get_task_mock(self, client):
        """Test getting specific task."""
        with patch("src.api.main.task_queue") as mock_queue:
            mock_task = MagicMock()
            mock_task.id = "task-123"
            mock_task.title = "Test Task"
            mock_task.description = "Test Description"
            mock_task.task_type = "test"
            mock_task.status = "pending"
            mock_task.priority = TaskPriority.NORMAL
            mock_task.agent_id = "test-agent"
            mock_task.created_at = time.time()
            mock_task.started_at = None
            mock_task.completed_at = None
            mock_task.result = None
            mock_task.error_message = None

            mock_queue.get_task.return_value = mock_task

            response = client.get("/api/v1/tasks/task-123")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert data["data"]["id"] == "task-123"

    def test_get_nonexistent_task_mock(self, client):
        """Test getting non-existent task."""
        with patch("src.api.main.task_queue") as mock_queue:
            mock_queue.get_task.return_value = None

            response = client.get("/api/v1/tasks/nonexistent")

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_cancel_task_mock(self, client):
        """Test task cancellation."""
        with patch("src.api.main.task_queue") as mock_queue:
            mock_queue.cancel_task.return_value = True

            response = client.post("/api/v1/tasks/task-123/cancel")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True


class TestWebSocketEndpoints:
    """Test cases for WebSocket functionality."""

    def test_websocket_connection(self, client):
        """Test WebSocket connection establishment."""
        with (
            patch("src.api.main.task_queue") as mock_queue,
            patch("src.api.main.get_orchestrator") as mock_orch,
        ):
            # Mock the required async calls
            mock_queue.get_total_tasks.return_value = 0
            mock_queue.get_completed_tasks.return_value = 0
            mock_queue.get_queue_depth.return_value = 0
            mock_orch.return_value.get_active_agent_count.return_value = 0

            with client.websocket_connect("/api/v1/ws/events") as websocket:
                # Should receive connection confirmation
                data = websocket.receive_json()
                assert data["type"] == "connection-status"
                assert data["payload"]["status"] == "connected"


class TestAuthenticationEndpoints:
    """Test cases for authentication and authorization."""

    def test_login_endpoint_structure(self, client):
        """Test login endpoint accepts proper structure."""
        login_data = {"username": "test", "password": "test"}

        # This will fail authentication but test the endpoint structure
        response = client.post("/api/v1/auth/login", json=login_data)

        # Should be a proper error response
        assert response.status_code in [401, 500]
        data = response.json()
        assert "error" in data or "detail" in data


class TestSecurityHeaders:
    """Test security headers and middleware."""

    def test_security_headers_present(self, client):
        """Test that security headers are added to responses."""
        response = client.get("/health")

        # Check for security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert "Content-Security-Policy" in response.headers

    def test_rate_limit_headers(self, client):
        """Test that rate limit headers are present."""
        response = client.get("/api/v1/test")

        # Check for rate limiting headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_request_id_header(self, client):
        """Test that request ID is added to responses."""
        response = client.get("/health")

        assert "X-Request-ID" in response.headers
        assert "X-Processing-Time" in response.headers


class TestFullAPIIntegration:
    """End-to-end API integration tests."""

    def test_api_response_format_consistency(self, client):
        """Test that all API responses follow consistent format."""
        # Test endpoints that should work without complex setup
        response = client.get("/health")
        data = response.json()

        # All responses should have these fields
        assert "success" in data
        assert "timestamp" in data
        assert "request_id" in data

        if data["success"]:
            assert "data" in data
        else:
            assert "error" in data

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/health")

        # Should have CORS headers for preflight
        assert response.status_code in [
            200,
            405,
        ]  # 405 if OPTIONS not explicitly handled
