"""End-to-end tests for core user workflows in LeanVibe Agent Hive 2.0.

Tests the complete user journey from system startup to task completion,
validating all critical user-facing functionality.
"""

import asyncio
import pytest
import httpx
import time
from typing import AsyncGenerator

from src.core.config import get_settings
from src.core.task_queue import Task, TaskPriority, TaskStatus


@pytest.fixture
async def api_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for API testing."""
    settings = get_settings()
    base_url = f"http://localhost:{settings.api_port}"

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        # Verify API is accessible
        try:
            response = await client.get("/api/v1/status")
            if response.status_code != 200:
                pytest.skip("API server not available for E2E testing")
        except httpx.ConnectError:
            pytest.skip("API server not running on expected port")

        yield client


class TestCoreUserWorkflows:
    """End-to-end tests for essential user workflows."""

    @pytest.mark.asyncio
    async def test_complete_task_workflow(self, api_client: httpx.AsyncClient):
        """Test complete task lifecycle: submit → assign → process → complete."""

        # 1. Submit a task via API
        task_data = {
            "title": "E2E Test Task",
            "description": "Testing end-to-end task workflow",
            "task_type": "test",
            "priority": "normal",
        }

        response = await api_client.post("/api/v1/tasks", json=task_data)
        assert response.status_code == 200

        result = response.json()
        assert result["success"] is True
        task_id = result["data"]["task_id"]
        assert task_id is not None

        # 2. Verify task appears in task list
        response = await api_client.get("/api/v1/tasks")
        assert response.status_code == 200

        tasks_result = response.json()
        assert tasks_result["success"] is True
        tasks = tasks_result["data"]["tasks"]

        # Find our task
        our_task = next((t for t in tasks if t["id"] == task_id), None)
        assert our_task is not None
        assert our_task["title"] == task_data["title"]
        assert our_task["status"] == "pending"

        # 3. Get specific task details
        response = await api_client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200

        task_details = response.json()
        assert task_details["success"] is True
        task_info = task_details["data"]["task"]
        assert task_info["title"] == task_data["title"]
        assert task_info["description"] == task_data["description"]

        # 4. Cancel task (testing cancellation workflow)
        response = await api_client.post(f"/api/v1/tasks/{task_id}/cancel")
        assert response.status_code == 200

        cancel_result = response.json()
        assert cancel_result["success"] is True

        # 5. Verify task is cancelled
        response = await api_client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200

        final_task = response.json()
        assert final_task["data"]["task"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_agent_management_workflow(self, api_client: httpx.AsyncClient):
        """Test agent lifecycle: list → spawn → monitor → stop."""

        # 1. List existing agents
        response = await api_client.get("/api/v1/agents")
        assert response.status_code == 200

        agents_result = response.json()
        assert agents_result["success"] is True
        initial_agents = agents_result["data"]["agents"]
        initial_count = len(initial_agents)

        # 2. Spawn a new agent
        agent_data = {"agent_type": "meta", "name": "e2e-test-agent"}

        response = await api_client.post("/api/v1/agents", json=agent_data)
        assert response.status_code == 200

        spawn_result = response.json()
        assert spawn_result["success"] is True
        agent_name = spawn_result["data"]["agent_name"]
        assert agent_name == "e2e-test-agent"

        # 3. Verify agent appears in agent list
        await asyncio.sleep(1)  # Allow time for agent registration

        response = await api_client.get("/api/v1/agents")
        assert response.status_code == 200

        updated_agents = response.json()
        agents = updated_agents["data"]["agents"]
        assert len(agents) >= initial_count  # Should have at least one more agent

        # Find our agent
        our_agent = next((a for a in agents if a["name"] == "e2e-test-agent"), None)
        assert our_agent is not None
        assert our_agent["type"] == "meta"

        # 4. Get specific agent details
        response = await api_client.get(f"/api/v1/agents/{agent_name}")
        assert response.status_code == 200

        agent_details = response.json()
        assert agent_details["success"] is True
        agent_info = agent_details["data"]["agent"]
        assert agent_info["name"] == agent_name
        assert agent_info["type"] == "meta"

        # 5. Check agent health
        response = await api_client.post(f"/api/v1/agents/{agent_name}/health")
        assert response.status_code == 200

        health_result = response.json()
        assert health_result["success"] is True

        # 6. Stop the agent
        response = await api_client.post(f"/api/v1/agents/{agent_name}/stop")
        assert response.status_code == 200

        stop_result = response.json()
        assert stop_result["success"] is True

    @pytest.mark.asyncio
    async def test_system_status_monitoring(self, api_client: httpx.AsyncClient):
        """Test system monitoring and health check workflows."""

        # 1. Get system status
        response = await api_client.get("/api/v1/status")
        assert response.status_code == 200

        status_result = response.json()
        assert status_result["success"] is True
        status = status_result["data"]["status"]

        # Verify core components are healthy
        assert "database" in status
        assert "redis" in status
        assert "api" in status

        # 2. Get system diagnostics
        response = await api_client.get("/api/v1/diagnostics")
        assert response.status_code == 200

        diagnostics = response.json()
        assert diagnostics["success"] is True
        diag_data = diagnostics["data"]

        # Verify diagnostic information is present
        assert "system" in diag_data
        assert "performance" in diag_data
        assert "errors" in diag_data

        # 3. Test performance metrics endpoint
        response = await api_client.get("/api/v1/performance/metrics")
        assert response.status_code == 200

        metrics = response.json()
        assert metrics["success"] is True
        metrics_data = metrics["data"]

        # Basic metrics should be available
        assert "uptime" in metrics_data
        assert "requests" in metrics_data

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, api_client: httpx.AsyncClient):
        """Test API error handling and validation."""

        # 1. Test invalid task submission
        invalid_task = {
            "title": "",  # Empty title should be invalid
            "description": "Test task",
            "task_type": "test",
        }

        response = await api_client.post("/api/v1/tasks", json=invalid_task)
        assert response.status_code == 400  # Bad request

        error_result = response.json()
        assert error_result["success"] is False
        assert "error" in error_result

        # 2. Test non-existent task retrieval
        fake_task_id = "00000000-0000-0000-0000-000000000000"
        response = await api_client.get(f"/api/v1/tasks/{fake_task_id}")
        assert response.status_code == 404  # Not found

        # 3. Test non-existent agent operations
        fake_agent_name = "non-existent-agent"
        response = await api_client.get(f"/api/v1/agents/{fake_agent_name}")
        assert response.status_code == 404

        # 4. Test invalid agent type
        invalid_agent = {"agent_type": "invalid_type", "name": "test-agent"}

        response = await api_client.post("/api/v1/agents", json=invalid_agent)
        assert response.status_code == 400

        error_result = response.json()
        assert error_result["success"] is False

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, api_client: httpx.AsyncClient):
        """Test system behavior under concurrent operations."""

        # Submit multiple tasks concurrently
        async def submit_task(task_num: int):
            task_data = {
                "title": f"Concurrent Task {task_num}",
                "description": f"Testing concurrent submission {task_num}",
                "task_type": "test",
                "priority": "normal",
            }
            response = await api_client.post("/api/v1/tasks", json=task_data)
            return response

        # Submit 5 tasks concurrently
        concurrent_tasks = [submit_task(i) for i in range(5)]
        responses = await asyncio.gather(*concurrent_tasks)

        # All should succeed
        for response in responses:
            assert response.status_code == 200
            result = response.json()
            assert result["success"] is True

        # Verify all tasks appear in the system
        response = await api_client.get("/api/v1/tasks")
        assert response.status_code == 200

        tasks_result = response.json()
        tasks = tasks_result["data"]["tasks"]

        # Should have at least 5 new tasks
        concurrent_task_titles = [f"Concurrent Task {i}" for i in range(5)]
        found_tasks = [t for t in tasks if t["title"] in concurrent_task_titles]
        assert len(found_tasks) >= 5

    @pytest.mark.asyncio
    async def test_authentication_workflow(self, api_client: httpx.AsyncClient):
        """Test authentication endpoints (basic validation)."""

        # Test login endpoint exists (even if not fully implemented)
        login_data = {"username": "test_user", "password": "test_password"}

        response = await api_client.post("/api/v1/auth/login", json=login_data)
        # Should return either success or proper error, not 404
        assert response.status_code in [200, 400, 401, 501]  # Not 404 (not found)

        # Test auth status endpoint
        response = await api_client.get("/api/v1/auth/me")
        assert response.status_code in [200, 401, 501]  # Endpoint should exist
