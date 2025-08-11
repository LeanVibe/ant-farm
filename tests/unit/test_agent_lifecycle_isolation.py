"""Comprehensive isolation tests for Agent lifecycle with database mocking.

These tests ensure Agent components operate correctly in complete isolation
from database, Redis, CLI tools, and other external dependencies. All external
operations are mocked to verify agent behavior without system dependencies.
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog

from src.agents.base_agent import BaseAgent, HealthStatus, TaskResult, ToolResult
from src.core.message_broker import Message, MessageType
from src.core.task_queue import Task
from tests.unit.test_component_isolation_framework import (
    ComponentIsolationTestFramework,
    ComponentTestConfig,
    create_isolated_agent,
)

logger = structlog.get_logger()


class MockTestAgent(BaseAgent):
    """Test agent implementation for isolation testing."""

    def __init__(
        self, name: str = "test_agent", agent_type: str = "test", role: str = "tester"
    ):
        super().__init__(name, agent_type, role, enhanced_communication=False)
        self.run_called = False
        self.run_duration = 0.1  # Short duration for testing
        self.tasks_processed = []
        self.messages_received = []

    async def run(self) -> None:
        """Simple run implementation for testing."""
        self.run_called = True

        # Simulate agent work
        await asyncio.sleep(self.run_duration)

        # Check for tasks periodically
        while self.status == "active":
            try:
                # Simulate checking for tasks
                await asyncio.sleep(0.01)

                # Break after short time for testing
                if time.time() - self.start_time > 0.5:
                    break

            except Exception as e:
                logger.error("Error in agent run loop", error=str(e))
                break

    async def _process_task_implementation(self, task: Task) -> TaskResult:
        """Override task processing for testing."""
        self.tasks_processed.append(task)

        # Simulate task processing
        await asyncio.sleep(0.01)

        # Return success for most tasks, failure for specific test cases
        if task.description == "FAIL_TEST":
            return TaskResult(success=False, error="Simulated task failure")

        return TaskResult(
            success=True,
            data={
                "task_id": task.id,
                "processed_by": self.name,
                "processing_time": 0.01,
            },
        )


class TestAgentLifecycleIsolation:
    """Comprehensive isolation tests for Agent lifecycle."""

    @pytest.fixture
    async def isolated_agent(self):
        """Fixture providing an isolated agent."""
        config = ComponentTestConfig(
            mock_redis=True,
            mock_database=True,
            mock_cli_tools=True,
            mock_network=True,
            record_interactions=True,
        )

        framework = ComponentIsolationTestFramework(config)

        async with framework.isolate_component(MockTestAgent) as _:
            # Create agent manually to control initialization
            agent = MockTestAgent()

            # Replace components with mocks
            agent.cli_tools = framework.mock_cli_tools

            yield agent, framework

    @pytest.mark.asyncio
    async def test_agent_initialization_isolation(self, isolated_agent):
        """Test agent initialization without external dependencies."""
        agent, framework = isolated_agent

        # Verify initial state
        assert agent.name == "test_agent"
        assert agent.agent_type == "test"
        assert agent.role == "tester"
        assert agent.status == "inactive"
        assert agent.tasks_completed == 0
        assert agent.tasks_failed == 0

        # Verify CLI tools were mocked
        assert agent.cli_tools is not None
        assert hasattr(agent.cli_tools, "interactions")

        # Verify no external calls during initialization
        framework.assert_no_external_calls()

    @pytest.mark.asyncio
    async def test_agent_startup_isolation(self, isolated_agent):
        """Test agent startup process in isolation."""
        agent, framework = isolated_agent

        # Mock the start method to avoid full initialization
        with (
            patch.object(agent, "_register_agent", new_callable=AsyncMock),
            patch("src.core.task_queue.task_queue.initialize", new_callable=AsyncMock),
            patch(
                "src.core.message_broker.message_broker.initialize",
                new_callable=AsyncMock,
            ),
            patch(
                "src.core.message_broker.message_broker.start_listening",
                new_callable=AsyncMock,
            ),
        ):
            # Start agent with mocked dependencies
            start_task = asyncio.create_task(agent.start())

            # Let it initialize briefly
            await asyncio.sleep(0.2)

            # Verify agent reached active status
            assert agent.status == "active"
            assert agent.run_called is True

            # Stop the agent
            agent.status = "stopping"
            await start_task

        # Verify cleanup was called
        assert agent.status == "inactive"

    @pytest.mark.asyncio
    async def test_cli_tool_execution_isolation(self, isolated_agent):
        """Test CLI tool execution in complete isolation."""
        agent, framework = isolated_agent

        # Execute CLI tool command
        result = await agent.execute_with_cli_tool(
            prompt="Write a simple Python function to add two numbers"
        )

        # Verify execution succeeded with mock
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert "Mock CLI response" in result.output
        assert result.tool_used in ["opencode", "claude"]
        assert result.execution_time > 0

        # Verify CLI interaction was recorded
        cli_interactions = [
            i
            for i in framework.mock_cli_tools.interactions
            if i.component == "cli_tools"
        ]
        assert len(cli_interactions) == 1
        assert cli_interactions[0].method == "execute_prompt"

        # Verify no external calls leaked
        framework.assert_no_external_calls()

    @pytest.mark.asyncio
    async def test_task_processing_isolation(self, isolated_agent):
        """Test task processing in isolation."""
        agent, framework = isolated_agent

        # Create test task
        test_task = Task(
            id=str(uuid.uuid4()),
            title="Test Task",
            description="A test task for isolation testing",
            task_type="development",
            payload={"test_data": "value"},
            priority=5,
        )

        # Mock task queue operations
        with (
            patch(
                "src.core.task_queue.task_queue.start_task", new_callable=AsyncMock
            ) as mock_start,
            patch(
                "src.core.task_queue.task_queue.complete_task", new_callable=AsyncMock
            ) as mock_complete,
        ):
            # Process task
            result = await agent.process_task(test_task)

            # Verify task processing
            assert result.success is True
            assert result.data["task_id"] == test_task.id
            assert result.data["processed_by"] == agent.name

            # Verify task was added to processed list
            assert len(agent.tasks_processed) == 1
            assert agent.tasks_processed[0] == test_task

            # Verify task queue operations were called
            mock_start.assert_called_once_with(test_task.id)
            mock_complete.assert_called_once()

            # Verify metrics were updated
            assert agent.tasks_completed == 1
            assert agent.tasks_failed == 0

    @pytest.mark.asyncio
    async def test_task_failure_handling_isolation(self, isolated_agent):
        """Test task failure handling in isolation."""
        agent, framework = isolated_agent

        # Create failing task
        failing_task = Task(
            id=str(uuid.uuid4()),
            title="Failing Task",
            description="FAIL_TEST",  # Special description to trigger failure
            task_type="test_failure",
            payload={"test": "failure"},
            priority=1,
        )

        # Mock task queue operations
        with (
            patch("src.core.task_queue.task_queue.start_task", new_callable=AsyncMock),
            patch(
                "src.core.task_queue.task_queue.fail_task", new_callable=AsyncMock
            ) as mock_fail,
        ):
            # Process failing task
            result = await agent.process_task(failing_task)

            # Verify task failure was handled
            assert result.success is False
            assert "Simulated task failure" in result.error

            # Verify metrics were updated
            assert agent.tasks_completed == 0
            assert agent.tasks_failed == 1

            # Verify task queue failure was called
            mock_fail.assert_called_once_with(failing_task.id, "Simulated task failure")

    @pytest.mark.asyncio
    async def test_message_handling_isolation(self, isolated_agent):
        """Test message handling in isolation."""
        agent, framework = isolated_agent

        # Test ping message
        ping_message = Message(
            id=str(uuid.uuid4()),
            from_agent="test_sender",
            to_agent=agent.name,
            topic="ping",
            message_type=MessageType.REQUEST,
            payload={},
            timestamp=time.time(),
        )

        # Handle ping message
        response = await agent.message_handler.handle_message(ping_message)

        # Verify ping response
        assert response is not None
        assert response.message_type == MessageType.REPLY
        assert response.payload["pong"] is True
        assert response.payload["status"] == agent.status
        assert "uptime" in response.payload

    @pytest.mark.asyncio
    async def test_health_check_isolation(self, isolated_agent):
        """Test agent health check in isolation."""
        agent, framework = isolated_agent

        # Perform health check
        health = await agent.health_check()

        # Verify health status
        assert isinstance(health, HealthStatus)

        # With mocked CLI tools, should be healthy if tools are available
        if framework.mock_cli_tools.available_tools:
            assert health in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        else:
            assert health == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_context_storage_isolation(self, isolated_agent):
        """Test context storage in isolation."""
        agent, framework = isolated_agent

        # Mock database manager
        mock_db_manager = MagicMock()
        mock_db_manager.store_context = AsyncMock(return_value="context_id_123")
        agent.async_db_manager = mock_db_manager
        agent.agent_uuid = "test_agent_uuid"

        # Store context
        context_id = await agent.store_context(
            content="Test context content",
            importance_score=0.8,
            category="test_category",
            metadata={"test": "metadata"},
        )

        # Verify context was stored
        assert context_id == "context_id_123"

        # Verify database call was made
        mock_db_manager.store_context.assert_called_once_with(
            agent_id="test_agent_uuid",
            content="Test context content",
            importance_score=0.8,
            category="test_category",
            topic=None,
            metadata={"test": "metadata"},
        )

    @pytest.mark.asyncio
    async def test_context_retrieval_isolation(self, isolated_agent):
        """Test context retrieval in isolation."""
        agent, framework = isolated_agent

        # Mock context engine
        mock_context_engine = MagicMock()
        mock_results = [
            MagicMock(context=MagicMock(content="Relevant context 1")),
            MagicMock(context=MagicMock(content="Relevant context 2")),
        ]
        mock_context_engine.retrieve_context = AsyncMock(return_value=mock_results)
        agent.context_engine = mock_context_engine
        agent.agent_uuid = "test_agent_uuid"

        # Retrieve context
        results = await agent.retrieve_context(
            query="test query", limit=5, category_filter="test_category"
        )

        # Verify context was retrieved
        assert len(results) == 2
        assert results[0].context.content == "Relevant context 1"

        # Verify context engine call was made
        mock_context_engine.retrieve_context.assert_called_once_with(
            query="test query",
            agent_id="test_agent_uuid",
            limit=5,
            category_filter="test_category",
            min_importance=0.0,
        )

    @pytest.mark.asyncio
    async def test_agent_capabilities_isolation(self, isolated_agent):
        """Test agent capabilities reporting in isolation."""
        agent, framework = isolated_agent

        # Get capabilities
        capabilities = agent.capabilities

        # Verify capabilities structure
        assert isinstance(capabilities, list)
        assert len(capabilities) > 0

        # Should include basic capabilities
        assert "context_management" in capabilities
        assert "messaging" in capabilities
        assert "persistent_sessions" in capabilities

        # Should include agent type and role
        assert agent.agent_type in capabilities
        assert agent.role in capabilities

        # Should include CLI tool capabilities if available
        if framework.mock_cli_tools.available_tools:
            cli_capabilities = [c for c in capabilities if c.startswith("cli_")]
            assert len(cli_capabilities) > 0

    @pytest.mark.asyncio
    async def test_message_sending_isolation(self, isolated_agent):
        """Test message sending in isolation."""
        agent, framework = isolated_agent

        # Mock message broker
        with patch(
            "src.core.message_broker.message_broker.send_message",
            new_callable=AsyncMock,
        ) as mock_send:
            mock_send.return_value = "message_id_123"

            # Send message
            message_id = await agent.send_message(
                to_agent="target_agent",
                topic="test_topic",
                content={"data": "test_message"},
                message_type=MessageType.DIRECT,
            )

            # Verify message was sent
            assert message_id == "message_id_123"

            # Verify broker call was made
            mock_send.assert_called_once_with(
                from_agent=agent.name,
                to_agent="target_agent",
                topic="test_topic",
                payload={"data": "test_message"},
                message_type=MessageType.DIRECT,
            )

    @pytest.mark.asyncio
    async def test_rate_limiting_isolation(self, isolated_agent):
        """Test CLI tool rate limiting in isolation."""
        agent, framework = isolated_agent

        # Set aggressive rate limiting for testing
        agent._max_calls_per_window = 2
        agent._rate_limit_window = 1.0

        # Make multiple CLI calls quickly
        start_time = time.time()

        for i in range(5):
            result = await agent.execute_with_cli_tool(f"Test prompt {i}")
            assert result.success is True

        end_time = time.time()
        duration = end_time - start_time

        # Should have been rate limited (took longer than without rate limiting)
        # With rate limiting, should take at least some time for multiple calls
        assert duration > 0.01  # Should take some time due to rate limiting

        # Verify all calls were eventually processed
        cli_interactions = framework.mock_cli_tools.interactions
        assert len(cli_interactions) == 5

    @pytest.mark.asyncio
    async def test_agent_cleanup_isolation(self, isolated_agent):
        """Test agent cleanup in isolation."""
        agent, framework = isolated_agent

        # Setup agent with some state
        agent.status = "active"
        agent.current_task_id = "test_task"

        # Mock CLI session
        mock_cli_session = MagicMock()
        agent.cli_session = mock_cli_session
        agent.cli_session_id = "test_session"

        # Mock persistent CLI manager
        mock_cli_manager = MagicMock()
        mock_cli_manager.close_session = AsyncMock()
        agent.persistent_cli = mock_cli_manager

        # Mock message broker
        with patch(
            "src.core.message_broker.message_broker.stop_listening",
            new_callable=AsyncMock,
        ) as mock_stop:
            # Cleanup agent
            await agent.cleanup()

            # Verify cleanup actions
            assert agent.status == "inactive"

            # Verify CLI session was closed
            mock_cli_manager.close_session.assert_called_once_with("test_session")

            # Verify message broker stopped listening
            mock_stop.assert_called_once_with(agent.name)

    @pytest.mark.asyncio
    async def test_collaboration_invitation_isolation(self, isolated_agent):
        """Test collaboration invitation handling in isolation."""
        agent, framework = isolated_agent

        # Create collaboration invitation message
        invitation_message = Message(
            id=str(uuid.uuid4()),
            from_agent="coordinator",
            to_agent=agent.name,
            topic="collaboration_invitation",
            message_type=MessageType.REQUEST,
            payload={
                "collaboration_id": "collab_123",
                "title": "Test Collaboration",
                "collaboration_type": "pair_programming",
                "required_capabilities": ["coding", "testing"],
                "your_tasks": [
                    {"description": "Write unit tests", "depends_on": []},
                    {"description": "Code review", "depends_on": ["Write unit tests"]},
                ],
            },
            timestamp=time.time(),
        )

        # Handle invitation
        response = await agent.message_handler.handle_message(invitation_message)

        # Verify response
        assert response is not None
        assert response.message_type == MessageType.REPLY
        assert "accepted" in response.payload

        # For base test agent, should accept if capabilities match
        # (base implementation accepts all invitations)
        assert response.payload["accepted"] is True

    @pytest.mark.asyncio
    async def test_error_recovery_isolation(self, isolated_agent):
        """Test agent error recovery in isolation."""
        agent, framework = isolated_agent

        # Simulate CLI tool error
        framework.simulate_redis_error("get", ConnectionError)

        # Try CLI operation (should handle error gracefully)
        result = await agent.execute_with_cli_tool("Test prompt after error")

        # Should still succeed with mock CLI tools
        assert result.success is True

        # Verify agent status remains stable
        assert agent.status != "error"

    @pytest.mark.asyncio
    async def test_performance_metrics_isolation(self, isolated_agent):
        """Test agent performance metrics in isolation."""
        agent, framework = isolated_agent

        # Process several tasks to generate metrics
        tasks = [
            Task(
                id=str(uuid.uuid4()),
                title=f"Task {i}",
                description=f"Test task {i}",
                task_type="test",
                payload={"index": i},
                priority=5,
            )
            for i in range(3)
        ]

        # Mock task queue operations
        with (
            patch("src.core.task_queue.task_queue.start_task", new_callable=AsyncMock),
            patch(
                "src.core.task_queue.task_queue.complete_task", new_callable=AsyncMock
            ),
        ):
            start_time = time.time()

            for task in tasks:
                await agent.process_task(task)

            end_time = time.time()

            # Verify metrics
            assert agent.tasks_completed == len(tasks)
            assert agent.tasks_failed == 0
            assert agent.total_execution_time > 0

            # Verify all tasks were processed
            assert len(agent.tasks_processed) == len(tasks)

            # Performance should be reasonable in isolation
            total_duration = end_time - start_time
            tasks_per_second = len(tasks) / total_duration
            assert tasks_per_second > 5  # Should process at least 5 tasks/second


class TestAgentErrorScenarios:
    """Test agent behavior under various error conditions."""

    @pytest.mark.asyncio
    async def test_database_connection_failure_isolation(self):
        """Test agent behavior when database connection fails."""
        config = ComponentTestConfig(mock_database=True)
        framework = ComponentIsolationTestFramework(config)

        async with framework.isolate_component(MockTestAgent) as _:
            agent = MockTestAgent()

            # Simulate database connection failure
            framework.simulate_database_error("execute", ConnectionError)

            # Agent should continue to function without database
            result = await agent.execute_with_cli_tool("Test prompt")
            assert result.success is True

            # Health check should show degraded status
            health = await agent.health_check()
            assert health in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    @pytest.mark.asyncio
    async def test_cli_tool_failure_isolation(self):
        """Test agent behavior when CLI tools fail."""
        config = ComponentTestConfig(mock_cli_tools=True)
        framework = ComponentIsolationTestFramework(config)

        async with framework.isolate_component(MockTestAgent) as _:
            agent = MockTestAgent()
            agent.cli_tools = framework.mock_cli_tools

            # Simulate all CLI tools being unavailable
            framework.mock_cli_tools.available_tools = {}
            framework.mock_cli_tools.preferred_tool = None

            # CLI execution should fail gracefully
            result = await agent.execute_with_cli_tool("Test prompt")
            assert result.success is False
            assert "No suitable CLI tool available" in result.error

            # Health check should show unhealthy status
            health = await agent.health_check()
            assert health == HealthStatus.UNHEALTHY


class TestAgentPerformance:
    """Performance tests for agent operations in isolation."""

    @pytest.mark.asyncio
    async def test_high_volume_task_processing_isolation(self):
        """Test agent performance with high volume task processing."""
        config = ComponentTestConfig(
            record_interactions=False
        )  # Disable for performance
        framework = ComponentIsolationTestFramework(config)

        async with framework.isolate_component(MockTestAgent) as _:
            agent = MockTestAgent()
            agent.run_duration = 0.001  # Very short run duration

            # Create many tasks
            task_count = 50
            tasks = [
                Task(
                    id=str(uuid.uuid4()),
                    title=f"Perf Task {i}",
                    description=f"Performance test task {i}",
                    task_type="performance",
                    payload={"index": i},
                    priority=5,
                )
                for i in range(task_count)
            ]

            # Mock task queue operations
            with (
                patch(
                    "src.core.task_queue.task_queue.start_task", new_callable=AsyncMock
                ),
                patch(
                    "src.core.task_queue.task_queue.complete_task",
                    new_callable=AsyncMock,
                ),
            ):
                start_time = time.time()

                # Process all tasks
                for task in tasks:
                    await agent.process_task(task)

                end_time = time.time()
                duration = end_time - start_time

                # Performance assertions
                tasks_per_second = task_count / duration
                assert tasks_per_second > 20  # Should process at least 20 tasks/second
                assert duration < 10.0  # Should complete within 10 seconds

                # Verify all tasks were processed successfully
                assert agent.tasks_completed == task_count
                assert agent.tasks_failed == 0

                logger.info(
                    f"Agent performance: {task_count} tasks in {duration:.2f}s "
                    f"({tasks_per_second:.1f} tasks/s)"
                )


if __name__ == "__main__":
    # Run individual test for debugging
    async def run_single_test():
        test_instance = TestAgentLifecycleIsolation()
        config = ComponentTestConfig()
        framework = ComponentIsolationTestFramework(config)

        async with framework.isolate_component(MockTestAgent) as _:
            agent = MockTestAgent()
            await test_instance.test_cli_tool_execution_isolation((agent, framework))
            print("Agent lifecycle test completed successfully")

    asyncio.run(run_single_test())
