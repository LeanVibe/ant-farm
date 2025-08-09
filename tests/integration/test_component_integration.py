"""Integration tests demonstrating component interactions."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.message_broker import MessageBroker, MessageHandler, MessageType
from src.core.orchestrator import AgentInfo, AgentRegistry, AgentStatus
from src.core.task_queue import Task, TaskPriority, TaskQueue, TaskStatus


@pytest.fixture
async def message_broker():
    """Create a real message broker for testing."""
    with patch("redis.asyncio.from_url") as mock_redis:
        # Mock Redis client
        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock(return_value=True)
        mock_redis_client.publish = AsyncMock()
        mock_redis_client.hset = AsyncMock()
        mock_redis_client.expire = AsyncMock()
        mock_redis_client.zadd = AsyncMock()
        mock_redis_client.lpush = AsyncMock()

        # Mock pubsub
        pubsub_mock = AsyncMock()
        pubsub_mock.subscribe = AsyncMock()
        pubsub_mock.unsubscribe = AsyncMock()
        mock_redis_client.pubsub = MagicMock(return_value=pubsub_mock)

        mock_redis.return_value = mock_redis_client

        broker = MessageBroker("redis://localhost:6379/1")
        await broker.initialize()
        return broker


@pytest.fixture
async def task_queue():
    """Create a real task queue for testing."""
    with patch("redis.asyncio.from_url") as mock_redis:
        # Mock Redis client
        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock(return_value=True)
        mock_redis_client.lpush = AsyncMock()
        mock_redis_client.blpop = AsyncMock(return_value=None)
        mock_redis_client.hset = AsyncMock()
        mock_redis_client.hgetall = AsyncMock(return_value={})
        mock_redis_client.zrem = AsyncMock()
        mock_redis_client.zadd = AsyncMock()
        mock_redis_client.zrange = AsyncMock(return_value=[])
        mock_redis_client.zcard = AsyncMock(return_value=0)
        mock_redis_client.expire = AsyncMock()

        mock_redis.return_value = mock_redis_client

        queue = TaskQueue("redis://localhost:6379/1")
        await queue.initialize()
        return queue


@pytest.fixture
def agent_registry():
    """Create an agent registry for testing."""
    with patch("src.core.orchestrator.get_database_manager") as mock_db:
        mock_db_manager = MagicMock()
        mock_db_manager.create_tables = MagicMock()
        mock_db_manager.get_session = MagicMock()
        mock_db.return_value = mock_db_manager

        registry = AgentRegistry("postgresql://test")
        return registry


class TestMessageBrokerTaskQueueIntegration:
    """Test integration between message broker and task queue."""

    @pytest.mark.asyncio
    async def test_task_assignment_workflow(self, message_broker, task_queue):
        """Test complete task assignment workflow using message broker."""
        # Create a task
        task = Task(
            title="Integration Test Task",
            description="Test task for integration testing",
            task_type="test",
            priority=TaskPriority.HIGH,
        )

        # Submit task to queue
        task_id = await task_queue.submit_task(task)
        assert task_id == task.id

        # Create a mock agent handler
        agent_handler = MessageHandler("test-agent")

        # Register handler for task assignments
        task_received = asyncio.Event()
        received_task = None

        async def handle_task_assignment(message):
            nonlocal received_task
            received_task = message.payload
            task_received.set()
            return {"status": "acknowledged"}

        agent_handler.register_handler("task_assignment", handle_task_assignment)

        # Start listening for messages
        await message_broker.start_listening("test-agent", agent_handler)

        # Simulate orchestrator assigning task via message broker
        await message_broker.send_message(
            from_agent="orchestrator",
            to_agent="test-agent",
            topic="task_assignment",
            payload={"task_id": task_id, "task": task.model_dump()},
            message_type=MessageType.DIRECT,
        )

        # Verify message was sent
        message_broker.redis_client.publish.assert_called()

        # Clean up
        await message_broker.stop_listening("test-agent")

    @pytest.mark.asyncio
    async def test_agent_heartbeat_workflow(self, message_broker, agent_registry):
        """Test agent heartbeat and status updates."""
        # Register an agent
        agent_info = AgentInfo(
            id="test-agent-id",
            name="test-agent",
            type="meta",
            role="meta",
            status=AgentStatus.STARTING,
            capabilities=["testing"],
            tmux_session="hive-test",
            last_heartbeat=time.time(),
            created_at=time.time(),
            tasks_completed=0,
            tasks_failed=0,
        )

        # Mock database operations for the test
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        agent_registry.db_manager.get_session.return_value = mock_session

        result = await agent_registry.register_agent(agent_info)
        assert result is True

        # Update agent status
        await agent_registry.update_agent_status("test-agent", AgentStatus.ACTIVE)

        # Verify agent is in registry
        retrieved_agent = await agent_registry.get_agent("test-agent")
        assert retrieved_agent is not None
        assert retrieved_agent.status == AgentStatus.ACTIVE

        # Simulate heartbeat via message broker
        await message_broker.send_message(
            from_agent="test-agent",
            to_agent="orchestrator",
            topic="heartbeat",
            payload={"status": "healthy", "timestamp": time.time()},
            message_type=MessageType.DIRECT,
        )

        # Verify message was sent
        message_broker.redis_client.publish.assert_called()

    @pytest.mark.asyncio
    async def test_task_completion_notification(self, message_broker, task_queue):
        """Test task completion notification workflow."""
        # Create and submit a task
        task = Task(
            title="Completion Test Task",
            description="Task for testing completion workflow",
            task_type="test",
            priority=TaskPriority.NORMAL,
        )

        task_id = await task_queue.submit_task(task)

        # Simulate task completion
        await task_queue.complete_task(task_id, {"result": "success"})

        # Notify orchestrator via message broker
        await message_broker.send_message(
            from_agent="test-agent",
            to_agent="orchestrator",
            topic="task_completed",
            payload={
                "task_id": task_id,
                "result": {"result": "success"},
                "completion_time": time.time(),
            },
            message_type=MessageType.DIRECT,
        )

        # Verify notification was sent
        message_broker.redis_client.publish.assert_called()

        # Verify task status is updated
        task_status = await task_queue.get_task_status(task_id)
        assert task_status.status == TaskStatus.COMPLETED
        assert task_status.result == {"result": "success"}


class TestOrchestrationWorkflow:
    """Test complete orchestration workflows."""

    @pytest.mark.asyncio
    async def test_agent_lifecycle_management(self, agent_registry, message_broker):
        """Test complete agent lifecycle through orchestrator."""
        # Create agent info
        agent_info = AgentInfo(
            id="lifecycle-agent-id",
            name="lifecycle-agent",
            type="qa",
            role="qa",
            status=AgentStatus.STARTING,
            capabilities=["testing", "quality_assurance"],
            tmux_session="hive-lifecycle",
            last_heartbeat=time.time(),
            created_at=time.time(),
            tasks_completed=0,
            tasks_failed=0,
        )

        # Mock database for testing
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        agent_registry.db_manager.get_session.return_value = mock_session

        # 1. Register agent
        result = await agent_registry.register_agent(agent_info)
        assert result is True

        # 2. Update to active status
        await agent_registry.update_agent_status("lifecycle-agent", AgentStatus.ACTIVE)

        # 3. Simulate agent communication
        await message_broker.send_message(
            from_agent="lifecycle-agent",
            to_agent="orchestrator",
            topic="agent_ready",
            payload={"capabilities": agent_info.capabilities},
            message_type=MessageType.DIRECT,
        )

        # 4. Update to busy status (working on task)
        await agent_registry.update_agent_status(
            "lifecycle-agent", AgentStatus.BUSY, "task-123"
        )

        # 5. Complete task and return to idle
        await agent_registry.update_agent_status("lifecycle-agent", AgentStatus.IDLE)

        # 6. Verify final state
        final_agent = await agent_registry.get_agent("lifecycle-agent")
        assert final_agent is not None
        assert final_agent.status == AgentStatus.IDLE
        assert final_agent.current_task_id == "task-123"  # Last assigned task ID

    @pytest.mark.asyncio
    async def test_error_handling_integration(
        self, message_broker, task_queue, agent_registry
    ):
        """Test error handling across components."""
        # Create a task that will fail
        task = Task(
            title="Failing Task",
            description="Task designed to fail for testing",
            task_type="test",
            priority=TaskPriority.HIGH,
            max_retries=1,
        )

        task_id = await task_queue.submit_task(task)

        # Simulate task failure
        await task_queue.fail_task(task_id, "Simulated failure for testing", retry=True)

        # Verify task is marked for retry
        task_status = await task_queue.get_task_status(task_id)
        assert task_status.status == TaskStatus.PENDING
        assert task_status.retry_count == 1
        assert "Simulated failure" in task_status.error_message

        # Simulate agent error notification
        await message_broker.send_message(
            from_agent="test-agent",
            to_agent="orchestrator",
            topic="task_failed",
            payload={
                "task_id": task_id,
                "error": "Simulated failure for testing",
                "retry_requested": True,
            },
            message_type=MessageType.DIRECT,
        )

        # Verify error notification was sent
        message_broker.redis_client.publish.assert_called()
