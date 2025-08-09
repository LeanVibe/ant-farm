"""Integration test for complete agent workflow: spawn → task → completion."""

import asyncio
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.core.async_db import AsyncDatabaseManager
from src.core.message_broker import Message, MessageBroker, MessageType
from src.core.task_queue import TaskQueue


class MockSimpleAgent:
    """Simple mock agent for testing workflows."""

    def __init__(self, name: str, agent_type: str = "test"):
        self.name = name
        self.agent_type = agent_type
        self.agent_id = str(uuid.uuid4())
        self.processed_tasks = []
        self.is_running = False
        self.status = "inactive"

        # Mock dependencies
        self.db_manager = None
        self.message_broker = None

    async def process_task(self, task_data: dict) -> dict:
        """Mock task processing."""
        task_id = task_data.get("task_id", str(uuid.uuid4()))
        task_description = task_data.get("description", "Test task")

        # Handle error conditions gracefully
        if "fail" in task_description:
            return {
                "success": False,
                "result": f"Task failed: {task_description}",
                "error": "Simulated task processing error",
                "agent_name": self.name,
            }

        # Simulate processing time
        await asyncio.sleep(0.1)

        # Record processed task
        self.processed_tasks.append(
            {
                "task_id": task_id,
                "description": task_description,
                "timestamp": datetime.now(UTC),
            }
        )

        return {
            "success": True,
            "result": f"Completed task: {task_description}",
            "output": f"Mock output for task {task_id}",
            "agent_name": self.name,
        }

    async def start(self):
        """Start the agent."""
        self.status = "active"
        self.is_running = True

        # Register with database if available
        if self.db_manager:
            await self.db_manager.register_agent(
                name=self.name, agent_type=self.agent_type, role="test_role"
            )

    async def stop(self):
        """Stop the agent."""
        self.status = "inactive"
        self.is_running = False

    async def _update_heartbeat(self):
        """Update heartbeat in database."""
        if self.db_manager:
            await self.db_manager.update_agent_heartbeat(self.name)

    async def store_context(
        self, content: str, importance_score: float = 0.5, category: str = "general"
    ):
        """Store context in database."""
        if self.db_manager:
            return await self.db_manager.store_context(
                agent_id=self.agent_id,
                content=content,
                importance_score=importance_score,
                category=category,
            )


@pytest.fixture
async def mock_database():
    """Create a mock database manager."""
    db = AsyncMock(spec=AsyncDatabaseManager)

    # Mock database operations
    db.register_agent.return_value = str(uuid.uuid4())
    db.update_agent_heartbeat.return_value = True
    db.get_agent_by_name.return_value = None
    db.store_context.return_value = str(uuid.uuid4())
    db.record_system_metric.return_value = str(uuid.uuid4())
    db.health_check.return_value = True

    return db


@pytest.fixture
async def mock_message_broker():
    """Create a mock message broker."""
    broker = AsyncMock(spec=MessageBroker)
    broker.send_message.return_value = str(uuid.uuid4())
    broker.start_listening.return_value = None
    broker.stop_listening.return_value = None
    broker.broadcast_message.return_value = str(uuid.uuid4())
    broker.multicast_message.return_value = str(uuid.uuid4())
    broker.send_request.return_value = {"status": "success"}
    broker.get_message_history.return_value = []
    broker.get_offline_messages.return_value = []
    broker.initialize.return_value = None
    broker.shutdown.return_value = None

    # Track sent messages
    broker.sent_messages = []

    async def track_send_message(message: Message):
        broker.sent_messages.append(message)
        return str(uuid.uuid4())

    broker.send_message.side_effect = track_send_message

    return broker


@pytest.fixture
async def mock_task_queue():
    """Create a mock task queue."""
    queue = AsyncMock(spec=TaskQueue)

    # Track submitted and completed tasks
    queue.submitted_tasks = []
    queue.completed_tasks = []

    async def track_submit_task(task_data: dict):
        task_id = str(uuid.uuid4())
        task_data["task_id"] = task_id
        queue.submitted_tasks.append(task_data)
        return task_id

    async def track_complete_task(task_id: str, result: dict):
        queue.completed_tasks.append({"task_id": task_id, "result": result})
        return True

    # Mock all TaskQueue methods
    queue.submit_task.side_effect = track_submit_task
    queue.complete_task.side_effect = track_complete_task
    queue.add_task.side_effect = track_submit_task
    queue.cancel_task.return_value = True
    queue.cleanup_expired_tasks.return_value = 0
    queue.fail_task.return_value = True
    queue.get_agent_active_task_count.return_value = 0
    queue.get_completed_tasks_count.return_value = 0
    queue.get_failed_tasks.return_value = []
    queue.get_failed_tasks_count.return_value = 0
    queue.get_queue_depth.return_value = 0
    queue.get_queue_stats.return_value = {
        "total": 0,
        "pending": 0,
        "active": 0,
        "completed": 0,
    }
    queue.get_task.return_value = None
    queue.get_total_tasks.return_value = 0
    queue.get_unassigned_tasks.return_value = []
    queue.initialize.return_value = None
    queue.list_tasks.return_value = []
    queue.start_task.return_value = True
    queue.update_task_status.return_value = True

    return queue


@pytest.fixture
async def test_agent(mock_database, mock_message_broker):
    """Create a test agent."""
    agent = MockSimpleAgent("test-agent-01", "test")

    # Inject dependencies
    agent.db_manager = mock_database
    agent.message_broker = mock_message_broker

    return agent


class TestCompleteAgentWorkflow:
    """Test complete agent workflow from spawn to task completion."""

    @pytest.mark.asyncio
    async def test_agent_spawn_lifecycle(
        self, test_agent, mock_database, mock_message_broker
    ):
        """Test agent spawning and lifecycle management."""

        # 1. Spawn agent
        await test_agent.start()

        # Verify agent registration
        mock_database.register_agent.assert_called_once()
        assert test_agent.is_running is True

        # 2. Agent should send heartbeat
        # Simulate heartbeat call
        await test_agent._update_heartbeat()
        mock_database.update_agent_heartbeat.assert_called()

        # 3. Stop agent
        await test_agent.stop()
        assert test_agent.is_running is False

    @pytest.mark.asyncio
    async def test_task_assignment_and_processing(self, test_agent, mock_task_queue):
        """Test task assignment and processing workflow."""

        # Start agent
        await test_agent.start()

        # Create test task
        task_data = {
            "description": "Test task for integration",
            "priority": 5,
            "metadata": {"test": True},
        }

        # Process task
        result = await test_agent.process_task(task_data)

        # Verify task was processed
        assert result["success"] is True
        assert "Test task for integration" in result["result"]
        assert result["agent_name"] == "test-agent-01"
        assert len(test_agent.processed_tasks) == 1

        await test_agent.stop()

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(
        self, mock_database, mock_message_broker, mock_task_queue
    ):
        """Test complete end-to-end agent workflow."""

        # 1. Create and spawn agent
        agent = MockSimpleAgent("integration-test-agent", "test")
        agent.db_manager = mock_database
        agent.message_broker = mock_message_broker

        await agent.start()

        # Verify agent is registered and running
        mock_database.register_agent.assert_called()
        assert agent.is_running is True

        # 2. Submit task to queue
        task_data = {
            "description": "End-to-end integration test task",
            "priority": 3,
            "requirements": ["testing"],
            "metadata": {"test_type": "integration", "expected_duration": 300},
        }

        task_id = await mock_task_queue.submit_task(task_data)
        assert task_id is not None
        assert len(mock_task_queue.submitted_tasks) == 1

        # 3. Agent processes task
        # Simulate task assignment to agent
        task_with_id = mock_task_queue.submitted_tasks[0]
        result = await agent.process_task(task_with_id)

        # Verify task processing
        assert result["success"] is True
        assert len(agent.processed_tasks) == 1

        # 4. Complete task in queue
        await mock_task_queue.complete_task(task_id, result)
        assert len(mock_task_queue.completed_tasks) == 1

        # 5. Agent sends completion message
        completion_message = Message(
            id=str(uuid.uuid4()),
            from_agent=agent.name,
            to_agent="task_coordinator",
            topic="task_completed",
            message_type=MessageType.NOTIFICATION,
            payload={
                "task_id": task_id,
                "result": result,
                "completion_time": datetime.now(UTC).isoformat(),
            },
            timestamp=datetime.now(UTC).timestamp(),
        )

        await mock_message_broker.send_message(completion_message)
        assert len(mock_message_broker.sent_messages) == 1

        # 6. Verify workflow completion
        completed_task = mock_task_queue.completed_tasks[0]
        assert completed_task["task_id"] == task_id
        assert completed_task["result"]["success"] is True

        sent_message = mock_message_broker.sent_messages[0]
        assert sent_message.topic == "task_completed"
        assert sent_message.payload["task_id"] == task_id

        # 7. Stop agent
        await agent.stop()
        assert agent.is_running is False

    @pytest.mark.asyncio
    async def test_multiple_agents_collaboration(
        self, mock_database, mock_message_broker, mock_task_queue
    ):
        """Test multiple agents working on related tasks."""

        # Create multiple agents
        agents = [
            MockSimpleAgent("agent-01", "architect"),
            MockSimpleAgent("agent-02", "developer"),
            MockSimpleAgent("agent-03", "qa"),
        ]

        # Start all agents
        for agent in agents:
            agent.db_manager = mock_database
            agent.message_broker = mock_message_broker
            await agent.start()

        # Verify all agents registered
        assert mock_database.register_agent.call_count == 3

        # Submit related tasks
        tasks = [
            {
                "description": "Design system architecture",
                "priority": 1,
                "agent_type": "architect",
            },
            {
                "description": "Implement user authentication",
                "priority": 2,
                "agent_type": "developer",
                "depends_on": ["task_1"],
            },
            {
                "description": "Test authentication system",
                "priority": 3,
                "agent_type": "qa",
                "depends_on": ["task_2"],
            },
        ]

        task_ids = []
        for task in tasks:
            task_id = await mock_task_queue.submit_task(task)
            task_ids.append(task_id)

        # Process tasks in sequence
        for i, agent in enumerate(agents):
            task_with_id = mock_task_queue.submitted_tasks[i]
            result = await agent.process_task(task_with_id)
            await mock_task_queue.complete_task(task_ids[i], result)

            # Send completion message
            completion_message = Message(
                id=str(uuid.uuid4()),
                from_agent=agent.name,
                to_agent="task_coordinator",
                topic="task_completed",
                message_type=MessageType.NOTIFICATION,
                payload={"task_id": task_ids[i], "result": result},
                timestamp=datetime.now(UTC).timestamp(),
            )
            await mock_message_broker.send_message(completion_message)

        # Verify all tasks completed
        assert len(mock_task_queue.completed_tasks) == 3
        assert len(mock_message_broker.sent_messages) == 3

        # Verify each agent processed one task
        for agent in agents:
            assert len(agent.processed_tasks) == 1

        # Stop all agents
        for agent in agents:
            await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_error_handling(self, test_agent, mock_task_queue):
        """Test agent error handling during task processing."""

        await test_agent.start()

        # Create a task that will cause an error
        failing_task = {
            "description": "This task will fail intentionally",
            "priority": 1,
        }

        # Process should handle error gracefully
        result = await test_agent.process_task(failing_task)

        # Agent should handle the error gracefully and return error result
        assert result["success"] is False
        assert "error" in result
        assert "failed" in result["result"].lower()

        await test_agent.stop()

    @pytest.mark.asyncio
    async def test_agent_context_storage(self, test_agent, mock_database):
        """Test agent context storage during task processing."""

        await test_agent.start()

        # Process task with context
        task_data = {
            "description": "Task requiring context storage",
            "priority": 2,
            "metadata": {"store_context": True},
        }

        result = await test_agent.process_task(task_data)
        assert result["success"] is True

        # Simulate context storage
        await test_agent.store_context(
            content="Task processing context",
            importance_score=0.8,
            category="task_execution",
        )

        # Verify context was stored
        mock_database.store_context.assert_called()

        await test_agent.stop()

    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, test_agent, mock_database):
        """Test system health monitoring during agent operation."""

        await test_agent.start()

        # Simulate health check
        health_status = await mock_database.health_check()
        assert health_status is True

        # Record system metrics
        await mock_database.record_system_metric(
            metric_name="agent_task_completion_time",
            metric_type="gauge",
            value=0.5,
            agent_id=str(test_agent.agent_id),
        )

        mock_database.record_system_metric.assert_called()

        await test_agent.stop()


class TestWorkflowPerformance:
    """Test workflow performance and scalability."""

    @pytest.mark.asyncio
    async def test_concurrent_task_processing(
        self, mock_database, mock_message_broker, mock_task_queue
    ):
        """Test processing multiple tasks concurrently."""

        # Create an agent capable of concurrent processing
        agent = MockSimpleAgent("concurrent-agent", "test")
        agent.db_manager = mock_database
        agent.message_broker = mock_message_broker

        await agent.start()

        # Create multiple tasks
        tasks = [
            {"description": f"Concurrent task {i}", "priority": i} for i in range(5)
        ]

        # Process tasks concurrently
        results = await asyncio.gather(*[agent.process_task(task) for task in tasks])

        # Verify all tasks completed successfully
        assert len(results) == 5
        for result in results:
            assert result["success"] is True

        assert len(agent.processed_tasks) == 5

        await agent.stop()

    @pytest.mark.asyncio
    async def test_workflow_timing(self, test_agent):
        """Test workflow timing and performance metrics."""

        start_time = datetime.now(UTC)

        await test_agent.start()

        # Process a task and measure timing
        task_start = datetime.now(UTC)

        task_data = {"description": "Performance timing test task", "priority": 1}

        result = await test_agent.process_task(task_data)

        task_end = datetime.now(UTC)
        processing_time = (task_end - task_start).total_seconds()

        # Verify reasonable processing time (should be fast for mock)
        assert processing_time < 1.0  # Less than 1 second
        assert result["success"] is True

        await test_agent.stop()

        total_time = (datetime.now(UTC) - start_time).total_seconds()
        assert total_time < 2.0  # Total workflow under 2 seconds
