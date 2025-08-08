"""Real system integration tests using actual Redis and PostgreSQL.

Tests the actual system components without mocks to validate
real-world behavior and performance.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import AsyncGenerator

from src.core.async_db import AsyncDatabaseManager
from src.core.task_queue import TaskQueue
from src.core.message_broker import MessageBroker
from src.core.task_queue import Task, TaskStatus, TaskPriority
from src.core.config import get_settings


@pytest.fixture
async def real_db() -> AsyncGenerator[AsyncDatabaseManager, None]:
    """Real database connection for integration testing."""
    settings = get_settings()
    db_manager = AsyncDatabaseManager()
    await db_manager.initialize()

    try:
        yield db_manager
    finally:
        await db_manager.close()


@pytest.fixture
async def real_redis() -> AsyncGenerator:
    """Real Redis connection for integration testing."""
    import redis.asyncio as redis

    settings = get_settings()

    # Parse Redis URL
    redis_client = redis.from_url(settings.redis_url, decode_responses=True)

    # Clean test keys before test
    await redis_client.flushdb()

    try:
        yield redis_client
    finally:
        await redis_client.flushdb()
        await redis_client.aclose()


@pytest.fixture
async def real_task_queue(real_redis) -> AsyncGenerator[TaskQueue, None]:
    """Real task queue using actual Redis."""
    # Use the same Redis URL as the real Redis client
    settings = get_settings()
    queue = TaskQueue(redis_url=settings.redis_url)
    await queue.initialize()

    try:
        yield queue
    finally:
        # Cleanup test data
        await real_redis.flushdb()


@pytest.fixture
async def real_message_broker(real_redis) -> AsyncGenerator[MessageBroker, None]:
    """Real message broker using actual Redis."""
    broker = MessageBroker(redis_client=real_redis)
    await broker.initialize()

    try:
        yield broker
    finally:
        await real_redis.flushdb()


class TestRealSystemIntegration:
    """Integration tests using real Redis and PostgreSQL."""

    @pytest.mark.asyncio
    async def test_real_task_queue_operations(self, real_task_queue: TaskQueue):
        """Test task queue operations with real Redis."""
        # Create a test task
        task = Task(
            title="Real integration test task",
            description="Testing real Redis integration",
            task_type="test",
            priority=TaskPriority.NORMAL,
            status=TaskStatus.PENDING,
        )

        # Submit task
        task_id = await real_task_queue.submit_task(task)
        assert task_id == task.id

        # Get task from queue
        retrieved_task = await real_task_queue.get_task()
        assert retrieved_task is not None
        assert retrieved_task.title == task.title
        assert retrieved_task.description == task.description

        # Start the task
        started = await real_task_queue.start_task(retrieved_task.id)
        assert started is True

        # Complete the task
        completed = await real_task_queue.complete_task(
            retrieved_task.id, {"result": "success"}
        )
        assert completed is True

    @pytest.mark.asyncio
    async def test_real_message_broker_pubsub(self, real_message_broker: MessageBroker):
        """Test message broker pub/sub with real Redis."""
        channel = "test_channel"
        received_messages = []

        # Define message handler
        async def message_handler(message: dict):
            received_messages.append(message)

        # Subscribe to channel
        await real_message_broker.subscribe(channel, message_handler)

        # Give subscription time to establish
        await asyncio.sleep(0.1)

        # Publish message
        test_message = {"type": "test", "data": "real integration test"}
        await real_message_broker.publish(channel, test_message)

        # Give message time to be delivered
        await asyncio.sleep(0.1)

        # Verify message received
        assert len(received_messages) == 1
        assert received_messages[0]["type"] == "test"
        assert received_messages[0]["data"] == "real integration test"

    @pytest.mark.asyncio
    async def test_real_database_task_persistence(self, real_db: AsyncDatabaseManager):
        """Test task persistence with real PostgreSQL."""
        # Create a test task
        task_data = {
            "title": "Database integration test",
            "description": "Testing real PostgreSQL integration",
            "task_type": "test",
            "priority": 5,
            "status": "pending",
            "assigned_agent_id": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        async with real_db.get_session() as session:
            # Insert task
            result = await session.execute(
                """
                INSERT INTO tasks (title, description, task_type, priority, status, assigned_agent_id, created_at, updated_at)
                VALUES (:title, :description, :task_type, :priority, :status, :assigned_agent_id, :created_at, :updated_at)
                RETURNING id
                """,
                task_data,
            )
            task_id = result.fetchone()[0]
            await session.commit()

            # Retrieve task
            result = await session.execute(
                "SELECT title, description, task_type, priority, status FROM tasks WHERE id = :id",
                {"id": task_id},
            )
            row = result.fetchone()

            assert row is not None
            assert row[0] == task_data["title"]
            assert row[1] == task_data["description"]
            assert row[2] == task_data["task_type"]
            assert row[3] == task_data["priority"]
            assert row[4] == task_data["status"]

    @pytest.mark.asyncio
    async def test_real_task_queue_priority_ordering(self, real_task_queue: TaskQueue):
        """Test priority ordering with real Redis."""
        # Create tasks with different priorities
        high_task = Task(
            title="High priority task",
            description="This should be processed first",
            task_type="test",
            priority=TaskPriority.HIGH,
            status=TaskStatus.PENDING,
        )

        low_task = Task(
            title="Low priority task",
            description="This should be processed last",
            task_type="test",
            priority=TaskPriority.LOW,
            status=TaskStatus.PENDING,
        )

        normal_task = Task(
            title="Normal priority task",
            description="This should be processed middle",
            task_type="test",
            priority=TaskPriority.NORMAL,
            status=TaskStatus.PENDING,
        )

        # Submit in reverse priority order
        await real_task_queue.submit_task(low_task)
        await real_task_queue.submit_task(normal_task)
        await real_task_queue.submit_task(high_task)

        # Get tasks and verify order (highest priority first)
        first_task = await real_task_queue.get_task()
        second_task = await real_task_queue.get_task()
        third_task = await real_task_queue.get_task()

        assert first_task.priority == TaskPriority.HIGH
        assert second_task.priority == TaskPriority.NORMAL
        assert third_task.priority == TaskPriority.LOW

    @pytest.mark.asyncio
    async def test_real_system_performance_baseline(
        self, real_task_queue: TaskQueue, real_message_broker: MessageBroker
    ):
        """Test system performance with real components."""
        import time

        # Performance test: enqueue/dequeue operations
        num_tasks = 100
        start_time = time.time()

        # Enqueue tasks
        for i in range(num_tasks):
            task = Task(
                title=f"Performance test task {i}",
                description=f"Performance testing task number {i}",
                task_type="performance",
                priority=TaskPriority.NORMAL,
                status=TaskStatus.PENDING,
            )
            await real_task_queue.enqueue(task)

        enqueue_time = time.time() - start_time

        # Dequeue tasks
        start_time = time.time()
        dequeued_count = 0

        while True:
            task = await real_task_queue.dequeue()
            if task is None:
                break
            dequeued_count += 1

        dequeue_time = time.time() - start_time

        # Performance assertions (should be reasonable for CI)
        assert enqueue_time < 5.0  # Should enqueue 100 tasks in under 5 seconds
        assert dequeue_time < 5.0  # Should dequeue 100 tasks in under 5 seconds
        assert dequeued_count == num_tasks  # All tasks should be processed

        # Message broker performance test
        num_messages = 50
        received_count = 0

        async def counter_handler(message: dict):
            nonlocal received_count
            received_count += 1

        await real_message_broker.subscribe("perf_test", counter_handler)
        await asyncio.sleep(0.1)  # Let subscription establish

        start_time = time.time()

        # Publish messages
        for i in range(num_messages):
            await real_message_broker.publish(
                "perf_test", {"id": i, "data": f"message {i}"}
            )

        # Wait for message processing
        await asyncio.sleep(1.0)

        publish_time = time.time() - start_time

        # Performance assertions
        assert publish_time < 3.0  # Should publish 50 messages in under 3 seconds
        assert received_count == num_messages  # All messages should be received
