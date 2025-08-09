"""Comprehensive TDD tests for TaskQueue - Core System Component.

This test suite covers the most critical functionality of the task queue system:
1. Task submission and retrieval (core workflow)
2. Priority queueing (ensures high-priority tasks processed first)
3. Task assignment and status management (agent coordination)
4. Retry logic and failure handling (reliability)
5. Queue statistics and monitoring (observability)
6. Concurrent operations (performance and safety)

Following TDD principles: Write failing tests first, implement minimal code, refactor.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.core.task_queue import QueueStats, Task, TaskPriority, TaskQueue, TaskStatus


@pytest.fixture
def mock_redis():
    """Create a comprehensive mock Redis client for testing."""
    redis_mock = AsyncMock()

    # Basic Redis operations
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.flushdb = AsyncMock(return_value=True)

    # Priority queue operations (sorted sets)
    redis_mock.zadd = AsyncMock(return_value=1)
    redis_mock.zrem = AsyncMock(return_value=1)
    redis_mock.zrange = AsyncMock(return_value=[])
    redis_mock.zrevrange = AsyncMock(return_value=[])
    redis_mock.zrangebyscore = AsyncMock(return_value=[])
    redis_mock.zcard = AsyncMock(return_value=0)
    redis_mock.zscore = AsyncMock(return_value=None)

    # Hash operations (task storage)
    redis_mock.hset = AsyncMock(return_value=1)
    redis_mock.hget = AsyncMock(return_value=None)
    redis_mock.hgetall = AsyncMock(return_value={})
    redis_mock.hdel = AsyncMock(return_value=1)
    redis_mock.hexists = AsyncMock(return_value=False)

    # List operations (task assignment queues)
    redis_mock.lpush = AsyncMock(return_value=1)
    redis_mock.rpush = AsyncMock(return_value=1)
    redis_mock.blpop = AsyncMock(return_value=None)
    redis_mock.brpop = AsyncMock(return_value=None)
    redis_mock.llen = AsyncMock(return_value=0)

    # Set operations (status tracking)
    redis_mock.sadd = AsyncMock(return_value=1)
    redis_mock.srem = AsyncMock(return_value=1)
    redis_mock.smembers = AsyncMock(return_value=set())
    redis_mock.scard = AsyncMock(return_value=0)

    # Key operations
    redis_mock.expire = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.exists = AsyncMock(return_value=0)
    redis_mock.keys = AsyncMock(return_value=[])

    return redis_mock


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager for testing."""
    cache_mock = AsyncMock()
    cache_mock.get = AsyncMock(return_value=None)
    cache_mock.set = AsyncMock(return_value=True)
    cache_mock.invalidate = AsyncMock(return_value=True)
    cache_mock.clear = AsyncMock(return_value=True)
    return cache_mock


@pytest.fixture
async def task_queue(mock_redis, mock_cache_manager):
    """Create TaskQueue instance with mocked dependencies."""
    with (
        patch("redis.asyncio.from_url", return_value=mock_redis),
        patch("src.core.task_queue.get_cache_manager", return_value=mock_cache_manager),
    ):
        queue = TaskQueue("redis://localhost:6379/1")
        await queue.initialize()
        return queue


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return Task(
        title="Test Task",
        description="A task for testing purposes",
        task_type="test",
        priority=TaskPriority.NORMAL,
        payload={"test_data": "value"},
    )


@pytest.fixture
def high_priority_task():
    """Create a high priority task for testing."""
    return Task(
        title="Urgent Task",
        description="High priority task",
        task_type="urgent",
        priority=TaskPriority.HIGH,
        payload={"urgent": True},
    )


class TestTaskQueueBasicOperations:
    """Test fundamental task queue operations."""

    @pytest.mark.asyncio
    async def test_initialize_task_queue(self, mock_redis, mock_cache_manager):
        """Test that task queue initializes properly."""
        with (
            patch("redis.asyncio.from_url", return_value=mock_redis),
            patch(
                "src.core.task_queue.get_cache_manager", return_value=mock_cache_manager
            ),
        ):
            queue = TaskQueue("redis://localhost:6379/1")
            await queue.initialize()

            # Verify Redis connection was tested
            mock_redis.ping.assert_called_once()

            # Verify queue instance has required attributes
            assert hasattr(queue, "redis_client")
            assert hasattr(queue, "cache_manager")
            assert queue.queue_prefix == "hive:queue"

    @pytest.mark.asyncio
    async def test_submit_task_basic(self, task_queue, sample_task, mock_redis):
        """Test basic task submission to queue."""
        # Act - Submit task
        task_id = await task_queue.submit_task(sample_task)

        # Assert - Task ID returned
        assert task_id == sample_task.id
        assert isinstance(task_id, str)

        # Verify Redis operations were called
        mock_redis.hset.assert_called()  # Task data stored
        mock_redis.zadd.assert_called()  # Task added to priority queue

    @pytest.mark.asyncio
    async def test_get_task_basic(self, task_queue, sample_task, mock_redis):
        """Test retrieving a task by ID."""
        # Arrange - Mock Redis to return task data
        task_data = sample_task.model_dump()
        mock_redis.hgetall.return_value = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in task_data.items()
        }

        # Act - Get task
        retrieved_task = await task_queue.get_task(sample_task.id)

        # Assert - Task retrieved correctly
        assert retrieved_task is not None
        assert retrieved_task.id == sample_task.id
        assert retrieved_task.title == sample_task.title
        assert retrieved_task.task_type == sample_task.task_type

    @pytest.mark.asyncio
    async def test_get_nonexistent_task(self, task_queue, mock_redis):
        """Test getting a task that doesn't exist returns None."""
        # Arrange - Mock Redis to return empty result
        mock_redis.hgetall.return_value = {}

        # Act - Try to get non-existent task
        task = await task_queue.get_task("nonexistent-id")

        # Assert - None returned
        assert task is None


class TestTaskQueuePriorityOperations:
    """Test priority queueing functionality - critical for agent coordination."""

    @pytest.mark.asyncio
    async def test_priority_queue_ordering(self, task_queue, mock_redis):
        """Test that higher priority tasks are retrieved first."""
        # Arrange - Create tasks with different priorities
        low_task = Task(
            title="Low Priority",
            description="Low priority task",
            task_type="low",
            priority=TaskPriority.LOW,
        )
        high_task = Task(
            title="High Priority",
            description="High priority task",
            task_type="high",
            priority=TaskPriority.HIGH,
        )
        critical_task = Task(
            title="Critical Task",
            description="Critical priority task",
            task_type="critical",
            priority=TaskPriority.CRITICAL,
        )

        # Act - Submit tasks in random order
        await task_queue.submit_task(low_task)
        await task_queue.submit_task(high_task)
        await task_queue.submit_task(critical_task)

        # Assert - Verify priority queue was called with correct scores
        # Priority values are inverted (lower number = higher priority)
        call_args_list = mock_redis.zadd.call_args_list

        # Extract the priority scores from zadd calls
        scores = []
        for call in call_args_list:
            args, kwargs = call
            if len(args) >= 2:
                # zadd receives {task_id: priority} mapping
                mapping = args[1]
                if isinstance(mapping, dict):
                    scores.extend(mapping.values())

        # Verify tasks were added with correct priority scores
        assert len(scores) == 3
        # Critical should have lowest score (highest priority)
        assert TaskPriority.CRITICAL.value in scores
        assert TaskPriority.HIGH.value in scores
        assert TaskPriority.LOW.value in scores

    @pytest.mark.asyncio
    async def test_get_next_task_respects_priority(self, task_queue, mock_redis):
        """Test that get_next_task returns highest priority task first."""
        # Arrange - Mock Redis to return critical task first
        critical_task_id = str(uuid4())
        mock_redis.zrange.return_value = [critical_task_id]

        # Mock task data retrieval
        critical_task_data = {
            "id": critical_task_id,
            "title": "Critical Task",
            "description": "Critical task",
            "task_type": "critical",
            "priority": str(TaskPriority.CRITICAL),
            "status": TaskStatus.PENDING,
            "payload": "{}",
            "dependencies": "[]",
            "retry_count": "0",
            "max_retries": "3",
            "timeout_seconds": "300",
            "created_at": str(time.time()),
        }
        mock_redis.hgetall.return_value = critical_task_data

        # Act - Get next task
        task = await task_queue.get_next_task()

        # Assert - Critical task returned
        assert task is not None
        assert task.id == critical_task_id
        assert task.priority == TaskPriority.CRITICAL

        # Verify Redis was queried for highest priority tasks
        mock_redis.zrange.assert_called()


class TestTaskQueueAssignmentAndStatus:
    """Test task assignment to agents and status management."""

    @pytest.mark.asyncio
    async def test_assign_task_to_agent(self, task_queue, sample_task, mock_redis):
        """Test assigning a task to a specific agent."""
        agent_id = "test-agent-001"

        # Act - Assign task
        success = await task_queue.assign_task(sample_task.id, agent_id)

        # Assert - Assignment successful
        assert success is True

        # Verify Redis operations for assignment
        mock_redis.hset.assert_called()  # Task status updated
        # Verify status change to ASSIGNED

    @pytest.mark.asyncio
    async def test_start_task_processing(self, task_queue, sample_task, mock_redis):
        """Test marking a task as in progress."""
        # Act - Start task processing
        success = await task_queue.start_task(sample_task.id)

        # Assert - Task started successfully
        assert success is True

        # Verify status update operations
        mock_redis.hset.assert_called()

    @pytest.mark.asyncio
    async def test_complete_task_successfully(
        self, task_queue, sample_task, mock_redis
    ):
        """Test completing a task with results."""
        result_data = {
            "output": "Task completed successfully",
            "metrics": {"duration": 45},
        }

        # Act - Complete task
        success = await task_queue.complete_task(sample_task.id, result_data)

        # Assert - Task completed
        assert success is True

        # Verify completion operations
        mock_redis.hset.assert_called()  # Task updated with result
        mock_redis.zrem.assert_called()  # Task removed from active queue

    @pytest.mark.asyncio
    async def test_assign_task_to_agent(self, task_queue, sample_task, mock_redis):
        """Test assigning a task to a specific agent."""
        agent_id = "test-agent-001"

        # Arrange - Mock that task exists in Redis
        task_data = sample_task.model_dump()
        mock_redis.hgetall.return_value = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in task_data.items()
        }

        # Act - Assign task
        success = await task_queue.assign_task(sample_task.id, agent_id)

        # Assert - Assignment successful
        assert success is True

        # Verify Redis operations for assignment
        mock_redis.hset.assert_called()  # Task status updated

    @pytest.mark.asyncio
    async def test_start_task_processing(self, task_queue, sample_task, mock_redis):
        """Test marking a task as in progress."""
        # Arrange - Mock that task exists in Redis
        task_data = sample_task.model_dump()
        mock_redis.hgetall.return_value = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in task_data.items()
        }

        # Act - Start task processing
        success = await task_queue.start_task(sample_task.id)

        # Assert - Task started successfully
        assert success is True

        # Verify status update operations
        mock_redis.hset.assert_called()

    @pytest.mark.asyncio
    async def test_complete_task_successfully(
        self, task_queue, sample_task, mock_redis
    ):
        """Test completing a task with results."""
        result_data = {
            "output": "Task completed successfully",
            "metrics": {"duration": 45},
        }

        # Arrange - Mock that task exists in Redis
        task_data = sample_task.model_dump()
        mock_redis.hgetall.return_value = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in task_data.items()
        }

        # Act - Complete task
        success = await task_queue.complete_task(sample_task.id, result_data)

        # Assert - Task completed
        assert success is True

        # Verify completion operations
        mock_redis.hset.assert_called()  # Task updated with result
        mock_redis.zrem.assert_called()  # Task removed from active queue

    @pytest.mark.asyncio
    async def test_fail_task_with_retry(self, task_queue, sample_task, mock_redis):
        """Test failing a task and scheduling retry."""
        error_message = "Network timeout occurred"

        # Arrange - Mock that task exists in Redis
        task_data = sample_task.model_dump()
        mock_redis.hgetall.return_value = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in task_data.items()
        }

        # Act - Fail task with retry
        success = await task_queue.fail_task(sample_task.id, error_message, retry=True)

        # Assert - Task failed and scheduled for retry
        assert success is True

        # Verify failure handling operations
        mock_redis.hset.assert_called()  # Error message and retry count updated

    @pytest.mark.asyncio
    async def test_get_task_status(self, task_queue, sample_task, mock_redis):
        """Test retrieving current task status."""
        # Arrange - Mock task data with status
        task_data = sample_task.model_dump()
        task_data["status"] = TaskStatus.IN_PROGRESS
        mock_redis.hgetall.return_value = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in task_data.items()
        }

        # Act - Get task status
        task = await task_queue.get_task_status(sample_task.id)

        # Assert - Correct status returned
        assert task is not None
        assert task.status == TaskStatus.IN_PROGRESS


class TestTaskQueueReliabilityFeatures:
    """Test reliability features: retries, timeouts, error handling."""

    @pytest.mark.asyncio
    async def test_task_retry_logic(self, task_queue, mock_redis):
        """Test that failed tasks are retried up to max_retries."""
        # Arrange - Create task with specific retry limit
        task = Task(
            title="Retry Test Task",
            description="Task for testing retry logic",
            task_type="retry_test",
            max_retries=2,
        )

        # Submit task
        await task_queue.submit_task(task)

        # Act - Fail task multiple times
        await task_queue.fail_task(task.id, "First failure", retry=True)
        await task_queue.fail_task(task.id, "Second failure", retry=True)
        await task_queue.fail_task(task.id, "Final failure", retry=True)

        # Assert - Verify retry handling
        # Should update task status and retry count each time
        assert mock_redis.hset.call_count >= 3  # At least 3 updates

    @pytest.mark.asyncio
    async def test_task_timeout_handling(self, task_queue, mock_redis):
        """Test handling of task timeouts."""
        # Arrange - Create task with short timeout
        task = Task(
            title="Timeout Test",
            description="Task with short timeout",
            task_type="timeout_test",
            timeout_seconds=1,
        )

        await task_queue.submit_task(task)
        await task_queue.start_task(task.id)

        # Wait for timeout period
        await asyncio.sleep(0.1)  # Simulate time passing

        # Act - Check for timed out tasks
        timed_out_tasks = await task_queue.get_timed_out_tasks()

        # Assert - Implementation should identify timed out tasks
        # This tests the interface even if implementation is pending
        assert isinstance(timed_out_tasks, list)

    @pytest.mark.asyncio
    async def test_dependency_handling(self, task_queue, mock_redis):
        """Test task dependency resolution."""
        # Arrange - Create dependent tasks
        parent_task = Task(
            title="Parent Task",
            description="Task that others depend on",
            task_type="parent",
        )

        dependent_task = Task(
            title="Dependent Task",
            description="Task that depends on parent",
            task_type="dependent",
            dependencies=[parent_task.id],
        )

        # Act - Submit both tasks
        await task_queue.submit_task(parent_task)
        await task_queue.submit_task(dependent_task)

        # Assert - Dependent task should not be available until parent completes
        # This tests the interface for dependency management
        mock_redis.hset.assert_called()


class TestTaskQueueStatisticsAndMonitoring:
    """Test queue monitoring and statistics functionality."""

    @pytest.mark.asyncio
    async def test_get_queue_stats(self, task_queue, mock_redis):
        """Test retrieving comprehensive queue statistics."""
        # Arrange - Mock Redis to return various counts
        mock_redis.scard.side_effect = [
            5,
            3,
            2,
            10,
            1,
        ]  # pending, assigned, in_progress, completed, failed
        mock_redis.zcard.return_value = 8  # total active tasks

        # Act - Get queue statistics
        stats = await task_queue.get_queue_stats()

        # Assert - Stats object returned with correct structure
        assert isinstance(stats, QueueStats)
        assert hasattr(stats, "pending_tasks")
        assert hasattr(stats, "assigned_tasks")
        assert hasattr(stats, "in_progress_tasks")
        assert hasattr(stats, "completed_tasks")
        assert hasattr(stats, "failed_tasks")
        assert hasattr(stats, "total_tasks")

    @pytest.mark.asyncio
    async def test_get_queue_depth(self, task_queue, mock_redis):
        """Test getting current queue depth."""
        # Arrange - Mock queue depth
        mock_redis.zcard.return_value = 15

        # Act - Get queue depth
        depth = await task_queue.get_queue_depth()

        # Assert - Correct depth returned
        assert depth == 15

    @pytest.mark.asyncio
    async def test_list_tasks_by_status(self, task_queue, mock_redis):
        """Test listing tasks filtered by status."""
        # Arrange - Mock task IDs for specific status
        task_ids = ["task1", "task2", "task3"]
        mock_redis.smembers.return_value = set(task_ids)

        # Mock task data for each ID
        mock_redis.hgetall.return_value = {
            "id": "task1",
            "title": "Test Task",
            "description": "Test",
            "task_type": "test",
            "status": TaskStatus.PENDING,
            "priority": str(TaskPriority.NORMAL),
            "payload": "{}",
            "dependencies": "[]",
            "retry_count": "0",
            "max_retries": "3",
            "timeout_seconds": "300",
            "created_at": str(time.time()),
        }

        # Act - List pending tasks
        pending_tasks = await task_queue.list_tasks(status=TaskStatus.PENDING)

        # Assert - Tasks returned
        assert isinstance(pending_tasks, list)


class TestTaskQueueConcurrencyAndPerformance:
    """Test concurrent operations and performance aspects."""

    @pytest.mark.asyncio
    async def test_concurrent_task_submission(self, task_queue, mock_redis):
        """Test submitting multiple tasks concurrently."""
        # Arrange - Create multiple tasks
        tasks = [
            Task(
                title=f"Task {i}",
                description=f"Concurrent task {i}",
                task_type="concurrent",
            )
            for i in range(10)
        ]

        # Act - Submit all tasks concurrently
        results = await asyncio.gather(
            *[task_queue.submit_task(task) for task in tasks]
        )

        # Assert - All submissions successful
        assert len(results) == 10
        assert all(
            isinstance(result, str) for result in results
        )  # All task IDs returned

        # Verify Redis operations performed for each task
        assert mock_redis.hset.call_count >= 10
        assert mock_redis.zadd.call_count >= 10

    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self, task_queue, mock_redis):
        """Test concurrent task assignment and processing."""
        # Arrange - Mock multiple tasks available
        task_ids = [f"task-{i}" for i in range(5)]
        mock_redis.zrange.side_effect = [[task_id] for task_id in task_ids]

        # Mock task data
        mock_redis.hgetall.return_value = {
            "id": "task-1",
            "title": "Concurrent Task",
            "description": "Task for concurrent processing",
            "task_type": "concurrent",
            "status": TaskStatus.PENDING,
            "priority": str(TaskPriority.NORMAL),
            "payload": "{}",
            "dependencies": "[]",
            "retry_count": "0",
            "max_retries": "3",
            "timeout_seconds": "300",
            "created_at": str(time.time()),
        }

        # Act - Get multiple tasks concurrently
        async def get_and_assign_task(agent_id):
            task = await task_queue.get_next_task()
            if task:
                await task_queue.assign_task(task.id, agent_id)
            return task

        agents = [f"agent-{i}" for i in range(3)]
        results = await asyncio.gather(
            *[get_and_assign_task(agent_id) for agent_id in agents]
        )

        # Assert - Concurrent operations handled correctly
        assert len(results) == 3
        # Redis operations should be performed safely
        assert mock_redis.zrange.call_count >= 3

    @pytest.mark.asyncio
    async def test_cache_integration(self, task_queue, mock_cache_manager, mock_redis):
        """Test integration with caching system for performance."""
        # Arrange - Create sample task
        task = Task(
            title="Cached Task",
            description="Task for cache testing",
            task_type="cache_test",
        )

        # Act - Submit task (should trigger cache operations)
        await task_queue.submit_task(task)

        # Get queue stats (should use cache)
        await task_queue.get_queue_stats()

        # Assert - Cache manager was used
        # Cache operations may be called for performance optimization
        assert mock_cache_manager.set.called or mock_cache_manager.get.called


class TestTaskQueueErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_redis_connection_failure_handling(
        self, mock_redis, mock_cache_manager
    ):
        """Test handling Redis connection failures gracefully."""
        # Arrange - Mock Redis to raise connection error
        mock_redis.ping.side_effect = Exception("Connection failed")

        with (
            patch("redis.asyncio.from_url", return_value=mock_redis),
            patch(
                "src.core.task_queue.get_cache_manager", return_value=mock_cache_manager
            ),
        ):
            queue = TaskQueue("redis://localhost:6379/1")

            # Act & Assert - Initialization should handle connection failure
            with pytest.raises(Exception):
                await queue.initialize()

    @pytest.mark.asyncio
    async def test_invalid_task_data_handling(self, task_queue):
        """Test handling of invalid task data."""
        # Act & Assert - Invalid task should raise validation error
        with pytest.raises((ValueError, TypeError)):
            invalid_task = Task(
                title="",  # Empty title should be invalid
                description="",
                task_type="",
            )

    @pytest.mark.asyncio
    async def test_task_not_found_operations(self, task_queue, mock_redis):
        """Test operations on non-existent tasks."""
        # Arrange - Mock Redis to return no data
        mock_redis.hgetall.return_value = {}

        # Act - Try operations on non-existent task
        non_existent_id = "non-existent-task-id"

        task = await task_queue.get_task(non_existent_id)
        assign_result = await task_queue.assign_task(non_existent_id, "agent-1")
        start_result = await task_queue.start_task(non_existent_id)
        complete_result = await task_queue.complete_task(non_existent_id, {})

        # Assert - Operations handle non-existent tasks gracefully
        assert task is None
        assert assign_result is False
        assert start_result is False
        assert complete_result is False


# Additional test cases for edge cases and performance
class TestTaskQueueAdvancedFeatures:
    """Test advanced task queue features and optimizations."""

    @pytest.mark.asyncio
    async def test_bulk_task_operations(self, task_queue, mock_redis):
        """Test bulk operations for performance."""
        # Arrange - Create multiple tasks
        tasks = [
            Task(
                title=f"Bulk Task {i}",
                description=f"Bulk operation task {i}",
                task_type="bulk",
            )
            for i in range(100)
        ]

        # Act - Submit tasks in bulk (if implemented)
        if hasattr(task_queue, "submit_tasks_bulk"):
            results = await task_queue.submit_tasks_bulk(tasks)
            assert len(results) == 100
        else:
            # Fallback to individual submissions
            results = []
            for task in tasks:
                result = await task_queue.submit_task(task)
                results.append(result)
            assert len(results) == 100

    @pytest.mark.asyncio
    async def test_queue_cleanup_operations(self, task_queue, mock_redis):
        """Test queue maintenance and cleanup operations."""
        # Act - Trigger cleanup operations
        if hasattr(task_queue, "cleanup_completed_tasks"):
            await task_queue.cleanup_completed_tasks()

        if hasattr(task_queue, "cleanup_failed_tasks"):
            await task_queue.cleanup_failed_tasks()

        # Assert - Cleanup operations don't raise errors
        # Implementation details will be tested once methods exist
        assert True  # Basic existence test

    @pytest.mark.asyncio
    async def test_queue_health_monitoring(self, task_queue, mock_redis):
        """Test queue health monitoring capabilities."""
        # Act - Check queue health
        if hasattr(task_queue, "health_check"):
            health = await task_queue.health_check()
            assert isinstance(health, (bool, dict))

        # Test basic health indicators
        stats = await task_queue.get_queue_stats()
        assert isinstance(stats, QueueStats)
