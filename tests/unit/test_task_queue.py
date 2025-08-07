import asyncio

import pytest

from src.core.task_queue import Task, TaskPriority, TaskQueue, TaskStatus


@pytest.fixture
async def task_queue_instance():
    """Fixture to create and initialize a TaskQueue instance for testing."""
    # Use a separate Redis database for testing
    queue = TaskQueue(redis_url="redis://localhost:6381/1")
    await queue.initialize()
    # Clear the test database before each test
    await queue.redis_client.flushdb()
    yield queue
    # Clean up after tests
    await queue.redis_client.flushdb()
    await queue.redis_client.close()


@pytest.mark.asyncio
async def test_submit_and_get_task(task_queue_instance: TaskQueue):
    """Test submitting a task and retrieving it from the queue."""
    task = Task(
        title="Test Task",
        description="A simple test task",
        task_type="test",
        priority=TaskPriority.NORMAL,
    )
    task_id = await task_queue_instance.submit_task(task)
    assert task_id == task.id

    retrieved_task = await task_queue_instance.get_task(
        "test_agent", priorities=[TaskPriority.NORMAL]
    )
    assert retrieved_task is not None
    assert retrieved_task.id == task.id
    assert retrieved_task.title == "Test Task"
    assert retrieved_task.status == TaskStatus.ASSIGNED
    assert retrieved_task.agent_id == "test_agent"


@pytest.mark.asyncio
async def test_task_priority(task_queue_instance: TaskQueue):
    """Test that tasks are retrieved in priority order."""
    low_priority_task = Task(
        title="Low Prio", description="low", task_type="test", priority=TaskPriority.LOW
    )
    high_priority_task = Task(
        title="High Prio",
        description="high",
        task_type="test",
        priority=TaskPriority.HIGH,
    )

    await task_queue_instance.submit_task(low_priority_task)
    await task_queue_instance.submit_task(high_priority_task)

    # First retrieved task should be the high priority one
    first_task = await task_queue_instance.get_task("agent1")
    assert first_task is not None
    assert first_task.id == high_priority_task.id

    # Second should be the low priority one
    second_task = await task_queue_instance.get_task("agent2")
    assert second_task is not None
    assert second_task.id == low_priority_task.id


@pytest.mark.asyncio
async def test_task_completion(task_queue_instance: TaskQueue):
    """Test marking a task as complete."""
    task = Task(title="Completable Task", description="d", task_type="test")
    task_id = await task_queue_instance.submit_task(task)
    retrieved_task = await task_queue_instance.get_task("agent")
    assert retrieved_task is not None

    await task_queue_instance.start_task(task_id)
    status_started = await task_queue_instance.get_task_status(task_id)
    assert status_started.status == TaskStatus.IN_PROGRESS

    result_data = {"output": "success"}
    await task_queue_instance.complete_task(task_id, result=result_data)

    status_completed = await task_queue_instance.get_task_status(task_id)
    assert status_completed.status == TaskStatus.COMPLETED
    assert status_completed.result == result_data
    assert status_completed.completed_at is not None


@pytest.mark.asyncio
async def test_task_failure_and_retry(task_queue_instance: TaskQueue):
    """Test task failure, retry logic, and permanent failure."""
    task = Task(
        title="Failing Task",
        description="d",
        task_type="test",
        max_retries=1,
        priority=TaskPriority.CRITICAL,
    )
    task_id = await task_queue_instance.submit_task(task)

    # First attempt (fails)
    retrieved_task = await task_queue_instance.get_task("agent")
    assert retrieved_task is not None
    await task_queue_instance.fail_task(task_id, "First failure", retry=True)

    # Check if it's requeued for retry
    status_after_fail = await task_queue_instance.get_task_status(task_id)
    assert status_after_fail.status == TaskStatus.PENDING
    assert status_after_fail.retry_count == 1

    # Second attempt (fails again, should be permanent)
    retrieved_again = await task_queue_instance.get_task(
        "agent", priorities=[TaskPriority.CRITICAL]
    )
    assert retrieved_again is not None
    await task_queue_instance.fail_task(task_id, "Second failure", retry=True)

    status_permanent_fail = await task_queue_instance.get_task_status(task_id)
    assert status_permanent_fail.status == TaskStatus.FAILED
    assert status_permanent_fail.error_message == "Second failure"


@pytest.mark.asyncio
async def test_task_dependencies(task_queue_instance: TaskQueue):
    """Test that a task with dependencies only runs after they are complete."""
    dep1 = Task(title="Dep 1", description="d1", task_type="test")
    dep2 = Task(title="Dep 2", description="d2", task_type="test")
    main_task = Task(
        title="Main Task",
        description="main",
        task_type="test",
        dependencies=[dep1.id, dep2.id],
    )

    await task_queue_instance.submit_task(dep1)
    await task_queue_instance.submit_task(dep2)
    await task_queue_instance.submit_task(main_task)

    # Main task should not be available yet
    assert await task_queue_instance.get_task("agent") is None

    # Complete first dependency
    retrieved_dep1 = await task_queue_instance.get_task("agent_dep")
    assert retrieved_dep1 is not None
    await task_queue_instance.complete_task(retrieved_dep1.id)

    # Main task should still not be available
    assert await task_queue_instance.get_task("agent") is None

    # Complete second dependency
    retrieved_dep2 = await task_queue_instance.get_task("agent_dep")
    assert retrieved_dep2 is not None
    await task_queue_instance.complete_task(retrieved_dep2.id)

    # Now the main task should be available
    retrieved_main = await task_queue_instance.get_task("agent")
    assert retrieved_main is not None
    assert retrieved_main.id == main_task.id


@pytest.mark.asyncio
async def test_cancel_task(task_queue_instance: TaskQueue):
    """Test cancelling a pending task."""
    task = Task(title="Cancellable Task", description="d", task_type="test")
    task_id = await task_queue_instance.submit_task(task)

    cancelled = await task_queue_instance.cancel_task(task_id)
    assert cancelled is True

    status = await task_queue_instance.get_task_status(task_id)
    assert status.status == TaskStatus.CANCELLED

    # Ensure it's not in the queue anymore
    assert await task_queue_instance.get_task("agent") is None


@pytest.mark.asyncio
async def test_queue_stats(task_queue_instance: TaskQueue):
    """Test the accuracy of queue statistics."""
    # Submit a variety of tasks
    await task_queue_instance.submit_task(
        Task(title="t1", description="d", task_type="t", priority=TaskPriority.HIGH)
    )
    await task_queue_instance.submit_task(
        Task(title="t2", description="d", task_type="t", priority=TaskPriority.NORMAL)
    )
    task3 = Task(title="t3", description="d", task_type="t")
    await task_queue_instance.submit_task(task3)

    # Process one task to completion
    t3_retrieved = await task_queue_instance.get_task("agent")
    await task_queue_instance.complete_task(t3_retrieved.id)

    stats = await task_queue_instance.get_queue_stats()
    assert stats.total_tasks == 3
    assert stats.pending_tasks == 1  # t2 is pending, t1 is retrieved but not started
    assert stats.assigned_tasks == 1
    assert stats.completed_tasks == 1
    assert stats.failed_tasks == 0
    assert stats.queue_size_by_priority[TaskPriority.HIGH] == 1
    assert stats.queue_size_by_priority[TaskPriority.NORMAL] == 1


@pytest.mark.asyncio
async def test_task_timeout_cleanup(task_queue_instance: TaskQueue):
    """Test that expired tasks are cleaned up and retried."""
    task = Task(
        title="Timeout Task", description="d", task_type="test", timeout_seconds=1
    )
    task_id = await task_queue_instance.submit_task(task)

    retrieved = await task_queue_instance.get_task("agent")
    await task_queue_instance.start_task(retrieved.id)

    # Wait for task to expire
    await asyncio.sleep(1.5)

    cleaned_count = await task_queue_instance.cleanup_expired_tasks()
    assert cleaned_count == 1

    status = await task_queue_instance.get_task_status(task_id)
    assert status.status == TaskStatus.PENDING  # Re-queued for retry
    assert status.retry_count == 1
    assert "Task timeout" in status.error_message


@pytest.mark.asyncio
async def test_get_unassigned_tasks(task_queue_instance: TaskQueue):
    """Test retrieving all unassigned tasks."""
    task1 = Task(title="Unassigned 1", description="d", task_type="t")
    task2 = Task(title="Unassigned 2", description="d", task_type="t")
    assigned_task = Task(title="Assigned", description="d", task_type="t")

    await task_queue_instance.submit_task(task1)
    await task_queue_instance.submit_task(task2)
    await task_queue_instance.submit_task(assigned_task)

    # Assign one task
    await task_queue_instance.get_task("test_agent")

    unassigned = await task_queue_instance.get_unassigned_tasks()
    assert len(unassigned) == 2
    unassigned_ids = {t.id for t in unassigned}
    assert task1.id in unassigned_ids
    assert task2.id in unassigned_ids
    assert assigned_task.id not in unassigned_ids


@pytest.mark.asyncio
async def test_get_total_tasks_and_completed_count(task_queue_instance: TaskQueue):
    """Test getting total and completed task counts."""
    for i in range(5):
        await task_queue_instance.submit_task(
            Task(title=f"T{i}", description="d", task_type="t")
        )

    assert await task_queue_instance.get_total_tasks() == 5
    assert await task_queue_instance.get_completed_tasks_count() == 0

    # Complete 2 tasks
    for _ in range(2):
        task = await task_queue_instance.get_task("agent")
        await task_queue_instance.complete_task(task.id)

    assert await task_queue_instance.get_total_tasks() == 5
    assert await task_queue_instance.get_completed_tasks_count() == 2
    assert await task_queue_instance.get_failed_tasks_count() == 0
