"""Integration tests for core system functionality.

Tests the complete workflow of:
1. System startup
2. Task submission
3. Agent coordination
4. Message passing
5. Task completion

This validates the most critical user journeys.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
import redis.asyncio as redis

from src.core.caching import get_cache_manager
from src.core.message_broker import MessageBroker, MessageType
from src.core.task_queue import Task, TaskPriority, TaskQueue, TaskStatus


class TestCoreSystemIntegration:
    """Integration tests for core system workflows."""

    @pytest.fixture
    async def redis_client(self):
        """Create a real Redis client for integration testing."""
        client = redis.Redis.from_url("redis://localhost:6381")
        # Clear test data
        await client.flushdb()
        yield client
        # Cleanup
        await client.flushdb()
        await client.close()

    @pytest.fixture
    async def task_queue(self, redis_client):
        """Create TaskQueue with real Redis."""
        queue = TaskQueue()
        queue.redis_client = redis_client
        await queue.initialize()
        return queue

    @pytest.fixture
    async def message_broker(self, redis_client):
        """Create MessageBroker with real Redis."""
        broker = MessageBroker()
        broker.redis_client = redis_client
        await broker.initialize()
        return broker

    @pytest.mark.asyncio
    async def test_complete_task_workflow(self, task_queue, message_broker):
        """Test complete task submission to completion workflow."""
        # Step 1: Submit a task
        task = Task(
            title="Integration Test Task",
            description="Test task for integration workflow",
            task_type="test_processing",
            priority=TaskPriority.HIGH,
        )

        task_id = await task_queue.submit_task(task)
        assert task_id is not None

        # Step 2: Verify task is in queue
        queue_depth = await task_queue.get_queue_depth()
        assert queue_depth == 1

        retrieved_task = await task_queue.get_task(task_id)
        assert retrieved_task is not None
        assert retrieved_task.status == TaskStatus.PENDING

        # Step 3: Get next task (simulating agent picking up work)
        next_task = await task_queue.get_next_task()
        assert next_task is not None
        assert next_task.id == task_id

        # Step 4: Assign task to agent
        agent_name = "test-agent-1"
        success = await task_queue.assign_task(task_id, agent_name)
        assert success is True

        # Step 5: Start task processing
        await task_queue.start_task(task_id)

        # Verify task status
        task_status = await task_queue.get_task_status(task_id)
        assert task_status == TaskStatus.IN_PROGRESS

        # Step 6: Simulate agent messaging about task progress
        message_sent = await message_broker.send_message(
            from_agent=agent_name,
            to_agent="orchestrator",
            topic="task_progress",
            payload={"task_id": task_id, "status": "processing", "progress": 50},
            message_type=MessageType.DIRECT,
        )
        assert message_sent is True

        # Step 7: Complete the task
        completion_result = {
            "status": "success",
            "output": "Task completed successfully",
        }
        success = await task_queue.complete_task(task_id, completion_result)
        assert success is True

        # Step 8: Verify final state
        final_task = await task_queue.get_task(task_id)
        assert final_task.status == TaskStatus.COMPLETED
        assert final_task.result == completion_result

        # Step 9: Verify queue is empty
        final_queue_depth = await task_queue.get_queue_depth()
        assert final_queue_depth == 0

        # Step 10: Send completion notification
        completion_sent = await message_broker.send_message(
            from_agent=agent_name,
            to_agent="orchestrator",
            topic="task_completed",
            payload={"task_id": task_id, "result": completion_result},
            message_type=MessageType.DIRECT,
        )
        assert completion_sent is True

    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self, task_queue, message_broker):
        """Test coordination between multiple agents."""
        # Submit multiple tasks
        tasks = []
        for i in range(3):
            task = Task(
                title=f"Multi-Agent Task {i + 1}",
                description=f"Task {i + 1} for multi-agent testing",
                task_type="parallel_processing",
                priority=TaskPriority.NORMAL,
            )
            task_id = await task_queue.submit_task(task)
            tasks.append(task_id)

        # Verify all tasks are queued
        queue_depth = await task_queue.get_queue_depth()
        assert queue_depth == 3

        # Simulate multiple agents picking up tasks
        agents = ["agent-1", "agent-2", "agent-3"]
        assigned_tasks = {}

        for i, agent in enumerate(agents):
            # Get next task
            next_task = await task_queue.get_next_task()
            assert next_task is not None

            # Assign to agent
            success = await task_queue.assign_task(next_task.id, agent)
            assert success is True
            assigned_tasks[agent] = next_task.id

            # Start processing
            await task_queue.start_task(next_task.id)

        # Verify all tasks are in progress
        for task_id in tasks:
            status = await task_queue.get_task_status(task_id)
            assert status == TaskStatus.IN_PROGRESS

        # Simulate agent communication during processing
        for agent, task_id in assigned_tasks.items():
            # Agent sends heartbeat
            heartbeat_sent = await message_broker.send_message(
                from_agent=agent,
                to_agent="orchestrator",
                topic="heartbeat",
                payload={
                    "agent": agent,
                    "status": "processing",
                    "current_task": task_id,
                    "timestamp": time.time(),
                },
                message_type=MessageType.DIRECT,
            )
            assert heartbeat_sent is True

        # Complete all tasks
        for agent, task_id in assigned_tasks.items():
            result = {
                "agent": agent,
                "status": "completed",
                "data": f"Result from {agent}",
            }
            success = await task_queue.complete_task(task_id, result)
            assert success is True

        # Verify all tasks completed
        for task_id in tasks:
            final_task = await task_queue.get_task(task_id)
            assert final_task.status == TaskStatus.COMPLETED
            assert final_task.result is not None

        # Verify queue is empty
        final_queue_depth = await task_queue.get_queue_depth()
        assert final_queue_depth == 0

    @pytest.mark.asyncio
    async def test_task_failure_and_retry(self, task_queue, message_broker):
        """Test task failure handling and retry logic."""
        # Submit task
        task = Task(
            title="Failure Test Task",
            description="Task that will fail and retry",
            task_type="failure_test",
            priority=TaskPriority.HIGH,
            max_retries=2,
        )

        task_id = await task_queue.submit_task(task)
        agent_name = "failure-test-agent"

        # Get and assign task
        next_task = await task_queue.get_next_task()
        await task_queue.assign_task(task_id, agent_name)
        await task_queue.start_task(task_id)

        # Fail the task (first attempt)
        await task_queue.fail_task(task_id, "Simulated failure", retry=True)

        # Task should still be available for retry
        retry_task = await task_queue.get_next_task()
        assert retry_task is not None
        assert retry_task.id == task_id

        # Assign and start retry
        await task_queue.assign_task(task_id, agent_name)
        await task_queue.start_task(task_id)

        # Fail again (second attempt)
        await task_queue.fail_task(task_id, "Second failure", retry=True)

        # Should still be available for final retry
        final_retry_task = await task_queue.get_next_task()
        assert final_retry_task is not None
        assert final_retry_task.id == task_id

        # Complete on final attempt
        await task_queue.assign_task(task_id, agent_name)
        await task_queue.start_task(task_id)
        success = await task_queue.complete_task(
            task_id, {"status": "success_after_retries"}
        )
        assert success is True

        # Verify final completion
        final_task = await task_queue.get_task(task_id)
        assert final_task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_system_broadcast_communication(self, message_broker):
        """Test system-wide broadcast messaging."""
        # Test system shutdown broadcast
        broadcast_sent = await message_broker.send_message(
            from_agent="system",
            to_agent="broadcast",
            topic="system_maintenance",
            payload={
                "message": "System maintenance in 5 minutes",
                "maintenance_time": time.time() + 300,
                "expected_duration": 1800,  # 30 minutes
            },
            message_type=MessageType.BROADCAST,
        )
        assert broadcast_sent is True

        # Test emergency stop broadcast
        emergency_sent = await message_broker.send_message(
            from_agent="orchestrator",
            to_agent="broadcast",
            topic="emergency_stop",
            payload={
                "reason": "Critical system error detected",
                "stop_all_tasks": True,
                "timestamp": time.time(),
            },
            message_type=MessageType.BROADCAST,
        )
        assert emergency_sent is True

    @pytest.mark.asyncio
    async def test_queue_statistics_and_monitoring(self, task_queue):
        """Test queue statistics for system monitoring."""
        # Submit tasks with different priorities
        task_ids = []

        # High priority task
        high_task = Task(
            title="High Priority Task",
            description="Critical task",
            task_type="critical",
            priority=TaskPriority.HIGH,
        )
        task_ids.append(await task_queue.submit_task(high_task))

        # Normal priority tasks
        for i in range(3):
            normal_task = Task(
                title=f"Normal Task {i + 1}",
                description=f"Regular task {i + 1}",
                task_type="regular",
                priority=TaskPriority.NORMAL,
            )
            task_ids.append(await task_queue.submit_task(normal_task))

        # Low priority task
        low_task = Task(
            title="Low Priority Task",
            description="Background task",
            task_type="background",
            priority=TaskPriority.LOW,
        )
        task_ids.append(await task_queue.submit_task(low_task))

        # Check queue statistics
        stats = await task_queue.get_queue_stats()
        assert stats is not None

        total_tasks = await task_queue.get_total_tasks()
        assert total_tasks == 5

        queue_depth = await task_queue.get_queue_depth()
        assert queue_depth == 5

        # Complete some tasks and check stats
        for i in range(2):
            next_task = await task_queue.get_next_task()
            await task_queue.assign_task(next_task.id, f"agent-{i}")
            await task_queue.start_task(next_task.id)
            await task_queue.complete_task(next_task.id, {"completed": True})

        # Check updated stats
        completed_count = await task_queue.get_completed_tasks()
        assert completed_count == 2

        remaining_depth = await task_queue.get_queue_depth()
        assert remaining_depth == 3
