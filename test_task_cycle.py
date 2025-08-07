#!/usr/bin/env python3
"""Test script for end-to-end task processing cycle."""

import asyncio
import json
import time
from src.core.task_queue import task_queue, Task, TaskPriority


async def test_task_cycle():
    """Test submitting a task and verify it can be retrieved."""

    # Initialize task queue
    print("Initializing task queue...")
    await task_queue.initialize()
    print("Task queue initialized")

    # Create a test task
    test_task = Task(
        title="Test Task",
        description="This is a test task for verifying the task processing cycle",
        task_type="development",
        payload={"test": True, "timestamp": time.time()},
        priority=TaskPriority.NORMAL,
    )

    # Submit the task
    print(f"Submitting task: {test_task.title}")
    task_id = await task_queue.submit_task(test_task)
    print(f"Task submitted with ID: {task_id}")

    # Wait a moment
    await asyncio.sleep(1)

    # Try to get the task as an agent would
    print("Attempting to get task as agent 'test-agent'...")
    retrieved_task = await task_queue.get_task("test-agent")

    if retrieved_task:
        print(f"✅ Task retrieved successfully!")
        print(f"   Task ID: {retrieved_task.id}")
        print(f"   Title: {retrieved_task.title}")
        print(f"   Status: {retrieved_task.status}")
        print(f"   Agent ID: {retrieved_task.agent_id}")

        # Test completing the task
        print("Completing the task...")
        result = {"status": "completed", "test_result": "success"}
        success = await task_queue.complete_task(retrieved_task.id, result)

        if success:
            print("✅ Task completed successfully!")
        else:
            print("❌ Failed to complete task")

    else:
        print("❌ No task retrieved - task cycle may have issues")

    # Get queue stats
    stats = await task_queue.get_queue_stats()
    print(f"\nQueue stats:")
    print(f"   Total tasks: {stats.total_tasks}")
    print(f"   Pending tasks: {stats.pending_tasks}")
    print(f"   Completed tasks: {stats.completed_tasks}")
    print(f"   Failed tasks: {stats.failed_tasks}")


if __name__ == "__main__":
    asyncio.run(test_task_cycle())
