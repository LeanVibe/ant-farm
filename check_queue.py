#!/usr/bin/env python3
"""Simple test to check Redis task queue status."""

import asyncio
from src.core.task_queue import task_queue


async def check_queue_status():
    """Check current queue status."""
    await task_queue.initialize()

    print("=== Current Queue Status ===")
    stats = await task_queue.get_queue_stats()
    print(f"Total tasks: {stats.total_tasks}")
    print(f"Pending tasks: {stats.pending_tasks}")
    print(f"Completed tasks: {stats.completed_tasks}")
    print(f"Failed tasks: {stats.failed_tasks}")

    # Check specific task
    task_id = "206c6db4-93ac-4d87-bc45-4570277082d0"
    task = await task_queue.get_task_status(task_id)
    if task:
        print(f"\nTask {task_id}:")
        print(f"  Status: {task.status}")
        print(f"  Agent: {task.agent_id}")
        print(f"  Retry count: {task.retry_count}")
        print(f"  Error: {task.error_message}")
    else:
        print(f"\nTask {task_id} not found")


if __name__ == "__main__":
    asyncio.run(check_queue_status())
