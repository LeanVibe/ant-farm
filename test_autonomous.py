#!/usr/bin/env python3
"""Test autonomous task execution with real agent."""

import asyncio
import json
import time
import subprocess
import signal
from src.core.task_queue import task_queue, Task, TaskPriority


async def test_autonomous_execution():
    """Test submitting a task and having a real agent process it."""

    print("=== Testing Autonomous Task Execution ===")

    # Initialize task queue
    await task_queue.initialize()

    # Start an agent in background
    print("Starting developer agent...")
    agent_process = subprocess.Popen(
        [
            "uv",
            "run",
            "python3",
            "start_agent.py",
            "--type",
            "developer",
            "--name",
            "auto-test-agent",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Give agent time to start
        await asyncio.sleep(8)

        # Create a simple task
        test_task = Task(
            title="Simple Echo Task",
            description="Echo 'Hello from autonomous agent!' to demonstrate task processing",
            task_type="development",
            payload={
                "action": "echo",
                "message": "Hello from autonomous agent!",
                "test_id": f"auto-{int(time.time())}",
            },
            priority=TaskPriority.HIGH,
        )

        print(f"Submitting task: {test_task.title}")
        task_id = await task_queue.submit_task(test_task)
        print(f"Task submitted with ID: {task_id}")

        # Wait for agent to process task
        print("Waiting for agent to process task...")
        for i in range(15):  # Wait up to 15 seconds
            await asyncio.sleep(1)

            # Check task status
            completed_task = await task_queue.get_task_status(task_id)
            if completed_task and completed_task.status == "completed":
                print(f"✅ Task completed by agent!")
                print(f"   Result: {completed_task.result}")
                break
            elif completed_task and completed_task.status == "failed":
                print(f"❌ Task failed: {completed_task.error_message}")
                break
            elif i % 3 == 0:
                print(
                    f"   Task status: {completed_task.status if completed_task else 'unknown'}"
                )
        else:
            print("⏰ Task did not complete within timeout")

        # Get final stats
        stats = await task_queue.get_queue_stats()
        print(f"\nFinal queue stats:")
        print(f"   Total tasks: {stats.total_tasks}")
        print(f"   Completed tasks: {stats.completed_tasks}")
        print(f"   Failed tasks: {stats.failed_tasks}")

    finally:
        # Clean up agent process
        print("Shutting down agent...")
        agent_process.send_signal(signal.SIGTERM)
        try:
            agent_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            agent_process.kill()
            agent_process.wait()


if __name__ == "__main__":
    asyncio.run(test_autonomous_execution())
