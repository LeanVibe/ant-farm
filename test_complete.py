#!/usr/bin/env python3
"""Test complete task processing cycle."""

import asyncio
import time
import subprocess
import signal
from src.core.task_queue import task_queue, Task, TaskPriority


async def test_complete_cycle():
    """Test complete task processing with detailed monitoring."""

    print("=== Complete Task Processing Test ===")

    # Initialize task queue
    await task_queue.initialize()

    # Clear any existing tasks (for clean test)
    print("Clearing existing tasks...")

    # Create a simple task
    test_task = Task(
        title="Complete Test Task",
        description="This task will test the complete processing cycle",
        task_type="development",
        payload={"action": "echo", "message": "Task processing works!"},
        priority=TaskPriority.HIGH,
    )

    print(f"Submitting task: {test_task.title}")
    task_id = await task_queue.submit_task(test_task)
    print(f"Task submitted with ID: {task_id}")

    # Start agent
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
            "complete-test-agent",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Monitor the task for 30 seconds
        print("Monitoring task processing...")

        for i in range(30):
            await asyncio.sleep(1)

            # Check task status
            task_status = await task_queue.get_task_status(task_id)
            if task_status:
                status = task_status.status
                agent = task_status.agent_id or "none"
                print(f"[{i + 1:2d}s] Status: {status:12s} | Agent: {agent}")

                if status == "completed":
                    print(f"✅ Task completed successfully!")
                    print(f"   Result: {task_status.result}")
                    break
                elif status == "failed":
                    print(f"❌ Task failed: {task_status.error_message}")
                    break
            else:
                print(f"[{i + 1:2d}s] Task not found")
        else:
            print("⏰ Task monitoring timeout")

        # Final status
        final_task = await task_queue.get_task_status(task_id)
        if final_task:
            print(f"\nFinal task status: {final_task.status}")
            print(f"Final agent: {final_task.agent_id}")
            if final_task.result:
                print(f"Final result: {final_task.result}")

    finally:
        # Clean up
        print("\nShutting down agent...")
        agent_process.send_signal(signal.SIGTERM)
        try:
            stdout, stderr = agent_process.communicate(timeout=5)
            if stdout:
                print("Agent stdout (last 500 chars):")
                print(stdout[-500:])
        except subprocess.TimeoutExpired:
            agent_process.kill()
            agent_process.wait()


if __name__ == "__main__":
    asyncio.run(test_complete_cycle())
