#!/usr/bin/env python3
"""
Echo Task Queue Integration Test
Demonstrates submitting and processing echo tasks through the task queue system.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))
from core.task_queue import Task, TaskPriority, task_queue


async def test_echo_task_queue():
    """Test echo task submission and processing through task queue."""
    print("🔗 Testing Echo Task with Task Queue Integration")

    try:
        # Initialize task queue
        await task_queue.initialize()
        print("✅ Task queue initialized")

        # Create echo task
        echo_task = Task(
            title="Simple Echo Task",
            description="Echo 'Hello from autonomous agent!' to demonstrate task processing",
            task_type="development",
            priority=TaskPriority.NORMAL,
            payload={"message": "Hello from autonomous agent!"},
        )

        # Submit task to queue
        task_id = await task_queue.submit_task(echo_task)
        print(f"📤 Task submitted with ID: {task_id}")

        # Get task status
        task_status = await task_queue.get_task_status(task_id)
        if task_status:
            print(f"📋 Task Status: {task_status.status}")
            print(f"   Priority: {task_status.priority}")
            print(f"   Created: {time.ctime(task_status.created_at)}")

        # Simulate agent getting task
        print("\n🤖 Simulating agent retrieving task...")
        retrieved_task = await task_queue.get_task("echo_agent_test")

        if retrieved_task:
            print(f"✅ Task retrieved by agent: {retrieved_task.agent_id}")
            print(f"   Task ID: {retrieved_task.id}")
            print(f"   Status: {retrieved_task.status}")
            print(f"   Message: {retrieved_task.payload.get('message')}")

            # Start task processing
            await task_queue.start_task(retrieved_task.id)
            print("🏃 Task marked as in progress")

            # Simulate processing (echo the message)
            message = retrieved_task.payload.get(
                "message", "Hello from autonomous agent!"
            )
            echo_response = f"🔊 ECHO: {message}"

            # Complete task with result
            result_data = {
                "echo_message": echo_response,
                "original_message": message,
                "timestamp": time.time(),
                "agent_name": "echo_agent_test",
            }

            await task_queue.complete_task(retrieved_task.id, result_data)
            print("✅ Task completed successfully")
            print(f"   Echo Result: {echo_response}")

            # Get final task status
            final_status = await task_queue.get_task_status(retrieved_task.id)
            if final_status:
                print(f"📊 Final Status: {final_status.status}")
                if final_status.result:
                    print(f"   Result: {final_status.result}")
        else:
            print("❌ No task retrieved from queue")

        # Get queue statistics
        stats = await task_queue.get_queue_stats()
        print("\n📈 Queue Statistics:")
        print(f"   Total tasks: {stats.total_tasks}")
        print(f"   Completed: {stats.completed_tasks}")
        print(f"   Failed: {stats.failed_tasks}")
        print(f"   Average completion time: {stats.average_completion_time:.3f}s")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()


async def main():
    """Main entry point."""
    print("🚀 Starting Echo Task Queue Integration Test")

    try:
        await test_echo_task_queue()

    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        print("👋 Test complete")


if __name__ == "__main__":
    asyncio.run(main())
