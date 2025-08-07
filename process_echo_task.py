#!/usr/bin/env python3
"""
Simple Echo Task Processor
Demonstrates task processing by echoing 'Hello from autonomous agent!'
"""

import asyncio
import sys
import time
from pathlib import Path

from agents.base_agent import BaseAgent, TaskResult
from core.task_queue import Task, TaskPriority

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


class EchoAgent(BaseAgent):
    """Simple agent that processes echo tasks."""

    def __init__(self):
        super().__init__(
            name="echo_agent", agent_type="development", role="Echo task processor"
        )

    async def run(self) -> None:
        """Main execution loop."""
        print(f"🤖 Echo Agent '{self.name}' started")
        print(f"🔧 Available CLI tools: {list(self.cli_tools.available_tools.keys())}")

        # Process the echo task directly
        await self.process_echo_task()

    async def process_echo_task(self) -> None:
        """Process the echo task directly."""
        # Create the echo task
        echo_task = Task(
            title="Simple Echo Task",
            description="Echo 'Hello from autonomous agent!' to demonstrate task processing",
            task_type="development",
            priority=TaskPriority.NORMAL,
            payload={"message": "Hello from autonomous agent!"},
        )

        print("\n📋 Processing Echo Task:")
        print(f"   Title: {echo_task.title}")
        print(f"   Description: {echo_task.description}")
        print(f"   Type: {echo_task.task_type}")
        print(f"   Priority: {echo_task.priority}")
        print(f"   Message: {echo_task.payload.get('message')}")

        # Process the task
        result = await self._process_echo_task_implementation(echo_task)

        # Display results
        print("\n✅ Task Processing Result:")
        print(f"   Success: {result.success}")
        if result.success:
            print(f"   Echo Message: {result.data.get('echo_message')}")
            print(f"   Timestamp: {result.data.get('timestamp')}")
            print(f"   Agent: {result.data.get('agent_name')}")
            if result.metrics:
                print(
                    f"   Execution Time: {result.metrics.get('execution_time', 0):.3f}s"
                )
        else:
            print(f"   Error: {result.error}")

    async def _process_echo_task_implementation(self, task: Task) -> TaskResult:
        """Custom implementation for echo tasks."""
        start_time = time.time()

        try:
            # Extract message from task payload
            message = task.payload.get("message", "Hello from autonomous agent!")

            # Process the echo
            echo_response = f"🔊 ECHO: {message}"

            # Simulate some processing time
            await asyncio.sleep(0.1)

            execution_time = time.time() - start_time

            # Return successful result
            return TaskResult(
                success=True,
                data={
                    "echo_message": echo_response,
                    "original_message": message,
                    "timestamp": time.time(),
                    "agent_name": self.name,
                    "task_id": task.id,
                },
                metrics={
                    "execution_time": execution_time,
                    "message_length": len(message),
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                success=False,
                error=f"Echo task failed: {str(e)}",
                metrics={"execution_time": execution_time},
            )


async def main():
    """Main entry point."""
    print("🚀 Starting Echo Task Processing Demo")

    try:
        # Create and start echo agent
        echo_agent = EchoAgent()
        await echo_agent.run()

    except KeyboardInterrupt:
        print("\n⏹️  Echo task processing interrupted")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        print("👋 Echo task processing complete")


if __name__ == "__main__":
    asyncio.run(main())
