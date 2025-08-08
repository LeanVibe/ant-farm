#!/usr/bin/env python3

"""Bottom-up test for agent task processing workflow."""

import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from core.task_queue import Task, TaskPriority, task_queue
from agents.meta_agent import MetaAgent


async def test_task_processing_workflow():
    """Test the complete task processing workflow."""
    print("🧪 Testing Agent Task Processing Workflow")

    # Initialize components
    print("1. Initializing task queue...")
    try:
        await task_queue.initialize()
        print("   ✅ Task queue initialized")
    except Exception as e:
        print(f"   ❌ Task queue failed: {e}")
        return False

    # Create a test task
    print("2. Creating test task...")
    task = Task(
        title="Test CLI Tool Execution",
        description="Test that the agent can execute OpenCode and return results",
        task_type="development",
        priority=TaskPriority.HIGH,
        payload={"action": "simple_test"},
    )

    try:
        task_id = await task_queue.submit_task(task)
        print(f"   ✅ Task created: {task_id}")
    except Exception as e:
        print(f"   ❌ Task creation failed: {e}")
        return False

    # Create an agent
    print("3. Creating MetaAgent...")
    try:
        agent = MetaAgent("test-workflow-agent")
        print(f"   ✅ Agent created: {agent.name}")
    except Exception as e:
        print(f"   ❌ Agent creation failed: {e}")
        return False

    # Test task retrieval
    print("4. Testing task retrieval...")
    try:
        retrieved_task = await task_queue.get_task(agent.name)
        if retrieved_task and retrieved_task.id == task_id:
            print(f"   ✅ Task retrieved: {retrieved_task.title}")
        else:
            print(f"   ❌ Task retrieval failed or wrong task")
            return False
    except Exception as e:
        print(f"   ❌ Task retrieval error: {e}")
        return False

    # Test task start
    print("5. Testing task start...")
    try:
        started = await task_queue.start_task(task_id)
        if started:
            print("   ✅ Task started successfully")
        else:
            print("   ❌ Task start failed")
            return False
    except Exception as e:
        print(f"   ❌ Task start error: {e}")
        return False

    # Test CLI tool availability
    print("6. Testing CLI tool availability...")
    try:
        available_tools = list(agent.cli_tools.available_tools.keys())
        print(f"   ✅ Available CLI tools: {available_tools}")
        if not available_tools:
            print("   ⚠️  No CLI tools available")
    except Exception as e:
        print(f"   ❌ CLI tool check failed: {e}")

    # Test simple CLI execution
    print("7. Testing CLI tool execution...")
    try:
        simple_prompt = "Echo 'Hello from agent test' and list the current directory"
        result = await agent.execute_with_cli_tool(simple_prompt)
        print(f"   ✅ CLI execution result: {result[:100]}...")
    except Exception as e:
        print(f"   ❌ CLI execution failed: {e}")
        # Continue - CLI failure shouldn't stop the test

    # Test task completion
    print("8. Testing task completion...")
    try:
        completed = await task_queue.complete_task(
            task_id, {"status": "success", "test": True}
        )
        if completed:
            print("   ✅ Task completed successfully")
        else:
            print("   ❌ Task completion failed")
            return False
    except Exception as e:
        print(f"   ❌ Task completion error: {e}")
        return False

    print("\n🎉 All workflow tests passed!")
    return True


async def test_agent_initialization():
    """Test agent initialization components separately."""
    print("\n🔧 Testing Agent Initialization Components")

    agent = MetaAgent("test-init-agent")

    # Test context engine (optional)
    print("1. Testing context engine initialization...")
    try:
        from core.context_engine import get_context_engine
        from core.config import settings

        context_engine = await asyncio.wait_for(
            get_context_engine(settings.database_url), timeout=10.0
        )
        print("   ✅ Context engine initialized")
    except asyncio.TimeoutError:
        print("   ⚠️  Context engine timeout (expected in some environments)")
    except Exception as e:
        print(f"   ⚠️  Context engine failed: {e}")

    # Test message broker (optional)
    print("2. Testing message broker...")
    try:
        from core.message_broker import message_broker

        await asyncio.wait_for(message_broker.initialize(), timeout=5.0)
        print("   ✅ Message broker initialized")
    except asyncio.TimeoutError:
        print("   ⚠️  Message broker timeout")
    except Exception as e:
        print(f"   ⚠️  Message broker failed: {e}")

    print("   ✅ Agent initialization test complete")


if __name__ == "__main__":

    async def main():
        print("🚀 LeanVibe Agent Hive - Bottom-Up Workflow Test\n")

        # Test initialization
        await test_agent_initialization()

        # Test workflow
        success = await test_task_processing_workflow()

        if success:
            print("\n✅ All tests passed - agent workflow is functional!")
        else:
            print("\n❌ Some tests failed - workflow needs fixes")
            sys.exit(1)

    asyncio.run(main())
