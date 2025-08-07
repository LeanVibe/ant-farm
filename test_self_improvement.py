#!/usr/bin/env python3
"""
Test script for the self-improvement endpoint.
This bypasses the full bootstrap and tests the MetaAgent directly.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sqlalchemy.orm import sessionmaker

from src.agents.meta_agent import MetaAgent
from src.core.config import get_settings
from src.core.models import Agent, Task, get_database_manager


async def test_self_improvement():
    """Test the self-improvement workflow directly."""
    print("🧪 Testing Self-Improvement Workflow")
    print("=" * 50)

    # Initialize settings
    settings = get_settings()

    # Initialize database
    print("1. Initializing database...")
    db_manager = get_database_manager(settings.database_url)
    db_manager.create_tables()

    # Create database session
    SessionLocal = sessionmaker(bind=db_manager.engine)
    session = SessionLocal()

    print("✓ Database ready")

    # Create MetaAgent directly
    print("2. Creating MetaAgent...")
    meta_agent = MetaAgent(name="meta-agent-test")

    # Register agent in database with proper UUID
    print("3. Registering agent in database...")
    import uuid

    agent_uuid = str(uuid.uuid4())

    agent_record = Agent(
        id=agent_uuid,
        name="meta-agent-test",
        type="meta",
        role="system_coordinator",
        capabilities={
            "self_improvement": True,
            "code_modification": True,
            "context_analysis": True,
        },
        status="active",
    )

    session.add(agent_record)
    session.commit()

    # Set the agent UUID so context retrieval works
    meta_agent.agent_uuid = agent_uuid

    print(f"✓ Agent registered with UUID: {agent_uuid}")

    # Initialize the agent
    await meta_agent.initialize()
    print("✓ MetaAgent created and initialized")
    # Create a test self-improvement task
    print("4. Creating test task...")

    test_task = Task(
        id=str(uuid.uuid4()),
        title="Add better error logging",
        description="Improve error logging in src/core/task_queue.py by adding more detailed error messages",
        type="self_improvement",
        priority=5,  # Medium priority
        status="pending",
        agent_id=agent_uuid,  # Assign to our registered agent
        payload={
            "improvements": {
                "target_file": "src/core/task_queue.py",
                "improvement_type": "error_handling",
            }
        },
    )

    # Process the task
    print("5. Processing self-improvement task...")
    try:
        result = await meta_agent._process_self_improvement_task(test_task)
        print("✓ Task processed successfully!")
        print(f"Result: {result}")

        return True

    except Exception as e:
        print(f"❌ Task processing failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print("6. Cleaning up...")
        session.close()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    success = asyncio.run(test_self_improvement())
    if success:
        print("\n🎉 Self-improvement test PASSED!")
        print("The system is ready for live API testing.")
    else:
        print("\n💥 Self-improvement test FAILED!")
        sys.exit(1)
