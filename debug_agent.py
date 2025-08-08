#!/usr/bin/env python3

"""Debug script to test agent initialization directly."""

import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from agents.meta_agent import MetaAgent


async def test_agent():
    """Test MetaAgent initialization and basic run."""
    print("Creating MetaAgent...")

    try:
        agent = MetaAgent("debug-meta-agent")
        print(f"Agent created. Status: {agent.status}")

        print("Starting agent...")
        # Run for just a few seconds
        agent_task = asyncio.create_task(agent.start())

        # Give it 5 seconds to initialize
        try:
            await asyncio.wait_for(agent_task, timeout=5)
        except asyncio.TimeoutError:
            print("Agent ran for 5 seconds successfully")
            agent.status = "stopping"
            try:
                await asyncio.wait_for(agent_task, timeout=2)
            except asyncio.TimeoutError:
                agent_task.cancel()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_agent())
