#!/usr/bin/env python3
"""Test script to spawn an agent directly via orchestrator."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.config import get_settings
from core.orchestrator import get_orchestrator


async def test_spawn_agent():
    """Test spawning a Meta Agent directly."""
    try:
        settings = get_settings()
        orchestrator = await get_orchestrator(settings.database_url, Path("."))

        # Start the orchestrator
        await orchestrator.start()

        print("🤖 Spawning Meta Agent...")
        agent_name = await orchestrator.spawn_agent("meta", "test-meta")

        if agent_name:
            print(f"✅ Agent '{agent_name}' spawned successfully!")

            # List agents
            agents = await orchestrator.registry.list_agents()
            print(f"📋 Active agents: {len(agents)}")
            for agent in agents:
                print(f"   • {agent.name} ({agent.type}) - {agent.status.value}")
        else:
            print("❌ Failed to spawn agent")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_spawn_agent())
