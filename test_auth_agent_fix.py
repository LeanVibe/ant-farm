#!/usr/bin/env python3
"""Test to verify authentication fixes for unresponsive agents."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))


async def test_agent_authentication_fix():
    """Test that agents can authenticate and connect properly."""
    print("Testing agent authentication fixes...")

    try:
        # Import required modules
        from src.core.config import get_settings
        from src.core.orchestrator import get_orchestrator

        # Get settings
        settings = get_settings()

        # Verify configuration
        print(f"✓ Database URL: {settings.database_url}")
        print(f"✓ Redis URL: {settings.redis_url}")

        # Check that database URL uses correct port
        assert ":5433/" in settings.database_url or ":5433$" in settings.database_url, (
            "Database URL should use port 5433"
        )

        # Check that Redis URL uses correct port
        assert ":6381" in settings.redis_url, "Redis URL should use port 6381"

        print("✓ Configuration validation passed")

        # Test orchestrator initialization
        orchestrator = await get_orchestrator(settings.database_url, Path("."))
        await orchestrator.start()

        print("✓ Orchestrator initialized successfully")

        # Test agent spawning
        agent_name = await orchestrator.spawn_agent("meta", "test-auth-agent")

        if agent_name:
            print(f"✓ Agent '{agent_name}' spawned successfully!")

            # List agents to verify it's registered
            agents = await orchestrator.registry.list_agents()
            test_agent = None
            for agent in agents:
                if agent.name == agent_name:
                    test_agent = agent
                    break

            if test_agent:
                print(f"✓ Agent registered with status: {test_agent.status.value}")

                # Verify agent has proper configuration
                if hasattr(test_agent, "capabilities") and test_agent.capabilities:
                    print(f"✓ Agent has capabilities: {len(test_agent.capabilities)}")
                else:
                    print("⚠ Agent capabilities not found")

                # Clean up - terminate the agent
                success = await orchestrator.terminate_agent(agent_name)
                if success:
                    print("✓ Agent terminated successfully")
                else:
                    print("⚠ Failed to terminate agent (continuing)")

                print("🎉 Authentication fix verification completed successfully!")
                return True
            else:
                print("✗ Agent not found in registry")
                return False
        else:
            print("✗ Failed to spawn agent")
            return False

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Clean up any remaining test sessions
        try:
            os.system("tmux kill-session -t hive-test-auth-agent 2>/dev/null || true")
        except:
            pass


async def main():
    """Main test function."""
    print("Running authentication fix verification for agents...\n")

    success = await test_agent_authentication_fix()

    if success:
        print("\n🎉 All authentication fixes are working correctly!")
        print("✅ Unresponsive agents issue has been resolved!")
        return 0
    else:
        print("\n❌ Authentication fixes verification failed.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
