#!/usr/bin/env python3
"""
PLAN.md Bootstrap Script for LeanVibe Agent Hive 2.0

This script implements the exact bootstrap workflow specified in PLAN.md Task 6.
It creates the initial self-improvement system ready for the first API task.

Workflow:
1. Initialize database connection and create tables
2. Run context population script to build initial knowledge base
3. Initialize and start the Orchestrator
4. Configure the Orchestrator to spawn one MetaAgent on startup
5. Start the FastAPI application server using uvicorn

After completion, the system will be ready for its first live test
of the self-improvement loop via POST /api/v1/tasks/self-improvement
"""

import asyncio
import sys
from pathlib import Path

import structlog

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

logger = structlog.get_logger()


async def main():
    """PLAN.md bootstrap workflow - minimal setup for self-improvement loop."""

    print("=" * 60)
    print("LEANVIBE AGENT HIVE 2.0 - PLAN.MD BOOTSTRAP")
    print("Setting up self-improvement system...")
    print("=" * 60)

    try:
        # Step 1: Initialize database
        print("\\n1. Database initialization...")
        from src.core.config import get_settings
        from src.core.models import get_database_manager

        settings = get_settings()
        db_manager = get_database_manager(settings.database_url)
        db_manager.create_tables()
        print("✓ Database tables created")

        # Step 2: Populate initial context (optional)
        print("\\n2. Populating initial context...")
        context_script = project_root / "scripts" / "populate_context.py"

        if context_script.exists():
            try:
                result = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(context_script),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=project_root,
                )
                stdout, stderr = await result.communicate()

                if result.returncode == 0:
                    print("✓ Context knowledge base populated")
                else:
                    print("⚠ Context population failed (continuing anyway)")
                    print(f"Error: {stderr.decode()[:200]}...")
            except Exception as e:
                print(f"⚠ Context population error: {e}")
        else:
            print("⚠ Context script not found (skipping)")

        # Step 3: Initialize Orchestrator
        print("\\n3. Starting orchestrator...")
        from src.core.orchestrator import get_orchestrator

        orchestrator = await get_orchestrator(settings.database_url, project_root)
        await orchestrator.start()
        print("✓ Orchestrator started")

        # Step 4: Spawn MetaAgent
        print("\\n4. Spawning MetaAgent...")
        try:
            # Check if meta-agent already exists
            existing_agent = await orchestrator.get_agent("meta-agent")
            if existing_agent and existing_agent.status == "active":
                print("✓ MetaAgent already active")
            else:
                session_name = await orchestrator.spawn_agent("meta", "meta-agent")
                print(f"✓ MetaAgent spawned (session: {session_name})")

                # Wait for agent to initialize
                await asyncio.sleep(3)

                # Verify agent is ready
                meta_agent = await orchestrator.get_agent("meta-agent")
                if meta_agent and meta_agent.status == "active":
                    print("✓ MetaAgent is ready for self-improvement tasks")
                else:
                    print("⚠ MetaAgent spawned but not yet active")

        except Exception as e:
            print(f"⚠ MetaAgent spawn error: {e}")
            print("System can still work, MetaAgent can be spawned later")

        # Step 5: Start API Server
        print("\\n5. Starting FastAPI server...")

        print("\\n" + "=" * 60)
        print("🚀 LEANVIBE AGENT HIVE 2.0 READY!")
        print("=" * 60)
        print("API Server starting on: http://localhost:8000")
        print("API Documentation: http://localhost:8000/api/docs")
        print("Dashboard: http://localhost:8000/dashboard/")
        print("")
        print("🎯 FIRST SELF-IMPROVEMENT TASK:")
        print("POST http://localhost:8000/api/v1/tasks/self-improvement")
        print("Content-Type: application/json")
        print("")
        print("{")
        print('  "title": "Refactor error handling",')
        print('  "description": "Improve error handling in the task queue system"')
        print("}")
        print("=" * 60)

        # Start the server
        import uvicorn

        from src.api.main import app

        config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, log_level="info")

        server = uvicorn.Server(config)
        await server.serve()

    except KeyboardInterrupt:
        print("\\n\\nBootstrap interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\\n\\nBootstrap failed: {e}")
        logger.error("Bootstrap error", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    # Configure basic logging
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the bootstrap
    asyncio.run(main())
