#!/usr/bin/env python3
"""
Bootstrap script for LeanVibe Agent Hive 2.0 - Main System Entry Point

This script implements the bootstrap workflow from docs/PLAN.md:
1. Initialize database connection and create tables
2. Run context population script
3. Initialize and start the Orchestrator
4. Configure Orchestrator to spawn one MetaAgent on startup
5. Start the FastAPI application server using uvicorn

This enables the first self-improvement task workflow.
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import structlog
import uvicorn

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.config import get_settings
from src.core.models import get_database_manager
from src.core.orchestrator import get_orchestrator
from src.core.task_queue import task_queue

logger = structlog.get_logger()


class SystemBootstrap:
    """Bootstrap the complete Agent Hive system for self-improvement."""

    def __init__(self):
        self.settings = get_settings()
        self.project_root = project_root
        self.orchestrator = None

    async def bootstrap_system(self) -> bool:
        """
        Complete system bootstrap following PLAN.md workflow.
        Returns True if successful, False otherwise.
        """
        try:
            logger.info("🚀 Starting LeanVibe Agent Hive 2.0 Bootstrap")

            # Step 1: Initialize database and create tables
            logger.info("📊 Step 1: Initializing database...")
            if not await self._initialize_database():
                return False

            # Step 2: Run context population
            logger.info("🧠 Step 2: Populating context engine...")
            if not await self._populate_context():
                return False

            # Step 3: Initialize and start Orchestrator
            logger.info("🎭 Step 3: Starting orchestrator...")
            if not await self._start_orchestrator():
                return False

            # Step 4: Spawn MetaAgent
            logger.info("🤖 Step 4: Spawning MetaAgent...")
            if not await self._spawn_meta_agent():
                return False

            # Step 5: Start FastAPI server
            logger.info("🌐 Step 5: Starting API server...")
            await self._start_api_server()

            return True

        except Exception as e:
            logger.error("❌ Bootstrap failed", error=str(e))
            return False

    async def _initialize_database(self) -> bool:
        """Initialize database connection and create tables."""
        try:
            # Create database manager and tables
            db_manager = get_database_manager(self.settings.database_url)
            db_manager.create_tables()

            logger.info("✅ Database tables created successfully")
            return True

        except Exception as e:
            logger.error("❌ Database initialization failed", error=str(e))
            logger.info(
                "💡 Make sure PostgreSQL is running: docker compose up -d postgres"
            )
            return False

    async def _populate_context(self) -> bool:
        """Run the context population script."""
        try:
            # Run context population script
            populate_script = self.project_root / "scripts" / "populate_context.py"

            result = subprocess.run(
                [sys.executable, str(populate_script)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info("✅ Context engine populated successfully")
                return True
            else:
                logger.error(
                    "❌ Context population failed",
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Context population timed out after 5 minutes")
            return False
        except Exception as e:
            logger.error("❌ Context population script failed", error=str(e))
            return False

    async def _start_orchestrator(self) -> bool:
        """Initialize and start the Orchestrator."""
        try:
            # Initialize task queue
            await task_queue.initialize()
            logger.info("✅ Task queue initialized")

            # Get orchestrator instance
            self.orchestrator = await get_orchestrator(
                self.settings.database_url, self.project_root
            )

            # Start orchestrator background tasks
            await self.orchestrator.start()
            logger.info("✅ Orchestrator started successfully")

            return True

        except Exception as e:
            logger.error("❌ Orchestrator initialization failed", error=str(e))
            return False

    async def _spawn_meta_agent(self) -> bool:
        """Configure Orchestrator to spawn one MetaAgent on startup."""
        try:
            if not self.orchestrator:
                raise RuntimeError("Orchestrator not initialized")

            # Spawn MetaAgent
            agent_name = await self.orchestrator.spawn_agent(
                agent_type="meta", agent_name="meta-agent-bootstrap"
            )

            if agent_name:
                logger.info("✅ MetaAgent spawned successfully", agent_name=agent_name)

                # Give the agent a moment to start
                await asyncio.sleep(2)
                return True
            else:
                logger.error("❌ Failed to spawn MetaAgent")
                return False

        except Exception as e:
            logger.error("❌ MetaAgent spawning failed", error=str(e))
            return False

    async def _start_api_server(self):
        """Start the FastAPI application server using uvicorn."""
        try:
            logger.info("🌐 Starting FastAPI server on http://localhost:8000")
            logger.info("📖 API Documentation: http://localhost:8000/api/docs")
            logger.info("🎯 Ready for first self-improvement task!")

            # Import the FastAPI app
            from src.api.main import app

            # Configure uvicorn
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=8000,
                log_level="info",
                reload=False,  # Disable reload for production-like behavior
            )

            server = uvicorn.Server(config)
            await server.serve()

        except Exception as e:
            logger.error("❌ API server failed to start", error=str(e))
            raise


async def main():
    """Main entry point for system bootstrap."""
    bootstrap = SystemBootstrap()

    try:
        success = await bootstrap.bootstrap_system()

        if not success:
            logger.error("💥 Bootstrap failed - system not ready")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("👋 Bootstrap interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error("💥 Unexpected bootstrap error", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    # Configure logging
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    print("🚀 LeanVibe Agent Hive 2.0 - System Bootstrap")
    print("=" * 50)
    print("This will initialize the complete self-improvement system.")
    print("Press Ctrl+C to cancel at any time.")
    print()

    asyncio.run(main())
