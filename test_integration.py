#!/usr/bin/env python3
"""
Integration test suite for LeanVibe Agent Hive 2.0
Tests core system components and their interactions.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

import redis.asyncio as redis

from src.core.config import settings
from src.core.models import get_database_manager
from src.core.task_queue import Task, TaskPriority, TaskQueue


class IntegrationTester:
    """Integration test runner for the system."""

    def __init__(self):
        self.results: dict[str, Any] = {}
        self.failures = 0
        self.successes = 0

    async def run_all_tests(self) -> bool:
        """Run all integration tests."""
        print("🧪 LeanVibe Agent Hive 2.0 - Integration Test Suite")
        print("=" * 60)

        tests = [
            ("Database Connection", self.test_database_connection),
            ("Redis Connection", self.test_redis_connection),
            ("Task Queue Operations", self.test_task_queue),
            ("API Components", self.test_api_components),
            ("Configuration", self.test_configuration),
        ]

        for test_name, test_func in tests:
            print(f"\n🔍 Testing: {test_name}")
            try:
                result = await test_func()
                if result:
                    print(f"✅ PASS: {test_name}")
                    self.successes += 1
                else:
                    print(f"❌ FAIL: {test_name}")
                    self.failures += 1
                self.results[test_name] = result
            except Exception as e:
                print(f"💥 ERROR: {test_name} - {str(e)}")
                self.failures += 1
                self.results[test_name] = False

        # Print summary
        print("\n" + "=" * 60)
        print(f"🏁 Test Results: {self.successes} passed, {self.failures} failed")

        if self.failures == 0:
            print("🎉 All tests passed! System is ready for autonomous operation.")
            return True
        else:
            print("⚠️ Some tests failed. Please address issues before proceeding.")
            return False

    async def test_database_connection(self) -> bool:
        """Test PostgreSQL database connection and schema."""
        try:
            db_manager = get_database_manager(settings.database_url)

            # Test connection
            from sqlalchemy import text

            session = db_manager.get_session()
            session.execute(text("SELECT 1"))
            session.close()

            # Test table creation
            db_manager.create_tables()

            print("  ✓ Database connection successful")
            print("  ✓ Schema creation successful")
            return True

        except Exception as e:
            print(f"  ✗ Database test failed: {e}")
            return False

    async def test_redis_connection(self) -> bool:
        """Test Redis connection and basic operations."""
        try:
            redis_client = redis.from_url(settings.redis_url, decode_responses=True)

            # Test connection
            await redis_client.ping()

            # Test basic operations
            test_key = "test:integration"
            await redis_client.set(test_key, "test_value")
            value = await redis_client.get(test_key)
            await redis_client.delete(test_key)

            if value != "test_value":
                raise ValueError("Redis value mismatch")

            await redis_client.aclose()

            print("  ✓ Redis connection successful")
            print("  ✓ Redis operations successful")
            return True

        except Exception as e:
            print(f"  ✗ Redis test failed: {e}")
            return False

    async def test_task_queue(self) -> bool:
        """Test task queue functionality."""
        try:
            # Create task queue instance
            task_queue = TaskQueue(settings.redis_url)
            await task_queue.initialize()

            # Create test task
            test_task = Task(
                title="Integration Test Task",
                description="Testing task queue system",
                task_type="test",
                priority=TaskPriority.HIGH,
            )

            # Test task submission
            task_id = await task_queue.submit_task(test_task)
            print(f"  ✓ Task submitted: {task_id}")

            # Test task retrieval
            retrieved_task = await task_queue.get_task("test_agent")
            if retrieved_task and retrieved_task.id == task_id:
                print("  ✓ Task retrieval successful")
            else:
                raise ValueError("Task retrieval failed")

            # Test task completion
            await task_queue.complete_task(task_id, {"status": "completed"})
            print("  ✓ Task completion successful")

            # Test queue stats
            stats = await task_queue.get_queue_stats()
            print(f"  ✓ Queue stats: {stats.total_tasks} total tasks")

            return True

        except Exception as e:
            print(f"  ✗ Task queue test failed: {e}")
            return False

    async def test_api_components(self) -> bool:
        """Test API component imports and basic functionality."""
        try:
            # Test imports
            from src.api.main import app
            from src.core.orchestrator import get_orchestrator

            print("  ✓ API components import successfully")

            # Test route registration
            routes = [route.path for route in app.routes if hasattr(route, "path")]
            expected_routes = [
                "/health",
                "/api/v1/status",
                "/api/v1/agents",
                "/api/v1/tasks",
            ]

            for route in expected_routes:
                if route not in routes:
                    raise ValueError(f"Missing expected route: {route}")

            print(f"  ✓ API routes registered: {len(routes)} routes")

            # Test orchestrator creation
            await get_orchestrator(settings.database_url, Path("."))
            print("  ✓ Orchestrator creation successful")

            return True

        except Exception as e:
            print(f"  ✗ API components test failed: {e}")
            return False

    async def test_configuration(self) -> bool:
        """Test system configuration."""
        try:
            # Test config validation
            issues = settings.validate_configuration()

            if issues:
                print("  ⚠️ Configuration issues found:")
                for issue in issues:
                    print(f"    - {issue}")
            else:
                print("  ✓ Configuration validation passed")

            # Test essential settings
            essential_settings = [
                ("database_url", settings.database_url),
                ("redis_url", settings.redis_url),
                ("project_root", settings.project_root),
            ]

            for setting_name, setting_value in essential_settings:
                if not setting_value:
                    raise ValueError(f"Essential setting missing: {setting_name}")

            print("  ✓ Essential settings present")

            # Test directory creation
            Path(settings.logs_dir).mkdir(exist_ok=True)
            Path(settings.get_workspace_path()).mkdir(exist_ok=True)
            print("  ✓ Directory creation successful")

            return len(issues) == 0

        except Exception as e:
            print(f"  ✗ Configuration test failed: {e}")
            return False


async def main():
    """Run integration tests."""
    tester = IntegrationTester()
    success = await tester.run_all_tests()

    if success:
        print("\n🚀 System ready for autonomous operation!")
        print("Next steps:")
        print("  1. Start infrastructure: make docker-up")
        print("  2. Start API: make start-api")
        print("  3. Bootstrap agents: make bootstrap")
        return 0
    else:
        print("\n🛠️ Please fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
