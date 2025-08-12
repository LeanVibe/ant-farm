#!/usr/bin/env python3
"""Test script to verify authentication fixes for agent runner."""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))


def test_configuration_loading():
    """Test that configuration loads correctly."""
    print("Testing configuration loading...")

    try:
        from src.core.config import get_settings

        settings = get_settings()

        print(f"✓ Database URL: {settings.database_url}")
        print(f"✓ Redis URL: {settings.redis_url}")

        # Check that database URL uses correct port
        assert ":5433/" in settings.database_url or ":5433$" in settings.database_url, (
            "Database URL should use port 5433"
        )

        # Check that Redis URL uses correct port
        assert ":6381" in settings.redis_url, "Redis URL should use port 6381"

        print("✓ Configuration validation passed")
        return True

    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False


def test_database_connection():
    """Test database connection with parsed configuration."""
    print("\nTesting database connection...")

    try:
        from src.core.config import get_settings

        settings = get_settings()

        # Parse database URL
        database_url = settings.database_url
        print(f"Debug: database_url = {database_url}")

        # Handle asyncpg URLs
        if database_url.startswith("postgresql+asyncpg://"):
            db_part = database_url[21:]  # Remove "postgresql+asyncpg://"
        elif database_url.startswith("postgresql://"):
            db_part = database_url[13:]  # Remove "postgresql://"
        elif database_url.startswith("postgres://"):
            db_part = database_url[11:]  # Remove "postgres://"
        else:
            # Default fallback
            db_host, db_port, db_name, db_user, db_pass = (
                "localhost",
                5433,
                "leanvibe_hive",
                "hive_user",
                "hive_pass",
            )
            print(f"Debug: using default values")

        if "db_part" in locals():
            print(f"Debug: db_part = {db_part}")

            if "@" in db_part:
                user_pass, host_db = db_part.split("@", 1)
                if ":" in user_pass:
                    db_user, db_pass = user_pass.split(":", 1)
                else:
                    db_user, db_pass = user_pass, ""
            else:
                db_user, db_pass = "hive_user", "hive_pass"
                host_db = db_part

            print(f"Debug: user_pass = {db_user}:{db_pass}, host_db = {host_db}")

            if ":" in host_db:
                host_port, db_name = (
                    host_db.split("/", 1)
                    if "/" in host_db
                    else (host_db, "leanvibe_hive")
                )
                host_parts = host_port.split(":")
                if len(host_parts) == 2:
                    db_host, db_port_str = host_parts
                    try:
                        db_port = int(db_port_str)
                    except ValueError:
                        db_port = 5433
                else:
                    db_host = host_parts[0]
                    db_port = 5433
            else:
                host_db_parts = (
                    host_db.split("/", 1)
                    if "/" in host_db
                    else [host_db, "leanvibe_hive"]
                )
                db_host = host_db_parts[0]
                db_name = host_db_parts[1]
                db_port = 5433

            print(
                f"Debug: db_host = {db_host}, db_port = {db_port}, db_name = {db_name}"
            )

        # Test connection
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_pass,
            cursor_factory=RealDictCursor,
        )

        print(f"✓ Connected to database {db_name} at {db_host}:{db_port}")
        conn.close()
        return True

    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


def test_redis_connection():
    """Test Redis connection with parsed configuration."""
    print("\nTesting Redis connection...")

    try:
        from src.core.config import get_settings

        settings = get_settings()

        # Parse Redis URL
        redis_url = settings.redis_url
        if redis_url.startswith("redis://"):
            redis_part = redis_url[8:]
            if "/" in redis_part:
                host_port = redis_part.split("/")[0]
            else:
                host_port = redis_part

            if ":" in host_port:
                host, port_str = host_port.split(":", 1)
                try:
                    port = int(port_str)
                except ValueError:
                    port = 6381
            else:
                host, port = host_port, 6381

        # Test connection
        import redis

        r = redis.Redis(host=host, port=port, decode_responses=True)
        r.ping()

        print(f"✓ Connected to Redis at {host}:{port}")
        r.close()
        return True

    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        return False


def test_agent_runner_parsing():
    """Test that agent runner can parse configuration correctly."""
    print("\nTesting agent runner configuration parsing...")

    try:
        # Import the parsing functions from agent runner
        sys.path.insert(0, str(Path(__file__).parent))

        from agent_runner import AgentRunner

        # Test database URL parsing
        db_url = "postgresql://hive_user:hive_pass@localhost:5433/leanvibe_hive"
        runner = AgentRunner("test", "test-runner")
        db_host, db_port, db_name, db_user, db_pass = runner._parse_database_url(db_url)

        assert db_host == "localhost"
        assert db_port == 5433
        assert db_name == "leanvibe_hive"
        assert db_user == "hive_user"
        assert db_pass == "hive_pass"

        print("✓ Database URL parsing works correctly")

        # Test Redis URL parsing
        redis_url = "redis://localhost:6381"
        redis_host, redis_port = runner._parse_redis_url(redis_url)

        assert redis_host == "localhost"
        assert redis_port == 6381

        print("✓ Redis URL parsing works correctly")
        return True

    except Exception as e:
        print(f"✗ Agent runner parsing failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing authentication fixes for agent runner...\n")

    tests = [
        test_configuration_loading,
        test_database_connection,
        test_redis_connection,
        test_agent_runner_parsing,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All authentication fixes are working correctly!")
        return 0
    else:
        print("❌ Some authentication fixes need attention.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
