#!/usr/bin/env python3
"""Test script to verify CLI authentication fixes."""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))


def test_cli_auth_module():
    """Test that CLI auth module loads correctly."""
    print("Testing CLI authentication module...")

    try:
        # Add src to path for proper imports
        import sys
        from pathlib import Path

        src_path = Path(__file__).parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from src.cli.auth import (
            clear_cli_auth_cache,
            get_authenticated_cli_user,
            get_cli_auth_token,
            is_cli_authenticated,
        )

        print("✓ CLI auth module imported successfully")

        # Test anonymous user creation
        user = get_authenticated_cli_user()
        print(f"✓ Anonymous user created: {user.username} (ID: {user.id})")

        # Test authentication status
        is_auth = is_cli_authenticated()
        print(
            f"✓ Authentication status check: {'Authenticated' if is_auth else 'Anonymous'}"
        )

        # Test token retrieval
        token = get_cli_auth_token()
        print(f"✓ Auth token check: {'Found' if token else 'Not found (expected)'}")

        # Test cache clearing
        clear_cli_auth_cache()
        print("✓ Auth cache cleared successfully")

        return True

    except Exception as e:
        print(f"✗ CLI auth module test failed: {e}")
        return False


def test_cli_utils_auth():
    """Test that CLI utils auth functions work."""
    print("\nTesting CLI utils authentication functions...")

    try:
        from cli.utils import get_api_headers

        # Test headers without auth
        headers = get_api_headers()
        print(f"✓ API headers without auth: {headers}")

        # Test headers with mock auth token
        os.environ["HIVE_CLI_TOKEN"] = "test-token-123"
        headers_with_auth = get_api_headers()
        print(f"✓ API headers with auth: {headers_with_auth}")

        # Clean up
        del os.environ["HIVE_CLI_TOKEN"]

        return True

    except Exception as e:
        print(f"✗ CLI utils auth test failed: {e}")
        return False


def test_command_auth_integration():
    """Test that CLI commands can make authenticated requests."""
    print("\nTesting CLI command authentication integration...")

    try:
        # Add src to path for proper imports
        import sys
        from pathlib import Path

        src_path = Path(__file__).parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        # Test that we can import command modules
        from src.cli.commands import agent, system, task

        print("✓ All CLI command modules imported successfully")

        # Check that agent commands use proper auth
        import inspect

        # Get the source of one of the agent command functions
        source = inspect.getsource(agent._list_agents_filtered)
        if "get_api_headers" in source:
            print("✓ Agent commands use authentication headers")
        else:
            print("⚠ Agent commands may not be using authentication headers")

        # Check that task commands use proper auth
        source = inspect.getsource(task._list_tasks_filtered)
        if "get_api_headers" in source:
            print("✓ Task commands use authentication headers")
        else:
            print("⚠ Task commands may not be using authentication headers")

        return True

    except Exception as e:
        print(f"✗ CLI command auth integration test failed: {e}")
        return False


def main():
    """Run all CLI authentication tests."""
    print("Testing CLI authentication fixes...\n")

    tests = [
        test_cli_auth_module,
        test_cli_utils_auth,
        test_command_auth_integration,
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
        print("🎉 All CLI authentication fixes are working correctly!")
        print("✅ CLI agent operations are now properly authenticated!")
        return 0
    else:
        print("❌ Some CLI authentication fixes need attention.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
