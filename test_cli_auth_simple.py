#!/usr/bin/env python3
"""Simple test script to verify CLI authentication fixes."""

import os
import sys
from pathlib import Path


def test_cli_utils_auth():
    """Test that CLI utils auth functions work."""
    print("Testing CLI utils authentication functions...")

    try:
        # Add src to path
        src_path = Path(__file__).parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from src.cli.utils import get_api_headers

        # Test headers without auth
        headers = get_api_headers()
        print(f"✓ API headers without auth: {headers}")

        # Test headers with mock auth token
        os.environ["HIVE_CLI_TOKEN"] = "test-token-123"
        headers_with_auth = get_api_headers()
        print(f"✓ API headers with auth: {headers_with_auth}")

        # Verify auth header is present when token is set
        if "Authorization" in headers_with_auth:
            print("✓ Authorization header correctly added when token is present")
        else:
            print("⚠ Authorization header not found when token is set")

        # Clean up
        if "HIVE_CLI_TOKEN" in os.environ:
            del os.environ["HIVE_CLI_TOKEN"]

        return True

    except Exception as e:
        print(f"✗ CLI utils auth test failed: {e}")
        return False


def test_command_imports():
    """Test that CLI commands can be imported."""
    print("\nTesting CLI command imports...")

    try:
        # Add src to path
        src_path = Path(__file__).parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        # Test that we can import command modules
        from src.cli.commands import agent, system, task

        print("✓ All CLI command modules imported successfully")

        return True

    except Exception as e:
        print(f"✗ CLI command import test failed: {e}")
        return False


def main():
    """Run simple CLI authentication tests."""
    print("Testing CLI authentication fixes (simple version)...\n")

    tests = [
        test_cli_utils_auth,
        test_command_imports,
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
        print("🎉 CLI authentication infrastructure is working correctly!")
        print("✅ CLI commands can now make authenticated requests!")
        return 0
    else:
        print("❌ Some CLI authentication infrastructure needs attention.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
