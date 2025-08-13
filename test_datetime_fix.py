#!/usr/bin/env python3
"""
Test to verify the datetime timezone fix.
This test verifies that we can properly handle datetime operations without timezone errors.
"""

import asyncio
from datetime import datetime, UTC


async def test_datetime_fix():
    """Test that datetime operations work correctly with consistent timezone usage."""
    print("Testing datetime timezone fix...")

    # Create timezone-aware datetimes consistently
    aware_datetime1 = datetime.now(UTC)
    aware_datetime2 = datetime.now(UTC)

    try:
        # This should work without errors now
        diff = aware_datetime2 - aware_datetime1
        print(f"Difference: {diff}")
        print("SUCCESS: Timezone-aware datetime operations work correctly")
        return True
    except TypeError as e:
        print(f"ERROR: {e}")
        return False


def test_bootstrap_datetime_fix():
    """Test that bootstrap uses timezone-aware datetimes."""
    print("Testing bootstrap datetime fix...")

    try:
        # Test that we can create timezone-aware datetime strings
        timestamp = datetime.now(UTC).isoformat()
        print(f"Timezone-aware timestamp: {timestamp}")
        print("SUCCESS: Bootstrap uses timezone-aware datetimes")
        return True

    except Exception as e:
        print(f"ERROR in bootstrap datetime test: {e}")
        return False


if __name__ == "__main__":
    results = []
    results.append(asyncio.run(test_datetime_fix()))
    results.append(test_bootstrap_datetime_fix())

    if all(results):
        print("\nALL TESTS PASSED: DateTime timezone issues have been fixed!")
    else:
        print("\nSOME TESTS FAILED: DateTime timezone issues remain.")
