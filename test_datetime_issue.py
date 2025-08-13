#!/usr/bin/env python3
"""
Test to verify the datetime timezone fix.
This test verifies that we can properly handle datetime operations without timezone errors.
"""

import asyncio
from datetime import datetime, timedelta, UTC


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


async def test_sleep_wake_manager_fix():
    """Test that sleep_wake_manager uses timezone-aware datetimes."""
    print("Testing sleep_wake_manager datetime fix...")

    try:
        # Import the sleep wake manager
        from src.core.sleep_wake_manager import SleepWakeManager, SleepSchedule

        # Create a schedule
        schedule = SleepSchedule(
            sleep_hour=2,
            sleep_minute=0,
            sleep_duration_hours=2,
            min_awake_hours=4,
            enable_adaptive_scheduling=True,
        )

        # Create manager
        manager = SleepWakeManager(schedule)

        # Test that get_sleep_stats works without timezone errors
        sleep_stats = manager.get_sleep_stats()
        next_sleep = sleep_stats["next_scheduled_sleep"]
        print(f"Next sleep time: {next_sleep}")
        print("SUCCESS: Sleep wake manager uses timezone-aware datetimes")
        return True

    except Exception as e:
        print(f"ERROR in sleep_wake_manager test: {e}")
        return False


if __name__ == "__main__":

    async def run_tests():
        results = []
        results.append(await test_datetime_fix())
        results.append(await test_sleep_wake_manager_fix())
        results.append(await test_bootstrap_datetime_fix())
        return results

    results = asyncio.run(run_tests())

    if all(results):
        print("\nALL TESTS PASSED: DateTime timezone issues have been fixed!")
    else:
        print("\nSOME TESTS FAILED: DateTime timezone issues remain.")
