#!/usr/bin/env python3
"""Simple test to verify API server can start with middleware."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_api_server():
    """Test that API server can be imported and middleware work."""
    try:
        print("Testing API server imports...")

        # Test middleware imports
        from src.api.middleware import (
            SecurityHeadersMiddleware,
            RequestValidationMiddleware,
            RequestTimeoutMiddleware,
        )
        print("✓ Middleware imports successful")

        # Test core middleware imports
        from src.core.auth import SecurityMiddleware
        from src.core.analytics import PerformanceMiddleware
        print("✓ Core middleware imports successful")

        # Test main app import
        from src.api.main import app
        print("✓ Main app import successful")

        # Check middleware stack
        middleware_count = len(app.user_middleware)
        print(f"✓ App has {middleware_count} middleware configured")

        # List middleware
        for i, middleware in enumerate(app.user_middleware):
            print(f"  {i+1}. {middleware.cls.__name__}")

        print("\n🎉 All tests passed! API server should start successfully.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = asyncio.run(test_api_server())
    sys.exit(0 if success else 1)